# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Memory-efficient chunked output projection and cross-entropy.

The full ``[batch, sequence, vocabulary]`` logits tensor is never
materialized. Output projection, FP32 cross-entropy, and first-order
gradients are evaluated one local sequence chunk at a time.

When the LM head is vocabulary sharded (``tp_size > 1``) the local weight holds
one vocabulary slice, so the cross-entropy normalization spans the
tensor-parallel group. That path takes the tensor-parallel mesh from the
caller and reuses the vocab-parallel kernel in
:mod:`hyper_parallel.components.losses._vocab_parallel_cross_entropy`.
"""

from __future__ import annotations

__all__ = [
    "ChunkedCausalLMLoss",
    "ChunkedCausalLMOutput",
    "chunked_cross_entropy",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# This loss is a PyTorch Trainer component and differentiates through
# torch.func; there is no platform-neutral autograd equivalent. PyTorch's
# Function metaclass supplies its public variadic dispatch contract.
# pylint: disable=forbidden-backend-import,not-callable,abstract-method,arguments-differ
import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional
from transformers.utils import ModelOutput

from hyper_parallel.components.losses._vocab_parallel_cross_entropy import (
    _resolve_class_mesh_dim,  # pylint: disable=protected-access
    vocab_parallel_cross_entropy_local,
)
from hyper_parallel.core.dtensor._utils import differentiable_all_reduce

# Model attribute holding the mesh that shards the bound LM head vocabulary.
_CHUNK_LOSS_TP_MESH_ATTR = "_hp_chunk_loss_tp_mesh"


@dataclass(frozen=True)
class _VocabParallelContext:
    """Vocabulary sharding of the LM-head weight used by one Chunk Loss call."""

    mesh: Any
    mesh_dim: int
    group: Any
    vocab_size: int
    tp_size: int


def _linear_cross_entropy_chunk(
    hidden_chunk: torch.Tensor,
    weight: torch.Tensor,
    target_chunk: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Return the FP32 summed cross-entropy for one sequence chunk."""
    logits = functional.linear(
        hidden_chunk.reshape(-1, hidden_chunk.size(-1)),
        weight,
    ).float()
    return functional.cross_entropy(
        logits,
        target_chunk.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )


def _vocab_parallel_chunk_loss(
    hidden_chunk: torch.Tensor,
    weight_shard: torch.Tensor,
    target_chunk: torch.Tensor,
    ignore_index: int,
    context: _VocabParallelContext,
) -> torch.Tensor:
    """Return the FP32 cross-entropy sum of one chunk over a sharded vocabulary.

    The local logits cover this rank's vocabulary slice only; the vocabulary
    normalization of the softmax and the target term are reduced over the
    tensor-parallel group, so the returned scalar is the same full-vocabulary
    sum on every rank of that group.
    """
    logits = functional.linear(
        hidden_chunk.reshape(-1, hidden_chunk.size(-1)),
        weight_shard,
    ).float()
    return vocab_parallel_cross_entropy_local(
        logits,
        target_chunk.reshape(-1),
        vocab_size=context.vocab_size,
        mesh=context.mesh,
        mesh_dim=context.mesh_dim,
        ignore_index=ignore_index,
        reduction="sum",
    ).reshape(())


def _chunk_grad_and_value(
    hidden_chunk: torch.Tensor,
    head_weight: torch.Tensor,
    target_chunk: torch.Tensor,
    ignore_index: int,
    context: _VocabParallelContext | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one chunk's ``(dHidden, dWeight, summed loss)``.

    The single-rank path keeps the ``torch.func`` transform. The vocabulary
    sharded path cannot: the cross-entropy reductions there are custom
    autograd Functions, which functorch transforms reject. Plain autograd over
    the chunk graph yields the same first-order gradients, and the chunk graph
    is released as soon as the gradients are taken, exactly as the transform
    does.
    """
    if context is None:
        (grad_hidden, grad_weight), chunk_loss = torch.func.grad_and_value(
            _linear_cross_entropy_chunk,
            argnums=(0, 1),
        )(hidden_chunk, head_weight, target_chunk, ignore_index)
        return grad_hidden, grad_weight, chunk_loss

    with torch.enable_grad():
        hidden_input = hidden_chunk.detach().requires_grad_(True)
        weight_input = head_weight.detach().requires_grad_(True)
        chunk_loss = _vocab_parallel_chunk_loss(
            hidden_input,
            weight_input,
            target_chunk,
            ignore_index,
            context,
        )
        grad_hidden, grad_weight = torch.autograd.grad(
            chunk_loss,
            (hidden_input, weight_input),
        )
    # Rank r only differentiates its own vocabulary slice, so its activation
    # gradient is a partial contribution. This loss keeps the Trainer's
    # non-loss-parallel contract of an LM head whose loss is identical on every
    # TP rank, which requires the full sum of those contributions.
    grad_hidden = differentiable_all_reduce(
        grad_hidden,
        op="sum",
        group=context.group,
    )
    return grad_hidden, grad_weight.detach(), chunk_loss.detach()


class _PrecomputedChunkLoss(torch.autograd.Function):
    """Precompute per-chunk gradients and replay them in outer backward."""

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx: Any,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
        ignore_index: int,
        vocab_context: _VocabParallelContext | None = None,
    ) -> torch.Tensor:
        """Compute the loss sum and retain accumulated first-order gradients.

        Args:
            ctx: Autograd context receiving the precomputed gradients.
            hidden_states: Local hidden states in ``[B, S, H]`` layout.
            head_weight: Vocabulary output weight or its local shard.
            targets: Integer targets aligned with ``hidden_states``.
            chunk_size: Maximum local sequence length evaluated per chunk.
            ignore_index: Target value excluded from the loss.
            vocab_context: Vocabulary sharding of ``head_weight``, or ``None``
                when it holds the whole vocabulary.

        Returns:
            The summed FP32 loss, disconnected from the chunk graphs.
        """
        grad_hidden = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)

        hidden_chunks = torch.split(hidden_states, chunk_size, dim=1)
        target_chunks = torch.split(targets, chunk_size, dim=1)
        grad_hidden_chunks = torch.split(grad_hidden, chunk_size, dim=1)
        for hidden_chunk, target_chunk, grad_hidden_chunk in zip(
                hidden_chunks, target_chunks, grad_hidden_chunks, strict=True
        ):
            chunk_grad_hidden, chunk_grad_weight, chunk_loss = _chunk_grad_and_value(
                hidden_chunk,
                head_weight,
                target_chunk,
                ignore_index,
                vocab_context,
            )
            grad_hidden_chunk.copy_(chunk_grad_hidden)
            grad_weight.add_(chunk_grad_weight)
            loss_sum.add_(chunk_loss)
            del chunk_grad_hidden, chunk_grad_weight, chunk_loss

        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss_sum

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_loss_sum: torch.Tensor | None) -> tuple:
        """Replay private gradients with the Trainer's upstream scale.

        Args:
            ctx: Autograd context holding the precomputed gradients.
            grad_loss_sum: Upstream gradient of the summed loss.

        Returns:
            Gradients for ``hidden_states`` and ``head_weight``, then ``None``
            for the non-differentiable arguments.
        """
        if grad_loss_sum is None:
            return None, None, None, None, None, None
        grad_hidden, grad_weight = ctx.saved_tensors
        return (
            grad_hidden * grad_loss_sum,
            grad_weight * grad_loss_sum,
            None,
            None,
            None,
            None,
        )


def _validate_chunk_options(chunk_size: int, ignore_index: int) -> None:
    """Validate scalar chunk-loss options."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(
            f"chunk_size must be an integer, got {type(chunk_size).__name__}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError(
            f"ignore_index must be an integer, got {type(ignore_index).__name__}"
        )


def _validate_chunk_shapes(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    head_weight: torch.Tensor,
) -> None:
    """Validate tensor ranks and logical dimensions."""
    if hidden_states.dim() != 3:
        raise ValueError(
            "hidden_states must have shape [batch, sequence, hidden], "
            f"got {tuple(hidden_states.shape)}"
        )
    if targets.dim() != 2 or hidden_states.shape[:2] != targets.shape:
        raise ValueError(
            "targets must match hidden_states batch/sequence dimensions, "
            f"got hidden={tuple(hidden_states.shape)} and targets={tuple(targets.shape)}"
        )
    if hidden_states.size(1) == 0 or hidden_states.size(2) == 0:
        raise ValueError(
            "hidden_states sequence and hidden dimensions must be non-empty"
        )
    if head_weight.dim() != 2 or head_weight.size(1) != hidden_states.size(2):
        raise ValueError(
            "head_weight must have shape [vocabulary, hidden] matching "
            f"hidden_states, got hidden={tuple(hidden_states.shape)} and "
            f"weight={tuple(head_weight.shape)}"
        )
    if head_weight.size(0) == 0:
        raise ValueError("head_weight vocabulary dimension must be non-empty")


def _validate_chunk_dtypes_and_device(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    head_weight: torch.Tensor,
) -> None:
    """Validate tensor dtypes and device ownership."""
    if not torch.is_floating_point(hidden_states) or not torch.is_floating_point(
        head_weight
    ):
        raise TypeError("hidden_states and head_weight must be floating-point tensors")
    if targets.dtype != torch.long:
        raise TypeError(f"targets must have dtype torch.long, got {targets.dtype}")
    if hidden_states.dtype != head_weight.dtype:
        raise TypeError(
            "hidden_states and head_weight must have the same dtype, "
            f"got {hidden_states.dtype} and {head_weight.dtype}"
        )
    if not hidden_states.device == targets.device == head_weight.device:
        raise ValueError(
            "hidden_states, targets, and head_weight must be on the same device, "
            f"got {hidden_states.device}, {targets.device}, and {head_weight.device}"
        )


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    head_weight: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    tp_mesh: Any = None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute summed linear cross-entropy without full-sequence logits.

    Args:
        hidden_states: Local hidden states in ``[B, S, H]`` layout.
        targets: Integer targets aligned with ``hidden_states`` in ``[B, S]``.
        head_weight: Vocabulary output weight in ``[V, H]`` layout; the local
            slice ``[V_local, H]`` when ``tp_mesh`` shards the vocabulary.
        chunk_size: Maximum local sequence length evaluated per chunk.
        ignore_index: Target value excluded from the summed loss.
        tp_mesh: Mesh whose single axis shards ``head_weight``'s vocabulary, or
            ``None`` for a full-vocabulary weight. A mesh with a single rank
            keeps the plain single-rank path.
        vocab_size: Global vocabulary size; defaults to ``V_local * tp_size``,
            which is exact for the even split the sharding planner produces.

    Returns:
        A scalar FP32 loss sum connected to ``hidden_states`` and
        ``head_weight``. With a sharded vocabulary the sum covers the whole
        vocabulary and is identical on every rank of ``tp_mesh``.

    Raises:
        ValueError: If the tensor-parallel mesh does not hold exactly one
            vocabulary-sharding axis, or the vocabulary is split unevenly.
    """
    _validate_chunk_options(chunk_size, ignore_index)
    _validate_chunk_shapes(hidden_states, targets, head_weight)
    _validate_chunk_dtypes_and_device(hidden_states, targets, head_weight)
    vocab_context = _resolve_vocab_parallel_context(head_weight, tp_mesh, vocab_size)
    return _PrecomputedChunkLoss.apply(
        hidden_states,
        head_weight,
        targets,
        chunk_size,
        ignore_index,
        vocab_context,
    )


def _resolve_vocab_parallel_context(
    head_weight: torch.Tensor,
    tp_mesh: Any,
    vocab_size: int | None,
) -> _VocabParallelContext | None:
    """Resolve the vocabulary sharding of ``head_weight``.

    Args:
        head_weight: Local LM-head weight, ``[V_local, H]``.
        tp_mesh: Mesh whose single axis shards the vocabulary, or ``None``.
        vocab_size: Global vocabulary size, or ``None`` to derive it.

    Returns:
        The vocabulary-parallel context, or ``None`` when the weight holds the
        whole vocabulary and the plain reduction order must be kept.

    Raises:
        ValueError: If no unique vocabulary-sharding axis can be identified, or
            the given global vocabulary size is not an even multiple of the
            local weight.
    """
    if tp_mesh is None:
        return None
    mesh_dim = _resolve_class_mesh_dim(tp_mesh, None)
    tp_size = int(tp_mesh.size(mesh_dim))
    if tp_size <= 1:
        return None
    local_vocab_size = int(head_weight.size(0))
    expected_vocab_size = local_vocab_size * tp_size
    if vocab_size is None:
        vocab_size = expected_vocab_size
    elif vocab_size != expected_vocab_size:
        raise ValueError(
            "chunked_cross_entropy requires an even vocabulary split: the "
            f"local weight holds {local_vocab_size} rows but vocab_size="
            f"{vocab_size} with tp_size={tp_size} does not describe that shard"
        )
    return _VocabParallelContext(
        mesh=tp_mesh,
        mesh_dim=mesh_dim,
        group=tp_mesh.get_group(mesh_dim),
        vocab_size=vocab_size,
        tp_size=tp_size,
    )


def _get_output_value(model_output: Any, name: str) -> Any:
    """Read one optional value from a mapping or Transformers output."""
    if isinstance(model_output, Mapping):
        return model_output.get(name)
    return getattr(model_output, name, None)


@dataclass
class ChunkedCausalLMOutput(ModelOutput):
    """Training output produced by a model-integrated Chunk Loss adapter."""

    loss_sum: torch.Tensor | None = None
    valid_token_count: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    aux_loss_coef: float | None = None
    logits: torch.Tensor | None = None
    past_key_values: Any | None = None
    hidden_states: Any | None = None
    attentions: Any | None = None
    router_logits: Any | None = None


class ChunkedCausalLMLoss(nn.Module):
    """Trainer-facing model-integrated Chunk Loss objective.

    The configured model-family adapter consumes the prepared model inputs,
    calls :func:`chunked_cross_entropy` before a full LM-head output exists,
    and returns :class:`ChunkedCausalLMOutput`. This module converts the local
    summed CE to one local token mean; global DP/CP weighting remains owned by
    ``mean_global_loss``.

    A vocabulary-sharded LM head (``tp_size > 1`` without loss parallelism) is
    supported: the adapter reaches the tensor-parallel mesh recorded by
    :meth:`bind_model` through :func:`chunk_loss_tp_mesh`, and the chunked
    cross-entropy then reduces the softmax normalization and the target term
    over that group.
    """

    def __init__(self, chunk_size: int = 1024, ignore_index: int = -100) -> None:
        """Initialize and validate the per-rank sequence chunk options."""
        super().__init__()
        _validate_chunk_options(chunk_size, ignore_index)
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self._model_is_bound = False

    @staticmethod
    def _parallel_size(distributed_setup: Any, name: str) -> int:
        """Read one topology size from the shared mesh context."""
        mesh = getattr(distributed_setup, "mesh_context", None)
        return int(getattr(mesh, name, 1))

    @staticmethod
    def _resolve_tp_mesh(distributed_setup: Any) -> Any:
        """Return the mesh that shards the LM-head vocabulary, or ``None``.

        Args:
            distributed_setup: Shared Trainer distributed setup.

        Returns:
            The ``tp`` sub-mesh when ``tp_size > 1``, otherwise ``None``.

        Raises:
            NotImplementedError: If pipeline or loss parallelism is active, or
                sequence parallelism shares the ``tp`` axis with the
                vocabulary.
            ValueError: If tensor parallelism is active without a device mesh
                carrying a ``tp`` axis.
        """
        mesh_context = getattr(distributed_setup, "mesh_context", None)
        tp_size = ChunkedCausalLMLoss._parallel_size(distributed_setup, "tp_size")
        pp_size = ChunkedCausalLMLoss._parallel_size(distributed_setup, "pp_size")
        loss_parallel = bool(getattr(mesh_context, "loss_parallel", False))
        if pp_size != 1 or loss_parallel:
            raise NotImplementedError(
                "ChunkedCausalLMLoss currently requires pp_size=1 and "
                f"loss_parallel=false; got pp_size={pp_size}, "
                f"loss_parallel={loss_parallel}"
            )
        if tp_size == 1:
            return None
        if bool(getattr(mesh_context, "sequence_parallel", False)):
            # Sequence parallelism spreads different tokens over the tp axis
            # while the vocabulary is sharded over that same axis, so the
            # cross-rank normalization would mix unrelated tokens.
            raise NotImplementedError(
                "ChunkedCausalLMLoss requires sequence_parallel=false with "
                f"tp_size={tp_size}: the vocabulary shards would otherwise "
                "have to normalize different sequence slices"
            )

        device_mesh = getattr(mesh_context, "device_mesh", None)
        mesh_dim_names = tuple(getattr(device_mesh, "mesh_dim_names", None) or ())
        if device_mesh is None or "tp" not in mesh_dim_names:
            raise ValueError(
                "ChunkedCausalLMLoss needs a DeviceMesh carrying a 'tp' axis to "
                f"locate the vocabulary shards when tp_size={tp_size}; got "
                f"mesh_dim_names={mesh_dim_names}"
            )
        tp_mesh = device_mesh["tp"]
        if int(tp_mesh.size()) != tp_size:
            raise ValueError(
                f"'tp' mesh axis holds {int(tp_mesh.size())} ranks but "
                f"mesh_context.tp_size is {tp_size}"
            )
        return tp_mesh

    def bind_model(self, model: nn.Module, distributed_setup: Any = None) -> None:
        """Bind the registered model-family Chunk Loss forward adapter.

        The vocabulary-sharding mesh resolved here is recorded on the model so
        :func:`chunk_loss_tp_mesh` can hand it to the adapter's
        :func:`chunked_cross_entropy` call.

        Args:
            model: Fully constructed model whose parameter structure is fixed.
            distributed_setup: Shared Trainer distributed setup.

        Raises:
            NotImplementedError: If PP or loss parallelism is active.
            ValueError: If the model family has no Chunk Loss adapter, or TP is
                active without a mesh carrying a ``tp`` axis.
        """
        # Trainer-side token aggregation currently follows the data contract's
        # canonical ignore value. The model-integrated objective must use the
        # same value or DP/CP weighting would count a different token set.
        from hyper_parallel.data.constants import IGNORE_INDEX  # pylint: disable=import-outside-toplevel

        if self.ignore_index != IGNORE_INDEX:
            raise ValueError(
                "ChunkedCausalLMLoss Trainer integration requires "
                f"ignore_index={IGNORE_INDEX}, got {self.ignore_index}"
            )

        setattr(model, _CHUNK_LOSS_TP_MESH_ATTR, self._resolve_tp_mesh(distributed_setup))

        # Lazy imports keep generic loss-package import independent of concrete
        # model families and their optional backend dependencies.
        from hyper_parallel.models.registry import get_model_adapter  # pylint: disable=import-outside-toplevel

        model_type = getattr(getattr(model, "config", None), "model_type", "")
        adapter_spec = get_model_adapter(model_type)
        provider = None if adapter_spec is None else adapter_spec.loss
        if provider is None:
            raise ValueError(
                "ChunkedCausalLMLoss requires a registered model-family loss "
                f"adapter, but model_type={model_type!r} provides none"
            )
        adapter = provider()
        bind_chunk_loss = getattr(adapter, "bind_chunk_loss", None)
        if not callable(bind_chunk_loss):
            raise TypeError(
                f"model_type={model_type!r} loss adapter must expose bind_chunk_loss(model)"
            )
        bind_chunk_loss(model)
        self._model_is_bound = True

    def prepare_model_inputs(
        self,
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Move labels to the model-family no-full-logits protocol."""
        if not self._model_is_bound:
            raise RuntimeError("ChunkedCausalLMLoss must be bound before model forward")
        targets = loss_inputs.get("shift_labels")
        loss_mask = loss_inputs.get("loss_mask")
        if not isinstance(targets, torch.Tensor):
            targets, loss_mask = self._shift_targets(loss_inputs)
        if loss_mask is not None:
            if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must be a Tensor with the same shape as targets")

        prepared = dict(model_inputs)
        # A legacy one-dictionary caller may leave loss-only fields in
        # model_inputs. Replace them with the adapter's explicit protocol so
        # labels and masks never leak into the decoder's generic **kwargs.
        prepared.pop("labels", None)
        prepared.pop("shift_labels", None)
        prepared.pop("loss_mask", None)
        prepared.update(
            {
                "chunk_loss_targets": targets,
                "chunk_loss_mask": loss_mask,
                "chunk_loss_chunk_size": self.chunk_size,
                "chunk_loss_ignore_index": self.ignore_index,
            }
        )
        return prepared

    @staticmethod
    def _causal_shift(values: torch.Tensor, fill_value: Any) -> torch.Tensor:
        """Return one next-token target per position, padded at the tail."""
        return functional.pad(values, (0, 1), value=fill_value)[..., 1:].contiguous()

    def _shift_targets(
        self,
        loss_inputs: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Any]:
        """Derive shifted targets from a batch that still carries raw labels.

        Batch producers that pre-shift publish ``shift_labels`` directly (see
        ``data/batching/get_batch.py``). The temporary VLM path keeps raw
        ``labels``, so the causal next-token shift its own loss applies — drop
        position 0 and pad the tail with the ignore value — is reproduced here,
        together with the matching mask alignment. Only the values handed to
        ``chunked_cross_entropy`` are affected; token accounting keeps reading
        the untouched ``loss_inputs``.
        """
        labels = loss_inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise ValueError(
                "ChunkedCausalLMLoss requires pre-shifted shift_labels or raw "
                "labels aligned with every local hidden position"
            )
        loss_mask = loss_inputs.get("loss_mask")
        if loss_mask is not None and (
            not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != labels.shape
        ):
            raise ValueError("loss_mask must be a Tensor with the same shape as labels")
        targets = self._causal_shift(labels, self.ignore_index)
        mask = None if loss_mask is None else self._causal_shift(loss_mask.to(torch.bool), False)
        return targets, mask

    def forward(  # pylint: disable=unused-argument
        self,
        *,
        model_output: Any,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize one adapter-produced CE sum exactly once."""
        logits = _get_output_value(model_output, "logits")
        if logits is not None:
            raise ValueError(
                "ChunkedCausalLMLoss model adapter materialized logits; expected logits=None"
            )
        loss_sum = _get_output_value(model_output, "loss_sum")
        valid_token_count = _get_output_value(model_output, "valid_token_count")
        if not isinstance(loss_sum, torch.Tensor) or loss_sum.numel() != 1:
            raise ValueError("ChunkedCausalLMLoss requires a scalar Tensor loss_sum")
        if not isinstance(valid_token_count, torch.Tensor) or valid_token_count.numel() != 1:
            raise ValueError(
                "ChunkedCausalLMLoss requires a scalar Tensor valid_token_count"
            )

        denominator = valid_token_count.clamp_min(1).to(loss_sum.dtype)
        local_loss = loss_sum / denominator
        aux_loss = _get_output_value(model_output, "aux_loss")
        if aux_loss is not None:
            if not isinstance(aux_loss, torch.Tensor) or aux_loss.numel() != 1:
                raise ValueError("ChunkedCausalLMLoss aux_loss must be a scalar Tensor")
            aux_loss_coef = _get_output_value(model_output, "aux_loss_coef")
            local_loss = local_loss + float(
                0.0 if aux_loss_coef is None else aux_loss_coef
            ) * aux_loss
        return torch.where(
            valid_token_count.ne(0),
            local_loss,
            torch.zeros_like(local_loss),
        )


def chunk_loss_tp_mesh(model: nn.Module) -> Any:
    """Return the mesh that shards ``model``'s LM head vocabulary.

    :meth:`ChunkedCausalLMLoss.bind_model` records the tensor-parallel mesh of
    a vocabulary-sharded LM head on the bound model; model-family adapters read
    it back and pass it to :func:`chunked_cross_entropy` as ``tp_mesh``.

    Args:
        model: Model bound by :class:`ChunkedCausalLMLoss`.

    Returns:
        The ``tp`` sub-mesh, or ``None`` for a full-vocabulary LM head.
    """
    return getattr(model, _CHUNK_LOSS_TP_MESH_ATTR, None)


__all__ = [
    "ChunkedCausalLMLoss",
    "ChunkedCausalLMOutput",
    "chunk_loss_tp_mesh",
    "chunked_cross_entropy",
]
