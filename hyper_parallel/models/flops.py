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
"""FLOPs-per-token estimation from model-configuration geometry.

Backend-free, pure-Python estimator used by the trainer's throughput / MFU
metrics. It reads HF-style config attributes (a config object or a plain
mapping both work) and applies the standard 6N convention:

- every matmul parameter contributes ``6 * params`` FLOPs per token
  (forward + backward);
- MoE models count only the *activated* experts (top-k routed + shared),
  plus the router gate projection;
- the LM-head logits projection (``hidden x vocab``) is included;
- when ``seq_len`` is known, the attention score/weight matmuls add
  ``6 * layers * heads * (qk_head_dim + v_head_dim) * seq_len`` per token.
  Following the torchtitan convention, causal-attention sparsity is NOT
  accounted for;
- activation-checkpoint recompute is never counted as useful FLOPs.

Multi-head lat attention (MLA, DeepSeek-V2/V3 and Kimi families) and GQA/MHA
projection layouts are both recognized; MoE fields from the DeepSeek
(``n_routed_experts`` / ``n_shared_experts`` / ``first_k_dense_replace``) and
Qwen-MoE (``num_experts`` / ``num_experts_per_tok`` /
``shared_expert_intermediate_size`` / ``decoder_sparse_step``) conventions are
both accepted. The estimate is architecture-agnostic and approximate by
design — model families that need an exact value may expose
``model.hp_flops_per_token``, which :func:`resolve_flops_per_token` prefers.

Vision towers are accounted separately, because their cost scales with the image
patches in the batch rather than with ``input_ids``: :func:`estimate_vision_flops`
returns a per-patch cost model that the environment meter evaluates against the
patches actually seen in each step.  Without it a VLM step that pushes 10 images
(16,560 patches) through the tower per rank is charged only for its language
tokens, which understates TFLOPS/MFU/HFU and makes image-heavy and text-only
configurations incomparable.
"""

from collections.abc import Mapping
from typing import Any, NamedTuple, Optional


_LAYER_FIELDS = ("num_hidden_layers", "n_layers", "num_layers")
_HIDDEN_FIELDS = ("hidden_size", "dim", "d_model", "n_embd")
_HEAD_FIELDS = ("num_attention_heads", "n_heads", "n_head")
_KV_HEAD_FIELDS = ("num_key_value_heads", "n_kv_heads")
_ROUTED_EXPERT_FIELDS = ("n_routed_experts", "num_experts")
_TOPK_FIELDS = ("num_experts_per_tok", "moe_topk", "moe_router_topk", "top_k", "topk")

# Vision-tower geometry.  Multimodal configs keep it in a sub-config whose field
# names are family specific: Kimi-K2.x uses ``vt_*``, most others HF defaults.
_VISION_CONFIG_FIELDS = ("vision_config", "vision_tower_config", "vision_encoder_config")
_VISION_LAYER_FIELDS = ("vt_num_hidden_layers", "num_hidden_layers", "n_layers", "depth", "num_layers")
_VISION_HIDDEN_FIELDS = ("vt_hidden_size", "hidden_size", "mm_hidden_size", "embed_dim", "dim")
_VISION_INTER_FIELDS = ("vt_intermediate_size", "intermediate_size", "mlp_hidden_size", "ffn_dim")
_VISION_HEAD_FIELDS = ("vt_num_attention_heads", "num_attention_heads", "n_heads", "num_heads")


def _read(config: Any, *names: str) -> Optional[Any]:
    """Return the first present, non-None field from a config object or mapping."""
    for name in names:
        if isinstance(config, Mapping):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _read_float(config: Any, *names: str) -> Optional[float]:
    """Return the first present field coerced to ``float``, else ``None``."""
    value = _read(config, *names)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attention_geometry(config: Any) -> Optional[tuple[int, int, int, float]]:
    """Return ``(num_heads, qk_head_dim, v_head_dim, per-layer projection params)``."""
    hidden = _read_float(config, *_HIDDEN_FIELDS)
    heads = _read_float(config, *_HEAD_FIELDS)
    if not hidden or not heads:
        return None
    heads = int(heads)
    kv_lora_rank = _read_float(config, "kv_lora_rank")
    if kv_lora_rank:
        qk_nope_dim = _read_float(config, "qk_nope_head_dim") or 0.0
        qk_rope_dim = _read_float(config, "qk_rope_head_dim") or 0.0
        qk_dim = qk_nope_dim + qk_rope_dim
        v_dim = _read_float(config, "v_head_dim") or qk_dim
        q_lora_rank = _read_float(config, "q_lora_rank")
        if q_lora_rank:
            q_params = hidden * q_lora_rank + q_lora_rank * heads * qk_dim
        else:
            q_params = hidden * heads * qk_dim
        kv_a_params = hidden * (kv_lora_rank + qk_rope_dim)
        kv_b_params = kv_lora_rank * heads * (qk_nope_dim + v_dim)
        o_params = heads * v_dim * hidden
        params = q_params + kv_a_params + kv_b_params + o_params
    else:
        head_dim = _read_float(config, "head_dim") or hidden / heads
        kv_heads = _read_float(config, *_KV_HEAD_FIELDS) or heads
        qk_dim = v_dim = head_dim
        q_params = hidden * heads * head_dim
        kv_params = 2 * hidden * kv_heads * head_dim
        o_params = heads * head_dim * hidden
        params = q_params + kv_params + o_params
    return heads, int(qk_dim), int(v_dim), params


def _mlp_active_params(config: Any, layers: int, hidden: float) -> Optional[float]:
    """Return per-token MLP parameters summed over all layers (MoE: activated only)."""
    intermediate = _read_float(config, "intermediate_size")
    routed_experts = _read_float(config, *_ROUTED_EXPERT_FIELDS)
    if not routed_experts:
        if not intermediate:
            return None
        return layers * 3 * hidden * intermediate
    topk = _read_float(config, *_TOPK_FIELDS)
    moe_ffn = _read_float(config, "moe_intermediate_size") or intermediate
    if not topk or not moe_ffn:
        return None
    dense_first = min(int(_read_float(config, "first_k_dense_replace") or 0), layers)
    sparse_step = max(int(_read_float(config, "decoder_sparse_step") or 1), 1)
    span = layers - dense_first
    moe_layers = span if sparse_step == 1 else (span + sparse_step - 1) // sparse_step
    dense_layers = layers - moe_layers
    per_moe_layer = topk * 3 * hidden * moe_ffn + hidden * routed_experts
    shared_experts = _read_float(config, "n_shared_experts")
    if shared_experts:
        per_moe_layer += shared_experts * 3 * hidden * moe_ffn
    else:
        shared_ffn = _read_float(config, "shared_expert_intermediate_size")
        if shared_ffn:
            per_moe_layer += 3 * hidden * shared_ffn
    total = moe_layers * per_moe_layer
    if intermediate:
        total += dense_layers * 3 * hidden * intermediate
    return total


def _geometry_config(config: Any) -> Any:
    """Return the config that carries the geometry.

    Multimodal wrappers (e.g. ``kimi_k25``) keep the language-model geometry
    under ``text_config``, leaving only wrapper fields at the top level; fall
    back to that sub-config so the estimator sees the real dimensions.
    """
    if _read(config, *_LAYER_FIELDS) and _read(config, *_HIDDEN_FIELDS):
        return config
    nested = _read(config, "text_config")
    return nested if nested is not None else config


def estimate_flops_per_token(config: Any, seq_len: Optional[int] = None) -> Optional[float]:
    """Estimate training FLOPs per token (6N convention) from config geometry.

    Args:
        config: HF-style model config; attribute access or plain mapping. A
            composite multimodal config is resolved via ``_geometry_config``.
        seq_len: Sequence length used to add the attention score/weight
            quadratic term; ``None`` reports the linear 6N part only.

    Returns:
        Estimated FLOPs per token, or ``None`` when the config lacks the
        geometry fields needed for an honest estimate.
    """
    config = _geometry_config(config)
    layers = _read_float(config, *_LAYER_FIELDS)
    hidden = _read_float(config, *_HIDDEN_FIELDS)
    if not layers or not hidden:
        return None
    layers = int(layers)
    attention = _attention_geometry(config)
    mlp_params = _mlp_active_params(config, layers, hidden)
    if attention is None or mlp_params is None:
        return None
    heads, qk_dim, v_dim, attn_params = attention
    active_params = layers * attn_params + mlp_params
    vocab = _read_float(config, "vocab_size")
    if vocab:
        active_params += hidden * vocab
    flops = 6.0 * active_params
    if seq_len:
        flops += 6.0 * layers * heads * (qk_dim + v_dim) * int(seq_len)
    return flops


def batch_seq_len(micro_batches: Any) -> Optional[int]:
    """Return the sequence length of the first micro-batch carrying ``input_ids``.

    Args:
        micro_batches: One micro-batch or a list of mapping micro-batches.

    Returns:
        The trailing ``input_ids`` dimension, or ``None`` when unavailable.
    """
    if micro_batches is None:
        return None
    if isinstance(micro_batches, Mapping):
        micro_batches = [micro_batches]
    for batch in micro_batches:
        if not isinstance(batch, Mapping):
            continue
        shape = getattr(batch.get("input_ids"), "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[-1])
    return None


def resolve_flops_per_token(
    model: Any,
    model_config: Any = None,
    seq_len: Optional[int] = None,
) -> Optional[float]:
    """Resolve FLOPs per token: the model's own property wins, else estimate.

    Args:
        model: Built model instance; a family may expose
            ``hp_flops_per_token`` with an exact derivation.
        model_config: HF-style config used by the generic estimator;
            defaults to ``model.config``.
        seq_len: Sequence length for the attention quadratic term.

    Returns:
        FLOPs per token, or ``None`` when neither source can provide one.
    """
    value = getattr(model, "hp_flops_per_token", None) if model is not None else None
    if value:
        return float(value)
    if model_config is None and model is not None:
        model_config = getattr(model, "config", None)
    if model_config is None:
        return None
    return estimate_flops_per_token(model_config, seq_len)


class VisionFlopsEstimate(NamedTuple):
    """Vision-tower training FLOPs, split into a per-patch and a quadratic part.

    The tower runs dense ops over every patch of every image in the batch, so its
    cost does not scale with ``input_ids`` and cannot be folded into
    ``flops_per_token``; it is accounted separately and added to the TFLOPS
    numerator by the environment meter.

    Evaluation is ``per_patch * total_patches + attention_coefficient * sum(p_i**2)``
    where ``p_i`` is the patch count of image *i*.  Attention is computed inside one
    image only, so the quadratic term uses each image's own patch count rather than
    the batch-wide patch total -- a batch of 10 x 1656 patches is 10x cheaper in
    attention than one 16560-patch image, and folding it into a single number would
    overstate the tower by the average patch count.
    """

    per_patch: float
    attention_coefficient: float
    total_params: float

    def flops(self, total_patches: int, patch_square_sum: Optional[int] = None) -> float:
        """Return training FLOPs for a batch of ``total_patches`` vision patches.

        Args:
            total_patches: Patch rows processed by the tower in the step.
            patch_square_sum: ``sum(p_i ** 2)`` over the batch's images; when
                omitted the attention part is skipped rather than guessed.

        Returns:
            Estimated forward+backward FLOPs for the vision tower.
        """
        flops = self.per_patch * float(total_patches)
        if patch_square_sum:
            flops += self.attention_coefficient * float(patch_square_sum)
        return flops


def estimate_vision_flops(config: Any) -> Optional[VisionFlopsEstimate]:
    """Estimate vision-tower FLOPs per patch from a multimodal config's geometry.

    Uses the same 6N convention as :func:`estimate_flops_per_token`: every matmul
    parameter costs ``6 * params`` FLOPs per patch (forward + backward), with a
    ViT layer counted as ``4 * hidden**2`` (q/k/v/out projections) plus
    ``2 * hidden * intermediate`` (ungated MLP).  The attention score/weight
    matmuls contribute ``6 * layers * heads * (qk_dim + v_dim) * p_i`` per patch,
    i.e. a quadratic term over each image's own patch count.  Patch-embedding,
    position embeddings (including Kimi's divided-fixed 2D/3D scheme) and the
    multimodal projector are not counted; they are a small constant next to the
    tower's depth.

    Args:
        config: HF-style multimodal config; attribute access or plain mapping.
            Text-only configs have no vision sub-config and yield ``None``, which
            keeps the metric unchanged for language-only training.

    Returns:
        The per-patch cost model, or ``None`` when no vision geometry is present.
    """
    if config is None:
        return None
    vision = _read(config, *_VISION_CONFIG_FIELDS)
    if vision is None:
        return None
    layers = _read_float(vision, *_VISION_LAYER_FIELDS)
    hidden = _read_float(vision, *_VISION_HIDDEN_FIELDS)
    if not layers or not hidden:
        return None
    intermediate = _read_float(vision, *_VISION_INTER_FIELDS) or 4.0 * hidden
    heads = _read_float(vision, *_VISION_HEAD_FIELDS) or 0.0
    layers = int(layers)
    params_per_layer = 4.0 * hidden * hidden + 2.0 * hidden * intermediate
    # heads * (qk_dim + v_dim) == 2 * hidden for a standard ViT, so the quadratic
    # coefficient needs only the head count being present, not its value.
    head_dim_sum = 2.0 * hidden if heads else 0.0
    return VisionFlopsEstimate(
        per_patch=6.0 * params_per_layer * layers,
        attention_coefficient=6.0 * layers * head_dim_sum if heads else 0.0,
        total_params=params_per_layer * layers,
    )


def resolve_vision_flops(
    model: Any,
    model_config: Any = None,
) -> Optional[VisionFlopsEstimate]:
    """Resolve the vision-tower cost model: the model's own value wins, else estimate.

    Args:
        model: Built model instance; a family may expose ``hp_vision_flops`` with
            an exact derivation.
        model_config: HF-style config used by the generic estimator; defaults to
            ``model.config``.

    Returns:
        The cost model, or ``None`` for a text-only model.
    """
    value = getattr(model, "hp_vision_flops", None) if model is not None else None
    if isinstance(value, VisionFlopsEstimate):
        return value
    if model_config is None and model is not None:
        model_config = getattr(model, "config", None)
    return estimate_vision_flops(model_config)


def resolve_recompute_factor(
    config: Any,
    override: Optional[float] = None,
) -> Optional[float]:
    """Return ``executed FLOPs / model FLOPs``, the MFU -> HFU multiplier.

    MFU divides the *useful* model FLOPs (6N: one forward and one backward per token) by
    the peak, so by definition it omits the activation-checkpoint recomputation.  HFU is
    the hardware view: every FLOP the device executes.  Full checkpointing runs the
    forward twice (6N -> 8N), so the multiplier is 4/3; measured on a profiled 2-SN step
    the MAC-busy fraction was 27.9% while the model MFU of the same window was 20.4%,
    a ratio of 1.37.

    Args:
        config: Trainer config (uses ``activation_checkpoint.mode`` and
            ``training.hfu_recompute_factor``).
        override: Explicit multiplier, when the caller already resolved one.

    Returns:
        The multiplier, or ``None`` when it cannot be established -- a selective schedule
        recomputes an unknown fraction, and reporting a guessed hardware-utilisation
        number is worse than reporting none.
    """
    if override:
        return float(override)
    training = getattr(config, "training", None)
    explicit = getattr(training, "hfu_recompute_factor", None)
    if explicit:
        return float(explicit)
    mode = getattr(getattr(config, "activation_checkpoint", None), "mode", None)
    if mode == "full":
        return 4.0 / 3.0
    if mode == "off" or mode is None:
        return 1.0
    return None
