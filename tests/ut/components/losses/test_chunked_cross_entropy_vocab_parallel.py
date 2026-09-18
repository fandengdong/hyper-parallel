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
"""Unit tests for chunked linear cross-entropy over a sharded vocabulary.

Gate-1: no process group and no accelerator. The tensor-parallel group is
simulated with one thread per rank and a barrier-backed all-reduce, so the
reduction, the vocabulary offsets and the collective symmetry are exercised
end to end on CPU.
"""
# pylint: disable=not-callable,protected-access

import importlib
import os
import threading
import unittest
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # pylint: disable=wrong-import-position
from torch import nn  # pylint: disable=wrong-import-position
from torch.nn import functional  # pylint: disable=wrong-import-position

from hyper_parallel.components.losses import ChunkedCausalLMLoss  # pylint: disable=wrong-import-position
from hyper_parallel.components.losses import chunk_loss_tp_mesh  # pylint: disable=wrong-import-position
from hyper_parallel.components.losses import chunked_cross_entropy  # pylint: disable=wrong-import-position
from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position

# The package re-exports a function named after this module, so ``import ... as``
# would bind the function; the platform patch needs the module object itself.
_LOSS_MODULE = importlib.import_module("hyper_parallel.components.losses.chunked_cross_entropy")
# The vocab-parallel cross entropy keeps its own module-level platform.
_KERNEL_MODULE = importlib.import_module(
    "hyper_parallel.components.losses._vocab_parallel_cross_entropy"
)

_VOCAB = 16
_HIDDEN = 6
_BATCH = 2
_SEQ = 9
_IGNORE_INDEX = -100
_TP_SIZE = 2
# Generous for a tiny CPU computation, short enough to fail a deadlock quickly.
_TP_TIMEOUT_SECONDS = 15


class _BarrierTpWorld:
    """One tensor-parallel group spanned by simulated ranks in one process.

    Every rank exchanging the same call in the same order meets at the barrier,
    so a rank that skips a collective deadlocks instead of silently producing a
    wrong result.
    """

    def __init__(self, size):
        self.size = size
        self.group = object()
        self.barrier = threading.Barrier(size, timeout=_TP_TIMEOUT_SECONDS)
        self.slots = [None] * size

    def all_reduce(self, rank, data, op):
        """Reduce ``data`` over the simulated group with plain differentiable ops."""
        self.slots[rank] = data
        self.barrier.wait()
        stacked = torch.stack(list(self.slots))
        reduced = stacked.amax(0) if op == "max" else stacked.sum(0)
        self.barrier.wait()
        return reduced


class _SimulatedTpMesh:
    """1-D device-mesh stand-in over :class:`_BarrierTpWorld`."""

    ndim = 1
    mesh_dim_names = ("tp",)

    def __init__(self, world, rank):
        self._world = world
        self._rank = rank

    def size(self, mesh_dim=None):
        """Return the tensor-parallel group size."""
        return self._world.size

    def get_local_rank(self, mesh_dim=None):
        """Return this simulated rank's coordinate on the mesh axis."""
        return self._rank

    def get_group(self, mesh_dim=None):
        """Return the simulated tensor-parallel group."""
        return self._world.group


_LOCAL_RANK = threading.local()


class _SimulatedTpPlatform:
    """Platform stand-in routing every reduction into :class:`_BarrierTpWorld`."""

    def __init__(self, world):
        self.world = world

    def differentiable_all_reduce(self, data, op, group):
        """Reduce over the simulated group, refusing any other group."""
        if group is not self.world.group:
            raise AssertionError("chunk loss reduction escaped the tensor-parallel group")
        return self.world.all_reduce(_LOCAL_RANK.rank, data, op)


class _MeshWithoutTpAxis:
    """2-D mesh stand-in whose axes carry no ``tp`` name."""

    ndim = 2
    mesh_dim_names = ("dp", "cp")

    def size(self, mesh_dim=None):
        """Return the axis size."""
        return 2


class _ForbiddenPlatform:
    """Platform stand-in that fails when a single-rank path reduces at all."""

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        raise AssertionError(
            "single-rank chunked cross entropy must not issue a distributed reduction"
        )


class _MeshOfSizeOne:
    """1-D mesh stand-in with a single rank: the plain path must be kept."""

    ndim = 1
    mesh_dim_names = ("tp",)

    def size(self, mesh_dim=None):
        """Return the mesh size."""
        return 1

    def get_local_rank(self, mesh_dim=None):
        """Return the only rank."""
        return 0

    def get_group(self, mesh_dim=None):
        """Return a group that must never be used."""
        raise AssertionError("a single-rank mesh must not be used for reductions")


def _eager_loss(hidden_states, head_weight, targets):
    """Return the full-vocabulary FP32 summed cross-entropy reference."""
    logits = functional.linear(hidden_states, head_weight).float()
    return functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=_IGNORE_INDEX,
        reduction="sum",
    )

def _frozen_linear_cross_entropy_chunk(hidden_chunk, weight, target_chunk, ignore_index):
    """Frozen copy of the single-rank chunk cross-entropy body."""
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


class _FrozenPrecomputedChunkLoss(torch.autograd.Function):
    """Frozen copy of the single-rank Chunk Loss autograd Function.

    Characterization reference: it reproduces the pre-vocabulary-parallel
    implementation bit for bit, so ``torch.equal`` against it pins that the
    single-rank path was not altered.
    """

    @staticmethod
    def forward(ctx, hidden_states, head_weight, targets, chunk_size, ignore_index):
        """Precompute per-chunk gradients exactly as the released version did."""
        grad_hidden = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)

        grad_and_value = torch.func.grad_and_value(
            _frozen_linear_cross_entropy_chunk,
            argnums=(0, 1),
        )
        hidden_chunks = torch.split(hidden_states, chunk_size, dim=1)
        target_chunks = torch.split(targets, chunk_size, dim=1)
        grad_hidden_chunks = torch.split(grad_hidden, chunk_size, dim=1)
        for hidden_chunk, target_chunk, grad_hidden_chunk in zip(
                hidden_chunks, target_chunks, grad_hidden_chunks, strict=True
        ):
            (chunk_grad_hidden, chunk_grad_weight), chunk_loss = grad_and_value(
                hidden_chunk,
                head_weight,
                target_chunk,
                ignore_index,
            )
            grad_hidden_chunk.copy_(chunk_grad_hidden)
            grad_weight.add_(chunk_grad_weight)
            loss_sum.add_(chunk_loss)
            del chunk_grad_hidden, chunk_grad_weight, chunk_loss

        ctx.save_for_backward(grad_hidden, grad_weight)
        return loss_sum

    @staticmethod
    def backward(ctx, grad_loss_sum):
        """Replay the frozen gradients with the upstream scale."""
        grad_hidden, grad_weight = ctx.saved_tensors
        return (
            grad_hidden * grad_loss_sum,
            grad_weight * grad_loss_sum,
            None,
            None,
            None,
        )


def _frozen_chunked_cross_entropy(hidden_states, targets, head_weight, chunk_size):
    """Run the frozen single-rank Chunk Loss."""
    return _FrozenPrecomputedChunkLoss.apply(
        hidden_states,
        head_weight,
        targets,
        chunk_size,
        _IGNORE_INDEX,
    )


def _chunk_loss_inputs(seed=11):
    """Return deterministic hidden states and targets for the kernel tests."""
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(_BATCH, _SEQ, _HIDDEN, generator=generator)
    head_weight = torch.randn(_VOCAB, _HIDDEN, generator=generator)
    targets = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=generator)
    targets[0, 2] = _IGNORE_INDEX
    targets[1, _SEQ - 1] = _IGNORE_INDEX
    return hidden_states, targets, head_weight


def _run_simulated_rank(world, rank, result, errors, inputs, chunk_size):
    """Run one simulated rank's chunk loss and record loss and gradients."""
    _LOCAL_RANK.rank = rank
    hidden_states, targets, weight_shard = inputs
    try:
        hidden_states = hidden_states.detach().clone().requires_grad_()
        weight_shard = weight_shard.detach().clone().requires_grad_()
        loss = chunked_cross_entropy(
            hidden_states,
            targets,
            weight_shard,
            chunk_size=chunk_size,
            ignore_index=_IGNORE_INDEX,
            tp_mesh=_SimulatedTpMesh(world, rank),
        )
        loss.backward()
        result[rank] = (loss.detach(), hidden_states.grad.clone(), weight_shard.grad.clone())
    except BaseException as error:  # pylint: disable=broad-except
        # Report the rank's own failure instead of the barrier timeout it causes.
        errors[rank] = error


def _patch_platforms(platform):
    """Route every reduction of the Chunk Loss path through one stand-in."""
    previous = (_LOSS_MODULE.platform, _KERNEL_MODULE.platform)
    _LOSS_MODULE.platform = platform
    _KERNEL_MODULE.platform = platform
    return previous


def _restore_platforms(previous):
    """Restore the platform objects captured by :func:`_patch_platforms`."""
    _LOSS_MODULE.platform, _KERNEL_MODULE.platform = previous


def _run_simulated_group(world, workers):
    """Run one worker per simulated rank in lockstep.

    Args:
        world: The barrier-backed tensor-parallel group to reduce over.
        workers: One callable per rank, taking ``(world, rank, result, errors)``.

    Returns:
        One per-rank result entry per worker.
    """
    result = [None] * world.size
    errors = [None] * world.size
    threads = [
        threading.Thread(target=worker, args=(world, rank, result, errors))
        for rank, worker in enumerate(workers)
    ]
    previous = _patch_platforms(_SimulatedTpPlatform(world))
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(_TP_TIMEOUT_SECONDS * 2)
        for rank, error in enumerate(errors):
            if error is not None:
                raise error
        for thread in threads:
            if thread.is_alive():
                raise AssertionError("simulated tensor-parallel ranks did not converge")
    finally:
        _restore_platforms(previous)
    return result


def _kernel_rank_worker(inputs, chunk_size):
    """Return a rank worker running the bare chunked cross-entropy."""

    def _worker(world, rank, result, errors):
        _run_simulated_rank(world, rank, result, errors, inputs, chunk_size)

    return _worker


def _simulate_sharded_forward(hidden_states, targets, head_weight, chunk_size):
    """Run every simulated rank of one tensor-parallel group, in lockstep."""
    world = _BarrierTpWorld(_TP_SIZE)
    shards = list(torch.chunk(head_weight, _TP_SIZE, dim=0))
    workers = [
        _kernel_rank_worker((hidden_states, targets, shards[rank]), chunk_size)
        for rank in range(_TP_SIZE)
    ]
    return _run_simulated_group(world, workers)


class TestChunkedCrossEntropySingleRankUnchanged(unittest.TestCase):
    """The full-vocabulary path must keep the released single-rank behaviour."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_rank_matches_frozen_implementation_bitwise(self):
        """Verify the single-rank loss and gradients are unchanged.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Recompute a chunked cross-entropy over uneven chunks and ignored targets.
        Expectation: Loss and gradients equal the frozen pre-change implementation bit for bit.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        for chunk_size in (1, 4, _SEQ):
            with self.subTest(chunk_size=chunk_size):
                hidden = hidden_states.clone().requires_grad_()
                weight = head_weight.clone().requires_grad_()
                loss = chunked_cross_entropy(
                    hidden, targets, weight, chunk_size=chunk_size, ignore_index=_IGNORE_INDEX
                )
                (loss * 0.37).backward()

                frozen_hidden = hidden_states.clone().requires_grad_()
                frozen_weight = head_weight.clone().requires_grad_()
                frozen_loss = _frozen_chunked_cross_entropy(
                    frozen_hidden, targets, frozen_weight, chunk_size
                )
                (frozen_loss * 0.37).backward()

                self.assertTrue(
                    torch.equal(loss, frozen_loss),
                    f"chunk_size={chunk_size}: loss {loss} != frozen loss {frozen_loss}",
                )
                self.assertTrue(
                    torch.equal(hidden.grad, frozen_hidden.grad),
                    f"chunk_size={chunk_size}: hidden gradient changed from the frozen implementation",
                )
                self.assertTrue(
                    torch.equal(weight.grad, frozen_weight.grad),
                    f"chunk_size={chunk_size}: weight gradient changed from the frozen implementation",
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_rank_matches_eager_full_logits_loss(self):
        """Verify the single-rank loss still matches the eager full-logits objective."""
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        hidden = hidden_states.clone().requires_grad_()
        weight = head_weight.clone().requires_grad_()
        loss = chunked_cross_entropy(
            hidden, targets, weight, chunk_size=4, ignore_index=_IGNORE_INDEX
        )
        reference = _eager_loss(hidden_states, head_weight, targets)
        torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_rank_issues_no_collective(self):
        """Verify the plain path performs no distributed reduction.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Compute the chunk loss with no mesh and with a single-rank mesh.
        Expectation: Neither call reaches a distributed all-reduce.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        for tp_mesh in (None, _MeshOfSizeOne()):
            with self.subTest(tp_mesh=tp_mesh):
                with mock.patch.object(_LOSS_MODULE, "platform", _ForbiddenPlatform()):
                    loss = chunked_cross_entropy(
                        hidden_states, targets, head_weight, chunk_size=4, tp_mesh=tp_mesh
                    )
                torch.testing.assert_close(
                    loss, _eager_loss(hidden_states, head_weight, targets), rtol=1e-6, atol=1e-6
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_rank_mesh_result_equals_no_mesh(self):
        """Verify a single-rank mesh is not routed into the sharded path."""
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        plain = chunked_cross_entropy(hidden_states, targets, head_weight, chunk_size=3)
        with_mesh = chunked_cross_entropy(
            hidden_states, targets, head_weight, chunk_size=3, tp_mesh=_MeshOfSizeOne()
        )
        self.assertTrue(
            torch.equal(plain, with_mesh),
            f"single-rank mesh loss {with_mesh} differs from plain loss {plain}",
        )


class TestChunkedCrossEntropyVocabParallel(unittest.TestCase):
    """A sharded vocabulary must reproduce the full-vocabulary objective."""

    def setUp(self):
        """Keep the module platform patchable per test."""
        self._loss_module_platform = _LOSS_MODULE.platform
        self.addCleanup(setattr, _LOSS_MODULE, "platform", self._loss_module_platform)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sharded_loss_and_gradients_match_single_rank(self):
        """Verify a vocabulary-sharded Chunk Loss equals the single-rank loss.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Split one LM head over a simulated two-rank tensor-parallel group.
        Expectation: Loss, hidden gradient and both weight shards match the single-rank result.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        reference_hidden = hidden_states.clone().requires_grad_()
        reference_weight = head_weight.clone().requires_grad_()
        reference_loss = chunked_cross_entropy(
            reference_hidden, targets, reference_weight, chunk_size=4, ignore_index=_IGNORE_INDEX
        )
        reference_loss.backward()

        result = _simulate_sharded_forward(hidden_states, targets, head_weight, chunk_size=4)

        for rank, (loss, grad_hidden, grad_weight) in enumerate(result):
            torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-5)
            torch.testing.assert_close(grad_hidden, reference_hidden.grad, rtol=1e-6, atol=1e-6)
            weight_shard = torch.chunk(reference_weight.grad, _TP_SIZE, dim=0)[rank]
            torch.testing.assert_close(grad_weight, weight_shard, rtol=1e-6, atol=1e-6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_every_rank_reports_the_same_loss_and_hidden_gradient(self):
        """Verify the sharded loss and hidden gradient are replicated on all ranks.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Compare the two simulated ranks of one tensor-parallel group.
        Expectation: Both ranks report the identical loss and hidden gradient.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        result = _simulate_sharded_forward(hidden_states, targets, head_weight, chunk_size=4)
        first_loss, first_grad_hidden, _ = result[0]
        for rank in range(1, _TP_SIZE):
            loss, grad_hidden, _ = result[rank]
            self.assertTrue(
                torch.equal(loss, first_loss),
                f"rank {rank} loss {loss} differs from rank 0 loss {first_loss}",
            )
            self.assertTrue(
                torch.equal(grad_hidden, first_grad_hidden),
                f"rank {rank} hidden gradient differs from rank 0",
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shard_owning_no_target_keeps_global_normalization(self):
        """Verify a shard without local targets still contributes the normalization.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Restrict every target to the lower vocabulary half owned by rank 0.
        Expectation: The loss and both gradients still match the single-rank result.
        """
        hidden_states, _, head_weight = _chunk_loss_inputs()
        generator = torch.Generator().manual_seed(29)
        targets = torch.randint(0, _VOCAB // 2, (_BATCH, _SEQ), generator=generator)
        targets[0, 5] = _IGNORE_INDEX

        reference_hidden = hidden_states.clone().requires_grad_()
        reference_weight = head_weight.clone().requires_grad_()
        reference_loss = chunked_cross_entropy(
            reference_hidden, targets, reference_weight, chunk_size=4, ignore_index=_IGNORE_INDEX
        )
        reference_loss.backward()

        result = _simulate_sharded_forward(hidden_states, targets, head_weight, chunk_size=4)

        for rank, (loss, grad_hidden, grad_weight) in enumerate(result):
            torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-5)
            torch.testing.assert_close(grad_hidden, reference_hidden.grad, rtol=1e-6, atol=1e-6)
            weight_shard = torch.chunk(reference_weight.grad, _TP_SIZE, dim=0)[rank]
            torch.testing.assert_close(grad_weight, weight_shard, rtol=1e-6, atol=1e-6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_all_ignored_targets_keep_ranks_symmetric(self):
        """Verify an all-padding batch stays finite and symmetric across ranks.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Mark every target of the batch as ignored and split the vocabulary.
        Expectation: Every rank reports a zero loss and zero gradients without diverging.
        """
        hidden_states, _, head_weight = _chunk_loss_inputs()
        targets = torch.full((_BATCH, _SEQ), _IGNORE_INDEX, dtype=torch.long)
        result = _simulate_sharded_forward(hidden_states, targets, head_weight, chunk_size=4)
        for rank, (loss, grad_hidden, grad_weight) in enumerate(result):
            self.assertTrue(
                torch.equal(loss, torch.zeros_like(loss)),
                f"rank {rank} loss {loss} is not exactly zero",
            )
            torch.testing.assert_close(grad_hidden, torch.zeros_like(grad_hidden), rtol=0, atol=0)
            torch.testing.assert_close(grad_weight, torch.zeros_like(grad_weight), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sharded_loss_matches_single_rank_in_bfloat16(self):
        """Verify the sharded path keeps its contract under bfloat16 training.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Repeat the sharded comparison with bfloat16 hidden states and weights.
        Expectation: The loss matches the single-rank bfloat16 loss within bfloat16 tolerance.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        hidden_states = hidden_states.to(torch.bfloat16)
        head_weight = head_weight.to(torch.bfloat16)
        reference_loss = chunked_cross_entropy(
            hidden_states, targets, head_weight, chunk_size=4, ignore_index=_IGNORE_INDEX
        )
        result = _simulate_sharded_forward(hidden_states, targets, head_weight, chunk_size=4)
        for loss, _, _ in result:
            self.assertEqual(loss.dtype, torch.float32)
            torch.testing.assert_close(loss, reference_loss, rtol=2e-2, atol=2e-2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reduction_uses_the_tp_group_only(self):
        """Verify every reduction targets the tensor-parallel group.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Replace the platform all-reduce with a group-checking stand-in.
        Expectation: The sharded path reduces over the TP group and refuses any other group.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        world = _BarrierTpWorld(_TP_SIZE)
        previous = _patch_platforms(_SimulatedTpPlatform(world))
        try:
            with self.assertRaises(AssertionError):
                _LOSS_MODULE.platform.differentiable_all_reduce(torch.zeros(1), "sum", object())
        finally:
            _restore_platforms(previous)
        self.assertIsNotNone(_simulate_sharded_forward(hidden_states, targets, head_weight, 4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_absent_tp_axis_raises(self):
        """Verify a mesh without a vocabulary-sharding axis fails fast.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Pass a two-dimensional mesh whose axes are not named 'tp'.
        Expectation: A ValueError explains that no class-sharding axis was found.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        with self.assertRaises(ValueError):
            chunked_cross_entropy(
                hidden_states, targets, head_weight, chunk_size=4, tp_mesh=_MeshWithoutTpAxis()
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_uneven_vocabulary_split_raises(self):
        """Verify an explicit vocabulary size must describe the local shard.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Pass a global vocabulary size that is not tp_size times the local shard.
        Expectation: A ValueError refuses to mask targets against a wrong vocabulary range.
        """
        hidden_states, targets, head_weight = _chunk_loss_inputs()
        shard = torch.chunk(head_weight, _TP_SIZE, dim=0)[0]
        world = _BarrierTpWorld(_TP_SIZE)
        with self.assertRaises(ValueError):
            chunked_cross_entropy(
                hidden_states,
                targets,
                shard,
                chunk_size=4,
                tp_mesh=_SimulatedTpMesh(world, 0),
                vocab_size=_VOCAB + 2,
            )


def _setup(tp_size=1, pp_size=1, loss_parallel=False, sequence_parallel=False, device_mesh=None):
    """Return a minimal distributed setup for loss binding."""
    mesh_context = type(
        "MeshContext",
        (),
        {
            "tp_size": tp_size,
            "pp_size": pp_size,
            "loss_parallel": loss_parallel,
            "sequence_parallel": sequence_parallel,
            "device_mesh": device_mesh,
        },
    )()
    return type("DistributedSetup", (), {"mesh_context": mesh_context})()


class _FakeDeviceMesh:
    """Device mesh stand-in exposing a ``tp`` sub-mesh lookup."""

    def __init__(self, mesh_dim_names, tp_mesh):
        self.mesh_dim_names = mesh_dim_names
        self._tp_mesh = tp_mesh

    def __getitem__(self, name):
        """Return the requested sub-mesh."""
        if name != "tp":
            raise KeyError(name)
        return self._tp_mesh


class _FakeSubMesh:
    """Sub-mesh stand-in with a fixed size."""

    def __init__(self, size):
        self._size = size

    def size(self, mesh_dim=None):
        """Return the sub-mesh size."""
        return self._size


class TestChunkedCausalLMLossBindingGuards(unittest.TestCase):
    """Unsupported parallelism must fail fast instead of computing a wrong loss."""

    def setUp(self):
        """Create a loss module and a bare model stand-in."""
        self.loss_fn = ChunkedCausalLMLoss(chunk_size=4)
        self.model = nn.Module()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pipeline_parallel_raises(self):
        """Verify pipeline parallelism is still refused with a clear error."""
        with self.assertRaises(NotImplementedError) as caught:
            self.loss_fn.bind_model(self.model, _setup(pp_size=2))
        self.assertIn("pp_size=1", str(caught.exception))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_loss_parallel_raises(self):
        """Verify loss parallelism is still refused with a clear error."""
        with self.assertRaises(NotImplementedError) as caught:
            self.loss_fn.bind_model(self.model, _setup(tp_size=2, loss_parallel=True))
        self.assertIn("loss_parallel=false", str(caught.exception))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tensor_parallel_without_mesh_raises(self):
        """Verify tensor parallelism without a device mesh cannot compute a loss."""
        with self.assertRaises(ValueError) as caught:
            self.loss_fn.bind_model(self.model, _setup(tp_size=2))
        self.assertIn("'tp' axis", str(caught.exception))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tensor_parallel_mesh_size_mismatch_raises(self):
        """Verify a mesh whose tp axis disagrees with the topology is refused."""
        device_mesh = _FakeDeviceMesh(("dp", "tp"), _FakeSubMesh(4))
        with self.assertRaises(ValueError) as caught:
            self.loss_fn.bind_model(self.model, _setup(tp_size=2, device_mesh=device_mesh))
        self.assertIn("tp_size is 2", str(caught.exception))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sequence_parallel_with_tensor_parallel_raises(self):
        """Verify sequence parallelism over the same axis as the vocabulary is refused.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Bind with tp_size=2 and sequence_parallel=true.
        Expectation: A NotImplementedError refuses to mix sequence and vocabulary shards.
        """
        device_mesh = _FakeDeviceMesh(("dp", "tp"), _FakeSubMesh(2))
        with self.assertRaises(NotImplementedError) as caught:
            self.loss_fn.bind_model(
                self.model, _setup(tp_size=2, sequence_parallel=True, device_mesh=device_mesh)
            )
        self.assertIn("sequence_parallel=false", str(caught.exception))


class _EmbeddingDecoder(nn.Module):
    """Minimal decoder stand-in: the Chunk Loss path only reads final hidden states."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids=None, **kwargs):
        """Return the last hidden state for the given token ids."""
        del kwargs
        return type("Outputs", (), {"last_hidden_state": self.embed(input_ids)})()


class _MinimalCausalLM(nn.Module):
    """Kimi-K2.6-shaped model stand-in carrying a bias-free LM head."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.config = type("Config", (), {"model_type": "kimi_k25"})()
        self.model = _EmbeddingDecoder(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def _model_rank_worker(model, loss_fn, prepared, labels):
    """Return a rank worker running the bound model and Chunk Loss objective."""

    def _worker(_world, rank, result, errors):
        _LOCAL_RANK.rank = rank
        try:
            output = model(**prepared, use_cache=False)
            loss = loss_fn(model_output=output, labels=labels)
            loss.backward()
            result[rank] = (loss.detach(), model.lm_head.weight.grad.clone())
        except BaseException as error:  # pylint: disable=broad-except
            # Report the rank's own failure instead of the barrier timeout it causes.
            errors[rank] = error

    return _worker


class TestChunkedCausalLMLossVocabParallelTraining(unittest.TestCase):
    """A TP-sharded LM head must train identically through the bound objective."""

    def _inputs(self):
        """Return deterministic token ids and labels with padding."""
        generator = torch.Generator().manual_seed(23)
        input_ids = torch.randint(0, _VOCAB, (1, _SEQ), generator=generator)
        labels = input_ids.clone()
        labels[0, 3] = _IGNORE_INDEX
        labels[0, _SEQ - 1] = _IGNORE_INDEX
        return input_ids, labels

    def _prepared_inputs(self, loss_fn, input_ids, labels):
        """Prepare the model-family Chunk Loss input protocol."""
        return loss_fn.prepare_model_inputs(
            {"input_ids": input_ids, "labels": labels},
            {"labels": labels, "loss_mask": labels.ne(_IGNORE_INDEX)},
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sharded_lm_head_trains_identically_to_full_head(self):
        """Verify a vocabulary-sharded LM head reproduces the full-head step.

        Feature: Chunk Loss vocabulary-parallel support.
        Description: Bind a full LM head first, then the same model with the head split over two ranks.
        Expectation: Loss and every LM-head weight shard match the full-head training step.
        """
        input_ids, labels = self._inputs()

        torch.manual_seed(41)
        reference_model = _MinimalCausalLM(_VOCAB, _HIDDEN)
        reference_loss_fn = ChunkedCausalLMLoss(chunk_size=4)
        reference_loss_fn.bind_model(reference_model, _setup())
        reference_output = reference_model(
            **self._prepared_inputs(reference_loss_fn, input_ids, labels), use_cache=False
        )
        reference_loss = reference_loss_fn(model_output=reference_output, labels=labels)
        reference_loss.backward()
        reference_grad = reference_model.lm_head.weight.grad.detach().clone()

        world = _BarrierTpWorld(_TP_SIZE)
        workers = []
        shards = torch.chunk(reference_model.lm_head.weight.detach(), _TP_SIZE, dim=0)
        for rank in range(_TP_SIZE):
            # Each rank resolves its own coordinate on the tp axis, which is
            # what selects the vocabulary slice the loss must score.
            tp_mesh = _SimulatedTpMesh(world, rank)
            model = _MinimalCausalLM(_VOCAB, _HIDDEN)
            model.load_state_dict(reference_model.state_dict())
            model.lm_head.weight.data = shards[rank].clone()
            loss_fn = ChunkedCausalLMLoss(chunk_size=4)
            loss_fn.bind_model(
                model, _setup(tp_size=_TP_SIZE, device_mesh=_FakeDeviceMesh(("dp", "tp"), tp_mesh))
            )
            self.assertIs(chunk_loss_tp_mesh(model), tp_mesh)
            prepared = self._prepared_inputs(loss_fn, input_ids, labels)
            workers.append(_model_rank_worker(model, loss_fn, prepared, labels))

        result = _run_simulated_group(world, workers)

        for rank, (loss, grad) in enumerate(result):
            torch.testing.assert_close(loss, reference_loss.detach(), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(
                grad,
                torch.chunk(reference_grad, _TP_SIZE, dim=0)[rank],
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertTrue(
            torch.equal(result[0][0], result[1][0]),
            f"rank 0 loss {result[0][0]} differs from rank 1 loss {result[1][0]}",
        )


if __name__ == "__main__":
    unittest.main()
