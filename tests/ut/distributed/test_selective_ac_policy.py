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
"""Unit tests for the selective activation-checkpointing matmul cadence.

Selective recomputation saves one in every ``save_every`` matmul outputs and
recomputes the rest, so the cadence is the knob that trades device memory for
recompute. The default (2) must stay bit-for-bit the historical behaviour.
"""
import os
import unittest

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import (  # noqa: E402  pylint: disable=wrong-import-position
    CheckpointPolicy,
)
from hyper_parallel.distributed.activation_checkpoint import (  # noqa: E402  pylint: disable=wrong-import-position
    _make_selective_checkpoint_policy_fn,
    _SELECTIVE_AC_MATMUL_OPS,
    _SELECTIVE_MATMUL_SAVE_EVERY,
)


class _Context:
    """Minimal checkpoint context carrying the phase flag the policy reads."""

    def __init__(self, is_recompute: bool = False) -> None:
        """Record which phase the traced op belongs to."""
        self.is_recompute = is_recompute


class TestSelectiveCheckpointCadence(unittest.TestCase):
    """Pin the save/recompute cadence and its default."""

    @staticmethod
    def _matmul_op():
        """Return an operator that the policy classifies as a matmul."""
        return sorted(_SELECTIVE_AC_MATMUL_OPS, key=str)[0]

    def test_default_cadence_is_every_other_matmul(self):
        """``save_every=2`` reproduces the historical alternating policy."""
        policy = _make_selective_checkpoint_policy_fn(2)
        ctx = _Context()
        decisions = [policy(ctx, self._matmul_op()) for _ in range(6)]
        self.assertEqual(
            decisions,
            [
                CheckpointPolicy.MUST_SAVE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_SAVE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_SAVE,
                CheckpointPolicy.MUST_RECOMPUTE,
            ],
        )

    def test_cadence_four_saves_one_in_four(self):
        """A larger cadence saves fewer activations (the memory-for-recompute dial)."""
        policy = _make_selective_checkpoint_policy_fn(4)
        ctx = _Context()
        decisions = [policy(ctx, self._matmul_op()) for _ in range(8)]
        self.assertEqual(
            decisions,
            [
                CheckpointPolicy.MUST_SAVE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_SAVE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_RECOMPUTE,
                CheckpointPolicy.MUST_RECOMPUTE,
            ],
        )

    def test_recompute_phase_counts_independently(self):
        """Forward and recompute phases must not share a matmul counter."""
        policy = _make_selective_checkpoint_policy_fn(4)
        op = self._matmul_op()
        self.assertEqual(policy(_Context(False), op), CheckpointPolicy.MUST_SAVE)
        self.assertEqual(policy(_Context(True), op), CheckpointPolicy.MUST_SAVE)

    def test_env_default_is_read_and_clamped_to_two(self):
        """The module default comes from the env knob and never drops below 2."""
        self.assertGreaterEqual(_SELECTIVE_MATMUL_SAVE_EVERY, 2)
        self.assertIsInstance(_SELECTIVE_MATMUL_SAVE_EVERY, int)

    def test_a_non_matmul_plain_op_is_recomputed(self):
        """Everything outside the save sets stays recomputed."""
        policy = _make_selective_checkpoint_policy_fn(2)
        decisions = set()
        for op in (torch.ops.aten.relu.default, torch.ops.aten.add.Tensor):
            decisions.add(policy(_Context(), op))
        self.assertEqual(decisions, {CheckpointPolicy.MUST_RECOMPUTE})


if __name__ == "__main__":
    unittest.main()
