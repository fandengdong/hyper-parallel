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
"""Unit tests for the MFU/HFU reporting convention.

MFU divides the *useful* model FLOPs by the peak, so by definition it omits the
activation-checkpoint recomputation forward.  HFU is the hardware view: every FLOP the
device executes, recompute included.  Full checkpointing runs the forward twice (6N -> 8N),
so HFU = MFU x 4/3; measured on a profiled 2-SN step the MAC-busy fraction was 27.9% while
the model MFU of the same window was 20.4% (ratio 1.37).

These tests pin the resolution rules, in particular that HFU is *not* reported when the
recomputed fraction is unknowable (a selective schedule without an explicit factor):
a wrong hardware-utilisation number is worse than a missing one.
"""
import os
import types
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.models.flops import resolve_recompute_factor  # noqa: E402
from hyper_parallel.trainer.callbacks.throughput_callback import (  # noqa: E402
    ThroughputMFUCallback,
)


class _Training:
    """Stand-in for the training config section."""

    def __init__(self, factor, peak_tflops=400.0):
        self.logging_steps = 1
        self.global_batch_size = 4096
        self.peak_tflops = peak_tflops
        self.hfu_recompute_factor = factor


class _Config:
    """Stand-in for the trainer config."""

    def __init__(self, mode, factor=None, peak_tflops=400.0):
        self.training = _Training(factor, peak_tflops)
        self.activation_checkpoint = types.SimpleNamespace(mode=mode)


class _Trainer:
    """Stand-in for the trainer."""

    def __init__(self, config):
        self.config = config


def _callback(mode, factor=None):
    """Return a callback whose trainer config carries the given AC mode."""
    callback = ThroughputMFUCallback.__new__(ThroughputMFUCallback)
    callback.trainer = _Trainer(_Config(mode, factor))
    callback._recompute_factor = None  # pylint: disable=protected-access
    callback._hfu_recompute_factor = factor  # pylint: disable=protected-access
    callback._warned_hfu = False  # pylint: disable=protected-access
    return callback


class TestRecomputeFactor(unittest.TestCase):
    """Pin ``executed FLOPs / model FLOPs`` for the HFU view."""

    def test_full_checkpointing_doubles_one_forward(self):
        """Full AC runs the forward twice: 6N -> 8N."""
        self.assertAlmostEqual(
            _callback("full")._resolve_recompute_factor(),  # pylint: disable=protected-access
            4.0 / 3.0,
        )

    def test_off_is_unity(self):
        """Without recomputation the device runs exactly the model FLOPs."""
        self.assertEqual(
            _callback("off")._resolve_recompute_factor(),  # pylint: disable=protected-access
            1.0,
        )

    def test_explicit_factor_wins(self):
        """A selective schedule knows its own fraction, so an explicit value is honoured."""
        self.assertEqual(
            _callback("selective", 1.5)._resolve_recompute_factor(),  # pylint: disable=protected-access
            1.5,
        )
        self.assertEqual(
            _callback("full", 1.25)._resolve_recompute_factor(),  # pylint: disable=protected-access
            1.25,
        )

    def test_selective_without_factor_reports_nothing(self):
        """An unknowable fraction must suppress HFU instead of guessing it."""
        self.assertIsNone(
            _callback("selective")._resolve_recompute_factor()  # pylint: disable=protected-access
        )

    def test_factor_is_cached(self):
        """Resolution happens once; later calls reuse it."""
        callback = _callback("full")
        first = callback._resolve_recompute_factor()  # pylint: disable=protected-access
        callback._hfu_recompute_factor = None  # pylint: disable=protected-access
        self.assertEqual(callback._resolve_recompute_factor(), first)  # pylint: disable=protected-access


class TestSharedResolver(unittest.TestCase):
    """The same rules must hold wherever HFU is reported (callback and env meter)."""

    def test_full_and_off(self):
        """Full checkpointing is 8N/6N; no checkpointing is exactly the model FLOPs."""
        self.assertAlmostEqual(resolve_recompute_factor(_Config("full")), 4.0 / 3.0)
        self.assertEqual(resolve_recompute_factor(_Config("off")), 1.0)

    def test_explicit_override_wins(self):
        """An explicit factor beats the mode-derived default."""
        self.assertEqual(resolve_recompute_factor(_Config("selective", 1.25)), 1.25)
        self.assertEqual(
            resolve_recompute_factor(_Config("full"), override=1.1), 1.1
        )

    def test_selective_without_override_is_unknown(self):
        """A selective schedule must not fabricate a hardware-utilisation number."""
        self.assertIsNone(resolve_recompute_factor(_Config("selective")))


if __name__ == "__main__":
    unittest.main()
