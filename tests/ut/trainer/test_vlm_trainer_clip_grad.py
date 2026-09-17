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
"""Gradient clipping is opt-out in the VLM training step.

``clip_grad_norm_`` walks every parameter, stacks a per-shard norm and
all-reduces it, so a run that does not want clipping should not pay for it. The
text path skips the call when ``max_grad_norm <= 0``; these tests pin the same
contract for ``VLMTrainer.train_step``, including the ``grad_norm`` the step
still reports to the callbacks and the training loop.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark


def _build_trainer(max_grad_norm):
    """Build a VLM trainer whose BaseTrainer is stubbed down to the step contract.

    Returns the trainer plus a dict recording the arguments ``train_step`` hands
    to ``on_step_end``. The FSDP unit pipeline is answered as "handled" so the
    step skips the optimizer loop, which this test does not exercise.
    """
    from hyper_parallel.trainer.vlm_trainer import VLMTrainer  # pylint: disable=C0415

    reported = {}

    def _on_step_end(**step_end):
        reported.update(step_end)

    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(max_grad_norm=max_grad_norm)
        ),
        num_micro_batches=1,
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        lr_scheduler=None,
        state=SimpleNamespace(global_step=0),
        get_batch=lambda _data_iterator: (
            {"input_ids": torch.zeros(1, 4, dtype=torch.long)},
            {"labels": torch.tensor([[1, 2, 3, 4]])},
        ),
        model_reshard=lambda _micro_step, _num_micro_steps: None,
        configure_fsdp_gradient_sync=lambda _micro_step, _num_micro_steps: None,
        forward_backward_step=lambda _model_inputs: (
            torch.tensor(1.0),
            {"lm_loss": torch.tensor(1.0)},
        ),
        on_step_begin=lambda **_kwargs: None,
        on_step_end=_on_step_end,
    )
    return trainer, reported


def _run_step(max_grad_norm, clip_result):
    """Run one stubbed VLM step and return the step result, reports and clip mock."""
    trainer, reported = _build_trainer(max_grad_norm)
    with mock.patch("hyper_parallel.trainer.vlm_trainer.synchronize"), \
            mock.patch(
                "hyper_parallel.trainer.runtime.fsdp.run_unit_pipelined_step",
                return_value=True,
            ), \
            mock.patch(
                "hyper_parallel.trainer.vlm_trainer.clip_grad_norm_",
                return_value=clip_result,
            ) as clip:
        return trainer.train_step(iter(())), reported, clip, trainer


class TestVlmTrainerGradientClip(unittest.TestCase):
    """``max_grad_norm`` below 1 switches the per-step clip off."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_positive_max_grad_norm_clips_with_the_configured_value(self):
        """A positive threshold still clips, with the configured value."""
        result, reported, clip, trainer = _run_step(1.0, torch.tensor(0.5))

        clip.assert_called_once_with(trainer.base.model, 1.0)
        self.assertAlmostEqual(result["grad_norm"], 0.5)
        self.assertAlmostEqual(reported["grad_norm"], 0.5)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_zero_max_grad_norm_skips_the_clip(self):
        """``0.0`` disables clipping without touching the gradients."""
        result, reported, clip, _trainer = _run_step(0.0, torch.tensor(0.5))

        clip.assert_not_called()
        self.assertEqual(result["grad_norm"], 0.0)
        self.assertEqual(reported["grad_norm"], 0.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_negative_max_grad_norm_skips_the_clip(self):
        """A negative threshold is also "clipping off", never a negated scale."""
        result, reported, clip, _trainer = _run_step(-1.0, torch.tensor(0.5))

        clip.assert_not_called()
        self.assertEqual(result["grad_norm"], 0.0)
        self.assertEqual(reported["grad_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
