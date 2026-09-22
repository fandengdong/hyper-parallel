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
"""Vision patches in the TFLOPS/MFU/HFU accounting.

A vision tower runs dense ops over every patch of every image, so its FLOPs do not
scale with ``input_ids`` at all.  Charging a VLM step for its language tokens only
understates the metric exactly where a packed 8K window puts ten images per rank
(16,560 patches) instead of one (1,656).
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.models.flops import VisionFlopsEstimate
from hyper_parallel.trainer.callbacks import environ_meter_callback
from hyper_parallel.trainer.callbacks.environ_meter_callback import (
    EnvironMeterCallback,
)


def _build_callback(vision=None, flops_per_token=1.0e9):
    """Build a meter around a stub trainer with exact, hand-checkable numbers."""
    model = SimpleNamespace(hp_flops_per_token=flops_per_token)
    if vision is not None:
        model.hp_vision_flops = vision
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(peak_tflops=400.0, hfu_recompute_factor=None),
            activation_checkpoint=SimpleNamespace(mode="full"),
        ),
        model=model,
        model_config=None,
        mesh=None,
        lr_scheduler=None,
        optimizer=None,
    )
    callback = EnvironMeterCallback(trainer)
    # Keep the accounting test device-independent: memory metrics need a real NPU.
    callback._memory_metrics = lambda: {}
    return callback


def _run_step(callback, batch, step_time=1.0):
    """Run one step with a pinned clock so the TFLOPS arithmetic is exact."""
    clock = mock.Mock()
    clock.perf_counter.side_effect = [0.0, step_time]
    with mock.patch.object(environ_meter_callback, "time", clock):
        callback.on_step_begin(None, micro_batches=[batch])
        callback.on_step_end(None, loss=1.0, loss_dict=None, grad_norm=1.0)
    return callback.trainer.step_env_metrics


def _batch(*, tokens=8192, supervised=100, grids=None, patches=None):
    """Build a micro-batch carrying text ids and optional vision media."""
    batch = {
        "input_ids": torch.zeros(1, tokens, dtype=torch.long),
        "labels": torch.full((1, tokens), -100, dtype=torch.long),
    }
    batch["labels"][0, :supervised] = 1
    if grids is not None:
        batch["image_grid_thw"] = torch.tensor(grids, dtype=torch.long)
    if patches is not None:
        batch["pixel_values"] = torch.zeros(patches, 3, 14, 14)
    return batch


class TestVisionPatchCounts(unittest.TestCase):
    """Patch counting reads the media grids, which is what the tower consumes."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_image_grid_products_and_squares(self):
        """Each ``(t, h, w)`` row is one image; the square sum drives attention."""
        batch = _batch(grids=[[1, 24, 69], [1, 16, 64]])

        self.assertEqual(
            EnvironMeterCallback._vision_patch_counts(batch),
            (1656 + 1024, 1656 ** 2 + 1024 ** 2),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_video_grid_is_counted_too(self):
        """Video media ride the same accounting through ``video_grid_thw``."""
        batch = _batch(grids=[[2, 8, 16]])
        batch.pop("image_grid_thw")
        batch["video_grid_thw"] = torch.tensor([[2, 8, 16]], dtype=torch.long)

        self.assertEqual(EnvironMeterCallback._vision_patch_counts(batch), (256, 256 ** 2))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pixel_values_fallback_and_absent_media(self):
        """No grid: patch rows are the fallback; no media at all: nothing counted."""
        batch = _batch(patches=100)
        self.assertEqual(EnvironMeterCallback._vision_patch_counts(batch), (100, 100 ** 2))

        # A malformed grid must not be trusted over the patch rows themselves.
        batch["image_grid_thw"] = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        self.assertEqual(EnvironMeterCallback._vision_patch_counts(batch), (100, 100 ** 2))

        self.assertIsNone(EnvironMeterCallback._vision_patch_counts(_batch()))


class TestVisionStepMetrics(unittest.TestCase):
    """MFU/HFU gain the tower's FLOPs instead of dropping them on the floor."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_vision_flops_are_added_to_tflops_mfu_and_hfu(self):
        """One 1656-patch image per rank costs 1.656 TFLOP/s at 1 s/step."""
        vision = VisionFlopsEstimate(per_patch=1.0e9, attention_coefficient=0.0, total_params=0.0)
        callback = _build_callback(vision=vision)

        metrics = _run_step(callback, _batch(grids=[[1, 24, 69]]))

        llm_tflops = 8192 * 1.0e9 / 1e12
        vision_tflops = 1656 * 1.0e9 / 1e12
        self.assertAlmostEqual(metrics["performance/tflops"], llm_tflops + vision_tflops, places=6)
        self.assertAlmostEqual(metrics["performance/tflops_vision"], vision_tflops, places=6)
        self.assertEqual(metrics["data/step_vision_patches"], 1656.0)
        mfu = (llm_tflops + vision_tflops) / 400.0
        self.assertAlmostEqual(metrics["performance/mfu"], mfu, places=9)
        # Full activation checkpointing re-runs the forward for both halves.
        self.assertAlmostEqual(metrics["performance/hfu"], mfu * 4.0 / 3.0, places=9)
        self.assertGreater(metrics["performance/mfu"], llm_tflops / 400.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_attention_term_uses_the_square_sum(self):
        """Ten images must be charged ten times the attention of one image."""
        vision = VisionFlopsEstimate(
            per_patch=0.0, attention_coefficient=1.0e9, total_params=0.0)
        callback = _build_callback(vision=vision)
        grids = [[1, 24, 69]] * 10

        metrics = _run_step(callback, _batch(grids=grids))

        self.assertAlmostEqual(
            metrics["performance/tflops_vision"], 10 * 1656 ** 2 * 1.0e9 / 1e12, places=6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_text_only_model_keeps_the_previous_metric(self):
        """No vision tower means no new keys and unchanged TFLOPS."""
        callback = _build_callback(vision=None)

        metrics = _run_step(callback, _batch(grids=[[1, 24, 69]]))

        self.assertNotIn("performance/tflops_vision", metrics)
        self.assertNotIn("data/step_vision_patches", metrics)
        self.assertAlmostEqual(metrics["performance/tflops"], 8192 * 1.0e9 / 1e12, places=6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_text_only_batch_of_a_vlm_reports_zero_patches(self):
        """A VLM step with no images reports the metric as zero, not absent."""
        vision = VisionFlopsEstimate(per_patch=1.0e9, attention_coefficient=0.0, total_params=0.0)
        callback = _build_callback(vision=vision)

        metrics = _run_step(callback, _batch())

        self.assertEqual(metrics["data/step_vision_patches"], 0.0)
        self.assertNotIn("performance/tflops_vision", metrics)
        self.assertAlmostEqual(metrics["performance/tflops"], 8192 * 1.0e9 / 1e12, places=6)


if __name__ == "__main__":
    unittest.main()
