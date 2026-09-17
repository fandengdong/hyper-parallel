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
"""Context-parallel target sharding in the temporary VLM batch path.

The VLM CP split keeps the media fields complete on every CP rank (the vision
tower and the full-sequence media scatter stay rank-local and consistent) and
slices only the loss targets to this rank's sequence window. The matching
input slice and offset mask are installed on the text tower by
``models/kimi_k25/adapter/distributed/context_parallel.py``.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

# Some CI pythons are built without liblzma: the lazy transformers ->
# torchvision import chain dies on "from _lzma import *". The VLM facade pulls
# transformers.AutoProcessor, so these tests run only where lzma is available.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None


class _FakeMesh:
    """Minimal mesh exposing the CP sizes read by the batch adapter."""

    def __init__(self, cp_size, cp_rank=0, pp_size=1):
        """Store the CP/PP topology used by VLMGetBatch validation."""
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.pp_size = pp_size


def _make_batch(seq_len=8, pad_tail=2):
    """Build one collated VLM batch with a padding tail on the labels."""
    batch_size = 2
    input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    labels = input_ids.clone()
    if pad_tail:
        labels[:, seq_len - pad_tail:] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": labels,
        "pixel_values": torch.randn(3, 4, 4),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestVlmGetBatchCpTargets(unittest.TestCase):
    """``VLMGetBatch`` shards loss targets per CP rank, media stays complete."""

    @staticmethod
    def _build(cp_size, cp_rank=0):
        from hyper_parallel.data.vlm.get_batch import VLMGetBatch

        return VLMGetBatch(
            mesh_context=_FakeMesh(cp_size, cp_rank),
            device=torch.device("cpu"),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp1_keeps_the_batch_untouched(self):
        """Without CP the adapter is a pass-through (regression guard)."""
        batch = _make_batch()
        model_inputs, loss_inputs = self._build(1)(None, external_batch=batch)

        self.assertTrue(torch.equal(loss_inputs["labels"], batch["labels"]))
        self.assertTrue(torch.equal(model_inputs["labels"], batch["labels"]))
        self.assertTrue(torch.equal(loss_inputs["loss_mask"], batch["labels"] >= 0))
        for field in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            self.assertTrue(torch.equal(model_inputs[field], batch[field]), field)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp2_slices_labels_to_the_rank_window(self):
        """Each rank keeps its own contiguous window of the loss targets."""
        seq_len = 8
        for cp_rank in (0, 1):
            with self.subTest(cp_rank=cp_rank):
                batch = _make_batch(seq_len=seq_len)
                expected = batch["labels"][:, cp_rank * 4:(cp_rank + 1) * 4]
                model_inputs, loss_inputs = self._build(2, cp_rank)(
                    None, external_batch=batch
                )

                self.assertEqual(tuple(loss_inputs["labels"].shape), (2, seq_len // 2))
                self.assertTrue(torch.equal(loss_inputs["labels"], expected))
                self.assertTrue(torch.equal(model_inputs["labels"], expected))
                self.assertTrue(torch.equal(loss_inputs["loss_mask"], expected >= 0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_keeps_media_fields_complete(self):
        """The vision inputs stay full-length so the media scatter is unsharded."""
        batch = _make_batch()
        model_inputs, _ = self._build(2, 1)(None, external_batch=batch)

        self.assertEqual(tuple(model_inputs["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(model_inputs["attention_mask"].shape), (2, 8))
        self.assertTrue(torch.equal(model_inputs["input_ids"], batch["input_ids"]))
        self.assertEqual(tuple(model_inputs["pixel_values"].shape), (3, 4, 4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_indivisible_sequence_length(self):
        """cp_size must divide the padded sequence length."""
        with self.assertRaisesRegex(ValueError, "divisible by cp_size"):
            self._build(3)(None, external_batch=_make_batch(seq_len=8))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_still_rejects_pipeline_parallelism(self):
        """PP stays out of scope for the temporary VLM batch path."""
        with self.assertRaisesRegex(NotImplementedError, "requires PP=1"):
            self._build(2).__class__(
                mesh_context=_FakeMesh(2, 0, pp_size=2), device=torch.device("cpu")
            )


if __name__ == "__main__":
    unittest.main()
