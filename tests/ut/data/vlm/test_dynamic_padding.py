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
"""Dynamic (batch-longest) padding for the temporary VLM data path.

The transform historically padded every sample to the fixed ``max_seq_len``,
so a micro-batch of short samples still paid ``max_seq_len`` worth of compute.
``pad_granularity`` lets the transform stop at the next multiple of a chosen
alignment and the collator then pads the micro-batch to its own longest sample.
Loss semantics are unchanged: padded label slots stay ``IGNORE_INDEX``.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

# The VLM facade pulls transformers.AutoProcessor, whose lazy import chain ends
# in torchvision; some CI pythons lack liblzma and cannot import it.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None

from hyper_parallel.data.constants import IGNORE_INDEX


def _sample(seq_len: int) -> dict:
    """Build one pre-collation sample whose text fields are ``seq_len`` long."""
    return {
        "input_ids": torch.arange(seq_len),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.zeros(seq_len, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(seq_len, dtype=torch.long),
        "pixel_values": torch.randn(3, 4),
    }


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestPadTarget(unittest.TestCase):
    """The transform stops padding at the granularity when one is configured."""

    @staticmethod
    def _transform(**kwargs):
        from hyper_parallel.data.vlm.kimi_transform import KimiVLMChatTransform

        return KimiVLMChatTransform(processor=object(), max_seq_len=8192, **kwargs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_without_granularity_pads_to_max_seq_len(self):
        """The historical fixed-length contract is preserved."""
        transform = self._transform()

        self.assertEqual(transform._pad_target(3000), 8192)
        self.assertEqual(transform._pad_target(8192), 8192)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_granularity_rounds_up_and_caps_at_max_seq_len(self):
        """Padded length is the next multiple of the granularity, capped."""
        transform = self._transform(pad_granularity=128)

        self.assertEqual(transform._pad_target(1), 128)
        self.assertEqual(transform._pad_target(128), 128)
        self.assertEqual(transform._pad_target(129), 256)
        self.assertEqual(transform._pad_target(3000), 3072)
        self.assertEqual(transform._pad_target(8189), 8192)
        self.assertEqual(transform._pad_target(9000), 8192)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_truncate_and_pad_pads_all_sequence_fields(self):
        """Every sequence field grows to the same granular length."""
        from hyper_parallel.data.vlm.kimi_transform import _SEQ_FIELDS

        transform = self._transform(pad_granularity=128)
        out = transform._truncate_and_pad(_sample(300))

        for field in _SEQ_FIELDS:
            self.assertEqual(int(out[field].shape[0]), 384, field)
        self.assertTrue(torch.equal(out["labels"][:300], torch.zeros(300, dtype=torch.long)))
        self.assertTrue((out["labels"][300:] == IGNORE_INDEX).all())
        self.assertEqual(int(out["pixel_values"].shape[0]), 3)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_bad_granularity(self):
        """A non-positive or non-integer granularity fails fast."""
        for bad in (0, -8, 1.5, True):
            with self.subTest(granularity=bad):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self._transform(pad_granularity=bad)


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestVlmCollatorBatchPadding(unittest.TestCase):
    """``VLMCollator(pad_to_batch=True)`` pads up to the micro-batch longest."""

    @staticmethod
    def _collator(**kwargs):
        from hyper_parallel.data.vlm.collator import VLMCollator

        return VLMCollator(**kwargs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pads_to_batch_longest_rounded_up(self):
        """The batch length is the longest sample rounded up to the granularity."""
        batch = self._collator(pad_to_batch=True, pad_granularity=128)(
            [_sample(300), _sample(1000)]
        )

        for field in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            self.assertEqual(tuple(batch[field].shape), (2, 1024), field)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pad_values_and_no_cross_sample_contamination(self):
        """Labels pad with IGNORE_INDEX, the rest with 0, content untouched."""
        batch = self._collator(pad_to_batch=True, pad_granularity=128)(
            [_sample(300), _sample(1000)]
        )

        self.assertTrue((batch["labels"][0][300:] == IGNORE_INDEX).all())
        self.assertTrue((batch["labels"][1][1000:] == IGNORE_INDEX).all())
        for field in ("input_ids", "attention_mask", "mm_token_type_ids"):
            self.assertTrue((batch[field][0][300:] == 0).all(), field)
            self.assertTrue((batch[field][1][1000:] == 0).all(), field)
        # The shorter row's real content must still be the shorter sample.
        self.assertTrue(torch.equal(batch["input_ids"][0][:300], torch.arange(300)))
        self.assertTrue(torch.equal(batch["input_ids"][1][:1000], torch.arange(1000)))
        self.assertEqual(tuple(batch["pixel_values"].shape), (6, 4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_sample_batch_keeps_content_length(self):
        """With MBS=1 the batch longest is the sample itself: no pad-only compute."""
        batch = self._collator(pad_to_batch=True, pad_granularity=128)([_sample(3000)])

        self.assertEqual(tuple(batch["input_ids"].shape), (1, 3072))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_disabled_by_default(self):
        """Without the flag the collator still requires equal lengths."""
        collator = self._collator()

        with self.assertRaises(RuntimeError):
            collator([_sample(300), _sample(1000)])
        batch = collator([_sample(300), _sample(300)])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 300))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_builder_wires_the_option_and_keeps_reserved_guards(self):
        """The builder forwards the new options and still rejects packing."""
        from hyper_parallel.data.vlm.collator import build_vlm_collator

        collator = build_vlm_collator(pad_to_batch=True, pad_granularity=64)
        self.assertTrue(collator.pad_to_batch)
        self.assertEqual(collator.pad_granularity, 64)
        with self.assertRaisesRegex(NotImplementedError, "packing"):
            build_vlm_collator(packing=True, pad_to_batch=True)
        with self.assertRaises(ValueError):
            build_vlm_collator(pad_to_batch=True, pad_granularity=0)


if __name__ == "__main__":
    unittest.main()
