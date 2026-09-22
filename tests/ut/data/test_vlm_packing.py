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
"""Unit tests for VeOmni-style VLM packing (fill the window, pad the remainder).

Real instruction samples are short (~175 tokens for COCO LLaVA-instruct) while the window is
8192, so padding each sample costs 97.7% of the compute.  Packing is only *sound* if the
sub-sequence boundaries travel with the batch, so these tests pin both halves: the window
filling arithmetic, and the ``cu_seq_lens`` contract that keeps attention block diagonal.
"""
# pylint: disable=wrong-import-position

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # noqa: E402

from hyper_parallel.data.vlm.packing import (  # noqa: E402
    VlmPackingCollator,
    block_diagonal_mask,
    pack_vlm_samples,
    sample_length,
)


def _sample(length, image_rows=0, offset=0):
    """Build one fake transformed sample of ``length`` tokens."""
    sample = {
        "input_ids": torch.arange(offset, offset + length),
        "labels": torch.arange(offset, offset + length),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(length, dtype=torch.long),
    }
    if image_rows:
        sample["pixel_values"] = torch.full((image_rows, 3), float(offset))
        sample["image_grid_thw"] = torch.tensor([[1, 2, image_rows // 2]])
    return sample


class TestWindowFilling(unittest.TestCase):
    """Greedy filling must never split a sample and must fill to the cap."""

    def test_short_samples_are_packed_until_the_window_is_full(self):
        """14 samples of 175 tokens fill one 2048-token window and leave a tail window."""
        samples = [_sample(175, offset=i * 175) for i in range(14)]
        windows = pack_vlm_samples(samples, max_seq_len=2048)
        # 11 x 175 = 1925 fits, the 12th would need 2100 > 2048.
        self.assertEqual(len(windows), 2)
        first = windows[0]["input_ids"]
        self.assertEqual(first.numel(), 2048)
        # 11 x 175 = 1925 real tokens, then 123 padded zeros.  (Counting non-zeros would
        # undercount by one, because the very first token id IS zero -- padding shares the
        # value, so compare the token stream instead.)
        self.assertTrue(torch.equal(first[0, :1925], torch.arange(1925)))
        self.assertTrue(bool((first[0, 1925:] == 0).all()))

    def test_only_the_remainder_is_padded(self):
        """The last window pads only what is missing, and its labels are ignored."""
        samples = [_sample(3000, offset=0), _sample(3000, offset=3000)]
        windows = pack_vlm_samples(samples, max_seq_len=8192)
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window["input_ids"].numel(), 8192)
        self.assertEqual(int((window["labels"] == -100).sum()), 8192 - 6000)

    def test_a_full_window_is_emitted_without_extra_padding(self):
        """Exactly-full windows leave no tail and need no synthetic boundary."""
        windows = pack_vlm_samples([_sample(4096), _sample(4096)], max_seq_len=8192)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["cu_seq_lens"].tolist(), [0, 4096, 8192])

    def test_oversized_sample_is_rejected_not_split(self):
        """A sample longer than the window is a transform bug, so it must raise."""
        with self.assertRaises(ValueError):
            pack_vlm_samples([_sample(9000)], max_seq_len=8192)


class TestBoundaries(unittest.TestCase):
    """``cu_seq_lens`` is what keeps attention from leaking across packed samples."""

    def test_boundaries_describe_every_sub_sequence(self):
        """Each packed sample contributes one boundary, in order."""
        windows = pack_vlm_samples([_sample(100), _sample(200), _sample(300)], max_seq_len=8192)
        self.assertEqual(windows[0]["cu_seq_lens"].tolist(), [0, 100, 300, 600, 8192])

    def test_padding_tail_is_its_own_boundary(self):
        """The padded remainder is covered by a synthetic final boundary."""
        windows = pack_vlm_samples([_sample(5000)], max_seq_len=8192)
        cu = windows[0]["cu_seq_lens"].tolist()
        self.assertEqual(cu[0], 0)
        self.assertEqual(cu[-1], 8192)
        self.assertIn(5000, cu)

    def test_boundaries_are_int32_and_monotone(self):
        """Attention kernels index with int32 and require non-decreasing ends."""
        windows = pack_vlm_samples([_sample(700), _sample(700)], max_seq_len=2048)
        cu = windows[0]["cu_seq_lens"]
        self.assertEqual(cu.dtype, torch.int32)
        self.assertTrue(bool((cu[1:] >= cu[:-1]).all()))


class TestFieldHandling(unittest.TestCase):
    """Text resets, modality order, and the length contract."""

    def test_positions_restart_inside_each_packed_sample(self):
        """A packed sample must not inherit the previous sample's positions."""
        first = _sample(4, offset=0)
        first["position_ids"] = torch.tensor([0, 1, 2, 3])
        second = _sample(3, offset=4)
        second["position_ids"] = torch.tensor([10, 11, 12])
        window = pack_vlm_samples([first, second], max_seq_len=8)[0]
        self.assertEqual(window["position_ids"][0, :7].tolist(), [0, 1, 2, 3, 0, 1, 2])

    def test_images_are_concatenated_in_packing_order(self):
        """Vision rows must follow the order of the image placeholders in the packed ids."""
        first = _sample(4, image_rows=2, offset=0)
        second = _sample(3, image_rows=1, offset=4)
        window = pack_vlm_samples([first, second], max_seq_len=8)[0]
        self.assertEqual(window["pixel_values"].shape[0], 3)
        self.assertEqual(window["pixel_values"][0, 0].item(), 0.0)
        self.assertEqual(window["pixel_values"][-1, 0].item(), 4.0)
        self.assertEqual(window["image_grid_thw"].shape[0], 2)

    def test_multidimensional_position_fields_keep_their_axes(self):
        """mRoPE positions are ``[3, S]``: packing must not flatten them into ``[3*S]``.

        The first launch died on all 256 ranks with "too many indices for tensor of dimension
        1" because the packing flattened these; a position field must be concatenated along
        the token axis only, and reset per sub-sequence on that same axis.
        """
        first = _sample(4, offset=0)
        first["position_ids"] = torch.stack([torch.tensor([0, 1, 2, 3])] * 3)
        second = _sample(3, offset=4)
        second["position_ids"] = torch.stack([torch.tensor([9, 10, 11])] * 3)
        window = pack_vlm_samples([first, second], max_seq_len=8)[0]
        self.assertEqual(window["position_ids"].shape, (1, 3, 8))
        self.assertEqual(window["position_ids"][0, 0, :7].tolist(), [0, 1, 2, 3, 0, 1, 2])
        self.assertEqual(window["position_ids"][0, 2, :7].tolist(), [0, 1, 2, 3, 0, 1, 2])

    def test_text_position_ids_of_varying_rank_still_pack(self):
        """A 1-D field and a 2-D field may coexist in one batch."""
        first = _sample(3, offset=0)
        first["text_position_ids"] = torch.tensor([0, 1, 2])
        second = _sample(3, offset=3)
        second["text_position_ids"] = torch.tensor([0, 1, 2])
        window = pack_vlm_samples([first, second], max_seq_len=8)[0]
        self.assertEqual(window["text_position_ids"].shape, (1, 8))

    def test_length_helper_reports_the_token_count(self):
        """The filler needs the token length, and a sample without ids is a bug."""
        self.assertEqual(sample_length(_sample(37)), 37)
        with self.assertRaises(ValueError):
            sample_length({})


class TestPackingCollator(unittest.TestCase):
    """The collator is the trainer-facing half: one step, one window, hard errors otherwise."""

    def test_one_step_becomes_one_window(self):
        """A selection inside the budget packs to exactly one padded window."""
        collator = VlmPackingCollator(max_seq_len=2048)
        window = collator([_sample(175, offset=i * 175) for i in range(10)])
        self.assertEqual(window["input_ids"].numel(), 2048)
        self.assertEqual(window["cu_seq_lens"][0].item(), 0)
        self.assertEqual(window["cu_seq_lens"][-1].item(), 2048)

    def test_oversized_selection_raises_instead_of_dropping(self):
        """Two windows mean the sampler budget disagrees with the window size."""
        collator = VlmPackingCollator(max_seq_len=512)
        with self.assertRaises(ValueError):
            collator([_sample(400), _sample(400)])

    def test_empty_selection_raises(self):
        """An empty step is a bug upstream, not something to pack."""
        with self.assertRaises(ValueError):
            VlmPackingCollator(max_seq_len=512)([])

    def test_builder_accepts_the_packing_flag(self):
        """``build_vlm_collator(packing=True)`` must return the packing collator, not raise."""
        from hyper_parallel.data.vlm.collator import build_vlm_collator  # pylint: disable=C0415

        collator = build_vlm_collator(packing=True, max_seq_len=8192)
        self.assertIsInstance(collator, VlmPackingCollator)
        self.assertEqual(collator.max_seq_len, 8192)

    def test_invalid_window_is_rejected(self):
        """A non-positive window cannot be packed into."""
        with self.assertRaises(ValueError):
            VlmPackingCollator(max_seq_len=0)


if __name__ == "__main__":
    unittest.main()


class TestBlockDiagonalAttention(unittest.TestCase):
    """The 0-card gate: packed attention must equal per-sample attention.

    Concatenating tokens without enforcing the boundaries would let a token attend to a
    different sample's tokens, which trains a different model silently.  These tests build the
    mask from ``cu_seq_lens`` and check the *numerical* consequence, not just the mask's shape.
    """

    @staticmethod
    def _sdpa_packed(q, k, v, cu):
        """Run SDPA once over the packed window with the block-diagonal mask."""
        total = int(cu[-1])
        mask = block_diagonal_mask(cu)
        # Causal *within* each sub-sequence: AND the block mask with a lower-triangular mask.
        causal = torch.tril(torch.ones(total, total, dtype=torch.bool))
        return torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=(mask & causal).unsqueeze(0).unsqueeze(0)
        )

    @staticmethod
    def _sdpa_per_sample(q, k, v, cu):
        """Run SDPA separately per sub-sequence and concatenate, as the ground truth."""
        outs = []
        start = 0
        for end in [int(x) for x in cu[1:]]:
            qs, ks, vs = q[..., start:end, :], k[..., start:end, :], v[..., start:end, :]
            length = end - start
            causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
            outs.append(
                torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                    qs, ks, vs, attn_mask=causal.unsqueeze(0).unsqueeze(0)
                )
            )
            start = end
        return torch.cat(outs, dim=-2)

    def test_mask_is_true_only_inside_a_sub_sequence(self):
        """The mask is exactly the block-diagonal indicator."""
        cu = torch.tensor([0, 2, 5], dtype=torch.int32)
        mask = block_diagonal_mask(cu)
        expected = torch.tensor(
            [[1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1],
             [0, 0, 1, 1, 1],
             [0, 0, 1, 1, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_packed_attention_equals_per_sample_attention(self):
        """Numerical gate: one packed SDPA == per-sample SDPA concatenated."""
        torch.manual_seed(0)
        cu = torch.tensor([0, 5, 9, 14], dtype=torch.int32)
        total = int(cu[-1])
        heads, dim = 2, 8
        q = torch.randn(1, heads, total, dim)
        k = torch.randn(1, heads, total, dim)
        v = torch.randn(1, heads, total, dim)
        packed = self._sdpa_packed(q, k, v, cu)
        reference = self._sdpa_per_sample(q, k, v, cu)
        self.assertTrue(
            torch.allclose(packed, reference, rtol=1e-6, atol=1e-6),
            f"max abs diff {(packed - reference).abs().max().item():.3e}",
        )

    def test_without_the_mask_the_samples_would_leak(self):
        """Sanity: the gate must be able to fail -- a plain causal mask changes the result."""
        torch.manual_seed(0)
        cu = torch.tensor([0, 5, 9], dtype=torch.int32)
        total = int(cu[-1])
        q = torch.randn(1, 1, total, 4)
        k = torch.randn(1, 1, total, 4)
        v = torch.randn(1, 1, total, 4)
        packed = self._sdpa_packed(q, k, v, cu)
        reference = self._sdpa_per_sample(q, k, v, cu)
        leaking = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=torch.tril(torch.ones(total, total, dtype=torch.bool)).view(1, 1, total, total)
        )
        self.assertFalse(torch.allclose(leaking, reference, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(packed, reference, rtol=1e-6, atol=1e-6))

    def test_collator_ships_the_mask(self):
        """A packed step carries both the boundaries and their enforcement."""
        window = VlmPackingCollator(max_seq_len=512)([_sample(200), _sample(200)])
        self.assertIn("block_diagonal_mask", window)
        mask = window["block_diagonal_mask"]
        self.assertEqual(mask.shape, (512, 512))
        self.assertTrue(bool(mask[0, 0]))
        self.assertFalse(bool(mask[0, 300]))

    def test_bad_boundaries_are_rejected(self):
        """Malformed boundaries must fail loudly rather than produce a wrong mask."""
        with self.assertRaises(ValueError):
            block_diagonal_mask(torch.tensor([1, 3], dtype=torch.int32))
        with self.assertRaises(ValueError):
            block_diagonal_mask(torch.tensor([0, 3, 3], dtype=torch.int32))
