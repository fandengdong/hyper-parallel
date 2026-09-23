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
"""Unit tests for the merged Omni/VLM packing path.

Real instruction samples are short (~175 tokens for COCO LLaVA-instruct) while
the window is 8192, so padding each sample costs most of the compute. The merged
design packs with :class:`SamplePacker` / :class:`FirstFitPackingSelector`:
candidate samples are concatenated in order up to the token budget, and the
packed batch carries ``cu_seq_lens`` so the model can build a block-diagonal
attention mask. These tests pin the field handling (text and modality) and the
``cu_seq_lens`` contract that keeps attention from leaking across samples.
"""
# pylint: disable=wrong-import-position

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # noqa: E402

from hyper_parallel.data.batching.build_collate_fn import build_omni_collate_fn
from hyper_parallel.data.batching.dynamic_batch import PackingCandidateBuffer
from hyper_parallel.data.batching.packing import FirstFitPackingSelector, SamplePacker
from hyper_parallel.data.batching.runtime_input import AttentionRuntime, RuntimeInputContext


def _sample(length, image_rows=0, offset=0):
    """Build one fake encoded sample of ``length`` tokens."""
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


class TestPackedFields(unittest.TestCase):
    """Text concatenates on the token axis, modality concatenates on dim 0."""

    def test_text_fields_concatenate_in_source_order(self):
        """Every token field follows the packed sample order."""
        packed = SamplePacker().pack_selected_samples([_sample(4, offset=0), _sample(3, offset=4)])

        self.assertEqual(packed["input_ids"].tolist(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(packed["labels"].tolist(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(packed["attention_mask"].tolist(), [1] * 7)
        self.assertEqual(packed["mm_token_type_ids"].tolist(), [0] * 7)

    def test_modality_fields_concatenate_on_dim_zero(self):
        """Vision rows must follow the image placeholders in the packed ids."""
        first = _sample(4, image_rows=2, offset=0)
        second = _sample(3, image_rows=1, offset=4)
        packed = SamplePacker().pack_selected_samples([first, second])

        self.assertEqual(tuple(packed["pixel_values"].shape), (3, 3))
        self.assertEqual(packed["pixel_values"][0, 0].item(), 0.0)
        self.assertEqual(packed["pixel_values"][-1, 0].item(), 4.0)
        self.assertEqual(tuple(packed["image_grid_thw"].shape), (2, 3))

    def test_boundaries_describe_every_sub_sequence(self):
        """Each packed sample contributes one boundary, in order."""
        packed = SamplePacker().pack_selected_samples(
            [_sample(100), _sample(200), _sample(300)]
        )
        boundaries = packed["cu_seq_lens"]

        self.assertEqual(boundaries.dtype, torch.int32)
        self.assertEqual(boundaries.tolist(), [0, 100, 300, 600])
        self.assertTrue(bool((boundaries[1:] >= boundaries[:-1]).all()))

    def test_packing_metadata_is_not_packed(self):
        """``packing_length`` and an incoming ``cu_seq_lens`` are scheduling-only."""
        first = _sample(4)
        first["packing_length"] = 4
        second = _sample(3, offset=4)
        second["packing_length"] = 3
        packed = SamplePacker().pack_selected_samples([first, second])

        self.assertNotIn("packing_length", packed)
        self.assertEqual(packed["cu_seq_lens"].tolist(), [0, 4, 7])

    def test_single_sample_pack_is_a_passthrough_concat(self):
        """One selected sample still yields one packed sample with a boundary."""
        packed = SamplePacker().pack_selected_samples([_sample(5)])

        self.assertEqual(packed["input_ids"].shape[0], 5)
        self.assertEqual(packed["cu_seq_lens"].tolist(), [0, 5])


class TestSelectorBudget(unittest.TestCase):
    """The selector fills the token budget in order and never splits a sample."""

    def test_first_fit_fills_the_budget_without_splitting(self):
        """11 x 175 = 1925 fits a 2048 budget; the 12th sample is left behind."""
        selector = FirstFitPackingSelector()
        samples = [_sample(175, offset=index * 175) for index in range(14)]

        selected = selector.select_samples_to_pack(samples, 2048)

        self.assertEqual(list(selected), list(range(11)))
        self.assertEqual(selector.get_sample_cost(samples[0]), 175)

    def test_a_single_oversized_sample_forms_its_own_group(self):
        """The first candidate is always accepted, even above the budget."""
        selector = FirstFitPackingSelector()

        self.assertEqual(list(selector.select_samples_to_pack([_sample(9000)], 2048)), [0])

    def test_packing_length_overrides_the_encoded_length(self):
        """A prepared sample can report a cheaper scheduling cost."""
        sample = _sample(100)
        sample["packing_length"] = 7

        self.assertEqual(FirstFitPackingSelector().get_sample_cost(sample), 7)

    def test_empty_budget_group_is_rejected(self):
        """Nothing to select is an upstream bug, not an empty packed window."""
        with self.assertRaises(ValueError):
            FirstFitPackingSelector().select_samples_to_pack([], 2048)
        with self.assertRaises(ValueError):
            PackingCandidateBuffer(
                token_budget=16,
                min_buffered_samples=1,
                packing_selector=FirstFitPackingSelector(),
            ).get_micro_batch()

    def test_candidate_buffer_selects_and_retains_the_remainder(self):
        """Selected candidates leave the buffer; the rest survive for the next step."""
        buffer = PackingCandidateBuffer(
            token_budget=10,
            min_buffered_samples=1,
            packing_selector=FirstFitPackingSelector(),
        )
        for sample in (_sample(6, offset=0), _sample(6, offset=6)):
            buffer.put_item(sample)

        selected = buffer.get_micro_batch()

        self.assertEqual([sample["input_ids"][0].item() for sample in selected], [0])
        self.assertEqual(len(buffer.buffer), 1)
        self.assertEqual(buffer.buffer_token_count, 6)

    def test_packing_one_window_collates_with_the_omni_collator(self):
        """A packed step is one sample, and OmniCollator accepts it unchanged."""
        packed = SamplePacker().pack_selected_samples([_sample(4), _sample(3, offset=4)])
        batch = build_omni_collate_fn()([packed])

        self.assertEqual(tuple(batch["input_ids"].shape), (1, 7))
        self.assertEqual(batch["cu_seq_lens"].reshape(-1).tolist(), [0, 4, 7])


def _attention_mask(boundaries, seq_length, reset_mask=True):
    """Build the runtime attention mask for one packed single-row batch."""
    runtime = AttentionRuntime(
        mode="dense",
        create_mask=True,
        reset_mask=reset_mask,
        sliding_window=None,
    )
    context = RuntimeInputContext(
        local_input_shape=(1, seq_length),
        parallel_ranks={"tp": 0, "cp": 0},
        parallel_sizes={"tp": 1, "cp": 1},
        options={},
    )
    batch = {"input_ids": torch.zeros(1, seq_length, dtype=torch.long), "cu_seq_lens": boundaries}
    runtime_inputs = runtime.build_runtime_inputs(batch=batch, context=context)
    return runtime_inputs["attention_mask"]


class TestBlockDiagonalAttention(unittest.TestCase):
    """``cu_seq_lens`` is what keeps attention from leaking across packed samples."""

    @staticmethod
    def _sdpa_packed(q, k, v, mask):
        """Run SDPA once over the packed window with the runtime block mask."""
        return torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=mask
        )

    @staticmethod
    def _sdpa_per_sample(q, k, v, cu):
        """Run SDPA separately per sub-sequence and concatenate, as the ground truth."""
        outs = []
        start = 0
        for end in [int(boundary) for boundary in cu[1:]]:
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
        """The runtime mask is exactly the block-diagonal causal indicator."""
        cu = torch.tensor([0, 2, 5], dtype=torch.int32)
        mask = _attention_mask(cu, 5)[0, 0]
        expected = torch.tensor(
            [[1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 1, 0],
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
        mask = _attention_mask(cu, total)

        packed = self._sdpa_packed(q, k, v, mask)
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

        reference = self._sdpa_per_sample(q, k, v, cu)
        leaking = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, attn_mask=torch.tril(torch.ones(total, total, dtype=torch.bool)).view(1, 1, total, total)
        )

        self.assertFalse(torch.allclose(leaking, reference, rtol=1e-6, atol=1e-6))
        self.assertTrue(
            torch.allclose(self._sdpa_packed(q, k, v, _attention_mask(cu, total)), reference,
                           rtol=1e-6, atol=1e-6)
        )

    def test_bad_boundaries_are_rejected(self):
        """Malformed boundaries must fail loudly rather than produce a wrong mask."""
        with self.assertRaises(ValueError):
            _attention_mask(torch.tensor([1, 3], dtype=torch.int32), 3)
        with self.assertRaises(ValueError):
            _attention_mask(torch.tensor([0, 3, 3], dtype=torch.int32), 3)
        with self.assertRaises(ValueError):
            _attention_mask(torch.tensor([0, 2], dtype=torch.int32), 5)


if __name__ == "__main__":
    unittest.main()
