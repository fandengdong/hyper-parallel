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
"""Unit tests for ``hyper_parallel.models.flops`` FLOPs-per-token estimation."""

import unittest
from types import SimpleNamespace

from hyper_parallel.models.flops import (
    VisionFlopsEstimate,
    batch_seq_len,
    estimate_flops_per_token,
    estimate_vision_flops,
    resolve_flops_per_token,
    resolve_vision_flops,
)
from tests.common.mark_utils import arg_mark


# Kimi-K2.6 full geometry (MLA + fine-grained MoE); the 6N expectation is the
# value cross-checked against the previously hand-tuned per-token FLOPs.
_KIMI_GEOMETRY = dict(
    hidden_size=7168,
    num_hidden_layers=61,
    vocab_size=163840,
    num_attention_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    n_routed_experts=384,
    num_experts_per_tok=8,
    n_shared_experts=1,
    first_k_dense_replace=1,
    max_position_embeddings=4096,
)
# 6 * (61 * attn_proj + 60 * (8 routed + 1 shared experts + router)
#      + 1 dense MLP + lm_head) with the geometry above.
_KIMI_EXPECTED_6N = 190_116_397_056.0
# Additional attention score/weight term: 6 * 61 * 64 * (192 + 128) * 512.
_KIMI_EXPECTED_SEQ512 = _KIMI_EXPECTED_6N + 3_837_788_160.0

# Kimi-K2.6 vision tower (SigLIP-SO400M geometry) as it appears in config.json:
# the ``vision_config`` sub-config uses family-specific ``vt_*`` field names.
_KIMI_VISION = dict(
    vt_hidden_size=1152,
    vt_intermediate_size=4304,
    vt_num_attention_heads=16,
    vt_num_hidden_layers=27,
    patch_size=14,
    merge_kernel_size=[2, 2],
)
# Per layer: 4 * 1152**2 (q/k/v/out) + 2 * 1152 * 4304 (ungated MLP) = 15_224_832;
# x 27 layers = 411_070_464 parameters; 6N per patch = 2_466_422_784 FLOPs.
_KIMI_VISION_PARAMS = 411_070_464.0
_KIMI_VISION_PER_PATCH = 2_466_422_784.0
# 6 * 27 layers * 16 heads * (72 + 72) == 6 * 27 * 2 * 1152 -> 373_248 per patch^2.
_KIMI_VISION_ATTENTION = 373_248.0


class TestEstimateFlopsPerToken(unittest.TestCase):
    """Geometry-driven estimation across attention and MoE conventions."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mla_moe_geometry_mapping_and_object(self):
        """MLA + DeepSeek-style MoE fields, accepted as mapping and object."""
        for config in (dict(_KIMI_GEOMETRY), SimpleNamespace(**_KIMI_GEOMETRY)):
            self.assertEqual(estimate_flops_per_token(config), _KIMI_EXPECTED_6N)
            self.assertEqual(
                estimate_flops_per_token(config, seq_len=512),
                _KIMI_EXPECTED_SEQ512,
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dense_gqa_geometry(self):
        """Dense GQA model: projections + dense MLP + lm_head + attention term."""
        config = SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=32,
            vocab_size=128256,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
        )
        # 6 * (32 * 41943040 attn + 32 * 3 * 4096 * 14336 MLP + 4096 * 128256)
        # + 6 * 32 * 32 * 256 * 2048 attention term.
        expected = 48_249_176_064.0
        self.assertEqual(estimate_flops_per_token(config, seq_len=2048), expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_qwen_moe_field_convention(self):
        """Qwen-MoE-style fields (num_experts/num_experts_per_tok, no shared)."""
        config = SimpleNamespace(
            hidden_size=2048,
            num_hidden_layers=48,
            vocab_size=151936,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=6144,
            moe_intermediate_size=768,
            num_experts=128,
            num_experts_per_tok=8,
        )
        self.assertEqual(estimate_flops_per_token(config), 18_249_940_992.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_insufficient_geometry_returns_none(self):
        """Missing layers/heads or MoE top-k yield None instead of a guess."""
        self.assertIsNone(estimate_flops_per_token(SimpleNamespace(hidden_size=128)))
        self.assertIsNone(estimate_flops_per_token(None))
        no_topk = dict(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            n_routed_experts=8,
        )
        self.assertIsNone(estimate_flops_per_token(no_topk))


class TestBatchSeqLen(unittest.TestCase):
    """Sequence-length extraction from micro-batches."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_seq_len_extraction(self):
        """The trailing input_ids dim of the first usable batch wins."""
        batch = {"input_ids": SimpleNamespace(shape=(2, 512))}
        self.assertEqual(batch_seq_len([batch]), 512)
        self.assertEqual(batch_seq_len(batch), 512)
        self.assertIsNone(batch_seq_len([{"labels": None}]))
        self.assertIsNone(batch_seq_len(None))


class TestResolveFlopsPerToken(unittest.TestCase):
    """Resolution policy: model property wins, config estimate is the fallback."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_model_property_wins(self):
        """A model-provided hp_flops_per_token overrides the estimate."""
        model = SimpleNamespace(hp_flops_per_token=1.0e11)
        value = resolve_flops_per_token(model, model_config=dict(_KIMI_GEOMETRY))
        self.assertEqual(value, 1.0e11)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_config_fallback_and_none(self):
        """Without the property, estimate from config; nothing -> None."""
        model = SimpleNamespace(config=SimpleNamespace(**_KIMI_GEOMETRY))
        self.assertEqual(resolve_flops_per_token(model), _KIMI_EXPECTED_6N)
        self.assertEqual(
            resolve_flops_per_token(model, seq_len=512), _KIMI_EXPECTED_SEQ512
        )
        self.assertIsNone(resolve_flops_per_token(None))


class TestEstimateVisionFlops(unittest.TestCase):
    """Vision-tower cost model: the tower scales with patches, not with tokens."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kimi_vt_prefixed_geometry(self):
        """Kimi's ``vt_*`` field names must resolve to the same 6N convention."""
        config = {"text_config": _KIMI_GEOMETRY, "vision_config": _KIMI_VISION}

        estimate = estimate_vision_flops(config)

        self.assertIsInstance(estimate, VisionFlopsEstimate)
        self.assertAlmostEqual(estimate.per_patch, _KIMI_VISION_PER_PATCH)
        self.assertAlmostEqual(estimate.attention_coefficient, _KIMI_VISION_ATTENTION)
        self.assertAlmostEqual(estimate.total_params, _KIMI_VISION_PARAMS)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_standard_hf_vision_config(self):
        """Object-style, HF-named vision geometry works without the vt_ prefix."""
        config = SimpleNamespace(
            vision_config=SimpleNamespace(
                hidden_size=768,
                num_hidden_layers=12,
                intermediate_size=3072,
                num_attention_heads=12,
            )
        )

        estimate = estimate_vision_flops(config)

        # 6 * 12 * (4 * 768**2 + 2 * 768 * 3072) = 509_607_936 per patch.
        self.assertAlmostEqual(estimate.per_patch, 509_607_936.0)
        # 6 * 12 * 2 * 768 = 110_592 per patch squared.
        self.assertAlmostEqual(estimate.attention_coefficient, 110_592.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_text_only_and_incomplete_configs_yield_none(self):
        """A language-only model must keep the metric exactly as it was."""
        self.assertIsNone(estimate_vision_flops(None))
        self.assertIsNone(estimate_vision_flops(_KIMI_GEOMETRY))
        self.assertIsNone(estimate_vision_flops({"vision_config": {"hidden_size": 32}}))
        self.assertIsNone(estimate_vision_flops({"vision_config": {"num_hidden_layers": 4}}))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_attention_term_uses_each_image_patch_count(self):
        """Ten 1656-patch images must not be charged as one 16560-patch image."""
        estimate = estimate_vision_flops({"vision_config": _KIMI_VISION})
        linear = estimate.per_patch * 16560

        many_small = estimate.flops(16560, 10 * 1656 ** 2)
        one_large = estimate.flops(16560, 16560 ** 2)

        self.assertAlmostEqual(many_small - linear, _KIMI_VISION_ATTENTION * 10 * 1656 ** 2)
        self.assertAlmostEqual(one_large - many_small, _KIMI_VISION_ATTENTION * (16560 ** 2 - 10 * 1656 ** 2))
        # A single image's cost scales linearly with its patch count.
        single = estimate.flops(1656, 1656 ** 2)
        self.assertAlmostEqual(estimate.flops(16560, 10 * 1656 ** 2), 10 * single)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_linear_only_when_patch_square_sum_is_unknown(self):
        """Without per-image grids the attention term is skipped, not guessed."""
        estimate = estimate_vision_flops({"vision_config": _KIMI_VISION})

        self.assertAlmostEqual(estimate.flops(4096), estimate.per_patch * 4096)
        self.assertAlmostEqual(estimate.flops(4096, None), estimate.per_patch * 4096)
        self.assertAlmostEqual(estimate.flops(4096, 0), estimate.per_patch * 4096)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_resolve_prefers_model_property_then_config(self):
        """An exact model-side model wins; otherwise the config geometry is used."""
        exact = VisionFlopsEstimate(per_patch=1.0, attention_coefficient=2.0, total_params=3.0)
        model = SimpleNamespace(hp_vision_flops=exact, config={"vision_config": _KIMI_VISION})

        self.assertIs(resolve_vision_flops(model), exact)

        model = SimpleNamespace(config={"vision_config": _KIMI_VISION})
        resolved = resolve_vision_flops(model)
        self.assertAlmostEqual(resolved.per_patch, _KIMI_VISION_PER_PATCH)

        self.assertIsNone(resolve_vision_flops(None))
        self.assertIsNone(resolve_vision_flops(SimpleNamespace(config=_KIMI_GEOMETRY)))
