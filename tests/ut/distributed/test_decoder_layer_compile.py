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
"""Unit tests for the decoder-layer compile contract resolution.

``get_compile_layers`` resolves, in order: the model-owned
``get_compile_layers()`` contract, then the generic HF-convention container
paths. The VLM families nest their text decoder under a tower attribute, so the
tests pin both that the text decoder (not the vision tower) is selected and that
plain text models keep resolving through the original paths.
"""
import os
import unittest
import unittest.mock

import torch
from torch import nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.compile import (  # noqa: E402  pylint: disable=wrong-import-position
    apply_compile,
    get_compile_layers,
)
from hyper_parallel.models.build_options import CompileConfig  # noqa: E402  pylint: disable=wrong-import-position


class _Tower(nn.Module):
    """A tower exposing ``layers`` — the shape both text and vision towers have."""

    def __init__(self, count: int, tag: str) -> None:
        """Build ``count`` layers and keep ``tag`` to identify the tower."""
        super().__init__()
        self.tag = tag
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(count)])


class _KimiK25Like(nn.Module):
    """``Kimi_K25ForConditionalGeneration``: text and vision towers under ``model``."""

    def __init__(self, text_layers: int = 4, vision_layers: int = 3) -> None:
        """Build the text decoder and the vision tower under ``model``."""
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _Tower(text_layers, "text")
        self.model.vision_tower = _Tower(vision_layers, "vision")
        self.lm_head = nn.Linear(2, 2)


class _DeclaringModel(nn.Module):
    """A model that owns the contract itself."""

    def __init__(self) -> None:
        """Build a fallback-visible container and a separate declared one."""
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
        self.declared = nn.ModuleList([nn.Linear(2, 2) for _ in range(5)])

    def get_compile_layers(self) -> nn.ModuleList:
        """Declare a container that no fallback path would find."""
        return self.declared


class TestGetCompileLayers(unittest.TestCase):
    """Pin the resolution order and the VLM text-tower preference."""

    def test_contract_wins_over_fallback_paths(self):
        """An explicit contract is authoritative, even when ``layers`` exists."""
        model = _DeclaringModel()
        layers = get_compile_layers(model)
        self.assertEqual(len(layers), 5)
        self.assertEqual([name for name, _ in layers], ["declared.0", "declared.1",
                                                        "declared.2", "declared.3",
                                                        "declared.4"])

    def test_vlm_selects_the_text_decoder_not_the_vision_tower(self):
        """Kimi-K2.5-style nesting resolves to the text tower's 61 layers."""
        model = _KimiK25Like(text_layers=4, vision_layers=3)
        layers = get_compile_layers(model)
        self.assertEqual(len(layers), 4)
        text_ids = {id(layer) for layer in model.model.language_model.layers}
        vision_ids = {id(layer) for layer in model.model.vision_tower.layers}
        for name, layer in layers:
            self.assertIn("language_model", name)
            self.assertNotIn("vision", name)
            self.assertIn(id(layer), text_ids)
            self.assertNotIn(id(layer), vision_ids)

    def test_vlm_with_one_extra_wrapper_level(self):
        """A wrapper above the VLM object still resolves to the text decoder."""

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                """Build the VLM under an extra wrapper level."""
                super().__init__()
                self.model = _KimiK25Like(text_layers=2, vision_layers=2)

        wrapper = Wrapper()
        layers = get_compile_layers(wrapper)
        self.assertEqual(len(layers), 2)
        text_ids = {id(layer) for layer in wrapper.model.model.language_model.layers}
        for _, layer in layers:
            self.assertIn(id(layer), text_ids)

    def test_plain_text_model_keeps_the_hf_conventions(self):
        """Text-only families resolve through ``layers`` and ``model.layers``."""

        class LlamaLike(nn.Module):
            def __init__(self) -> None:
                """Build a text-only model with a top-level ``layers`` container."""
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])

        class NestedLike(nn.Module):
            def __init__(self) -> None:
                """Build a text-only model with the layers under ``model``."""
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])

        self.assertEqual(len(get_compile_layers(LlamaLike())), 3)
        self.assertEqual(len(get_compile_layers(NestedLike())), 2)

    def test_unsupported_model_raises(self):
        """A model with no contract and no known container fails loudly."""

        class Opaque(nn.Module):
            def __init__(self) -> None:
                """Build a model with no recognisable layer container."""
                super().__init__()
                self.body = nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "no decoder-layer compile contract"):
            get_compile_layers(Opaque())


class TestApplyCompile(unittest.TestCase):
    """Exercise the plumbing end to end with a backend that needs no codegen."""

    @staticmethod
    def _model() -> nn.Module:
        class Tiny(nn.Module):
            def __init__(self) -> None:
                """Build three stacked linear decoder layers."""
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Apply the three layers in order."""
                for layer in self.layers:
                    x = layer(x)
                return x

        torch.manual_seed(0)
        return Tiny()

    def test_disabled_compile_returns_the_model_unchanged(self):
        """``enabled=False`` must not touch the module tree."""
        model = self._model()
        self.assertIs(apply_compile(model, CompileConfig(enabled=False)), model)

    def test_enabled_compile_wraps_each_layer_and_keeps_forward_working(self):
        """Every resolved layer is compiled and the forward output is preserved."""
        model = self._model()
        compiled: list[str] = []

        original_compile = nn.Module.compile

        def _record(self, *args, **kwargs):
            compiled.append("call")
            return original_compile(self, *args, **kwargs)

        x = torch.ones(1, 2)
        expected = model(x)
        with unittest.mock.patch.object(nn.Module, "compile", _record):
            apply_compile(model, CompileConfig(enabled=True, backend="eager"))
        # One compile per declared decoder layer, and a still-working forward.
        self.assertEqual(len(compiled), 3)
        torch.testing.assert_close(model(x), expected)


if __name__ == "__main__":
    unittest.main()
