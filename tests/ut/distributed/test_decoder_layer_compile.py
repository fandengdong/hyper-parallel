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

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import (  # noqa: E402  pylint: disable=wrong-import-position
    checkpoint_wrapper,
)
from hyper_parallel.distributed.compile import (  # noqa: E402  pylint: disable=wrong-import-position
    apply_compile,
    get_compile_layers,
    _resolve_compile_targets,
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


class _NamedBlockTower(nn.Module):
    """A text tower whose wrapped blocks are named ``self_attn`` / ``mlp``."""

    def __init__(self, count: int = 2) -> None:
        """Build ``count`` layers, each holding two wrapped blocks."""
        super().__init__()
        self.tag = "text"
        self.layers = nn.ModuleList()
        for _ in range(count):
            layer = nn.Module()
            layer.self_attn = checkpoint_wrapper(nn.Linear(2, 2))
            layer.mlp = checkpoint_wrapper(nn.Linear(2, 2))
            self.layers.append(layer)


class _NamedBlockVlm(nn.Module):
    """A VLM-shaped model with named, wrapped text blocks."""

    def __init__(self, count: int = 2) -> None:
        """Build the wrapped text tower under ``model``."""
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _NamedBlockTower(count)


class TestCompileInsideAcInclude(unittest.TestCase):
    """Pin the FQN filter that keeps MoE blocks out of the compiled graph.

    Compiling a MoE block makes execution fail inside the fused grouped GEMM
    (``call aclnnGroupedMatmulV5 failed, error code is 1``), so the filter exists to
    compile only the blocks whose glue is safe to fuse (e.g. attention).
    """

    def _resolve(self, model, include):
        """Resolve targets with the inside-AC env set and ``include`` as the filter."""
        os.environ["HP_COMPILE_INSIDE_AC"] = "1"
        previous = os.environ.get("HP_COMPILE_INSIDE_AC_INCLUDE")
        os.environ["HP_COMPILE_INSIDE_AC_INCLUDE"] = include
        try:
            return _resolve_compile_targets(model)
        finally:
            os.environ.pop("HP_COMPILE_INSIDE_AC")
            if previous is None:
                os.environ.pop("HP_COMPILE_INSIDE_AC_INCLUDE", None)
            else:
                os.environ["HP_COMPILE_INSIDE_AC_INCLUDE"] = previous

    def test_filter_keeps_only_matching_blocks(self):
        """Only the attention blocks survive the filter."""
        targets = self._resolve(_NamedBlockVlm(count=2), r"self_attn$")
        self.assertEqual(len(targets), 2)
        for path, module in targets:
            self.assertTrue(path.endswith("self_attn"))
            self.assertIsInstance(module, nn.Linear)

    def test_filter_matching_nothing_falls_back(self):
        """An over-restrictive filter degrades to the default discovery, not to nothing."""
        model = _NamedBlockVlm(count=2)
        targets = self._resolve(model, r"does_not_match_anything")
        self.assertEqual(
            [module for _, module in targets],
            [module for _, module in get_compile_layers(model)],
        )

    def test_invalid_regex_falls_back_instead_of_raising(self):
        """A bad regex must widen the target set, never break the run."""
        model = _NamedBlockVlm(count=2)
        targets = self._resolve(model, r"self_attn[")
        self.assertEqual(len(targets), 4)


class _WrappedTower(nn.Module):
    """A text tower whose layers are activation-checkpoint wrappers."""

    def __init__(self, count: int = 3) -> None:
        """Wrap ``count`` linear layers in checkpoint wrappers."""
        super().__init__()
        self.tag = "text"
        self.layers = nn.ModuleList([checkpoint_wrapper(nn.Linear(2, 2)) for _ in range(count)])


class _WrappedVlm(nn.Module):
    """A VLM-shaped model whose text decoder layers are checkpoint wrappers."""

    def __init__(self, count: int = 3) -> None:
        """Build the wrapped text tower under ``model``."""
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _WrappedTower(count)


class TestResolveCompileTargets(unittest.TestCase):
    """Pin the HP_COMPILE_INSIDE_AC target selection.

    Compiling a decoder layer while activation checkpointing is on puts the
    checkpoint context manager inside the traced region, and dynamo refuses to enter
    a ``_GeneratorContextManager``: every layer breaks at its entry and runs eagerly,
    so nothing is fused.  The env moves the compile target inside the wrapper.
    """

    def _targets(self, model):
        """Resolve targets with the env unset, then set, restoring the environment."""
        previous = os.environ.pop("HP_COMPILE_INSIDE_AC", None)
        try:
            default_targets = _resolve_compile_targets(model)
        finally:
            if previous is not None:
                os.environ["HP_COMPILE_INSIDE_AC"] = previous
        os.environ["HP_COMPILE_INSIDE_AC"] = "1"
        try:
            inner_targets = _resolve_compile_targets(model)
        finally:
            if previous is None:
                os.environ.pop("HP_COMPILE_INSIDE_AC", None)
            else:
                os.environ["HP_COMPILE_INSIDE_AC"] = previous
        return default_targets, inner_targets

    def test_default_keeps_the_decoder_layers(self):
        """With the env unset the targets are exactly what discovery returns."""
        model = _WrappedVlm()
        default_targets, _ = self._targets(model)
        self.assertEqual(
            [module for _, module in default_targets],
            [module for _, module in get_compile_layers(model)],
        )

    def test_env_targets_the_module_inside_each_wrapper(self):
        """With the env set the checkpoint wrapper is skipped in favour of its inner module."""
        model = _WrappedVlm(count=3)
        _, inner_targets = self._targets(model)
        self.assertEqual(len(inner_targets), 3)
        for _, module in inner_targets:
            self.assertFalse(hasattr(module, "_checkpoint_wrapped_module"))
            self.assertIsInstance(module, nn.Linear)

    def test_env_falls_back_when_nothing_is_wrapped(self):
        """An unwrapped model still resolves through the normal discovery."""
        model = _KimiK25Like(text_layers=2, vision_layers=1)
        _, inner_targets = self._targets(model)
        self.assertEqual(
            [module for _, module in inner_targets],
            [module for _, module in get_compile_layers(model)],
        )


class _LayerWithAttn(nn.Module):
    """A whole decoder layer holding an attention child and an MoE child."""

    def __init__(self) -> None:
        """Build one attention module and one MoE stand-in."""
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Linear(2, 2)


class _WrappedLayerVlm(nn.Module):
    """A VLM whose checkpoint-wrapped blocks are *whole layers*, as the real model is."""

    def __init__(self, count: int = 3) -> None:
        """Wrap ``count`` whole decoder layers."""
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [checkpoint_wrapper(_LayerWithAttn()) for _ in range(count)]
        )


class TestCompileDescendantTargets(unittest.TestCase):
    """Pin ``HP_COMPILE_DESCENDANTS``, the fix for a silently wrong compile target.

    The first 2-SN compile arms asked for ``attn`` and got 108 blocks -- which the config
    shows are exactly the vision tower's 108 attention leaf specs, a frozen subtree worth
    under 1% of the step.  The text decoder's blocks are *whole layers* whose FQN carries
    no ``attn``, so a block-level filter can never reach the text attention.  These tests
    keep both the trap and the fix on the record.
    """

    def _resolve(self, model, include, descendants):
        """Resolve targets with the inside-AC env, the filter and the descendant switch."""
        keys = {
            "HP_COMPILE_INSIDE_AC": "1",
            "HP_COMPILE_INSIDE_AC_INCLUDE": include,
            "HP_COMPILE_DESCENDANTS": descendants,
        }
        previous = {key: os.environ.get(key) for key in keys}
        os.environ.update(keys)
        try:
            return _resolve_compile_targets(model)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_block_level_filter_cannot_reach_the_text_attention(self):
        """Without the switch, ``self_attn$`` matches no block and widens to the default.

        This is the trap itself: the request looks satisfied, the log reports how many
        blocks were kept, and the compiled subtree is not the one that was asked for.
        """
        model = _WrappedLayerVlm(count=3)
        targets = self._resolve(model, r"self_attn$", "0")
        self.assertNotIn("self_attn", [path for path, _ in targets])

    def test_descendant_filter_reaches_the_attention_inside_wrapped_layers(self):
        """With the switch, the filter selects the attention child of every layer."""
        model = _WrappedLayerVlm(count=3)
        targets = self._resolve(model, r"self_attn$", "1")
        self.assertEqual(len(targets), 3)
        for path, module in targets:
            self.assertTrue(path.endswith(".self_attn"), path)
            self.assertIsInstance(module, nn.Linear)

    def test_descendant_filter_excludes_the_moe_child(self):
        """The MoE (which crashes under compile) must not come along for the ride."""
        model = _WrappedLayerVlm(count=2)
        targets = self._resolve(model, r"self_attn$", "1")
        self.assertFalse(any(path.endswith(".mlp") for path, _ in targets))

    def test_outermost_match_wins(self):
        """A pattern matching both a layer and its children compiles the layer once."""
        model = _WrappedLayerVlm(count=2)
        targets = self._resolve(model, r"language_model", "1")
        self.assertEqual(len(targets), 2)
        for path, _ in targets:
            self.assertNotIn("self_attn", path)


class TestNpuFlattenedDimsShim(unittest.TestCase):
    """Pin the workaround for torch_npu's composite-index KeyError.

    ``detect_flattened_dims`` records composite sub-expressions such as ``x1 - 1`` as if
    they were axis variables and then looks them up in the range-tree dictionaries, which
    raises ``KeyError: 'x1 - 1'`` and kills the run in the first compiled backward.  The
    shim must defuse exactly that case, leave ordinary indices untouched, and be
    idempotent.
    """

    def _fake_npu(self, behaviour):
        """Install a fake ``torch_npu._inductor.codegen.ir`` module and return it."""
        import sys
        import types

        module = types.ModuleType("torch_npu._inductor.codegen.ir")
        module.detect_flattened_dims = behaviour
        for name in ("torch_npu", "torch_npu._inductor", "torch_npu._inductor.codegen"):
            parent = types.ModuleType(name)
            parent.__path__ = []  # type: ignore[attr-defined]
            sys.modules.setdefault(name, parent)
        sys.modules["torch_npu._inductor.codegen.ir"] = module
        return module

    def _install(self, module):
        """Install the shim with its idempotence flag reset for this test."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        shim._INSTALLED = False  # pylint: disable=protected-access
        self.addCleanup(setattr, shim, "_INSTALLED", False)
        return shim.install_npu_flattened_dims_shim()

    def test_composite_index_is_defused_instead_of_raising(self):
        """The KeyError becomes "no flattened dim" and the caller keeps running."""
        def behaviour(kernel, index):
            raise KeyError("x1 - 1")

        module = self._fake_npu(behaviour)
        self.assertTrue(self._install(module))
        self.assertEqual(module.detect_flattened_dims(None, "x1 - 1"), {})

    def test_ordinary_index_passes_through(self):
        """A detector that works must not be altered."""
        def behaviour(kernel, index):
            return {"x1": "unchanged"}

        module = self._fake_npu(behaviour)
        self._install(module)
        self.assertEqual(module.detect_flattened_dims(None, "x1"), {"x1": "unchanged"})

    def test_install_is_idempotent(self):
        """Installing twice must not stack wrappers (the second call is the real one)."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        module = self._fake_npu(lambda kernel, index: {})
        self.assertTrue(self._install(module))
        wrapped = module.detect_flattened_dims
        self.assertTrue(shim.install_npu_flattened_dims_shim())
        self.assertIs(module.detect_flattened_dims, wrapped)

    def test_env_switch_gates_the_shim(self):
        """The workaround is opt-in, so an arm that forgets it keeps the old behaviour."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        previous = os.environ.pop("HP_COMPILE_NPU_SHIM", None)
        try:
            self.assertFalse(shim.shim_is_requested())
            os.environ["HP_COMPILE_NPU_SHIM"] = "1"
            self.assertTrue(shim.shim_is_requested())
        finally:
            if previous is None:
                os.environ.pop("HP_COMPILE_NPU_SHIM", None)
            else:
                os.environ["HP_COMPILE_NPU_SHIM"] = previous


class TestNpuKernelFallback(unittest.TestCase):
    """Pin the per-kernel fallback used to survive torch_npu's NoTritonConfigsError.

    One generated kernel (the attention backward's fused
    ``add_cat_mul_neg_slice_slice_backward_view``) has no tiling config that compiles, and
    it aborts the whole run.  torch_npu's ``force_fallback_kernel_id`` exists to pay the
    eager path for that one kernel while the other eight fused kernels of the same graph
    stay compiled, so the plumbing (parse + set) is pinned here.
    """

    def _fake_config(self):
        """Install a fake ``torch_npu._inductor.config`` and return it."""
        import sys
        import types

        config = types.ModuleType("torch_npu._inductor.config")
        config.force_fallback_kernel_id = []
        package = types.ModuleType("torch_npu._inductor")
        package.__path__ = []  # type: ignore[attr-defined]
        package.config = config
        top = types.ModuleType("torch_npu")
        top.__path__ = []  # type: ignore[attr-defined]
        top._inductor = package  # pylint: disable=protected-access
        sys.modules["torch_npu"] = top
        sys.modules["torch_npu._inductor"] = package
        sys.modules["torch_npu._inductor.config"] = config
        return config

    def test_empty_spec_is_a_no_op(self):
        """Nothing configured means nothing changed."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        config = self._fake_config()
        self.assertFalse(shim.apply_npu_kernel_fallbacks(""))
        self.assertEqual(config.force_fallback_kernel_id, [])

    def test_kernel_ids_are_parsed(self):
        """A comma-separated list becomes the id list torch_npu expects."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        config = self._fake_config()
        self.assertTrue(shim.apply_npu_kernel_fallbacks("8, 12"))
        self.assertEqual(config.force_fallback_kernel_id, [8, 12])

    def test_all_is_passed_through(self):
        """``all`` keeps torch_npu's own meaning (fall back everything)."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        config = self._fake_config()
        self.assertTrue(shim.apply_npu_kernel_fallbacks("all"))
        self.assertEqual(config.force_fallback_kernel_id, "all")

    def test_env_spec_is_read_and_trimmed(self):
        """The env value is what the arm sets, whitespace and all."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        previous = os.environ.get("HP_COMPILE_NPU_FALLBACK_KERNELS")
        os.environ["HP_COMPILE_NPU_FALLBACK_KERNELS"] = " 8 "
        try:
            self.assertEqual(shim.fallback_spec(), "8")
        finally:
            if previous is None:
                os.environ.pop("HP_COMPILE_NPU_FALLBACK_KERNELS", None)
            else:
                os.environ["HP_COMPILE_NPU_FALLBACK_KERNELS"] = previous


class TestRopeEagerBreak(unittest.TestCase):
    """Pin the rotary graph break that removes the un-lowerable fusion group.

    ``rotate_half`` is ``cat(-x[half:], x[:half])`` and the RoPE application is
    ``(q * cos) + (rotate_half(q) * sin)``.  Inductor fuses that whole region into one
    kernel whose NPU lowering is broken both ways -- no valid Triton tiling config, and an
    invalid broadcast on the FX fallback path (``[7, 32, 2]`` vs ``[7, 32, 1]``).  Running
    the rotary eagerly keeps the rest of the attention compiled, so the break has to be
    real: this test traces a caller and asserts that no rotary op survives in the graph.
    """

    MODULE = "transformers.models.deepseek_v3.modeling_deepseek_v3"

    def setUp(self):
        """Install a fake transformers rotary module, restoring sys.modules afterwards."""
        import sys
        import types

        self._saved = {name: sys.modules.get(name) for name in
                       ("transformers", "transformers.models", self.MODULE.rsplit(".", 1)[0], self.MODULE)}

        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        module = types.ModuleType(self.MODULE)
        module.apply_rotary_pos_emb = lambda q, k, cos, sin, unsqueeze_dim=1: (
            (q * cos) + (rotate_half(q) * sin),
            (k * cos) + (rotate_half(k) * sin),
        )
        for parent in ("transformers", "transformers.models"):
            holder = types.ModuleType(parent)
            holder.__path__ = []  # type: ignore[attr-defined]
            sys.modules.setdefault(parent, holder)
        sys.modules[self.MODULE.rsplit(".", 1)[0]] = types.ModuleType(self.MODULE.rsplit(".", 1)[0])
        sys.modules[self.MODULE] = module
        self.module = module

        def restore():
            for name, previous in self._saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.addCleanup(restore)

    def test_rotary_ops_leave_the_compiled_graph(self):
        """With the break installed, no mul/cat/neg/slice from the rotary is compiled."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        self.assertEqual(shim.install_rope_eager_break(), 1)

        class Caller(nn.Module):
            """Calls the rotary and then does something trivially compilable."""

            def forward(self, q, k, cos, sin):
                """Rotate, then add a scalar so a graph still forms."""
                out, _ = self.module_ref.apply_rotary_pos_emb(q, k, cos, sin)
                return out + 1.0

        Caller.module_ref = self.module
        args = tuple(torch.randn(2, 4) for _ in range(4))
        explanation = torch._dynamo.explain(  # pylint: disable=protected-access
            Caller().forward, *args
        )
        operators = {
            str(node.target).split(".")[-1]
            for graph in explanation.graphs
            for node in graph.graph.nodes
            if node.op == "call_function"
        }
        self.assertFalse(
            operators & {"cat", "neg", "slice", "slice_backward", "mul"},
            f"rotary ops must not be compiled, found {sorted(operators)}",
        )

    def test_install_is_idempotent(self):
        """A second install must not double-wrap the entry point."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        self.assertEqual(shim.install_rope_eager_break(), 1)
        self.assertEqual(shim.install_rope_eager_break(), 0)

    def test_env_switch_gates_the_break(self):
        """The break is opt-in so an arm that forgets it keeps the old behaviour."""
        import hyper_parallel.distributed.npu_inductor_shim as shim

        previous = os.environ.pop("HP_COMPILE_ROPE_EAGER", None)
        try:
            self.assertFalse(shim.rope_eager_is_requested())
            os.environ["HP_COMPILE_ROPE_EAGER"] = "1"
            self.assertTrue(shim.rope_eager_is_requested())
        finally:
            if previous is None:
                os.environ.pop("HP_COMPILE_ROPE_EAGER", None)
            else:
                os.environ["HP_COMPILE_ROPE_EAGER"] = previous


if __name__ == "__main__":
    unittest.main()
