# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s4_ep.py: merged core suite file.

Sources: test_s4_local_compute_fn.py, test_s4_moe_gate_compile.py, test_s5_hf_native_moe.py, test_s6_ep_extend.py

Feature grouping: the original 51 atomic cases are merged into 10 cases by
feature family. Within each family all atomic checks run sequentially, with
section comments ``# ── case: <atomic case name> ──`` marking their origin and
assertions carrying a case-identification message; atomic assertions
(including pytest.raises match checks) are fully covered with no loss.
"""

# Injection factories intentionally keep the complete runtime callback signature,
# and these tests directly validate private planner metadata.
# pylint: disable=unused-argument,protected-access

import functools
import importlib
import math
from unittest import mock
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from hyper_parallel.distributed.expert_parallel import recipes as ep_compute
from hyper_parallel.distributed.expert_parallel.recipes import routed_only_ep_compute_fn
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
    _balanced_router,
    _sigmoid_group_router,
    _softmax_topk_router,
    _topk_router_module,
)
from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.distributed.expert_parallel.experts import (
    _argsort_keys,
    _expert_token_counts,
    _local_swiglu_expert_forward,
    _pack_fused_dispatch,
    _prepare_ep_dispatch,
    _unpack_fused_dispatch,
    resolve_swiglu_weights,
)
from hyper_parallel.distributed.recipe_spec import (
    local_compute,
)
from hyper_parallel.distributed._builder.precompiled_boundary import PrecompiledBoundary
from hyper_parallel.models.qwen3_moe.adapter.distributed import (
    expert_parallel as qwen3_adapter_ep,
)
from hyper_parallel.distributed._builder.parameter_sharding import (
    _StackedExperts,
    _stack_moe_experts,
)
from hyper_parallel.distributed._builder.applier import (
    _apply_phase_c,
    _expert_mesh_layout,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _rewrap_local_outputs,
    _wrap_local_region_forward,
)
from hyper_parallel.distributed._builder.rule_resolver import (
    _resolve_local_compute_fn,
)
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import (
    CP,
    EP,
    ModuleShardingSpec,
    TP,
    _normalize_out_fields,
)
from hyper_parallel.distributed._builder.default_templates import TEMPLATES
from hyper_parallel.distributed._builder.planner import ShardingPlanner
try:
    from hyper_parallel.trainer.config import Target
    _HAS_TRAINER_CONFIG = True
except ImportError:
    # trainer.config pulls in replacement / checkpoint conversion, which
    # require a newer transformers than some CI gates provide.
    _HAS_TRAINER_CONFIG = False
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Replicate,
    Shard,
)
from tests.ut.dual_mode_dtensor.conftest import _ensure_pg


# ==========================================================================
# Shared helpers (merged from the private helpers of the original test
# classes/modules)
# ==========================================================================

class _TinyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = _TinyMod()

    def forward(self, x):
        return self.mod(x)


class _TinyMultiOutputMod(nn.Module):
    def forward(self, x):
        return x, x + 1, x + 2


def _identity_spec():
    return _normalize_out_fields(ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    ))


def _wrap_region(mod, spec, mesh, validate_mode):
    """Wrap the local region along the apply path: boundary + resolve + wrap."""
    boundary = PrecompiledBoundary(spec, mesh, ("tp",))
    compute_fn = _resolve_local_compute_fn(
        mod, spec, mesh, ("tp",), expert_mesh=None)
    _wrap_local_region_forward(
        mod, boundary, spec, mesh, ("tp",),
        validate_mode=validate_mode, compute_fn=compute_fn)


class _TinyMoeMod(nn.Module):
    """Minimal MoE-shaped module for archetype factory tests (gate + experts
    [+ shared_expert + shared_expert_gate])."""
    def __init__(self, with_shared=False):
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        if with_shared:
            self.shared_expert = nn.Linear(4, 4)
            self.shared_expert_gate = nn.Linear(4, 1, bias=False)


class _FakeEpMesh:
    def __getitem__(self, name):
        assert name == "ep"
        return self

    def get_group(self, name):
        return f"group-{name}"

    def size(self):
        return 2


def _capture_ep_primitives(monkeypatch):
    """Patch EP primitives and return the metadata captured by the fakes."""
    captured = {}

    def fake_compute(module, hidden_states, *, router_fn, ep_group):
        captured.update(router_fn=router_fn, ep_group=ep_group)
        return hidden_states

    def fake_bind(module, ep_size):
        captured.update(bound_module=module, ep_size=ep_size)

    monkeypatch.setattr(ep_compute, "ep_routed_forward", fake_compute)
    monkeypatch.setattr(ep_compute, "bind_local_expert_forward", fake_bind)
    return captured


class _FakeMesh:
    mesh_dim_names = ("tp", "ep")


def _moe_gate_spec():
    t = TEMPLATES["moe_gate"]
    spec = ModuleShardingSpec(
        in_src=t.sp_in_src,
        in_dst=t.sp_in_dst,
        out_src=t.sp_out_src,
        out_dst=t.sp_out_dst,
    )
    return _normalize_out_fields(spec)


def _meta_mesh(shape, names):
    """Metadata-only mesh (planner tests need no real process groups, but
    DeviceMesh construction requires the default PG to exist -- same
    rationale as make_mesh's _ensure_pg)."""
    _ensure_pg()
    n = 1
    for s in shape:
        n *= s
    return init_device_mesh("cpu", tuple(shape), mesh_dim_names=tuple(names),
                            rank_list=tuple(range(n)), init_backend=False)


# ==========================================================================
# Family 1: declaration resolution / region_dispatch injection discipline /
# template boundary compilation
# Sources: TestRewrapLocalOutputs, TestRegionDispatchDeclaration, moe_gate template
# ==========================================================================

def test_declaration_and_region_dispatch(make_mesh, monkeypatch):
    """Declarative contract family: out_names rewrapping, region_dispatch
    explicit-declaration discipline, moe_gate template EP redistribute
    compilation."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_preserves_list_and_wraps_all_declared_outputs ──
    # Every declared Tensor output is wrapped and list remains list.
    spec = ModuleShardingSpec(
        out_src={
            "hidden": {TP: Replicate()},
            "aux": {TP: Shard(0)},
        },
        out_names=["hidden", "aux", "metadata"],
    )
    calls = []

    def fake_from_local(tensor, device_mesh, placements):
        calls.append((tensor, device_mesh, placements))
        return f"wrapped-{len(calls)}"

    monkeypatch.setattr(
        "hyper_parallel.distributed._builder.forward_rewriter.DTensor.from_local",
        fake_from_local,
    )
    outputs = [torch.ones(2), torch.zeros(2), None]
    result = _rewrap_local_outputs(outputs, spec, mesh, ("tp",), "TestModule")

    assert isinstance(result, list), "case: preserves_list_and_wraps_all_declared_outputs"
    assert result == ["wrapped-1", "wrapped-2", None], \
        "case: preserves_list_and_wraps_all_declared_outputs"
    assert len(calls) == 2, "case: preserves_list_and_wraps_all_declared_outputs"

    # ── case: test_declared_output_index_out_of_range_fails ──
    # A stale out_names contract fails with boundary context.
    spec = ModuleShardingSpec(
        out_src={"aux": {TP: Replicate()}},
        out_names=["hidden", "aux"],
    )
    with pytest.raises(ValueError, match="TestModule.*only 1 output"):
        _rewrap_local_outputs(
            (torch.ones(2),), spec, mesh, ("tp",), "TestModule")

    # ── case: test_local_compute_fn_without_region_dispatch_fails ──
    # region_dispatch injection discipline (no default): declaring an
    # injection must be explicit.
    @local_compute
    def my_fn(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x
        return compute_fn
    spec = _identity_spec()
    spec.local_compute_fn = my_fn
    with pytest.raises(ValueError, match="region_dispatch"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_redundant_true_without_injection_fails ──
    # region_dispatch=True without an injection -> fail-fast (a plain
    # boundary passes through naturally, so the declaration is redundant).
    model = _TinyModel()
    spec = _identity_spec()
    spec.region_dispatch = True
    plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
    with pytest.raises(ValueError, match="redundant"):
        _apply_phase_c(model, plan, mesh, validate_mode=False)

    # ── case: test_moe_gate_out_plan_has_ep_redistribute ──
    # moe_gate template EP redistribute (out_dst {EP: Shard(0)}) compilation.
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.out_plan) == 1, "case: moe_gate_out_plan_has_ep_redistribute"
    op = b.out_plan[0]
    ep_idx = ("tp", "ep").index("ep")
    # out_dst EP dim: Replicate -> Shard(0)
    assert op.src_placements[ep_idx] == Replicate(), \
        "case: moe_gate_out_plan_has_ep_redistribute"
    assert op.dst_placements[ep_idx] == Shard(0), \
        "case: moe_gate_out_plan_has_ep_redistribute"
    assert op.collective_type == "redistribute", \
        "case: moe_gate_out_plan_has_ep_redistribute"

    # ── case: test_moe_gate_in_plan_tp_allgather ──
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.in_plan) == 1, "case: moe_gate_in_plan_tp_allgather"
    assert b.in_plan[0].collective_type == "all_gather", \
        "case: moe_gate_in_plan_tp_allgather"


# ==========================================================================
# Family 2: local_compute_fn resolution chain + Target factory form
# Sources: TestResolveLocalComputeFn (resolution), TestTargetLocalComputeFn
# (happy path)
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_local_compute_fn_resolution(make_mesh):
    """Contract checks for the local-compute factory resolution chain and the
    Target factory form (the YAML sharding.injections carrier)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_user_fn_wins_even_with_ep_size ──
    # local_compute_fn is ring 1 of the resolution chain: the user fn is
    # returned directly even when the _ep_size metadata is present (the
    # built-in EP auto-injection path has been removed; _ep_size only drives
    # parameter sharding).
    built = []

    @local_compute
    def my_fn(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x
        built.append(compute_fn)
        return compute_fn

    spec = _identity_spec()
    spec.local_compute_fn = my_fn
    spec.region_dispatch = False
    spec._ep_size = 2
    fn = _resolve_local_compute_fn(
        _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
    assert isinstance(fn, functools.partial), \
        "case: user_fn_wins_even_with_ep_size"
    # Factory form: the factory is invoked once at apply time, and the
    # partial binds the compute_fn returned by the factory.
    assert len(built) == 1 and fn.func is built[0], \
        "case: user_fn_wins_even_with_ep_size"

    # ── case: test_region_dispatch_resolves_to_module_forward ──
    # region_dispatch pure gating (no user fn) -> the module's own forward.
    mod = _TinyMod()
    spec = _identity_spec()
    spec.region_dispatch = False
    fn = _resolve_local_compute_fn(
        mod, spec, mesh, ("tp",), expert_mesh=None)
    assert fn == mod.forward, \
        "case: region_dispatch_resolves_to_module_forward"  # pylint: disable=comparison-with-callable

    # ── cases: three forms in which resolution yields None (table-driven) ──
    # test_ep_size_alone_returns_none: after the rework, _ep_size>0 no longer
    # injects any compute -- with no local_compute_fn and
    # region_dispatch=False -> None (the apply-side preflight fail-fasts on
    # this).
    # test_inner_wrapper_does_not_resolve_module_forward: when an
    # inner_wrapper hosts the local computation, the whole module forward is
    # not chosen as the skeleton.
    # test_no_declaration_returns_none: neither source present -> None (the
    # module takes no skeleton, and the gate derives to False).
    none_cases = [
        ("ep_size_alone_returns_none",
         lambda s: setattr(s, "_ep_size", 2)),
        ("inner_wrapper_does_not_resolve_module_forward",
         lambda s: (setattr(s, "region_dispatch", False),
                    setattr(s, "inner_wrapper", "sdpa_hf"))),
        ("no_declaration_returns_none", lambda s: None),
    ]
    for label, mutate in none_cases:
        spec = _identity_spec()
        mutate(spec)
        fn = _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
        assert fn is None, f"case: {label}"

    # ── case: test_target_factory_built_with_context ──
    # Target factory: built at apply time (the generic context
    # module/mesh/expert_mesh is filtered by signature), and the returned
    # compute fn is partial-bound with module.
    seen = []

    @local_compute
    def my_factory(module, mesh, tp_mesh, cp_mesh, ep_mesh):
        seen.append((module, mesh, tp_mesh, cp_mesh, ep_mesh))

        def compute_fn(mod, x):
            return mod.lin(x) * 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = Target(
        my_factory, target_path="tests.my_factory")
    spec.region_dispatch = False
    fn = _resolve_local_compute_fn(
        mod, spec, mesh, ("tp",), expert_mesh=None)
    assert isinstance(fn, functools.partial), \
        "case: target_factory_built_with_context"
    assert seen and seen[0][0] is mod, \
        "case: target_factory_built_with_context"   # module context injected
    assert seen[0][2] is mesh["tp"], \
        "case: target_factory_built_with_context"   # tp_mesh filled by framework
    assert seen[0][3] is None, \
        "case: target_factory_built_with_context"   # no cp axis -> cp_mesh=None
    assert seen[0][4] is None, \
        "case: target_factory_built_with_context"   # no EP -> ep_mesh=None
    x = torch.randn(2, 4)
    torch.testing.assert_close(fn(x), mod.lin(x) * 2,
                               msg="case: target_factory_built_with_context")

    # ── case: test_config_keys_pass_through_untouched ──
    # Config keys are purely user-owned: the framework only fills the context
    # and performs no auto-population -- an unconfigured key reaches the
    # factory with its default (None); a configured key passes through
    # untouched.
    seen = []

    @local_compute
    def cfg_factory(mesh, tp_mesh, cp_mesh, ep_mesh, block_size=None):
        seen.append(block_size)

        def compute_fn(mod, x):
            return x
        return compute_fn

    spec = _identity_spec()
    spec.local_compute_fn = Target(
        cfg_factory, target_path="tests.cfg_factory")
    spec.region_dispatch = False
    _resolve_local_compute_fn(
        _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
    assert seen == [None], \
        "case: config_keys_pass_through_untouched"  # framework does not fill config keys

    spec2 = _identity_spec()
    spec2.local_compute_fn = Target(
        cfg_factory, target_path="tests.cfg_factory", block_size=128)
    spec2.region_dispatch = False
    _resolve_local_compute_fn(
        _TinyMod(), spec2, mesh, ("tp",), expert_mesh=None)
    assert seen[-1] == 128, \
        "case: config_keys_pass_through_untouched"  # explicit user config passes through


# ==========================================================================
# Family 3: custom compute_fn executes inside the local region
# (combination-scenario level, kept standalone)
# Sources: TestResolveLocalComputeFn::test_derived_gate_via_apply_path,
#          TestLocalRegionWithCustomComputeFn
# ==========================================================================

def test_custom_compute_fn_executes_in_region(make_mesh):
    """End-to-end execution of a custom compute_fn: apply derives the gated
    injection, and in both production and validate modes the compute_fn
    receives (module, local tensor)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_derived_gate_via_apply_path ──
    # Derived gating end-to-end: region_dispatch=False + local_compute_fn ->
    # _apply_phase_c still injects the skeleton and executes the custom fn
    # (the gate does not read the stored bool).
    calls = []

    @local_compute
    def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            calls.append(x)
            return module.lin(x) * 3
        return compute_fn

    model = _TinyModel()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute
    spec.region_dispatch = False   # injection discipline: explicit declaration (black-box hosting)
    plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
    _apply_phase_c(model, plan, mesh, validate_mode=False)

    x = torch.randn(2, 4)
    out = model.mod(x)
    assert len(calls) == 1, \
        "case: derived_gate_via_apply_path"   # custom fn executed -> skeleton injected
    torch.testing.assert_close(out, model.mod.lin(x) * 3,
                               msg="case: derived_gate_via_apply_path")

    # ── case: test_custom_compute_fn_runs_in_region ──
    # production: the custom compute_fn receives (module, local tensor), and
    # its output returns as local via the skeleton boundary exit.
    calls = []

    @local_compute
    def my_compute2(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            calls.append((module, x))
            return module.lin(x) * 2   # custom logic: scale by 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute2
    spec.region_dispatch = False
    _wrap_region(mod, spec, mesh, validate_mode=False)

    x = torch.randn(2, 4)
    with mock.patch.object(
            DTensor, "from_local",
            side_effect=AssertionError("production local-region must not re-wrap output")) as from_local:
        out = mod(x)
    from_local.assert_not_called()
    assert calls and calls[0][0] is mod, "case: custom_compute_fn_runs_in_region"
    torch.testing.assert_close(out, mod.lin(x) * 2,
                               msg="case: custom_compute_fn_runs_in_region")

    # ── case: test_custom_compute_fn_validate_mode ──
    # validate: DTensor inputs are unwrapped by the skeleton -- the compute_fn
    # still receives a local tensor (no mode awareness), and the exit rewraps
    # then unwraps on return.
    seen = []

    @local_compute
    def my_compute3(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            seen.append(x)
            return module.lin(x)
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute3
    spec.region_dispatch = False
    _wrap_region(mod, spec, mesh, validate_mode=True)

    x = torch.randn(2, 4)
    out = mod(x)
    assert len(seen) == 1, "case: custom_compute_fn_validate_mode"
    assert not isinstance(seen[0], DTensor), \
        "case: custom_compute_fn_validate_mode"   # always local inside compute_fn
    assert not isinstance(out, DTensor), \
        "case: custom_compute_fn_validate_mode"   # skeleton exit always unwraps
    torch.testing.assert_close(out, mod.lin(x),
                               msg="case: custom_compute_fn_validate_mode")

    # ── case: test_multi_output_validate_mode_returns_local_structure ──
    # Every declared output is rewrapped for placement accounting inside the
    # boundary, then recursively unwrapped before the local-region call returns.
    mod = _TinyMultiOutputMod()
    layout = {TP: Shard(1)}
    spec = ModuleShardingSpec(
        in_src={"x": layout},
        in_dst={"x": layout},
        out_src={"first": layout, "second": layout, "nested": layout},
        out_dst={"first": layout, "second": layout, "nested": layout},
        region_dispatch=False,
    )
    _wrap_region(mod, spec, mesh, validate_mode=True)

    first, second, third = mod(x)
    assert not isinstance(first, DTensor), \
        "case: multi_output_validate_mode_returns_local_structure"
    assert not isinstance(second, DTensor), \
        "case: multi_output_validate_mode_returns_local_structure"
    assert not isinstance(third, DTensor), \
        "case: multi_output_validate_mode_returns_local_structure"
    torch.testing.assert_close(first, x)
    torch.testing.assert_close(second, x + 1)
    torch.testing.assert_close(third, x + 2)


# ==========================================================================
# Family 4: error path A -- local_compute_fn / Target contract fail-fast
# Sources: the 6 raises cases of TestTargetLocalComputeFn
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_local_compute_fn_contract_errors(make_mesh):
    """local_compute_fn injection discipline and contract fail-fast
    (sequential pytest.raises)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_target_bad_return_raises ──
    # Target factory returns a non-callable -> TypeError (contract: must
    # return a compute fn).
    @local_compute
    def bad_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
        return 42

    spec = _identity_spec()
    spec.local_compute_fn = Target(bad_factory, target_path="tests.bad")
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="local_compute_fn"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_target_undecorated_factory_raises ──
    # Injection discipline: Target pointing at an undecorated factory ->
    # fail-fast hinting at @local_compute.
    spec = _identity_spec()
    spec.local_compute_fn = Target(lambda: 42, target_path="tests.bad")
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="@local_compute"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_plain_callable_undecorated_raises ──
    # Injection discipline: an undecorated callable -> fail-fast hinting at
    # @local_compute.
    spec = _identity_spec()
    spec.local_compute_fn = lambda module, x: x
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="@local_compute"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_compute_fn_param_mismatch_raises ──
    # Principle 1: compute fn parameters not matching the original forward ->
    # fail-fast at apply time.
    @local_compute
    def bad_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, hidden):   # original forward's parameter name is x
            return hidden
        return compute_fn

    spec = _identity_spec()
    spec.local_compute_fn = bad_compute
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="same-named"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_target_typo_config_key_raises ──
    # Configuring a key the factory does not declare (typo in rounter) ->
    # fail-fast listing the legal parameters -- config keys bind by name, and
    # a typo must not be silently swallowed.
    spec = _identity_spec()
    spec.local_compute_fn = Target(
        routed_only_ep_compute_fn,
        target_path="hyper_parallel.distributed."
                    "recipes.routed_only_ep_compute_fn",
        blok_size="oops")                     # typo: should be block_size
    spec.region_dispatch = False
    with pytest.raises(ValueError, match="undeclared keys"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_target_reserved_context_key_raises ──
    # Context keys are framework-reserved names: a user configuring mesh in
    # Target -> fail-fast (the mesh family may only be filled by the
    # framework).
    spec = _identity_spec()
    spec.local_compute_fn = Target(
        routed_only_ep_compute_fn,
        target_path="hyper_parallel.distributed."
                    "recipes.routed_only_ep_compute_fn",
        mesh="oops")
    spec.region_dispatch = False
    with pytest.raises(ValueError, match="framework-reserved context keys"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)


# ==========================================================================
# Family 5: built-in EP archetype factories (qwen2/qwen3/mixtral/custom
# combination sweep)
# Sources: the 6 happy-path cases of TestEpArchetypeFactories
# ==========================================================================

def test_ep_archetype_factories(make_mesh, monkeypatch):
    """Built-in EP archetype factories (recipes.py): use of the mesh family
    context + explicit router selection + each archetype's combine formula
    (accuracy_fix_plan.md section 3 E2)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_mesh_family_used_directly ──
    # ep_mesh is filled by the framework; the factory directly takes
    # ep_mesh.get_group("ep") and hands it to ep_routed_forward; the router
    # is embedded (default softmax top-k); no tp_group -- the TP
    # communication of a nested boundary is self-contained by the
    # sub-boundary (contract in expert_parallel.experts).
    module = _TinyMoeMod()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.routed_only_ep_compute_fn(
        module=module, mesh=mesh, tp_mesh=mesh["tp"], cp_mesh=None,
        ep_mesh=_FakeEpMesh())
    compute_fn(module, torch.randn(2, 4))
    assert captured["ep_group"] == "group-ep", "case: mesh_family_used_directly"
    assert captured["bound_module"] is module, "case: mesh_family_used_directly"
    assert captured["ep_size"] == 2, "case: mesh_family_used_directly"
    # Embedded router: default softmax top-k (the framework does not decide
    # the router; spec has no such field)
    assert captured["router_fn"] is _softmax_topk_router, \
        "case: mesh_family_used_directly"

    # ── case: test_qwen2moe_factory_combines_shared_and_gate ──
    # qwen2moe archetype combine formula: routed + sigmoid(gate(x)) * shared(x)
    # -- the shared_expert call is an ordinary submodule call (nested
    # boundary contract), with no compensating communication.
    captured = _capture_ep_primitives(monkeypatch)
    module = _TinyMoeMod(with_shared=True)
    compute_fn = ep_compute.qwen2moe_ep_compute_fn(
        module=module, mesh=None, tp_mesh=None, cp_mesh=None,
        ep_mesh=_FakeEpMesh())
    x = torch.randn(2, 4)
    out = compute_fn(module, x)
    expected = x + torch.sigmoid(module.shared_expert_gate(x)) * module.shared_expert(x)
    torch.testing.assert_close(
        out, expected, msg="case: qwen2moe_factory_combines_shared_and_gate")
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen2moe"], \
        "case: qwen2moe_factory_combines_shared_and_gate"

    # ── case: test_qwen3_factory_embeds_topk_router ──
    # Qwen3-MoE uses its explicit TopKRouter factory, migrated to the model
    # adapter in M3; the generic combine skeleton stays in recipes.py, so the
    # monkeypatched recipes primitives are still what the factory binds.
    module = _TinyMoeMod()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = qwen3_adapter_ep.qwen3moe_ep_compute_fn(
        module=module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_FakeEpMesh(),
    )
    compute_fn(module, torch.randn(2, 4))

    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen3moe"], \
        "case: qwen3_factory_embeds_topk_router"
    assert captured["bound_module"] is module, \
        "case: qwen3_factory_embeds_topk_router"

    # ── case: test_mixtral_factory_uses_tuple_router_and_training_jitter ──
    # Mixtral 5.12 uses its tuple router and jitters the expert input in training.
    module = _TinyMoeMod()
    module.jitter_noise = 0.2
    module.train()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.mixtral_ep_compute_fn(
        module=module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_FakeEpMesh(),
    )
    hidden_states = torch.ones(2, 4)
    torch.manual_seed(17)
    output = compute_fn(module, hidden_states)
    torch.manual_seed(17)
    expected = hidden_states * torch.empty_like(hidden_states).uniform_(0.8, 1.2)

    torch.testing.assert_close(
        output, expected,
        msg="case: mixtral_factory_uses_tuple_router_and_training_jitter")
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["mixtral"], \
        "case: mixtral_factory_uses_tuple_router_and_training_jitter"

    # ── case: test_mixtral_factory_disables_jitter_in_eval ──
    # Mixtral evaluation preserves hidden states even when jitter is configured.
    module = _TinyMoeMod()
    module.jitter_noise = 0.2
    module.eval()
    _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.mixtral_ep_compute_fn(
        module=module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_FakeEpMesh(),
    )
    hidden_states = torch.randn(2, 4)
    torch.testing.assert_close(
        compute_fn(module, hidden_states), hidden_states,
        msg="case: mixtral_factory_disables_jitter_in_eval")

    # ── case: test_custom_factory_embeds_its_router ──
    # The router is part of the injected function: a custom factory
    # references a MOE_ROUTER_ADAPTERS adapter by name and writes it into its
    # own compute fn -- the framework takes no part in the selection.
    captured = _capture_ep_primitives(monkeypatch)

    @local_compute
    def qwen3moe_ep_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
        ep_group = ep_mesh.get_group("ep")

        def compute_fn(module, hidden_states):
            return ep_compute.ep_routed_forward(
                module, hidden_states,
                router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
                ep_group=ep_group)
        return compute_fn

    fn = qwen3moe_ep_factory(
        mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=_FakeEpMesh())
    fn(_TinyMoeMod(), torch.randn(2, 4))
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen3moe"], \
        "case: custom_factory_embeds_its_router"
    assert captured["ep_group"] == "group-ep", \
        "case: custom_factory_embeds_its_router"


# ==========================================================================
# Family 6: error path B -- factory / stack / planner fail-fast
# Sources: the 2 raises cases of TestEpArchetypeFactories,
#          test_stack_moe_experts_rejects_bias, test_planner_ep_extend_invalid
# ==========================================================================

def test_factory_and_planner_error_paths(monkeypatch, tiny_hf_native_moe):
    """Fail-fast for factory interface assertions / missing ep_mesh / stack
    bias restriction / planner EP-extension parameter validation (sequential
    pytest.raises)."""
    # ── case: test_planner_ep_extend_invalid ──
    # ep_size exceeding the dense region / not dividing it / num_experts not
    # divisible -> ValueError.
    # mesh (1,2) D=2: ep=4 > D -> error
    mesh = _meta_mesh((1, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    # mesh (4,2) D=8: ep=3 does not divide D -> error
    mesh8 = _meta_mesh((4, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=3)
    # mesh (4,2) D=8: ep=8 is legal but num_experts=4 not divisible by ep=8 -> error
    with pytest.raises(ValueError, match="num_experts"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=8)

    # ── case: test_interface_assertion_fails_fast ──
    # Wrong archetype chosen (module lacks shared_expert/shared_expert_gate)
    # -> ValueError at apply time, listing the module's actual submodule
    # names.
    _capture_ep_primitives(monkeypatch)
    module = _TinyMoeMod(with_shared=False)
    with pytest.raises(ValueError, match="shared_expert") as exc_info:
        ep_compute.qwen2moe_ep_compute_fn(
            module=module, mesh=None, tp_mesh=None, cp_mesh=None,
            ep_mesh=_FakeEpMesh())
    msg = str(exc_info.value)
    assert "gate" in msg and "experts" in msg, \
        "case: interface_assertion_fails_fast"   # actual submodule names visible

    # ── case: test_factory_requires_ep_mesh ──
    # Non-EP boundary (framework fills ep_mesh=None) -> config error
    # fail-fast.
    mesh = type("M", (), {"mesh_dim_names": ("tp",)})()
    with pytest.raises(ValueError, match="ep_mesh"):
        ep_compute.routed_only_ep_compute_fn(
            module=_TinyMoeMod(), mesh=mesh, tp_mesh=None, cp_mesh=None,
            ep_mesh=None)

    # ── case: test_stack_moe_experts_rejects_bias ──
    # An expert with bias -> NotImplementedError (v1 limitation).
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    mlp.experts[0].gate_proj.bias = nn.Parameter(torch.zeros(32))
    ep_stack = {"experts.gate_proj": [f"experts.{i}.gate_proj.weight" for i in range(4)]}
    with pytest.raises(NotImplementedError, match="bias"):
        _stack_moe_experts(mlp, ep_stack)


# ==========================================================================
# Family 7: region_dispatch=True dispatch-through validation
# Sources: TestLocalRegionDispatchThrough (combination-scenario level, kept
# as its own family)
# ==========================================================================

def test_region_dispatch_through(make_mesh):
    """region_dispatch=True: validate dispatches through the injected
    function (pure standard ops), strategy propagation covers the injected
    code, and out_src is upgraded from declarative rewrapping to true
    validation; production behavior is unchanged."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_dispatch_through_validate ──
    # validate: DTensor enters the injected function directly (no to_local),
    # and the propagation result matching the out_src declaration -> passes;
    # production behavior unchanged.
    seen = {}

    @local_compute
    def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            seen["x_is_dtensor"] = isinstance(x, DTensor)
            return x * 2 + x              # pure pointwise: dispatchable
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute
    spec.region_dispatch = True       # injected code uses pure standard ops
    _wrap_region(mod, spec, mesh, validate_mode=True)

    # The boundary entry of a size-1 mesh does not wrap into DTensor
    # (degenerate skip) -- feed a DTensor directly (equivalent to the input
    # passed from the outer boundary in the D-14 nested scenario).
    x = torch.randn(2, 4)
    dt = DTensor.from_local(x, mesh, (Shard(1),))
    out = mod(dt)
    assert seen["x_is_dtensor"] is True, \
        "case: dispatch_through_validate"     # validate dispatch-through: injected fn sees DTensor
    assert not isinstance(out, DTensor), \
        "case: dispatch_through_validate"     # skeleton exit always unwraps
    torch.testing.assert_close(out, x * 3, msg="case: dispatch_through_validate")

    # ── case: test_dispatch_through_out_src_mismatch_fails ──
    # True validation: the propagation result (pointwise -> Shard(1)
    # preserved) disagrees with the declared out_src (Shard(0)) ->
    # fail-fast -- black-box mode cannot catch this class of injected-code
    # bug.
    @local_compute
    def my_compute2(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x * 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.out_src = {"output": {TP: Shard(0)}}   # a lying declaration
    spec.local_compute_fn = my_compute2
    spec.region_dispatch = True
    _wrap_region(mod, spec, mesh, validate_mode=True)

    dt = DTensor.from_local(torch.randn(2, 4), mesh, (Shard(1),))
    with pytest.raises(Exception, match="out_src"):
        mod(dt)

    # ── case: test_dispatch_through_production_unchanged ──
    # production: region_dispatch=True changes no branching (local
    # pass-through).
    seen = {}

    @local_compute
    def my_compute3(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            seen["x_is_dtensor"] = isinstance(x, DTensor)
            return x * 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute3
    spec.region_dispatch = True
    _wrap_region(mod, spec, mesh, validate_mode=False)

    x = torch.randn(2, 4)
    out = mod(x)
    assert seen["x_is_dtensor"] is False, \
        "case: dispatch_through_production_unchanged"   # production always local
    torch.testing.assert_close(out, x * 2,
                               msg="case: dispatch_through_production_unchanged")


# ==========================================================================
# Family 8: planner EP marking + stack handler
# Sources: test_planner_marks_hf_native_moe, test_planner_no_mark_without_ep,
#          test_planner_pre_stacked_d10_ep_extend, test_stack_moe_experts
# ==========================================================================

def test_planner_ep_marking_and_stack(tiny_hf_native_moe, tiny_moe, make_mesh):
    """Planner EP marking for HF-native / custom-named MoE (stacked metadata
    + TP-extend-EP contract) and the _stack_moe_experts stacking handler."""
    # ── case: test_planner_marks_hf_native_moe ──
    # per-expert params + ep>1 -> stacked metadata + TP-extend-EP contract
    # (D-09a/D-10).
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)

    spec = plan.modules["model.layers.0.mlp"]
    # Numeric-field guard in effect: the boundary aggregates at mlp, with no
    # per-expert boundaries
    assert not any("experts.0" in fqn for fqn in plan.modules), \
        "case: planner_marks_hf_native_moe"

    # stacked entries (D-10 TP-extend-EP: only {EP: S0} expert-dim sharding,
    # no TP key, no second axis)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_marks_hf_native_moe"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_marks_hf_native_moe"

    # per-expert entries removed; router fully replicated
    assert not any("experts.0" in k for k in spec.params), \
        "case: planner_marks_hf_native_moe"
    assert spec.params["gate.weight"][TP] == Replicate(), \
        "case: planner_marks_hf_native_moe"

    # _ep_stack metadata: stacked name -> source paths ordered by expert idx
    assert set(spec._ep_stack) == {
        "experts.gate_proj", "experts.up_proj", "experts.down_proj"}, \
        "case: planner_marks_hf_native_moe"
    assert spec._ep_stack["experts.gate_proj"] == [
        f"experts.{i}.gate_proj.weight" for i in range(4)], \
        "case: planner_marks_hf_native_moe"
    # TP-extend-EP: _ep_size = ep_size, boundary identity
    assert spec._ep_size == 2, "case: planner_marks_hf_native_moe"
    assert spec.in_dst["x_BLD"][TP] == Shard(1), \
        "case: planner_marks_hf_native_moe"

    # ── case: test_planner_no_mark_without_ep ──
    # ep=1 -> no stacking, per-expert entries kept (TP-only semantics
    # correct).
    mesh1 = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh1, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_stack == {}, "case: planner_no_mark_without_ep"
    assert "experts.0.gate_proj.weight" in spec.params, \
        "case: planner_no_mark_without_ep"
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0), \
        "case: planner_no_mark_without_ep"

    # ── case: test_planner_pre_stacked_d10_ep_extend ──
    # Custom naming (experts.w1 3D) -> D-10 TP-extend-EP path:
    # {EP: Shard(0)}, no TP key, SP-in identity boundary, _ep_stack empty
    # (already stacked).
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 2, "case: planner_pre_stacked_d10_ep_extend"
    assert spec._ep_stack == {}, "case: planner_pre_stacked_d10_ep_extend"
    # Custom-named w1/w2/w3 -> expert params only {EP: Shard(0)}, no TP key
    for proj in ("w1", "w2", "w3"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_pre_stacked_d10_ep_extend"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_pre_stacked_d10_ep_extend"
    assert spec.in_dst["x_BLD"][TP] == Shard(1), \
        "case: planner_pre_stacked_d10_ep_extend"   # SP-in identity

    # ── case: test_stack_moe_experts ──
    # Stacking handler: stacked values == original per-expert values,
    # original params removed.
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    orig = {
        proj: torch.stack([getattr(mlp.experts[i], proj).weight.data
                           for i in range(4)])
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
    ep_stack = plan.modules["model.layers.0.mlp"]._ep_stack

    _stack_moe_experts(mlp, ep_stack)

    assert isinstance(mlp.experts, _StackedExperts), "case: stack_moe_experts"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = getattr(mlp.experts, proj)
        assert stacked.shape == orig[proj].shape, "case: stack_moe_experts"
        torch.testing.assert_close(stacked, orig[proj],
                                   msg="case: stack_moe_experts")
    # Original per-expert params removed
    assert not any("experts.0" in n
                   for n, _ in mlp.named_parameters()), "case: stack_moe_experts"


# ==========================================================================
# Family 9: router adapters / SwiGLU weight resolution / expert forward
# utilities
# Sources: test_softmax_topk_router, testresolve_swiglu_weights_*,
#          test_local_expert_forward_uses_declared_activation,
#          test_topk_router_module_adapter, test_sigmoid_group_router_adapter
# ==========================================================================

def test_router_and_expert_utils(tiny_hf_native_moe, tiny_hf_batched_moe):
    """Router adapters (softmax top-k / TopKRouter module / sigmoid group)
    plus SwiGLU weight resolution and local expert forward."""
    # ── case: test_softmax_topk_router ──
    # The default adapter matches the routing semantics of the toy model's
    # forward.
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    topk_idx, topk_w = _softmax_topk_router(mlp, hidden)
    logits = mlp.gate(hidden).view(-1, 4)
    w = logits.softmax(-1)
    ref_w, ref_idx = w.topk(2, dim=-1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    assert torch.equal(topk_idx, ref_idx), "case: softmax_topk_router"
    torch.testing.assert_close(topk_w, ref_w, msg="case: softmax_topk_router")
    assert MOE_ROUTER_ADAPTERS["default"] is _softmax_topk_router, \
        "case: softmax_topk_router"

    # ── case: testresolve_swiglu_weights_two_naming_families ──
    # Both naming families gate/up/down_proj and w1/w2/w3 resolve; a missing
    # matrix raises.
    class Holder(nn.Module):  # pylint: disable=abstract-method
        pass

    h1 = Holder()
    h1.gate_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.up_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.down_proj = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = resolve_swiglu_weights(h1)
    assert g is h1.gate_proj and u is h1.up_proj and d is h1.down_proj, \
        "case: resolve_swiglu_weights_two_naming_families"

    h2 = Holder()
    h2.w1 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w3 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w2 = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = resolve_swiglu_weights(h2)
    assert g is h2.w1 and u is h2.w3 and d is h2.w2, \
        "case: resolve_swiglu_weights_two_naming_families"

    with pytest.raises(NotImplementedError, match="SwiGLU"):
        resolve_swiglu_weights(Holder())

    # ── case: testresolve_swiglu_weights_fused_layout ──
    # D-11 fused layout: gate_up_proj + down_proj -> (fused, None, down).
    h = Holder()
    h.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
    h.down_proj = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = resolve_swiglu_weights(h)
    assert g is h.gate_up_proj and u is None and d is h.down_proj, \
        "case: resolve_swiglu_weights_fused_layout"

    # automodel naming (gate_and_up_projs/down_projs) is isomorphic
    h2 = Holder()
    h2.gate_and_up_projs = nn.Parameter(torch.randn(4, 16, 8))
    h2.down_projs = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = resolve_swiglu_weights(h2)
    assert g is h2.gate_and_up_projs and u is None and d is h2.down_projs, \
        "case: resolve_swiglu_weights_fused_layout"

    # ── case: test_local_expert_forward_uses_declared_activation ──
    # EP expert computation honors the model activation instead of forcing SiLU.
    class Experts(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.local_expert_count = 1
            self.gate_up_proj = nn.Parameter(torch.randn(1, 8, 4))
            self.down_proj = nn.Parameter(torch.randn(1, 4, 4))
            self._ep_act_fn = torch.tanh

    experts = Experts()
    hidden_states = torch.randn(3, 4)
    expert_indices = torch.zeros(3, dtype=torch.long)
    output = _local_swiglu_expert_forward(experts, hidden_states, expert_indices)
    gate_states, up_states = F.linear(hidden_states, experts.gate_up_proj[0]).chunk(2, dim=-1)
    expected = F.linear(torch.tanh(gate_states) * up_states, experts.down_proj[0])
    torch.testing.assert_close(
        output, expected, msg="case: local_expert_forward_uses_declared_activation")

    mlp = tiny_hf_batched_moe.model.layers[0].mlp

    # ── case: test_topk_router_module_adapter ──
    # Qwen2/Qwen3/Mixtral adapter: directly takes the indices and scores of
    # the TopKRouter.
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    idx, w = _topk_router_module(mlp, hidden)
    _, ref_w, ref_idx = mlp.gate(hidden)
    assert torch.equal(idx, ref_idx), "case: topk_router_module_adapter"
    torch.testing.assert_close(w, ref_w, msg="case: topk_router_module_adapter")
    assert MOE_ROUTER_ADAPTERS["qwen2moe"] is _topk_router_module, \
        "case: topk_router_module_adapter"
    assert MOE_ROUTER_ADAPTERS["qwen3moe"] is _topk_router_module, \
        "case: topk_router_module_adapter"
    assert MOE_ROUTER_ADAPTERS["mixtral"] is _topk_router_module, \
        "case: topk_router_module_adapter"

    # ── case: test_sigmoid_group_router_adapter ──
    # deepseekv3/glm4moe adapter: sigmoid + correction bias + norm + scaling
    # (n_group=1 skips the group filter), matching a hand-computed reference.
    class Gate(nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(e, h) * 0.02)
            self.register_buffer("e_score_correction_bias", torch.randn(e) * 0.01)

        def forward(self, x):
            return F.linear(  # pylint: disable=not-callable
                x.view(-1, x.shape[-1]).float(), self.weight.float()
            )

    class MoE(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.gate = Gate(4, 16)
            self.top_k = 2
            self.n_group = 1
            self.norm_topk_prob = True
            self.routed_scaling_factor = 2.5

    torch.manual_seed(5)
    moe = MoE()
    hidden = torch.randn(2, 3, 16)
    idx, w = _sigmoid_group_router(moe, hidden)

    logits = moe.gate(hidden)
    scores = logits.sigmoid()
    choice = scores + moe.gate.e_score_correction_bias
    ref_idx = choice.topk(2, dim=-1, sorted=False)[1]
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 2.5
    assert torch.equal(idx, ref_idx), "case: sigmoid_group_router_adapter"
    torch.testing.assert_close(w, ref_w.to(w.dtype),
                               msg="case: sigmoid_group_router_adapter")

    moe.n_group = 2
    moe.topk_group = 1
    idx, w = _sigmoid_group_router(moe, hidden)
    group_scores = choice.view(-1, 2, 2).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = group_scores.topk(1, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, 2, 2).reshape(-1, 4)
    grouped_choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
    ref_idx = grouped_choice.topk(2, dim=-1, sorted=False)[1]
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 2.5
    assert torch.equal(idx, ref_idx), "case: sigmoid_group_router_group_filter"
    torch.testing.assert_close(
        w, ref_w.to(w.dtype), msg="case: sigmoid_group_router_group_filter"
    )


def test_fix_router_balanced_load(monkeypatch):
    """``fix_router`` variant: round-robin assignment keeps the real top-k
    weights but hands every destination rank exactly T*K/EP tokens."""
    # ── case: balanced_router_even_load ──
    # 12 tokens x top_k 2 = 24 slots over 8 experts -> 3 slots per expert,
    # against a gate whose own preference would be anything but uniform.
    class Gate(nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(e, h) * 0.02)
            self.register_buffer("e_score_correction_bias", torch.randn(e) * 0.01)

        def forward(self, x):
            return F.linear(  # pylint: disable=not-callable
                x.view(-1, x.shape[-1]).float(), self.weight.float()
            )

    class MoE(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.gate = Gate(8, 16)
            self.num_experts = 8
            self.top_k = 2

    torch.manual_seed(7)
    moe = MoE()
    hidden = torch.randn(12, 16)
    idx, w = _balanced_router(moe, hidden)
    assert idx.shape == (12, 2), "case: balanced_router_even_load"
    counts = torch.bincount(idx.reshape(-1), minlength=8)
    assert counts.tolist() == [3] * 8, "case: balanced_router_even_load"
    assert torch.equal(idx, (torch.arange(24) % 8).view(12, 2)), \
        "case: balanced_router_even_load"

    # ── case: balanced_router_even_destination_load ──
    # EP shape: 4 destinations x 2 local experts. 10 tokens x top_k 2 = 20
    # slots -> exactly 5 per destination. A per-expert round robin (i % 8)
    # replays the same residues on every rank instead, so the leftover piles
    # onto the first destinations: 6, 6, 4, 4 -- and it is the busiest
    # destination that sets the step time.
    ep_moe = MoE()
    ep_moe.experts = nn.Module()
    ep_moe.experts.local_expert_count = 2
    ep_hidden = torch.randn(10, 16)
    idx, _ = _balanced_router(ep_moe, ep_hidden)
    per_destination = torch.bincount(idx.reshape(-1) // 2, minlength=4)
    assert per_destination.tolist() == [5, 5, 5, 5], \
        "case: balanced_router_even_destination_load"
    naive = torch.bincount(
        (torch.arange(20) % 8) // 2, minlength=4)
    assert naive.tolist() == [6, 6, 4, 4], \
        "case: balanced_router_even_destination_load"

    # Only the assignment is replaced: the true top-k weights are kept, so
    # the gate keeps its gradient and the router GEMM stays in the profile.
    _, ref_w = _sigmoid_group_router(moe, hidden)
    torch.testing.assert_close(w, ref_w, msg="case: balanced_router_even_load")
    assert MOE_ROUTER_ADAPTERS["deepseekv3_fixed"] is _balanced_router, \
        "case: balanced_router_even_load"

    # ── case: fix_router_selects_adapter ──
    # The switch is the ONLY difference in the factory: off keeps the real
    # sigmoid-group router, on swaps in the balanced one.
    module = _TinyMoeMod()
    module.shared_experts = nn.Identity()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.deepseekv3_ep_compute_fn(
        module=module, mesh=None, tp_mesh=None, cp_mesh=None,
        ep_mesh=_FakeEpMesh())
    compute_fn(module, torch.randn(2, 4))
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["deepseekv3"], \
        "case: fix_router_selects_adapter"

    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.deepseekv3_ep_compute_fn(
        module=module, mesh=None, tp_mesh=None, cp_mesh=None,
        ep_mesh=_FakeEpMesh(), fix_router=True)
    compute_fn(module, torch.randn(2, 4))
    assert captured["router_fn"] is _balanced_router, \
        "case: fix_router_selects_adapter"


def test_argsort_keys_aicore_path(monkeypatch):
    """``_argsort_keys`` keeps the order via a float32 key, and only when the
    key range is exactly representable (that is what moves the kernel from
    AI_CPU to AICore on the EP dispatch path)."""
    # ── case: argsort_fp32_key_same_order ──
    # Dispatch keys are dest_rank * E + expert_idx, bounded by ep_size * E; real
    # ties are expected, so compare the sorted KEY SEQUENCE rather than the
    # permutation (the order of equal keys is not part of the contract).
    torch.manual_seed(11)
    keys = torch.randint(0, 128 * 384, (256,), dtype=torch.int64)
    reference = keys.argsort()

    monkeypatch.setattr(ep_experts, "_SORT_FP32_ENABLED", True)
    got = _argsort_keys(keys, bound=128 * 384)
    assert got.dtype == reference.dtype, "case: argsort_fp32_key_same_order"
    assert torch.equal(keys[got], keys[reference]), \
        "case: argsort_fp32_key_same_order"

    # ── case: argsort_fp32_key_falls_back_out_of_range ──
    # The largest integer float32 holds exactly is 2**24 - 1, so a bound above
    # 2**24 must keep the exact int64 path.
    big = torch.tensor([2 ** 24 + 1, 3, 2 ** 24, 2 ** 24 + 2], dtype=torch.int64)
    assert torch.equal(_argsort_keys(big, bound=2 ** 25), big.argsort()), \
        "case: argsort_fp32_key_falls_back_out_of_range"
    # At the limit itself (keys < 2**24) the float32 path stays exact.
    edge = torch.tensor([2 ** 24 - 1, 5, 0, 2 ** 24 - 2], dtype=torch.int64)
    assert torch.equal(_argsort_keys(edge, bound=2 ** 24), edge.argsort()), \
        "case: argsort_fp32_key_falls_back_out_of_range"


def test_expert_token_counts_without_bincount(monkeypatch):
    """EP routed-token counts come from a device-side histogram instead of
    ``torch.bincount``, whose NPU implementation reads the input's min and max
    back to the host on every call (two blocking D2H drains per MoE layer in
    the profiled K2.6 step). Values, length and dtype are unchanged and the
    grouped-GEMM path never reads a scalar back to the host."""
    bincount_calls = []
    real_bincount = torch.bincount

    def spy_bincount(*args, **kwargs):
        bincount_calls.append(1)
        return real_bincount(*args, **kwargs)

    monkeypatch.setattr(torch, "bincount", spy_bincount)

    # ── case: expert_token_counts_match_bincount ──
    # Same values and the same minlength contract (an expert that got no tokens
    # keeps its zero slot); int64 output whatever the index dtype.
    indices = torch.tensor([0, 2, 2, 1, 2, 0], dtype=torch.int64)
    reference = real_bincount(indices, minlength=4)
    counts = _expert_token_counts(indices, 4)
    assert counts.dtype == torch.int64, \
        f"case: expert_token_counts_match_bincount: dtype={counts.dtype}"
    assert counts.shape == reference.shape, \
        f"case: expert_token_counts_match_bincount: shape={tuple(counts.shape)}, " \
        f"expected={tuple(reference.shape)}"
    assert torch.equal(counts, reference), \
        f"case: expert_token_counts_match_bincount: counts={counts.tolist()}, " \
        f"expected={reference.tolist()}"
    assert torch.equal(_expert_token_counts(indices.to(torch.int32), 4), reference), \
        f"case: expert_token_counts_match_bincount: int32 index mismatch, " \
        f"counts={counts.tolist()}"
    assert torch.equal(_expert_token_counts(indices.new_zeros(0), 3),
                       torch.zeros(3, dtype=torch.int64)), \
        "case: expert_token_counts_match_bincount: empty input must keep minlength"
    assert not bincount_calls, \
        f"case: expert_token_counts_match_bincount: torch.bincount called " \
        f"{len(bincount_calls)} time(s)"

    # ── case: grouped_experts_no_host_readback ──
    # Grouped GEMM consumes the counts as its group-list source on device: each
    # expert block must hold exactly that expert's rows, and no scalar may be
    # read back to the host anywhere in the path.
    def reject_readback(*args, **kwargs):
        raise AssertionError("host scalar readback on the grouped expert path")

    class GroupedExperts(nn.Module):  # pylint: disable=abstract-method
        def __init__(self, local_expert_count):
            super().__init__()
            self.local_expert_count = local_expert_count
            self.ep_use_grouped_gemm = True
            self.counts = None
            self.seen_rows = None

        def forward_expert_major(self, x, num_tokens_per_expert, scores=None):
            self.counts = num_tokens_per_expert.clone()
            self.seen_rows = x.clone()
            return x

    grouped = GroupedExperts(3)
    # Row i of `dispatched` starts with i * 4, so a block's rows identify the
    # source token they came from.
    dispatched = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
    expert_id = torch.tensor([2, 0, 2, 1, 2, 0, 1], dtype=torch.int64)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(torch.Tensor, "item", reject_readback)
        guard.setattr(torch.Tensor, "tolist", reject_readback)
        output = _local_swiglu_expert_forward(grouped, dispatched, expert_id)
    grouped_counts = real_bincount(expert_id, minlength=3)
    assert torch.equal(grouped.counts, grouped_counts), \
        f"case: grouped_experts_no_host_readback: counts={grouped.counts.tolist()}, " \
        f"expected={grouped_counts.tolist()}"
    assert torch.equal(grouped.counts.cumsum(0), torch.tensor([2, 4, 7])), \
        f"case: grouped_experts_no_host_readback: group_list={grouped.counts.cumsum(0).tolist()}"
    for expert_index, block_size in enumerate(grouped_counts.tolist()):
        starts = grouped_counts.cumsum(0).tolist()
        begin = starts[expert_index] - block_size
        block_rows = grouped.seen_rows[begin:begin + block_size, 0].tolist()
        source_rows = (expert_id == expert_index).nonzero().reshape(-1).tolist()
        assert sorted(block_rows) == sorted(row * 4 for row in source_rows), \
            f"case: grouped_experts_no_host_readback: expert {expert_index} block rows " \
            f"{block_rows}, expected {sorted(row * 4 for row in source_rows)}"
    assert torch.equal(output, dispatched), \
        f"case: grouped_experts_no_host_readback: output={output.tolist()}, " \
        f"expected={dispatched.tolist()}"

    # ── case: eager_expert_forward_matches_bincount_reference ──
    # The eager (non-grouped) path still needs host ints; it now reads them in
    # one transfer, so compare against a slicing reference built from bincount.
    class EagerExperts(nn.Module):  # pylint: disable=abstract-method
        def __init__(self, local_expert_count):
            super().__init__()
            self.local_expert_count = local_expert_count
            self.gate_up_proj = nn.Parameter(torch.randn(local_expert_count, 8, 4))
            self.down_proj = nn.Parameter(torch.randn(local_expert_count, 4, 4))
            self.ep_act_fn = torch.tanh

    torch.manual_seed(13)
    eager = EagerExperts(3)
    eager_id = torch.tensor([1, 2, 1, 0, 2, 2, 1], dtype=torch.int64)
    got = _local_swiglu_expert_forward(eager, dispatched, eager_id)
    order = _argsort_keys(eager_id, bound=3)
    sorted_states = dispatched[order]
    pieces = []
    begin = 0
    for expert_index, block_size in enumerate(
            real_bincount(eager_id, minlength=3).tolist()):
        chunk = sorted_states[begin:begin + block_size]
        gate_states, up_states = F.linear(  # pylint: disable=not-callable
            chunk, eager.gate_up_proj[expert_index]).chunk(2, dim=-1)
        pieces.append(F.linear(  # pylint: disable=not-callable
            torch.tanh(gate_states) * up_states, eager.down_proj[expert_index]))
        begin += block_size
    expected = torch.empty_like(dispatched)
    expected[order] = torch.cat(pieces)
    torch.testing.assert_close(
        got, expected,
        msg=f"case: eager_expert_forward_matches_bincount_reference: "
            f"got={got.tolist()}, expected={expected.tolist()}")

    # ── case: dispatch_send_counts_no_bincount ──
    # The ragged all-to-all still needs host counts, but they must come from the
    # same device histogram (one transfer) rather than a bincount drain.
    exchanged = {}

    def fake_all_to_all_single(output_tensor, input_tensor, group=None):
        exchanged["group"] = group
        exchanged["send"] = input_tensor.clone()
        output_tensor.copy_(input_tensor)

    monkeypatch.setattr(ep_experts.dist, "all_to_all_single",
                        fake_all_to_all_single)
    hidden = torch.randn(3, 4)
    topk_index = torch.tensor([[0, 3], [4, 5], [1, 2]], dtype=torch.int64)
    topk_weight = torch.rand(3, 2)
    dispatch = _prepare_ep_dispatch(
        hidden, topk_index, topk_weight,
        local_expert_count=2, global_expert_count=8, ep_size=4,
        ep_group="group-ep",
    )
    destination = topk_index.reshape(-1) // 2
    expected_counts = real_bincount(destination, minlength=4).tolist()
    assert dispatch.send_counts == expected_counts, \
        f"case: dispatch_send_counts_no_bincount: send_counts={dispatch.send_counts}, " \
        f"expected={expected_counts}"
    # the fake exchange is the identity, so recv_counts mirrors send_counts
    assert dispatch.receive_counts == expected_counts, \
        f"case: dispatch_send_counts_no_bincount: recv_counts={dispatch.receive_counts}, " \
        f"expected={expected_counts}"
    assert exchanged["send"].dtype == torch.int64, \
        f"case: dispatch_send_counts_no_bincount: dtype={exchanged['send'].dtype}"
    source_indices = dispatch.source_indices
    dispatch_order = dispatch.dispatch_order
    dispatched_states = dispatch.states
    dispatched_indices = dispatch.expert_indices
    assert torch.equal(dispatched_states, hidden[source_indices[dispatch_order]]), \
        f"case: dispatch_send_counts_no_bincount: dispatched rows=" \
        f"{dispatched_states.tolist()}, expected={hidden[source_indices[dispatch_order]].tolist()}"
    assert dispatched_indices.shape == (6, 1), \
        f"case: dispatch_send_counts_no_bincount: expert indices shape=" \
        f"{tuple(dispatched_indices.shape)}"
    assert not bincount_calls, \
        f"case: dispatch_send_counts_no_bincount: torch.bincount called " \
        f"{len(bincount_calls)} time(s)"


# ==========================================================================
# Family 10: planner TP-extend-EP contracts + expert mesh layout
# Sources: test_planner_ep_extend_contract, test_planner_ep1_no_extend,
#          test_planner_batched_contract, test_planner_batched_ep1_no_mark,
#          test_expert_mesh_layout_mapping
# ==========================================================================

def test_planner_ep_extend_contracts(tiny_hf_native_moe, tiny_hf_batched_moe,
                                     make_mesh):
    """D-10/D-11 TP-extend-EP planner contracts (identity boundary, expert
    params sharded only by {EP: Shard(0)}, router fully replicated) and the
    derived expert mesh layout mapping."""
    # ── case: test_planner_ep_extend_contract ──
    # mesh (dp=4, tp=2), ep=4 -> extended EP groups {0,1,2,3}/{4,5,6,7}
    # (user example).
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4, \
        "case: planner_ep_extend_contract"   # ep_size is the extended EP group size
    assert spec._ep_stack, "case: planner_ep_extend_contract"  # stacked metadata unchanged

    # expert params: only {EP: Shard(0)} (expert-dim sharding), no TP key,
    # no second axis
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_ep_extend_contract"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_ep_extend_contract"
        assert len(p) == 2, \
            "case: planner_ep_extend_contract"   # only CP(Replicate) + EP keys

    # router fully replicated (local chunk computation)
    assert spec.params["gate.weight"][TP] == Replicate(), \
        "case: planner_ep_extend_contract"

    # Boundary contract identity (SP-in): in_dst/out_src/out_dst all
    # TP Shard(1)
    assert spec.in_dst["x_BLD"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"
    assert spec.out_src["output"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"
    assert spec.out_dst["output"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"
    # in_src unchanged from the upstream contract (chained validation passes)
    assert spec.in_src["x_BLD"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"

    # ── case: test_planner_ep1_no_extend ──
    # ep=1 -> no TP-extend-EP, per-expert entries kept (TP-only semantics
    # correct).
    mesh1 = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh1, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0, "case: planner_ep1_no_extend"
    assert spec._ep_stack == {}, "case: planner_ep1_no_extend"
    assert "experts.0.gate_proj.weight" in spec.params, \
        "case: planner_ep1_no_extend"
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0), \
        "case: planner_ep1_no_extend"

    # ── case: test_planner_batched_contract ──
    # D-11 batched layout (experts.gate_up_proj [E,2I,H]): no stacking
    # needed, marked directly with {EP: Shard(0)}; arch=qwen3moe ->
    # TopKRouter module adapter.
    plan = ShardingPlanner().plan(tiny_hf_batched_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4, "case: planner_batched_contract"
    assert spec._ep_stack == {}, \
        "case: planner_batched_contract"   # batched is stacked by nature; no stacking needed

    # expert params: only {EP: Shard(0)} (expert-dim sharding), no TP key,
    # no second axis
    for proj in ("gate_up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_batched_contract"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_batched_contract"
        assert len(p) == 2, "case: planner_batched_contract"

    # router (TopKRouter.weight) fully replicated; boundary identity
    assert spec.params["gate.weight"][TP] == Replicate(), \
        "case: planner_batched_contract"
    assert spec.in_dst["x_BLD"][TP] == Shard(1), "case: planner_batched_contract"
    assert spec.out_src["output"][TP] == Shard(1), \
        "case: planner_batched_contract"
    assert spec.out_dst["output"][TP] == Shard(1), \
        "case: planner_batched_contract"

    # ── case: test_planner_batched_ep1_no_mark ──
    # batched layout with ep=1 -> no marking (_ep_size == 0).
    #
    # Writing fused weights as bare TP Shard(1) (the old D-08 semantics) is
    # fail-fasted by _finalize_fused_expert_tp_guard -- contiguous block
    # sharding is incompatible with the in-forward chunk; a legal
    # configuration must override to TP Replicate (guard solution 2, and
    # out_src's TP must likewise be changed from the template-derived Partial
    # to Replicate).
    def rep():
        return {TP: Replicate(), CP: Replicate()}
    overrides = {"*.mlp": ModuleShardingSpec(
        params={
            "gate.weight": rep(),
            "experts.gate_up_proj": rep(),
            "experts.down_proj": rep(),
        },
        out_src={"output": rep()},
    )}
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_hf_batched_moe, mesh1, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0, "case: planner_batched_ep1_no_mark"
    assert spec.params["experts.gate_up_proj"][TP] == Replicate(), \
        "case: planner_batched_ep1_no_mark"

    # ── case: test_expert_mesh_layout_mapping ──
    # Derived expert mesh: the dense region is flattened into (edp, ep), and
    # an EP group is ep_size consecutive ranks in the flattened order
    # (spanning the TP group first, then extending to adjacent dp ranks).
    # mesh rank = d*2 + t
    # ep=4 (user example): EP groups {0,1,2,3} / {4,5,6,7} -- spanning
    # 2 TP groups x 2 dp
    shape, names, rank_list = _expert_mesh_layout(mesh, ("dp", "tp"), 4)
    assert shape == (2, 4), "case: expert_mesh_layout_mapping"
    assert names == ("edp", "ep"), "case: expert_mesh_layout_mapping"
    assert rank_list == (0, 1, 2, 3, 4, 5, 6, 7), \
        "case: expert_mesh_layout_mapping"

    # ep=2: EP groups {0,1}/{2,3}/{4,5}/{6,7} -- i.e. the TP groups
    shape, names, _ = _expert_mesh_layout(mesh, ("dp", "tp"), 2)
    assert shape == (4, 2), "case: expert_mesh_layout_mapping"
    assert names == ("edp", "ep"), "case: expert_mesh_layout_mapping"

    # Non-divisible -> error
    with pytest.raises(ValueError, match="must divide"):
        _expert_mesh_layout(mesh, ("dp", "tp"), 3)


# ==========================================================================
# Family 11: opt-in fused states+indices dispatch (HP_EP_FUSED_DISPATCH)
# The routed dispatch sends the hidden states and the expert indices of a token
# in two exchanges that carry identical token counts; the fused mode packs both
# payloads into one row and exchanges the row once. These cases pin what the
# switch promises: the routed tokens are unchanged, one exchange replaces two,
# an unset HP_EP_FUSED_DISPATCH means off, and the unpack stays view-only so
# the lazy wait (the shared-expert overlap window) is untouched.
# ==========================================================================

class _FusedDispatchMoe(nn.Module):
    """MoE double: the routed dispatch reads only ``experts.local_expert_count``."""

    def __init__(self, local_expert_count):
        super().__init__()
        self.experts = nn.Module()
        self.experts.local_expert_count = local_expert_count


class _FakeEpGroup:
    """Size+rank EP group double (the role _FakeEpMesh's group plays)."""

    def __init__(self, rank, size):
        self.rank = rank
        self._size = size

    def size(self):
        return self._size


class _FakeEpA2AWorld:
    """One-process EP world double for the routed token exchange.

    Ranks run one after another in this process, so no transfer can happen
    inside a call: every rank records the chunks it split out, and
    :meth:`settle` files chunk ``i`` of each receive buffer from rank ``i`` --
    the ``all_to_all`` contract ``_EPAllToAllUneven`` implements. The routing
    plan belongs to the test, so the per-rank counts stay unequal, including a
    destination that receives nothing at all.

    ``wrap_async`` hands back a real ``AsyncCollectiveTensor``, which exercises
    the lazy-wait behaviour of the production exchange without a process group.
    """

    def __init__(self, ep_size, send_plan, wrap_async=False):
        self.ep_size = ep_size
        self.send_plan = send_plan
        self.wrap_async = wrap_async
        self.sent = [[] for _ in range(ep_size)]
        self.handles = []

    def count_exchange(self, output_tensor, input_tensor, group=None):
        """Stand-in for the counts all_to_all_single in _prepare_ep_dispatch."""
        rank = group.rank
        sends = [int(count) for count in input_tensor.tolist()]
        assert sends == self.send_plan[rank], \
            (f"case: fused_dispatch_counts_match_plan: rank {rank} computed "
             f"send_counts={sends}, plan={self.send_plan[rank]}")
        output_tensor.copy_(torch.tensor(
            [self.send_plan[src][rank] for src in range(self.ep_size)],
            dtype=output_tensor.dtype, device=output_tensor.device))

    def token_exchange(self, rank, tensor, send_counts, recv_counts):
        """Stand-in for ep_all_to_all_async: record the send, return the buffer."""
        buffer = tensor.new_empty((sum(recv_counts),) + tuple(tensor.shape[1:]))
        self.sent[rank].append((tensor.split(send_counts), list(recv_counts), buffer))
        handle = AsyncCollectiveTensor(buffer) if self.wrap_async else buffer
        self.handles.append(handle)
        return handle

    def settle(self):
        """File every recorded chunk into its destination's receive buffer."""
        for call in range(len(self.sent[0])):
            chunks = {}
            for rank in range(self.ep_size):
                for dest, chunk in enumerate(self.sent[rank][call][0]):
                    chunks[(rank, dest)] = chunk
            for rank in range(self.ep_size):
                _, recv_counts, buffer = self.sent[rank][call]
                pieces = [chunks[(src, rank)] for src in range(self.ep_size)]
                for src, piece in enumerate(pieces):
                    assert piece.shape[0] == recv_counts[src], \
                        (f"case: fused_dispatch_counts_match_plan: rank {rank} expects "
                         f"{recv_counts[src]} rows from rank {src}, got {piece.shape[0]}")
                buffer.copy_(torch.cat(pieces))


def _fused_dispatch_plan():
    """Routing plan: per-rank expert index of every (token, top-k slot).

    Expert ``e`` is dispatched to destination ``e // local_expert_count``; the
    plan keeps some destinations far busier than others and leaves destination
    2 fed by nobody, so both "unequal split counts" and "a rank that receives
    zero tokens" are part of the scenario.
    """
    local_expert_count = 2
    slots = {
        0: [0, 1, 0, 3, 1, 0, 2, 3],
        1: [4, 5, 6, 4, 5, 6, 7, 4],
        2: [6, 7, 7, 6, 6, 7, 7, 6],   # nothing routes to destination 2
        3: [1, 0, 0, 1, 0, 1, 1, 0],
    }
    send_plan = {
        rank: [sum(1 for expert in experts if expert // local_expert_count == dest)
               for dest in range(len(slots))]
        for rank, experts in slots.items()
    }
    return slots, send_plan, local_expert_count


def _fused_dispatch_inputs():
    """Scenario for the exchange cases: routing plan plus per-rank inputs.

    The plan is the unequal one (destination 0 gets 5+8 routed rows while
    destination 2 gets none), so a packing that misplaces a row or an offset
    cannot hide.
    """
    slots, send_plan, local_expert_count = _fused_dispatch_plan()
    experts_per_token, hidden_size = 2, 3
    ep_size = len(slots)
    token_count = len(slots[0]) // experts_per_token
    torch.manual_seed(23)
    return {
        "slots": slots,
        "send_plan": send_plan,
        "local_expert_count": local_expert_count,
        "ep_size": ep_size,
        "experts_per_token": experts_per_token,
        "hidden_size": hidden_size,
        "row_count": len(slots[0]),
        "hidden": {rank: torch.randn(2, 2, hidden_size, dtype=torch.bfloat16)
                   for rank in range(ep_size)},
        "topk": {rank: torch.tensor(experts, dtype=torch.int64).view(
            token_count, experts_per_token) for rank, experts in slots.items()},
        "weights": {rank: torch.rand(token_count, experts_per_token)
                    for rank in range(ep_size)},
    }


def _run_routed_dispatch(monkeypatch, case, fused, wrap_async=False):
    """Drive every rank's routed dispatch once and settle the world.

    ``fake_async`` stands in for ``ep_all_to_all_async`` at the module call
    site, so the switch is the only difference between the two modes.
    """
    world = _FakeEpA2AWorld(case["ep_size"], case["send_plan"], wrap_async=wrap_async)
    exchanges = []

    def fake_async(tensor, send_counts, recv_counts, group, **kwargs):
        del kwargs  # the fused dispatch opts out of pending handles; a tensor is what it needs
        exchanges.append((group.rank, tuple(tensor.shape), tensor.dtype))
        return world.token_exchange(group.rank, tensor, send_counts, recv_counts)

    def router_for(rank):
        def router(module, hidden_states):
            return case["topk"][rank], case["weights"][rank]
        return router

    monkeypatch.setattr(ep_experts, "ep_all_to_all_async", fake_async)
    monkeypatch.setattr(ep_experts.dist, "all_to_all_single", world.count_exchange)
    monkeypatch.setattr(ep_experts, "_FUSED_DISPATCH_ENABLED", fused)
    states, indices = {}, {}
    for rank in range(case["ep_size"]):
        state = ep_experts.ep_routed_dispatch(
            _FusedDispatchMoe(case["local_expert_count"]),
            case["hidden"][rank],
            router_fn=router_for(rank),
            ep_group=_FakeEpGroup(rank, case["ep_size"]),
        )
        states[rank] = state.received_states
        indices[rank] = state.received_indices
    world.settle()
    return world, states, indices, exchanges


def test_fused_dispatch_one_exchange_per_pass(monkeypatch):
    """The fused mode replaces the two routed exchanges with one, and that one
    row carries the states' bytes plus the indices' bytes."""
    case = _fused_dispatch_inputs()
    ep_size = case["ep_size"]
    hidden_size = case["hidden_size"]
    row_count = case["row_count"]

    # ── case: fused_dispatch_counts_match_plan ──
    # The scenario is the unequal one: destination 0 gets 5+8 rows while
    # destination 2 gets none.
    assert case["send_plan"] == {0: [5, 3, 0, 0], 1: [0, 0, 5, 3],
                                2: [0, 0, 0, 8], 3: [8, 0, 0, 0]}, \
        f"case: fused_dispatch_counts_match_plan: send_plan={case['send_plan']}"

    world_off, _, _, calls_off = _run_routed_dispatch(monkeypatch, case, False)
    world_on, _, _, calls_on = _run_routed_dispatch(monkeypatch, case, True)

    # ── case: fused_dispatch_one_exchange_per_pass ──
    # The switch is the only difference in the schedule: OFF issues the states
    # and then the indices (two exchanges per rank), ON issues one packed row
    # per rank.
    assert [len(world_off.sent[rank]) for rank in range(ep_size)] == [2] * ep_size, \
        (f"case: fused_dispatch_one_exchange_per_pass: OFF exchanges per rank="
         f"{[len(world_off.sent[rank]) for rank in range(ep_size)]}, expected 2 each")
    assert [len(world_on.sent[rank]) for rank in range(ep_size)] == [1] * ep_size, \
        (f"case: fused_dispatch_one_exchange_per_pass: ON exchanges per rank="
         f"{[len(world_on.sent[rank]) for rank in range(ep_size)]}, expected 1 each")
    assert len(calls_off) == 2 * ep_size and len(calls_on) == ep_size, \
        (f"case: fused_dispatch_one_exchange_per_pass: ep_all_to_all_async calls "
         f"OFF={len(calls_off)}, ON={len(calls_on)}, expected "
         f"{2 * ep_size}/{ep_size}")
    assert [call[:2] for call in calls_off] == [
        (rank, shape)
        for rank in range(ep_size)
        for shape in ((row_count, hidden_size), (row_count, 1))
    ], f"case: fused_dispatch_one_exchange_per_pass: OFF calls={calls_off}"
    on_widths = {call[1][1] for call in calls_on}
    assert {call[0] for call in calls_on} == set(range(ep_size)) and len(on_widths) == 1, \
        f"case: fused_dispatch_one_exchange_per_pass: ON calls={calls_on}"
    states_bytes = hidden_size * torch.bfloat16.itemsize
    index_dtype_bytes = torch.int64.itemsize
    packed_row_bytes = on_widths.pop() * torch.bfloat16.itemsize
    assert packed_row_bytes >= states_bytes + index_dtype_bytes, \
        (f"case: fused_dispatch_one_exchange_per_pass: packed row is "
         f"{packed_row_bytes} bytes, expected at least "
         f"{states_bytes + index_dtype_bytes}")


def test_fused_dispatch_equivalence_across_ranks(monkeypatch):
    """HP_EP_FUSED_DISPATCH=1 delivers exactly the routed tokens the two
    separate exchanges deliver: same shapes, same dtypes, same values."""
    case = _fused_dispatch_inputs()
    ep_size = case["ep_size"]
    hidden_size = case["hidden_size"]
    hidden = case["hidden"]
    slots = case["slots"]
    local_expert_count = case["local_expert_count"]
    experts_per_token = case["experts_per_token"]
    send_plan = case["send_plan"]

    _, states_off, indices_off, _ = _run_routed_dispatch(monkeypatch, case, False)
    _, states_on, indices_on, _ = _run_routed_dispatch(monkeypatch, case, True)

    # ── case: fused_dispatch_matches_split_exchanges ──
    # Element-wise identical routed states/indices per rank, which is what makes
    # the switch safe to enable: same dtype, same shape, same values.
    for rank in range(ep_size):
        expected_rows = sum(send_plan[src][rank] for src in range(ep_size))
        assert states_on[rank].shape == states_off[rank].shape == \
            (expected_rows, hidden_size), \
            (f"case: fused_dispatch_matches_split_exchanges: rank {rank} shapes "
             f"fused={tuple(states_on[rank].shape)}, "
             f"split={tuple(states_off[rank].shape)}, "
             f"expected={(expected_rows, hidden_size)}")
        assert states_on[rank].dtype is states_off[rank].dtype, \
            (f"case: fused_dispatch_matches_split_exchanges: rank {rank} dtypes "
             f"fused={states_on[rank].dtype}, split={states_off[rank].dtype}")
        assert torch.equal(states_on[rank], states_off[rank]), \
            (f"case: fused_dispatch_matches_split_exchanges: rank {rank} states "
             f"fused={states_on[rank].tolist()}, split={states_off[rank].tolist()}")
        assert indices_on[rank].shape == indices_off[rank].shape == (expected_rows,), \
            (f"case: fused_dispatch_matches_split_exchanges: rank {rank} index shapes "
             f"fused={tuple(indices_on[rank].shape)}, "
             f"split={tuple(indices_off[rank].shape)}, expected={(expected_rows,)}")
        assert indices_on[rank].dtype is torch.int64, \
            (f"case: fused_dispatch_matches_split_exchanges: rank {rank} index dtype "
             f"fused={indices_on[rank].dtype}, split={indices_off[rank].dtype}")
        assert torch.equal(indices_on[rank], indices_off[rank]), \
            (f"case: fused_dispatch_matches_split_exchanges: rank {rank} indices "
             f"fused={indices_on[rank].tolist()}, "
             f"split={indices_off[rank].tolist()}")

    # ── case: fused_dispatch_delivers_the_routed_rows ──
    # Independent anchor (it also validates the world double itself): what a
    # rank receives is the multiset of rows routed to it, so a packing bug
    # cannot pass by matching its own output.
    for rank in range(ep_size):
        routed_rows, routed_experts = [], []
        for src in range(ep_size):
            flat = hidden[src].reshape(-1, hidden_size)
            for slot, expert in enumerate(slots[src]):
                if expert // local_expert_count == rank:
                    routed_rows.append(flat[slot // experts_per_token])
                    routed_experts.append(expert)
        assert states_on[rank].shape[0] == len(routed_rows), \
            (f"case: fused_dispatch_delivers_the_routed_rows: rank {rank} received "
             f"{states_on[rank].shape[0]} rows, routed={len(routed_rows)}")
        if not routed_rows:
            continue
        expected_states = torch.stack(routed_rows)
        expected_indices = torch.tensor(routed_experts, dtype=torch.int64)
        assert torch.equal(states_on[rank].sort(dim=0).values,
                           expected_states.sort(dim=0).values), \
            (f"case: fused_dispatch_delivers_the_routed_rows: rank {rank} states "
             f"received={states_on[rank].tolist()}, routed={expected_states.tolist()}")
        assert torch.equal(indices_on[rank].sort().values, expected_indices.sort().values), \
            (f"case: fused_dispatch_delivers_the_routed_rows: rank {rank} indices "
             f"received={indices_on[rank].tolist()}, routed={expected_indices.tolist()}")


def test_fused_dispatch_keeps_the_lazy_wait(monkeypatch):
    """The fused unpack is view-only, so the exchange's wait still lands on the
    first real consumer instead of on the dispatch call itself."""
    case = _fused_dispatch_inputs()
    ep_size = case["ep_size"]
    _, states_off, indices_off, _ = _run_routed_dispatch(monkeypatch, case, False)

    # ── case: fused_dispatch_keeps_the_lazy_wait ──
    # A real AsyncCollectiveTensor stands in for the collective handle: pack and
    # unpack must be pure views, so nothing enqueues the wait between the
    # exchange and the independent work the caller runs before the experts read
    # the result. A materializing op in the unpack (a reshape that copies, a
    # contiguous()) would complete the handle right here and serialize the
    # shared-expert overlap away -- the very thing the fused path must keep.
    world_lazy, states_lazy, indices_lazy, _ = _run_routed_dispatch(
        monkeypatch, case, True, wrap_async=True)
    assert len(world_lazy.handles) == ep_size, \
        (f"case: fused_dispatch_keeps_the_lazy_wait: handles={len(world_lazy.handles)}, "
         f"expected={ep_size}")
    for handle in world_lazy.handles:
        assert handle.completed is False, \
            "case: fused_dispatch_keeps_the_lazy_wait: the unpack waited on the exchange"
    for rank in range(ep_size):
        assert states_lazy[rank].completed is False, \
            (f"case: fused_dispatch_keeps_the_lazy_wait: rank {rank} states view "
             f"completed={states_lazy[rank].completed}, expected=False")
        assert indices_lazy[rank].completed is False, \
            (f"case: fused_dispatch_keeps_the_lazy_wait: rank {rank} index view "
             f"completed={indices_lazy[rank].completed}, expected=False")
        # The experts' first read is what materializes the pending exchange.
        materialized_states = states_lazy[rank] + 0
        materialized_indices = indices_lazy[rank] - 0
        assert torch.equal(materialized_states, states_off[rank]), \
            (f"case: fused_dispatch_keeps_the_lazy_wait: rank {rank} states after the "
             f"wait={materialized_states.tolist()}, split={states_off[rank].tolist()}")
        assert torch.equal(materialized_indices, indices_off[rank]), \
            (f"case: fused_dispatch_keeps_the_lazy_wait: rank {rank} indices after the "
             f"wait={materialized_indices.tolist()}, split={indices_off[rank].tolist()}")
        assert states_lazy[rank].completed is True, \
            (f"case: fused_dispatch_keeps_the_lazy_wait: rank {rank} states view "
             f"completed={states_lazy[rank].completed}, expected=True")
        assert indices_lazy[rank].completed is True, \
            (f"case: fused_dispatch_keeps_the_lazy_wait: rank {rank} index view "
             f"completed={indices_lazy[rank].completed}, expected=True")


class _IdentityEpA2AWorld:
    """One-rank EP world double whose exchange is the identity.

    With ``ep_size=1`` every expert is local, so the all-to-all moves nothing;
    handing the tensor straight back keeps the run free of the in-place copy the
    multi-rank double needs to fill its receive buffers, which is what makes the
    autograd graph here the production one -- the exchange is an autograd
    Function and the pack/unpack are ordinary ops on its input and output.
    """

    def __init__(self):
        self.calls = []

    def count_exchange(self, output_tensor, input_tensor, group=None):
        """One rank: the counts come back unchanged."""
        output_tensor.copy_(input_tensor)

    def token_exchange(self, tensor):
        """Record the exchanged shape and return the tensor (nothing moves)."""
        self.calls.append(tuple(tensor.shape))
        return tensor


class _IdentityA2AFunction(torch.autograd.Function):
    """Differentiable stand-in for ``ep_all_to_all_async``.

    What the gradient case needs is an exchange that behaves like the real one
    for autograd: a Function whose backward is the reverse exchange. On the
    one-rank world the forward and the backward are both the identity, so this
    is the real contract with the transfer factored out -- the packed row must
    stay on a differentiable *float* path, and a byte-view packing
    (``view(uint8)``) severs the graph right here without changing a single
    forward value.
    """

    @staticmethod
    def forward(ctx, tensor, world):  # pylint: disable=arguments-differ
        """Hand the tensor to the world's identity exchange."""
        return world.token_exchange(tensor)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Identity backward (the one-rank exchange moved nothing)."""
        return grad_output, None


def test_fused_dispatch_gradient_matches_split_exchanges(monkeypatch):
    """The fused dispatch keeps the routed branch differentiable: the gradient
    that reaches the local hidden states is the one the two separate exchanges
    produce. A packing that reinterpreted the states as bytes would detach the
    branch (every forward value unchanged, no gradient at all), which is
    invisible to any value-only comparison."""
    ep_size = 1
    local_expert_count = 4   # every expert is local -> the whole routing is rank 0's
    hidden_size = 3
    slots = [[0, 1], [2, 3], [0, 3], [1, 2]]

    def run(fused):
        torch.manual_seed(37)
        hidden = torch.randn(2, 2, hidden_size, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.rand(len(slots), len(slots[0]))
        topk = torch.tensor(slots, dtype=torch.int64)
        world = _IdentityEpA2AWorld()

        def fake_async(tensor, send_counts, recv_counts, group, **kwargs):
            del kwargs  # the fused dispatch opts out of pending handles
            return _IdentityA2AFunction.apply(tensor, world)

        monkeypatch.setattr(ep_experts, "ep_all_to_all_async", fake_async)
        monkeypatch.setattr(ep_experts.dist, "all_to_all_single", world.count_exchange)
        monkeypatch.setattr(ep_experts, "_FUSED_DISPATCH_ENABLED", fused)
        state = ep_experts.ep_routed_dispatch(
            _FusedDispatchMoe(local_expert_count),
            hidden,
            router_fn=lambda module, hidden_states: (topk, weights),
            ep_group=_FakeEpGroup(0, ep_size),
        )
        # ── case: fused_dispatch_keeps_the_routed_branch_differentiable ──
        # The routed states are a live graph node in both modes: the experts'
        # input must never arrive detached (that would train the routed branch
        # with a silently zero activation gradient).
        assert state.received_states.requires_grad is True, \
            (f"case: fused_dispatch_keeps_the_routed_branch_differentiable: fused={fused} "
             f"received_states.requires_grad={state.received_states.requires_grad}, "
             f"expected=True")
        loss = (state.received_states.float()
                * state.flattened_expert_weights.reshape(-1, 1)).sum()
        loss.backward()
        return hidden.grad.clone(), state.received_states.detach().clone()

    grad_off, states_off = run(False)
    grad_on, states_on = run(True)

    # ── case: fused_dispatch_gradient_matches_split_exchanges ──
    assert torch.equal(states_on, states_off), \
        (f"case: fused_dispatch_gradient_matches_split_exchanges: states fused="
         f"{states_on.tolist()}, split={states_off.tolist()}")
    assert torch.equal(grad_on, grad_off), \
        (f"case: fused_dispatch_gradient_matches_split_exchanges: gradient fused="
         f"{grad_on.tolist()}, split={grad_off.tolist()}")
    assert bool(grad_on.abs().sum() > 0), \
        (f"case: fused_dispatch_gradient_matches_split_exchanges: gradient fused="
         f"{grad_on.tolist()} is all zero, so this case proves nothing")


def test_fused_dispatch_pack_round_trip():
    """The fused row is byte-exact: pack/unpack restores dtype, shape and every
    value -- including non-square, empty and above-bf16-range payloads."""
    rows = 6

    # ── case: fused_pack_round_trip_non_square ──
    # H=3 is odd and H=5 is not a multiple of the index slot, so both need the
    # head pad; H=7168 is the model's real EP dispatch width.
    for state_dtype, index_dtype in ((torch.bfloat16, torch.int64),
                                     (torch.float32, torch.int64),
                                     (torch.bfloat16, torch.int32)):
        for hidden_size in (3, 5, 7168):
            torch.manual_seed(29)
            states = torch.randn(rows, hidden_size, dtype=state_dtype)
            indices = torch.randint(0, 8, (rows, 1), dtype=index_dtype)
            packed = _pack_fused_dispatch(states, indices)
            back_states, back_indices = _unpack_fused_dispatch(
                packed, hidden_size=hidden_size, index_dtype=index_dtype)

            # Layout derived here from the alignment rules, not from the source:
            # the index region must start at a byte offset that is a multiple of
            # the index element size, and the row must hold whole state elements.
            state_bytes = hidden_size * states.element_size()
            index_bytes = indices.element_size()
            align = math.lcm(states.element_size(), index_bytes)
            head_bytes = (-state_bytes) % align
            assert packed.shape[1] * states.element_size() == state_bytes + head_bytes + align, \
                (f"case: fused_pack_round_trip_non_square: {state_dtype} states / "
                 f"{index_dtype} indices, H={hidden_size}: row="
                 f"{packed.shape[1] * states.element_size()} bytes, expected="
                 f"{state_bytes + head_bytes + align}")
            assert torch.equal(
                packed.view(torch.uint8)[:, state_bytes + head_bytes:
                                         state_bytes + head_bytes + index_bytes],
                indices.view(torch.uint8)), \
                (f"case: fused_pack_round_trip_non_square: {state_dtype} states / "
                 f"{index_dtype} indices, H={hidden_size}: index bytes were converted "
                 f"instead of copied")
            assert back_states.dtype is state_dtype and back_states.shape == states.shape, \
                (f"case: fused_pack_round_trip_non_square: states restored as "
                 f"{back_states.dtype}{tuple(back_states.shape)}, expected "
                 f"{state_dtype}{tuple(states.shape)}")
            assert torch.equal(back_states, states), \
                (f"case: fused_pack_round_trip_non_square: {state_dtype} states round "
                 f"tripped as {back_states.tolist()}, expected {states.tolist()}")
            assert back_indices.dtype is index_dtype and back_indices.shape == (rows,), \
                (f"case: fused_pack_round_trip_non_square: indices restored as "
                 f"{back_indices.dtype}{tuple(back_indices.shape)}, expected "
                 f"{index_dtype}{(rows,)}")
            assert torch.equal(back_indices, indices.reshape(-1)), \
                (f"case: fused_pack_round_trip_non_square: {index_dtype} indices round "
                 f"tripped as {back_indices.tolist()}, expected "
                 f"{indices.reshape(-1).tolist()}")

    # ── case: fused_pack_keeps_indices_above_bf16_range ──
    # 1024 experts: bf16 holds only integers up to 256 exactly, so an
    # implementation that casts the index into the packed dtype silently
    # corrupts the routing above that -- the index must travel as bytes.
    torch.manual_seed(31)
    expert_count = 1024
    indices = torch.tensor([257, 511, 1000, 1023, 0, 512], dtype=torch.int64).view(rows, 1)
    states = torch.randn(rows, 4, dtype=torch.bfloat16)
    back_states, back_indices = _unpack_fused_dispatch(
        _pack_fused_dispatch(states, indices), hidden_size=4, index_dtype=torch.int64)
    assert expert_count > 256, \
        f"case: fused_pack_keeps_indices_above_bf16_range: expert_count={expert_count}"
    assert torch.equal(back_indices, indices.reshape(-1)), \
        (f"case: fused_pack_keeps_indices_above_bf16_range: indices round tripped as "
         f"{back_indices.tolist()}, expected {indices.reshape(-1).tolist()}")
    assert not torch.equal(back_indices, indices.reshape(-1).to(torch.bfloat16).to(torch.int64)), \
        (f"case: fused_pack_keeps_indices_above_bf16_range: the payload survives a bf16 "
         f"cast ({indices.reshape(-1).to(torch.bfloat16).to(torch.int64).tolist()}), so "
         f"this case cannot tell the byte copy from a numeric cast")
    assert torch.equal(back_states, states), \
        (f"case: fused_pack_keeps_indices_above_bf16_range: states round tripped as "
         f"{back_states.tolist()}, expected {states.tolist()}")

    # ── case: fused_pack_round_trip_empty ──
    # A rank that routes nothing and a rank that receives nothing: both sides
    # stay empty instead of driving an offset past the buffer.
    for state_dtype in (torch.bfloat16, torch.float32):
        states = torch.randn(0, 3, dtype=state_dtype)
        indices = torch.zeros(0, 1, dtype=torch.int64)
        packed = _pack_fused_dispatch(states, indices)
        back_states, back_indices = _unpack_fused_dispatch(
            packed, hidden_size=3, index_dtype=torch.int64)
        assert packed.shape[0] == 0, \
            (f"case: fused_pack_round_trip_empty: packed rows={packed.shape[0]}, expected=0")
        assert back_states.shape == (0, 3) and back_states.dtype is state_dtype, \
            (f"case: fused_pack_round_trip_empty: empty states restored as "
             f"{back_states.dtype}{tuple(back_states.shape)}, expected "
             f"{state_dtype}{(0, 3)}")
        assert back_indices.shape == (0,) and back_indices.dtype is torch.int64, \
            (f"case: fused_pack_round_trip_empty: empty indices restored as "
             f"{back_indices.dtype}{tuple(back_indices.shape)}, expected "
             f"{torch.int64}{(0,)}")
        assert torch.equal(back_states, states) and torch.equal(back_indices,
                                                               indices.reshape(-1)), \
            (f"case: fused_pack_round_trip_empty: empty round trip changed the payload: "
             f"states={back_states.tolist()}, indices={back_indices.tolist()}")


def test_fused_dispatch_flag_default_off():
    """The fused dispatch is opt-in: HP_EP_FUSED_DISPATCH is read once at import
    and an unset variable keeps the two separate exchanges in place."""
    with pytest.MonkeyPatch.context() as guard:
        # ── case: fused_dispatch_default_off ──
        guard.delenv("HP_EP_FUSED_DISPATCH", raising=False)
        reloaded = importlib.reload(ep_experts)
        assert reloaded._FUSED_DISPATCH_ENABLED is False, \
            (f"case: fused_dispatch_default_off: unset HP_EP_FUSED_DISPATCH imported as "
             f"{reloaded._FUSED_DISPATCH_ENABLED}, expected False")

        # ── case: fused_dispatch_opt_in ──
        guard.setenv("HP_EP_FUSED_DISPATCH", "1")
        reloaded = importlib.reload(ep_experts)
        assert reloaded._FUSED_DISPATCH_ENABLED is True, \
            (f"case: fused_dispatch_opt_in: HP_EP_FUSED_DISPATCH=1 imported as "
             f"{reloaded._FUSED_DISPATCH_ENABLED}, expected True")
        guard.setenv("HP_EP_FUSED_DISPATCH", "0")
        reloaded = importlib.reload(ep_experts)
        assert reloaded._FUSED_DISPATCH_ENABLED is False, \
            (f"case: fused_dispatch_opt_in: HP_EP_FUSED_DISPATCH=0 imported as "
             f"{reloaded._FUSED_DISPATCH_ENABLED}, expected False")
    importlib.reload(ep_experts)
