# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""expert_parallel.routing: MoE router adapters (05 §6.4.7, D-09).

Routing semantics are model-specific (softmax/sigmoid, top-k normalization,
scaling), while the expert MLP structure is uniform (SwiGLU). A factory picks
an adapter from ``MOE_ROUTER_ADAPTERS`` BY NAME in its own code — the choice
is explicit, never inferred. Each adapter maps
``(module, hidden) -> (topk_idx [T,K] int64, topk_w [T,K] float)``.

Split out of components/distributed/ep_utils.py in stage 4e.
"""

import torch


def _softmax_topk_router(module, hidden_states):
    """default adapter: softmax -> topk -> normalize by sum (Mixtral/Qwen3 semantics).

    top_k source: config.num_experts_per_tok / config.top_k / module.top_k (default 2);
    normalization switch: config.norm_topk_prob (default True).
    """
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    if gate is None:
        raise AttributeError(
            f"{type(module).__name__}: router not found (neither gate nor router "
            "attribute exists); please register a custom MOE_ROUTER_ADAPTERS entry"
        )
    cfg = getattr(module, "config", None)
    logits = gate(hidden_states)
    logits = logits.view(-1, logits.shape[-1])
    top_k = (getattr(cfg, "num_experts_per_tok", None)
             or getattr(cfg, "top_k", None)
             or getattr(module, "top_k", 2))
    weights = logits.softmax(-1)
    topk_w, topk_idx = weights.topk(int(top_k), dim=-1)
    if getattr(cfg, "norm_topk_prob", True):
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-20)
    return topk_idx, topk_w


def _topk_router_module(module, hidden_states):
    """Qwen2/Qwen3/Mixtral adapter: gate is a TopKRouter module (after the HF 2025
    refactor); forward directly returns (logits, scores [T,K], indices [T,K])
    -- take the latter two."""
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    out = gate(hidden_states)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        _, scores, indices = out
        return indices, scores
    raise TypeError(
        f"{type(module).__name__}: TopKRouter should return (logits, scores, indices), "
        f"got {type(out).__name__} -- use the default adapter or register a custom adapter"
    )


def _mask_scores_by_group(scores, n_group, topk_group):
    """Keep scores only in the highest-scoring expert groups."""
    expert_count = scores.shape[-1]
    group_scores = (
        scores.view(-1, n_group, expert_count // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_indices = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_indices, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, expert_count // n_group)
        .reshape(-1, expert_count)
    )
    return scores.masked_fill(~score_mask.bool(), float("-inf"))


def _sigmoid_group_router(module, hidden_states):
    """deepseekv3/glm4moe adapter: sigmoid + e_score_correction_bias +
    group-limited topk + (optional) normalization + routed_scaling_factor
    (step-by-step consistent with HF DeepseekV3MoE.route_tokens_to_experts /
    Glm4MoeMoE).

    Parameter source: the module's own attributes take precedence
    (n_group/topk_group/top_k/norm_topk_prob/routed_scaling_factor), falling
    back to module.config; when n_group is missing or <=1, the group-limited
    filter is skipped.
    """
    gate = getattr(module, "gate", None)
    if gate is None:
        gate = getattr(module, "router", None)
    cfg = getattr(module, "config", None)

    def _attr(name, default=None):
        v = getattr(module, name, None)
        if v is None and cfg is not None:
            v = getattr(cfg, name, None)
        return default if v is None else v

    logits = gate(hidden_states)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.view(-1, logits.shape[-1]).float()
    scores = logits.sigmoid()
    bias = getattr(gate, "e_score_correction_bias", None)
    scores_for_choice = scores + bias if bias is not None else scores

    n_group = int(_attr("n_group", 0) or 0)
    topk_group = int(_attr("topk_group", 0) or 0)
    top_k = int(_attr("top_k", None) or _attr("num_experts_per_tok", 2))
    if n_group > 1 and topk_group > 0:
        scores_for_choice = _mask_scores_by_group(
            scores_for_choice, n_group, topk_group
        )

    topk_idx = scores_for_choice.topk(top_k, dim=-1, sorted=False)[1]
    topk_w = scores.gather(1, topk_idx)
    if _attr("norm_topk_prob", False):
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(_attr("routed_scaling_factor", 1.0))
    return topk_idx, topk_w


def _global_expert_count(module):
    """Model-level routed expert count (mirrors ``experts._get_global_expert_count``).

    Kept local so this module stays a torch-only leaf (it is imported by
    model adapters that do not want the expert compute machinery).
    """
    for owner in (getattr(module, "experts", None), module):
        count = getattr(owner, "num_experts", None)
        if count is not None:
            return int(count)
    cfg = getattr(module, "config", None)
    for name in ("num_experts", "n_routed_experts"):
        count = getattr(cfg, name, None)
        if count is not None:
            return int(count)
    raise ValueError(
        f"{type(module).__name__}: cannot determine the global routed expert count"
    )


def _balanced_router(module, hidden_states):
    """deepseekv3 adapter variant with a deliberately uniform expert load.

    Real sigmoid-group routing, but the chosen experts are replaced by a
    round-robin assignment that spreads the local slots over the **destination
    ranks**: local slot ``i`` goes to one of the ``L`` experts owned by
    destination ``i mod Q``, where ``Q = E/L`` and ``L`` is the per-rank expert
    count (``module.experts.local_expert_count``). Every rank therefore
    receives exactly ``T*K/EP`` tokens instead of a data-dependent count.

    Spreading by *rank* rather than by expert is what makes this exact: the
    per-expert token count cannot be made uniform at all when ``E`` does not
    divide ``T*K``, and a per-expert round robin (``i mod E``) replays the same
    residue pattern on every rank, so the leftover slots pile onto the same
    fixed group of ranks (measured: 0.8% above the mean on the busiest rank,
    2.4% spread between rank groups). Going through the destination first
    absorbs that leftover per rank instead.

    Why: unbalanced routing is the dominant source of step-time jitter on MoE
    training runs — the slowest rank in a step is the one whose experts drew
    the most tokens, so identical configurations can differ by tens of percent
    step to step. Measured on the 293B/18-layer config, the real router put
    **6.3x** the mean token count on the busiest rank (leaving some experts
    empty) while the balanced one stayed at 1.00x. Flattening the load makes a
    step-to-step comparison converge in 2-3 steps instead of needing a dozen,
    which is what makes small (1-3%) kernel-level effects measurable at all.

    **Benchmark-only.** The assignment ignores the gate scores, so:
    - ``loss`` / ``grad_norm`` / any convergence signal is meaningless here;
    - expert hit distributions and "effective MFU" are optimistic (the load
      is artificially flat), so numbers from this mode must not be reported
      as an efficiency result.

    Correctness must be validated with the switch OFF (``fix_router:
    False``), where the real ``_sigmoid_group_router`` runs unchanged. The
    true ``topk_w`` are kept so the gate still receives gradient and the
    router GEMM stays in the step's compute profile.
    """
    topk_idx, topk_w = _sigmoid_group_router(module, hidden_states)
    token_count, experts_per_token = topk_idx.shape
    expert_count = _global_expert_count(module)
    # Set by the EP binder (bind_local_expert_forward) before any forward; a
    # missing value degrades to a plain per-expert round robin.
    local_count = getattr(
        getattr(module, "experts", None), "local_expert_count", None) or 1
    if expert_count % local_count != 0:
        raise ValueError(
            f"num_experts ({expert_count}) must be divisible by the local "
            f"expert count ({local_count})"
        )
    # Spread the slots over DESTINATION RANKS first (slot % destinations), then
    # over that rank's local experts: every rank receives exactly one out of
    # every `destinations` slots.
    destinations = expert_count // local_count
    slots = torch.arange(
        token_count * experts_per_token, device=topk_idx.device)
    balanced_idx = ((slots % destinations) * local_count
                    + (slots // destinations) % local_count)
    return balanced_idx.view(
        token_count, experts_per_token).to(topk_idx.dtype), topk_w


MOE_ROUTER_ADAPTERS = {
    "default": _softmax_topk_router,
    "qwen2moe": _topk_router_module,
    "qwen2_moe": _topk_router_module,
    "qwen3moe": _topk_router_module,
    "qwen3_moe": _topk_router_module,
    "mixtral": _topk_router_module,
    "deepseekv3": _sigmoid_group_router,
    "deepseek_v3": _sigmoid_group_router,
    "deepseekv3_fixed": _balanced_router,
    "deepseek_v3_fixed": _balanced_router,
    "glm4moe": _sigmoid_group_router,
    "glm4_moe": _sigmoid_group_router,
}
