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
"""Unit tests for the MoE expert capacity limit (``HP_EP_CAPACITY_FACTOR``).

Real routing is skewed (the busiest rank measured at 6x the mean), and a rank's
expert buffers are sized by the rows it *receives*, so the skewed case -- not the
balanced one -- decides whether the MoE step fits in memory.  Capping each expert
at ``capacity_factor x (T*K / E)`` slots bounds it directly.

These tests pin the contract the memory bound rests on:

* ``apply_capacity_limit`` is a pure function of ``(topk_idx, factor, num_experts)``:
  shape- and dtype-preserving, reproducible, and cheap enough that no host sync
  happens while building the mask;
* every expert keeps exactly ``min(count_e, ceil(factor * T*K / E))`` slots, so a
  skewed expert really is the one that gets capped;
* ``None`` / non-positive / unparseable configuration is *off* and drops nothing;
* the mask compacts the routed slots before the exchange, so a dropped slot never
  becomes a row of the dispatch.
"""
import contextlib
import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.distributed.expert_parallel.experts import (
    _aggregate_ep_outputs,
    _prepare_ep_dispatch,
    _resolve_capacity_factor,
    _routed_slot_token_ids,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    _expert_major_order,
    apply_capacity_limit,
)

_ENV_KNOB = "HP_EP_CAPACITY_FACTOR"


@contextlib.contextmanager
def _capacity_env(value):
    """Set (or clear, with ``None``) the capacity knob and restore it afterwards."""
    previous = os.environ.get(_ENV_KNOB)
    if value is None:
        os.environ.pop(_ENV_KNOB, None)
    else:
        os.environ[_ENV_KNOB] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ENV_KNOB, None)
        else:
            os.environ[_ENV_KNOB] = previous


def _counts_per_expert(keep, topk_idx, num_experts):
    """Return the kept-slot count of every expert, in expert order."""
    flat_keep = keep.reshape(-1)
    flat_expert = topk_idx.reshape(-1)
    return [int((flat_keep & (flat_expert == expert)).sum()) for expert in range(num_experts)]


def _capacity_of(factor, slots, num_experts):
    """The cap ``apply_capacity_limit`` documents: ``ceil(factor * slots / E)``."""
    return int(math.ceil(float(factor) * slots / float(num_experts)))


def _assignment(counts):
    """Build a ``[T, 1]`` assignment whose per-expert hit counts match ``counts``."""
    experts = []
    for expert, count in enumerate(counts):
        experts.extend([expert] * count)
    return torch.tensor(experts, dtype=torch.int64).unsqueeze(1)


# Skewed plan: expert 0 draws 4 of the 8 slots, expert 3 draws none.
_SKEWED = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 2]], dtype=torch.int64)
_SKEWED_SLOTS = _SKEWED.numel()
_SKEWED_EXPERTS = 4
_SKEWED_COUNTS = [4, 3, 1, 0]


class TestApplyCapacityLimit(unittest.TestCase):
    """The mask itself: shape, counts, reproducibility, off-switch."""

    def test_disabled_factors_keep_every_slot(self):
        """Feature: capacity limit off.

        Description: Call the mask builder with ``None`` and with non-positive factors.
        Expectation: Every slot survives, the shape and bool dtype are preserved, and
            the reported drop count is zero -- no host readback has to happen to say so.
        """
        for factor in (None, 0.0, -1.0):
            with self.subTest(factor=factor):
                keep, dropped = apply_capacity_limit(_SKEWED, factor, _SKEWED_EXPERTS)
                self.assertEqual(tuple(keep.shape), tuple(_SKEWED.shape))
                self.assertEqual(keep.dtype, torch.bool)
                self.assertTrue(bool(keep.all()))
                self.assertEqual(float(dropped), 0.0)

    def test_every_expert_keeps_at_most_the_capacity(self):
        """Feature: capacity limit on.

        Description: Mask the skewed plan at several factors.
        Expectation: No expert keeps more than ``ceil(factor * T*K / E)`` slots.
        """
        for factor in (0.25, 0.5, 1.0, 1.5, 2.0):
            with self.subTest(factor=factor):
                keep, _ = apply_capacity_limit(_SKEWED, factor, _SKEWED_EXPERTS)
                capacity = _capacity_of(factor, _SKEWED_SLOTS, _SKEWED_EXPERTS)
                counts = _counts_per_expert(keep, _SKEWED, _SKEWED_EXPERTS)
                for expert, kept in enumerate(counts):
                    self.assertLessEqual(
                        kept, capacity,
                        f"expert {expert} kept {kept} slots, capacity is {capacity}",
                    )

    def test_kept_count_is_the_capped_expert_load(self):
        """Feature: capacity limit on.

        Description: Mask the skewed plan and compare per-expert kept counts against
            ``min(count_e, capacity)``.
        Expectation: Every expert keeps exactly the smaller of its load and the
            capacity, so the busiest expert is the one bounded and an unloaded expert
            keeps nothing.
        """
        factor = 1.0
        capacity = _capacity_of(factor, _SKEWED_SLOTS, _SKEWED_EXPERTS)
        keep, _ = apply_capacity_limit(_SKEWED, factor, _SKEWED_EXPERTS)
        expected = [min(count, capacity) for count in _SKEWED_COUNTS]
        self.assertEqual(_counts_per_expert(keep, _SKEWED, _SKEWED_EXPERTS), expected)
        # 8 slots, 5 of them inside the cap of 2 per expert.
        self.assertEqual(sum(expected), 5)

    def test_dropped_count_is_the_mask_complement(self):
        """Feature: capacity limit on.

        Description: Compare the reported drop count with the number of ``False``
            entries in the mask, across factors.
        Expectation: The two agree, so callers logging the drop rate are logging the
            mask they actually applied.
        """
        for factor in (0.25, 1.0, 1.5):
            with self.subTest(factor=factor):
                keep, dropped = apply_capacity_limit(_SKEWED, factor, _SKEWED_EXPERTS)
                self.assertEqual(int(dropped), int((~keep).sum()))

    def test_a_generous_factor_drops_nothing(self):
        """Feature: capacity limit on but slack.

        Description: Use a factor far above the skew of the plan.
        Expectation: Nothing is dropped, so turning the knob on cannot silently
            change a run whose routing already fits.
        """
        keep, dropped = apply_capacity_limit(_SKEWED, 8.0, _SKEWED_EXPERTS)
        self.assertTrue(bool(keep.all()))
        self.assertEqual(int(dropped), 0)

    def test_mask_is_reproducible_for_the_same_input(self):
        """Feature: capacity limit determinism.

        Description: Build the mask twice from the same plan and factor.
        Expectation: Identical masks, so a resumed or replayed step drops the same
            slots -- the intra-expert tie order is the only degree of freedom and it is
            deterministic for a given input and backend.
        """
        first, first_dropped = apply_capacity_limit(_SKEWED, 0.75, _SKEWED_EXPERTS)
        second, second_dropped = apply_capacity_limit(_SKEWED, 0.75, _SKEWED_EXPERTS)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(int(first_dropped), int(second_dropped))

    def test_a_larger_factor_drops_less(self):
        """Feature: capacity limit monotonicity.

        Description: Raise the factor from below the mean load to well above the skew.
        Expectation: The dropped count never grows with the factor, and a factor above
            the skew drops nothing -- so the knob moves in one direction only.
        """
        plan = _assignment([8, 0, 0, 0])
        dropped = [
            int(apply_capacity_limit(plan, factor, 4)[1])
            for factor in (0.5, 1.0, 2.0, 4.0, 8.0)
        ]
        self.assertEqual(dropped, sorted(dropped, reverse=True))
        self.assertEqual(dropped[-1], 0)

    def test_a_token_keeps_a_slot_while_an_expert_has_room(self):
        """Feature: capacity limit per-expert policy.

        Description: Give every token the same two-expert assignment.
        Expectation: The cap is applied per expert, so a token routed to two experts
            still contributes through whichever slot survived.
        """
        plan = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        keep, _ = apply_capacity_limit(plan, 1.0, 2)
        self.assertTrue(bool((keep.sum(dim=1) >= 1).all()))

    def test_mask_shape_follows_the_routed_shape(self):
        """Feature: capacity limit shape contract.

        Description: Mask plans with a different ``[T, K]`` shape.
        Expectation: The mask keeps the caller's shape, which is what lets the caller
            index the routed slots flat.
        """
        for shape in ((4, 2), (8, 1), (3, 4)):
            with self.subTest(shape=shape):
                total = shape[0] * shape[1]
                plan = torch.arange(total, dtype=torch.int64).reshape(shape) % 4
                keep, _ = apply_capacity_limit(plan, 1.0, 4)
                self.assertEqual(tuple(keep.shape), shape)

    def test_non_positive_expert_count_raises(self):
        """Feature: capacity limit input validation.

        Description: Ask for a cap over zero experts.
        Expectation: A ValueError names the argument instead of dividing by zero.
        """
        for num_experts in (0, -3):
            with self.subTest(num_experts=num_experts):
                with self.assertRaisesRegex(ValueError, "num_experts"):
                    apply_capacity_limit(_SKEWED, 1.0, num_experts)

    def test_expert_major_order_is_a_permutation(self):
        """Feature: capacity limit helper.

        Description: Build the expert-major permutation of a skewed plan.
        Expectation: It is a permutation of ``range(T*K)`` that never interleaves two
            experts, which is what makes "kept per expert" a contiguous block count.
        """
        flat = _SKEWED.reshape(-1)
        order = _expert_major_order(flat, _SKEWED_EXPERTS)
        self.assertEqual(sorted(int(x) for x in order.tolist()), list(range(_SKEWED_SLOTS)))
        ordered_experts = flat[order].tolist()
        self.assertEqual(ordered_experts, sorted(ordered_experts))


class TestResolveCapacityFactor(unittest.TestCase):
    """Configuration resolution: env override, module attribute, off by default."""

    def test_env_override_wins_over_the_module_attribute(self):
        """Feature: capacity factor resolution.

        Description: Set both the knob and the module attribute to different values.
        Expectation: The env override wins, which is what makes a sweep possible
            without editing YAML.
        """
        with _capacity_env("2.5"):
            self.assertEqual(
                _resolve_capacity_factor(self._method(0.5)), 2.5)

    def test_module_attribute_is_used_when_the_env_is_absent(self):
        """Feature: capacity factor resolution.

        Description: Leave the knob unset and configure the MoE block instead.
        Expectation: The block's own ``capacity_factor`` is used.
        """
        with _capacity_env(None):
            self.assertEqual(
                _resolve_capacity_factor(self._method(1.25)), 1.25)

    def test_unconfigured_is_off(self):
        """Feature: capacity factor resolution.

        Description: Resolve with neither the knob nor the attribute present.
        Expectation: ``None``, i.e. the previous behaviour exactly.
        """
        with _capacity_env(None):
            self.assertIsNone(_resolve_capacity_factor(self._method(None)))
            self.assertIsNone(_resolve_capacity_factor(SimpleNamespace()))

    def test_non_positive_values_are_off(self):
        """Feature: capacity factor resolution.

        Description: Configure zero or a negative factor, by knob and by attribute.
        Expectation: Both mean "off" rather than "drop every token".
        """
        for value in ("0", "-2"):
            with self.subTest(env=value), _capacity_env(value):
                self.assertIsNone(_resolve_capacity_factor(self._method(1.5)))
        with _capacity_env(None):
            self.assertIsNone(_resolve_capacity_factor(self._method(0.0)))
            self.assertIsNone(_resolve_capacity_factor(self._method(-1.0)))

    def test_invalid_env_value_is_off(self):
        """Feature: capacity factor resolution.

        Description: Set the knob to a non-number.
        Expectation: The limit stays off instead of raising from inside the forward.
        """
        with _capacity_env("not-a-number"):
            with mock.patch.object(ep_experts.logger, "warning") as warn:
                self.assertIsNone(_resolve_capacity_factor(self._method(1.5)))
            warn.assert_called_once()

    def test_blank_env_is_treated_as_unset(self):
        """Feature: capacity factor resolution.

        Description: Set the knob to whitespace only.
        Expectation: It falls through to the module attribute rather than parsing.
        """
        with _capacity_env("   "):
            self.assertEqual(
                _resolve_capacity_factor(self._method(0.75)), 0.75)
            self.assertIsNone(_resolve_capacity_factor(self._method(None)))

    @staticmethod
    def _method(value):
        """A MoE-block stand-in exposing ``experts.capacity_factor`` (or not)."""
        if value is None:
            return SimpleNamespace(experts=SimpleNamespace())
        return SimpleNamespace(experts=SimpleNamespace(capacity_factor=value))


class TestDispatchCapacityCompaction(unittest.TestCase):
    """The mask must compact the routed slots before anything is exchanged."""

    _HIDDEN = 4
    _LOCAL_EXPERTS = 2
    _EP_SIZE = 2
    _GLOBAL_EXPERTS = 4

    def setUp(self) -> None:
        """Isolate the dispatch from the static-plan cache and the collective."""
        # The plan cache keys on the routing shape, not on the capacity mask, so a cached
        # plan would hide exactly the compaction these tests are about.
        self._patchers = [
            mock.patch.object(ep_experts, "get_static_plan", lambda key: None),
            mock.patch.object(ep_experts, "store_static_plan", lambda *args, **kwargs: None),
            mock.patch.object(ep_experts.dist, "all_to_all_single", self._identity_exchange),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        # One fixed hidden batch: ``states`` can then be compared between calls.
        torch.manual_seed(7)
        self.hidden = torch.randn(4, self._HIDDEN)

    @staticmethod
    def _identity_exchange(output_tensor, input_tensor, group=None):
        """Stand in for the counts exchange: mirror the send counts back."""
        del group
        output_tensor.copy_(input_tensor)

    def _dispatch(self, topk_indices, topk_weights, keep=None):
        """Run ``_prepare_ep_dispatch`` on a small CPU plan and keep the input too."""
        dispatch = _prepare_ep_dispatch(
            self.hidden,
            topk_indices,
            topk_weights,
            local_expert_count=self._LOCAL_EXPERTS,
            global_expert_count=self._GLOBAL_EXPERTS,
            ep_size=self._EP_SIZE,
            ep_group="ep-group",
            keep=keep,
        )
        return self.hidden, dispatch

    def _plan(self):
        """A skewed ``[T, K]`` top-k plan with float weights."""
        torch.manual_seed(0)
        return _SKEWED.clone(), torch.rand(4, 2)

    def test_dropped_slots_never_become_rows(self):
        """Feature: capacity-limited dispatch.

        Description: Dispatch a skewed plan through the mask builder and then through
            ``_prepare_ep_dispatch``.
        Expectation: The dispatch carries exactly the kept slots -- their expert ids,
            weights and source tokens -- so a dropped slot is never exchanged.
        """
        topk_indices, topk_weights = self._plan()
        keep, _ = apply_capacity_limit(topk_indices, 1.0, self._GLOBAL_EXPERTS)
        flat_keep = keep.reshape(-1)
        hidden, dispatch = self._dispatch(topk_indices, topk_weights, keep=keep)

        kept = int(flat_keep.sum())
        order = dispatch.dispatch_order
        expected_sources = _routed_slot_token_ids(
            4, topk_indices.shape[1], topk_indices.device)[flat_keep]
        expected_experts = topk_indices.reshape(-1)[flat_keep]
        expected_weights = topk_weights.reshape(-1)[flat_keep].to(hidden.dtype)

        self.assertEqual(dispatch.states.shape[0], kept)
        self.assertTrue(torch.equal(dispatch.source_indices, expected_sources))
        self.assertTrue(torch.equal(dispatch.expert_weights, expected_weights))
        # ``expert_indices`` is the destination-sorted row, and ``states`` follows that sort.
        self.assertTrue(torch.equal(
            dispatch.expert_indices, expected_experts[order].unsqueeze(-1)))
        self.assertTrue(torch.equal(
            dispatch.states, hidden[expected_sources[order]]))
        self.assertEqual(sum(dispatch.send_counts), kept)

    def test_no_mask_dispatches_every_slot(self):
        """Feature: capacity-limited dispatch off.

        Description: Dispatch the same plan without a mask.
        Expectation: Every routed slot becomes a row, so the knob is the only thing
            that changes the row count.
        """
        topk_indices, topk_weights = self._plan()
        hidden, dispatch = self._dispatch(topk_indices, topk_weights, keep=None)
        total = topk_indices.numel()
        order = dispatch.dispatch_order
        expected_sources = _routed_slot_token_ids(
            4, topk_indices.shape[1], topk_indices.device)
        self.assertEqual(dispatch.states.shape[0], total)
        self.assertTrue(torch.equal(dispatch.source_indices, expected_sources))
        self.assertTrue(torch.equal(
            dispatch.expert_indices,
            topk_indices.reshape(-1)[order].unsqueeze(-1)))
        self.assertTrue(torch.equal(
            dispatch.states, hidden[expected_sources[order]]))
        self.assertEqual(sum(dispatch.send_counts), total)

    def test_an_all_true_mask_is_bit_identical_to_no_mask(self):
        """Feature: capacity limit off by default.

        Description: Dispatch once with ``keep=None`` and once with an all-True mask.
        Expectation: Every field is bit-identical, so "off" is a true no-op rather than
            an extra identity-selection pass.
        """
        topk_indices, topk_weights = self._plan()
        _, dense = self._dispatch(topk_indices, topk_weights, keep=None)
        _, all_keep = self._dispatch(
            topk_indices,
            topk_weights,
            keep=torch.ones_like(topk_indices, dtype=torch.bool),
        )
        for field in ("source_indices", "expert_weights", "expert_indices",
                      "dispatch_order", "states"):
            self.assertTrue(
                torch.equal(getattr(dense, field), getattr(all_keep, field)),
                f"{field} differs between the dense and all-True dispatches",
            )

    def test_aggregation_matches_the_dense_drop_reference(self):
        """Feature: capacity-limited aggregation.

        Description: Aggregate synthetic expert outputs over the compacted dispatch and
            compare against the dense reference with every dropped slot's weight zeroed.
        Expectation: The two agree, so dropping removes a slot's contribution and
            nothing else -- the surviving arithmetic is unchanged.
        """
        topk_indices, topk_weights = self._plan()
        keep, _ = apply_capacity_limit(topk_indices, 1.0, self._GLOBAL_EXPERTS)
        _, dispatch = self._dispatch(topk_indices, topk_weights, keep=keep)

        rows = dispatch.source_indices.numel()
        expert_out = torch.arange(rows, dtype=torch.float32).unsqueeze(1) * torch.ones(
            1, self._HIDDEN)
        aggregated = _aggregate_ep_outputs(
            expert_out,
            dispatch.expert_weights,
            dispatch.source_indices,
            dispatch.dispatch_order,
            (1, 4, self._HIDDEN),
        )

        flat_keep = keep.reshape(-1)
        kept_positions = flat_keep.nonzero().reshape(-1)
        order = dispatch.dispatch_order
        # ``expert_out`` arrives in exchange order, so it lands on the slots ``order`` names.
        dense_rows = torch.zeros(topk_indices.numel(), self._HIDDEN)
        dense_rows[kept_positions[order]] = expert_out
        dense_weights = torch.zeros(topk_indices.numel(), dtype=self.hidden.dtype)
        dense_weights[kept_positions[order]] = dispatch.expert_weights[order]
        dense_source = _routed_slot_token_ids(
            4, topk_indices.shape[1], topk_indices.device)
        reference = torch.zeros(4, self._HIDDEN)
        reference.index_add_(0, dense_source, dense_rows * dense_weights.unsqueeze(-1))
        self.assertTrue(torch.allclose(aggregated.view(4, self._HIDDEN), reference))

    def test_dropping_does_not_change_the_surviving_rows(self):
        """Feature: capacity-limited dispatch equivalence.

        Description: Compare the masked dispatch against the unmasked one, restricted
            to the kept slots.
        Expectation: The filtered arrays are exactly the unmasked ones indexed by the
            mask, and the exchange order is the unmasked order with the dropped rows
            removed -- the cap drops slots, it never reorders or rescales them.
        """
        topk_indices, topk_weights = self._plan()
        keep, _ = apply_capacity_limit(topk_indices, 1.0, self._GLOBAL_EXPERTS)
        flat_keep = keep.reshape(-1)
        _, masked = self._dispatch(topk_indices, topk_weights, keep=keep)
        _, full = self._dispatch(topk_indices, topk_weights, keep=None)

        self.assertEqual(sum(masked.send_counts), int(flat_keep.sum()))
        self.assertTrue(torch.equal(
            masked.source_indices, full.source_indices[flat_keep]))
        self.assertTrue(torch.equal(
            masked.expert_weights, full.expert_weights[flat_keep]))

        def _ordered_pairs(dispatch):
            """``(expert, source token)`` per dispatch row, in exchange order."""
            order = dispatch.dispatch_order.tolist()
            experts = dispatch.expert_indices.reshape(-1).tolist()
            sources = dispatch.source_indices.tolist()
            return [(expert, sources[slot]) for expert, slot in zip(experts, order)]

        kept_full_rows = [
            row for row, slot in zip(_ordered_pairs(full), full.dispatch_order.tolist())
            if bool(flat_keep[slot])
        ]
        self.assertEqual(_ordered_pairs(masked), kept_full_rows)


if __name__ == "__main__":
    unittest.main()
