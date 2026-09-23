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
"""Unit tests for the Power-Sign Momentum (PSM) optimizer."""

import unittest

import torch
from torch import nn

from hyper_parallel.core.optimizer import fused_psm
from hyper_parallel.core.optimizer.psm import PSM
from hyper_parallel.core.fully_shard.state_dict_utils import _infer_state_keys
from hyper_parallel.components.optim.builders import PSM as PSMBuilder
from tests.common.mark_utils import arg_mark


class TestPSMUpdate(unittest.TestCase):
    """The update rule must match sign(m) * |m|^beta exactly."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_single_step_matches_formula(self):
        """m = g on the first step, so theta -= lr * sign(g) * |g|^beta."""
        param = nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = PSM([param], lr=0.1, gamma=0.9, beta=0.1)
        param.grad = torch.tensor([2.0, -0.5])

        optimizer.step()

        momentum = torch.tensor([2.0, -0.5])
        expected = torch.tensor([1.0, -2.0]) - 0.1 * momentum.sign() * momentum.abs().pow(0.1)
        self.assertTrue(torch.allclose(param.detach(), expected))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_momentum_accumulates_across_steps(self):
        """A second step must use gamma * m_prev + g, not the bare gradient."""
        param = nn.Parameter(torch.zeros(1))
        optimizer = PSM([param], lr=0.5, gamma=0.9, beta=0.1)
        param.grad = torch.ones(1)
        optimizer.step()
        first = param.detach().clone()
        param.grad = torch.ones(1)
        optimizer.step()

        second_momentum = 0.9 * 1.0 + 1.0
        expected_delta = 0.5 * second_momentum ** 0.1
        self.assertAlmostEqual(param.item(), (first - expected_delta).item(), places=6)
        self.assertAlmostEqual(optimizer.state[param]["exp_avg"].item(), second_momentum, places=6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_weight_decay_adds_parameter_term(self):
        """weight_decay contributes wd * theta inside the update, not to the gradient."""
        plain = nn.Parameter(torch.tensor([3.0]))
        decayed = nn.Parameter(torch.tensor([3.0]))
        opt_plain = PSM([plain], lr=0.1, weight_decay=0.0)
        opt_decay = PSM([decayed], lr=0.1, weight_decay=0.2)
        plain.grad = torch.tensor([1.0])
        decayed.grad = torch.tensor([1.0])

        opt_plain.step()
        opt_decay.step()

        momentum = 1.0
        base = 0.1 * momentum ** 0.1
        self.assertAlmostEqual(plain.item(), 3.0 - base, places=6)
        self.assertAlmostEqual(decayed.item(), 3.0 - (base + 0.1 * 0.2 * 3.0), places=6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_state_is_single_momentum_tensor(self):
        """PSM must declare and materialize only exp_avg (half of Adam's state)."""
        param = nn.Parameter(torch.zeros(1))
        optimizer = PSM([param], lr=0.1)
        param.grad = torch.ones(1)
        optimizer.step()

        self.assertEqual(PSM.optim_state_keys, ("exp_avg",))
        self.assertEqual(sorted(optimizer.state[param].keys()), ["exp_avg"])
        self.assertEqual(_infer_state_keys(optimizer), ["exp_avg"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_hyperparameters_rejected(self):
        """Negative lr and out-of-range gamma are configuration errors."""
        param = nn.Parameter(torch.zeros(1))
        with self.assertRaises(ValueError):
            PSM([param], lr=-1.0)
        with self.assertRaises(ValueError):
            PSM([param], lr=0.1, gamma=1.0)


class TestPSMForeachEquivalence(unittest.TestCase):
    """The batched foreach path must match the elementwise path exactly."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_foreach_matches_elementwise_over_several_steps(self):
        """Both update paths must produce identical parameters and momentum."""
        torch.manual_seed(0)
        params_a = [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.randn(5))]
        params_b = [nn.Parameter(p.detach().clone()) for p in params_a]
        opt_a = PSM(params_a, lr=0.05, gamma=0.9, beta=0.1, weight_decay=0.02)
        opt_b = PSM(params_b, lr=0.05, gamma=0.9, beta=0.1, weight_decay=0.02)
        opt_a._foreach_supported = True    # pylint: disable=protected-access
        opt_b._foreach_supported = False   # pylint: disable=protected-access

        for _ in range(3):
            for pa, pb in zip(params_a, params_b):
                grad = torch.randn_like(pa)
                pa.grad = grad.clone()
                pb.grad = grad.clone()
            opt_a.step()
            opt_b.step()

        for pa, pb in zip(params_a, params_b):
            self.assertTrue(torch.equal(pa.detach(), pb.detach()))
            self.assertTrue(torch.equal(opt_a.state[pa]["exp_avg"], opt_b.state[pb]["exp_avg"]))


class TestPSMFusedKernel(unittest.TestCase):
    """The fused native path must match the fallback path, and degrade safely."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_matches_fallback_over_several_steps(self):
        """Momentum must stay bit-identical and parameters agree to one ulp."""
        if not fused_psm.is_available():
            self.skipTest(f"fused kernel unavailable: {fused_psm.unavailable_reason()}")
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                params_a = [nn.Parameter(torch.randn(3, 4).to(dtype)), nn.Parameter(torch.randn(5).to(dtype))]
                params_b = [nn.Parameter(p.detach().clone()) for p in params_a]
                opt_a = PSM(params_a, lr=0.05, gamma=0.9, beta=0.1, weight_decay=0.02)
                opt_b = PSM(params_b, lr=0.05, gamma=0.9, beta=0.1, weight_decay=0.02)
                opt_a._fused_supported = True     # pylint: disable=protected-access
                opt_b._fused_supported = False    # pylint: disable=protected-access

                for _ in range(3):
                    for pa, pb in zip(params_a, params_b):
                        grad = torch.randn_like(pa)
                        pa.grad = grad.clone()
                        pb.grad = grad.clone()
                    opt_a.step()
                    opt_b.step()

                for pa, pb in zip(params_a, params_b):
                    # The momentum is the persistent state, so it must match exactly; the
                    # parameter update may differ in the last ulp because the native kernel
                    # calls the host libm power while the fallback uses torch's kernels.
                    self.assertTrue(torch.equal(opt_a.state[pa]["exp_avg"], opt_b.state[pb]["exp_avg"]))
                    self.assertTrue(
                        torch.allclose(pa.detach().float(), pb.detach().float(), rtol=tolerance,
                                       atol=tolerance)
                    )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unsupported_layout_is_rejected_untouched(self):
        """Unsupported dtypes or layouts must be reported without modifying state."""
        param = nn.Parameter(torch.randn(4, dtype=torch.float64))
        param.grad = torch.ones_like(param)
        momentum = torch.zeros_like(param)
        before = param.detach().clone()

        accepted = fused_psm.update([param], [param.grad], [momentum], 0.1, 0.9, 0.1, 0.01)

        self.assertFalse(accepted)
        self.assertTrue(torch.equal(param.detach(), before), "parameter must be untouched")
        self.assertTrue(torch.equal(momentum, torch.zeros_like(momentum)), "momentum must be untouched")

        view = nn.Parameter(torch.randn(6, 4).t())
        view.grad = torch.ones_like(view)
        self.assertFalse(fused_psm.update([view], [view.grad], [torch.zeros_like(view)], 0.1, 0.9, 0.1, 0.01))

        if fused_psm.is_available():
            half = nn.Parameter(torch.randn(8, dtype=torch.bfloat16))
            half.grad = torch.ones_like(half)
            self.assertTrue(
                fused_psm.update([half], [half.grad], [torch.zeros_like(half)], 0.1, 0.9, 0.1, 0.01),
                "bfloat16 host-offloaded tensors are the primary target of the kernel",
            )


class TestPSMBuilder(unittest.TestCase):
    """The YAML-targeted builder must reuse the AdamW parameter routing."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_build_splits_decay_groups_and_sets_hyperparameters(self):
        """Bias/norm parameters must land in the no-decay group with weight_decay 0."""
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        builder = PSMBuilder(
            psm_config=dict(psm_lr=1.0e-5, psm_gamma=0.9, psm_beta=0.1, psm_weight_decay=0.01),
            model=model,
            no_decay_params=["bias", "norm", "ln_"],
        )

        chained = builder.get_optimizer()
        self.assertEqual(list(chained.optimizers_dict.keys()), ["psm"])
        optimizer = chained.optimizers_dict["psm"]
        self.assertIsInstance(optimizer, PSM)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0e-5)
        self.assertEqual(optimizer.param_groups[0]["beta"], 0.1)
        decay_flags = sorted(group["weight_decay"] for group in optimizer.param_groups)
        self.assertEqual(decay_flags, [0.0, 0.01])
        # Every trainable parameter must be routed exactly once.
        routed = sum(len(group["params"]) for group in optimizer.param_groups)
        self.assertEqual(routed, sum(1 for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
