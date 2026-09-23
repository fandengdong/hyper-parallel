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
"""Power-Sign Momentum (PSM) optimizer, ported from MindSpeed-LLM.

The update rule is::

    m_t     = gamma * m_{t-1} + g_t
    theta_t = theta_{t-1} - lr * (sign(m_t) * |m_t|^beta + weight_decay * theta_{t-1})

Unlike Adam it keeps a **single** state tensor per parameter (the momentum ``exp_avg``)
instead of the ``(exp_avg, exp_avg_sq)`` pair, so its optimizer state is half the size --
the reason it is interesting for a 1T-parameter run whose optimizer state dominates host
memory.  It is also *not* scale invariant like Adam: the step magnitude is ``|m|^beta``,
so it depends on the raw gradient scale (an lr matched to Adam is a first cut, not
necessarily PSM's best lr).
"""

import logging
from typing import Any, Iterable, Optional

import torch
from torch.optim import Optimizer

from .dtensor_compat import to_local_if_dtensor
from . import fused_psm

logger = logging.getLogger(__name__)


class PSM(Optimizer):
    """Power-Sign Momentum optimizer.

    Args:
        params: Iterable of parameters or parameter groups to optimize.
        lr: Learning rate.
        gamma: Momentum factor.
        beta: Power-law exponent applied to the momentum magnitude.
        weight_decay: Decoupled weight-decay coefficient (applied inside the update).
    """

    #: Optimizer-state tensors PSM materializes, consumed by the distributed
    #: checkpoint estimator so it does not assume an Adam-shaped state pair.
    optim_state_keys = ("exp_avg",)

    def __init__(
            self,
            params: Iterable[Any],
            lr: float,
            gamma: float = 0.9,
            beta: float = 0.1,
            weight_decay: float = 0.0,
    ) -> None:
        """Validate hyperparameters and register the parameter groups.

        Args:
            params: Iterable of parameters or parameter groups to optimize.
            lr: Learning rate; must be non-negative.
            gamma: Momentum factor; must lie in ``[0, 1)``.
            beta: Power-law exponent applied to the momentum magnitude.
            weight_decay: Decoupled weight-decay coefficient.

        Raises:
            ValueError: If ``lr`` is negative or ``gamma`` is outside ``[0, 1)``.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (momentum): {gamma}")
        defaults = dict(lr=lr, gamma=gamma, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # The fallback path is probed once, on a scratch tensor rather than by catching a
        # failure mid-update, so a backend gap cannot leave momentum half-applied.  Note
        # that the batched ``torch._foreach_*`` form is only marginally faster than the
        # elementwise one -- both still walk the whole tensor once per op, which on
        # host-offloaded parameters is the real cost.  The fused kernel probed below is
        # what actually removes that traffic.
        self._foreach_supported: Optional[bool] = None
        # Probed once: the fused native kernel (see ``csrc/fused_psm.cpp``) replaces the
        # whole op chain with a single read/write pass per tensor, which is the form
        # ``torch._fused_adamw_`` uses to stay fast on host-offloaded parameters. It is
        # preferred whenever the toolchain and the float32 CPU layout allow it.
        self._fused_supported: Optional[bool] = None

    def _supports_fused(self) -> bool:
        """Return whether the fused native update is available on this host."""
        if self._fused_supported is None:
            self._fused_supported = fused_psm.is_available()
            if not self._fused_supported:
                logger.info(
                    "PSM: fused native update unavailable (%s); falling back to the batched "
                    "torch._foreach_* path",
                    fused_psm.unavailable_reason(),
                )
        return self._fused_supported

    def _update_fused(self, group: dict) -> bool:
        """Apply one group's update with the fused native kernel.

        Args:
            group: Parameter group to update.

        Returns:
            ``True`` when the kernel performed the update. ``False`` means the group
            was not modified -- any momentum buffer materialized here is still zero,
            exactly what the fallback path would create -- so the caller may hand the
            group to a fallback path without re-applying momentum.
        """
        if not self._supports_fused():
            return False
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return True
        for param in params:
            if len(self.state[param]) == 0:
                self.state[param]["exp_avg"] = torch.zeros_like(param)
        # The kernel addresses raw buffers, and a DTensor reports the global element
        # count while pointing at one local shard, so every tensor has to be resolved to
        # its local shard first -- otherwise the sizes disagree and the group is refused.
        return fused_psm.update(
            [to_local_if_dtensor(param.data) for param in params],
            [to_local_if_dtensor(param.grad) for param in params],
            [to_local_if_dtensor(self.state[param]["exp_avg"]) for param in params],
            group["lr"],
            group["gamma"],
            group["beta"],
            group["weight_decay"],
        )

    def _supports_foreach(self, param: Any) -> bool:
        """Return whether the batched foreach ops work on this parameter's backend."""
        if self._foreach_supported is None:
            try:
                scratch = torch.zeros(1, dtype=param.dtype, device=param.device)
                torch._foreach_mul_([scratch], 1.0)
                self._foreach_supported = True
                logger.info("PSM: using the batched foreach update path on %s", param.device)
            except Exception:  # pylint: disable=broad-except - any backend gap means fallback
                logger.warning(
                    "PSM: torch._foreach_* is unavailable on %s; using the elementwise path",
                    param.device,
                )
                self._foreach_supported = False
        return self._foreach_supported

    def _update_foreach(self, group: dict) -> None:
        """Apply one group's update with batched foreach ops."""
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        for param in params:
            if len(self.state[param]) == 0:
                self.state[param]["exp_avg"] = torch.zeros_like(param)
        exp_avgs = [self.state[param]["exp_avg"] for param in params]
        torch._foreach_mul_(exp_avgs, group["gamma"])
        torch._foreach_add_(exp_avgs, grads)
        update = torch._foreach_mul(
            torch._foreach_sign(exp_avgs),
            torch._foreach_pow(torch._foreach_abs(exp_avgs), group["beta"]),
        )
        if group["weight_decay"] != 0:
            torch._foreach_add_(update, params, alpha=group["weight_decay"])
        torch._foreach_add_(params, update, alpha=-group["lr"])

    def _update_elementwise(self, group: dict) -> None:
        """Apply one group's update parameter by parameter (fallback path)."""
        lr = group["lr"]
        gamma = group["gamma"]
        beta = group["beta"]
        weight_decay = group["weight_decay"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param)
            exp_avg = state["exp_avg"]
            exp_avg.mul_(gamma).add_(grad)
            # sign(m) * |m|^beta keeps the direction of the momentum while the power law
            # rescales its magnitude; |m|^0 is plain signSGD.
            update = exp_avg.sign() * exp_avg.abs().pow(beta)
            if weight_decay != 0:
                update.add_(param, alpha=weight_decay)
            param.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Apply one PSM update to every parameter that received a gradient.

        Args:
            closure: Optional closure that re-evaluates the model and returns the
                loss (enables gradients for the call, matching ``torch.optim``).

        Returns:
            The closure's loss when one was supplied, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self._update_fused(group):
                continue
            first = next((p for p in group["params"] if p.grad is not None), None)
            if first is None:
                continue
            if self._supports_foreach(first):
                self._update_foreach(group)
            else:
                self._update_elementwise(group)
        return loss
