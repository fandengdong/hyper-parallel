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
"""Adamw optimizer."""

from typing import Any, Iterable, List

import torch


def adamw(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        step: int,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool
) -> None:
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    step_tensor = torch.tensor(step, dtype=torch.int64, device=params[0].device)
    state_steps = [step_tensor] * len(params)

    torch._fused_adamw_(  # pylint: disable=protected-access
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs if amsgrad else [],
        state_steps,
        amsgrad=amsgrad,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize
    )


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer implementation."""

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            maximize=False
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    def __str__(self):
        return super().__repr__()

    __repr__ = __str__

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.advance_step_counters()
        return self._step_param_subset(None, closure)

    def advance_step_counters(self) -> None:
        """Advance every parameter group's step counter by one.

        ``step`` performs this itself. A caller that splits one logical step
        into several :meth:`step_subset` calls must call this once before the
        first subset, so that every subset uses the same step value for bias
        correction and the split stays numerically identical.
        """
        for group in self.param_groups:
            group['step'] = (group.get('step') or 0) + 1

    def step_subset(self, params: Iterable[Any], closure: Any = None) -> Any:
        """Perform one optimization step over ``params`` only.

        Parameters this optimizer does not own -- for example the private
        parameters of another optimizer in a chain -- and parameters without a
        gradient are skipped, so the caller may pass a model-side subset.

        The group step counters are not advanced here: one logical optimizer
        step may be split into several subset calls, and every call must use
        the same step value. Call :meth:`advance_step_counters` once first.

        Args:
            params: Iterable of parameters to update.
            closure: Optional callable that reevaluates the model. Unlike in
                :meth:`step` it runs once per subset call.

        Returns:
            The closure result when a closure is given, otherwise ``None``.

        Raises:
            RuntimeError: When :meth:`advance_step_counters` has not run yet.
        """
        for group in self.param_groups:
            if group.get('step') is None:
                raise RuntimeError(
                    'AdamW.step_subset needs advance_step_counters() before the first subset step'
                )
        return self._step_param_subset({id(param) for param in params}, closure)

    def _step_param_subset(self, param_ids, closure):
        """Run the AdamW update for the parameters selected by ``param_ids``.

        Args:
            param_ids: Ids of the parameters to update, or ``None`` to update
                every parameter of this optimizer.
            closure: Optional callable that reevaluates the model.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []

            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            current_rank_params = group['params']
            for p in current_rank_params:
                if param_ids is not None and id(p) not in param_ids:
                    continue
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            if params_with_grad:
                adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    group['step'],
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize']
                )

        return loss
