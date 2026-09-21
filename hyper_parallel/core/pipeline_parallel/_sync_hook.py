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
"""Autograd synchronization hooks for pipeline overlap."""
from typing import Any

import torch

from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookCoordinator, HookRole


class _SyncHookFunction(torch.autograd.Function):
    """Autograd identity that fires HookCoordinator rendezvous on fwd/bwd.

    Uses a **4-hook** design (``A``, ``B``, ``C``, ``D``) with pure
    COMM / COMPUTE roles — no NONE role.  Every rendezvous is a strict
    COMM + COMPUTE pair, guaranteeing NCCL-first dispatch ordering at
    **all** points including layer boundaries.

    Hook placement per MoE layer::

        [A] → dispatch → [B] → module → [C] → combine → [D] → (Attention) → [A_next]

    At layer boundaries (D / A hooks), the Attention that runs between
    layers is treated as COMPUTE, and the combine / combine.bwd is treated
    as COMM, so the coordinator enforces comm-first ordering even across
    layer transitions.
    """

    # 4-hook role tables: (prev_role_idx, next_role_idx).
    # Index encoding: 1 = COMM, 2 = COMPUTE.
    #
    _FWD_ROLES = {
        #         (prev, next)      prev op          next op
        "A": (2, 1),   # COMPUTE, COMM     Attention   | dispatch
        "B": (1, 2),   # COMM, COMPUTE     dispatch    | module
        "C": (2, 1),   # COMPUTE, COMM     module      | combine
        "D": (1, 2),   # COMM, COMPUTE     combine     | Attention
    }
    _BWD_ROLES = {
        "D": (2, 1),   # COMPUTE, COMM     Attn.bwd    | combine.bwd
        "C": (1, 2),   # COMM, COMPUTE     combine.bwd | module.bwd
        "B": (2, 1),   # COMPUTE, COMM     module.bwd  | dispatch.bwd
        "A": (1, 2),   # COMM, COMPUTE     dispatch.bwd| Attn.bwd
    }

    _ROLE_CACHE = (None, HookRole.COMM, HookRole.COMPUTE)

    @staticmethod
    def _role_enum(idx: int):
        return _SyncHookFunction._ROLE_CACHE[idx]

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, hook_name: str, coordinator: HookCoordinator) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Identity forward that fires a HookCoordinator rendezvous.

        Notifies the previous op's role and rendezvouses for the next op's
        role per the ``_FWD_ROLES`` table.  ``"D_LAST"`` is a sentinel
        meaning "skip this rendezvous" (last layer's closing D — no
        Attention follows).

        Args:
            ctx:         Autograd context, stores ``hook_name`` and
                         ``coordinator`` for the backward pass.
            x:           Input tensor, returned unchanged.
            hook_name:   One of ``"A"``, ``"B"``, ``"C"``, ``"D"``,
                         ``"D_LAST"``.
            coordinator: The :class:`HookCoordinator` driving the rendezvous.

        Returns:
            ``x`` unchanged.
        """
        ctx.hook_name = hook_name
        ctx.coordinator = coordinator

        if not coordinator.is_enabled():
            return x

        if hook_name == "D_LAST":
            # ``D_LAST`` marks the last layer's closing D hook — no
            # Attention follows in this chunk, so the rendezvous is
            # meaningless and is skipped.  We still
            # ``notify_dispatched(COMM)`` so the COMPUTE side of the
            # preceding ``C`` rendezvous unblocks early, letting
            # BWD's Attn.bwd_last overlap with FWD's post-combine
            # work — Torch autograd is thread-safe so this concurrent
            # FWD-record + BWD-replay is fine.
            prev_idx, _ = _SyncHookFunction._FWD_ROLES["D"]
            role_of = _SyncHookFunction._role_enum
            coordinator.notify_dispatched(role_of(prev_idx))
            return x

        prev_idx, next_idx = _SyncHookFunction._FWD_ROLES[hook_name]
        role_of = _SyncHookFunction._role_enum
        coordinator.notify_dispatched(role_of(prev_idx))
        coordinator.rendezvous(role_of(next_idx))
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Identity backward that fires a HookCoordinator rendezvous.

        Mirror of :meth:`forward` using the ``_BWD_ROLES`` table.
        ``"D_LAST"`` skips the rendezvous because this is the first BWD
        hook to fire and ``combine.bwd`` has already dispatched freely
        before any rendezvous can happen.

        Args:
            ctx:         Autograd context with ``hook_name`` and
                         ``coordinator`` saved during forward.
            grad_output: Gradient w.r.t. the forward output, returned
                         unchanged.

        Returns:
            ``(grad_output, None, None)`` — gradients only flow back to
            the tensor input, ``hook_name`` and ``coordinator`` are
            non-tensor inputs.
        """
        hook_name = ctx.hook_name
        coordinator = ctx.coordinator

        if not coordinator.is_enabled():
            return grad_output, None, None

        if hook_name == "D_LAST":
            # First BWD hook to fire; combine.bwd has already
            # dispatched freely before any rendezvous can happen.
            # Skipping here is safe on Torch because CUDA streams
            # are process-wide and the NCCL FIFO order is consistent
            # across ranks regardless of which thread launched
            # combine.bwd.
            return grad_output, None, None

        prev_idx, next_idx = _SyncHookFunction._BWD_ROLES[hook_name]
        role_of = _SyncHookFunction._role_enum
        coordinator.notify_dispatched(role_of(prev_idx))
        coordinator.rendezvous(role_of(next_idx))
        return grad_output, None, None
