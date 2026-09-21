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
"""Autograd and rendezvous regression tests for pipeline overlap hooks."""
import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.core.pipeline_parallel._sync_hook import _SyncHookFunction
from hyper_parallel.core.pipeline_parallel.comm_compute_overlap import CommComputeOverlap
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookRole


class TestSyncHook(unittest.TestCase):
    """Check gradient flow across warmup and overlapped backward windows."""

    def test_warmup_forward_records_hooks_for_later_backward(self):
        """A forward without overlap must still record synchronization for backward."""
        coordinator = Mock()
        coordinator.is_enabled.return_value = False
        tensor = torch.randn(3, requires_grad=True)
        output = _SyncHookFunction.apply(tensor, "A", coordinator)
        coordinator.rendezvous.assert_not_called()

        coordinator.is_enabled.return_value = True
        output.square().sum().backward()

        torch.testing.assert_close(tensor.grad, 2 * tensor.detach())
        coordinator.notify_dispatched.assert_called_once_with(HookRole.COMM)
        coordinator.rendezvous.assert_called_once_with(HookRole.COMPUTE)

    def test_last_layer_hook_preserves_gradient_without_boundary_rendezvous(self):
        """The final combine must notify dispatch and pass gradients without waiting."""
        coordinator = Mock()
        coordinator.is_enabled.return_value = True
        tensor = torch.randn(3, requires_grad=True)

        _SyncHookFunction.apply(tensor, "D_LAST", coordinator).sum().backward()

        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
        coordinator.notify_dispatched.assert_called_once_with(HookRole.COMM)
        coordinator.rendezvous.assert_not_called()

    def test_wrapped_dispatch_and_combine_preserve_tuple_outputs_and_gradients(self):
        """Hooked expert callables retain auxiliary values and autograd connectivity."""
        overlap = CommComputeOverlap()
        dispatch = overlap.wrap_dispatch(lambda value: (value * 2, "routing"))
        combine = overlap.wrap_combine(lambda value: value * 3, is_last_layer=True)
        tensor = torch.randn(3, requires_grad=True)

        output, auxiliary = dispatch(tensor)
        combine(output).sum().backward()

        self.assertEqual(auxiliary, "routing")
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 6))
