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
"""Micro-batch dimension normalization and native tensor splitting tests."""
import pytest
import torch

from hyper_parallel.core.pipeline_parallel._microbatch import _MicroBatch
from hyper_parallel.core.pipeline_parallel.scheduler import PipelineScheduleRuntime
from hyper_parallel.core.pipeline_parallel.utils import BatchDimSpec

_norm_args = PipelineScheduleRuntime._normalize_args_batch_dim
_norm_kwargs = PipelineScheduleRuntime._normalize_kwargs_batch_dim


def _dims(spec_seq):
    """Map a tuple of ``BatchDimSpec | None`` to its ``.batch_dim`` ints for asserts."""
    return tuple(None if s is None else s.batch_dim for s in spec_seq)


def test_normalize_args_bare_scalar_wrapped():
    """
    Feature: args_batch_dim normalization
    Description: A single-input model may pass a bare int / BatchDimSpec (no list).
    Expectation: Wrapped into a one-element tuple of BatchDimSpec.
    """
    assert _dims(_norm_args(0)) == (0,)
    assert _dims(_norm_args(3)) == (3,)
    assert _dims(_norm_args(BatchDimSpec(2))) == (2,)


def test_normalize_args_list_int_and_none():
    """
    Feature: args_batch_dim normalization
    Description: List/tuple entries may be plain int, BatchDimSpec, or None.
    Expectation: ints become BatchDimSpec, None and BatchDimSpec pass through.
    """
    assert _dims(_norm_args([0, 1])) == (0, 1)
    assert _dims(_norm_args((0, None, BatchDimSpec(2)))) == (0, None, 2)
    assert _dims(_norm_args([BatchDimSpec(1), 0])) == (1, 0)


def test_normalize_args_none_and_empty_and_from_tuple():
    """
    Feature: args_batch_dim normalization
    Description: None / empty stay benign; legacy ``from_tuple`` still works.
    Expectation: None -> None, [] -> (), from_tuple unchanged.
    """
    assert _norm_args(None) is None
    assert _norm_args([]) == ()
    assert _dims(_norm_args(BatchDimSpec.from_tuple((0, 1)))) == (0, 1)


@pytest.mark.parametrize("bad", [True, False, 1.5, "0", object()])
def test_normalize_args_rejects_bad_top_type(bad):
    """
    Feature: args_batch_dim normalization
    Description: A non-int/spec/list scalar (incl. bool) is rejected at the top.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_args(bad)


@pytest.mark.parametrize("seq", [[0, True], [1.5], [BatchDimSpec(0), "x"]])
def test_normalize_args_rejects_bad_element(seq):
    """
    Feature: args_batch_dim normalization
    Description: A bad element inside the list (incl. bool) is rejected.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_args(seq)


def test_normalize_kwargs_int_and_spec():
    """
    Feature: kwargs_batch_dim normalization
    Description: Dict values may be plain int or BatchDimSpec.
    Expectation: ints become BatchDimSpec; legacy from_dict still works.
    """
    out = _norm_kwargs({"x": 0, "y": BatchDimSpec(1)})
    assert {k: v.batch_dim for k, v in out.items()} == {"x": 0, "y": 1}
    out2 = _norm_kwargs(BatchDimSpec.from_dict({"a": 2}))
    assert out2["a"].batch_dim == 2


def test_normalize_kwargs_none_passthrough():
    """
    Feature: kwargs_batch_dim normalization
    Description: None stays None.
    Expectation: None -> None.
    """
    assert _norm_kwargs(None) is None


@pytest.mark.parametrize("bad", [[0], (0,), 0, "x"])
def test_normalize_kwargs_rejects_non_dict(bad):
    """
    Feature: kwargs_batch_dim normalization
    Description: A non-dict is rejected.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_kwargs(bad)


@pytest.mark.parametrize("bad", [True, 1.5, "0"])
def test_normalize_kwargs_rejects_bad_value(bad):
    """
    Feature: kwargs_batch_dim normalization
    Description: A bad dict value (incl. bool) is rejected.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_kwargs({"x": bad})



def test_native_microbatch_splits_args_and_kwargs():
    """Split different batch axes and preserve an explicitly unsplit tensor."""
    tensor = torch.arange(24).reshape(6, 4)
    shared = torch.arange(4)
    split = _MicroBatch(3, (BatchDimSpec(0), BatchDimSpec(-1)), {"labels": BatchDimSpec(1)})
    args, kwargs = split((tensor, shared), {"labels": tensor.t()})
    for index in range(3):
        torch.testing.assert_close(args[index][0], tensor[index * 2:(index + 1) * 2])
        assert args[index][1] is shared
        torch.testing.assert_close(kwargs[index]["labels"], tensor.t()[:, index * 2:(index + 1) * 2])


def test_native_microbatch_rejects_indivisible_batch():
    """Reject a batch whose size cannot be divided equally among micro-batches."""
    with pytest.raises(ValueError, match="not divisible"):
        _MicroBatch(3)((torch.ones(4, 2),), {})
