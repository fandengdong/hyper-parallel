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
"""``profiling.rank`` accepts several ranks, not just one.

A single-rank trace cannot separate a straggler from a local barrier: if the only
captured rank waits inside a collective, the wait looks identical whether one peer
arrived late or every peer did. Recording a few ranks spread across the job is what
turns that ambiguity into evidence, and it is the prerequisite for judging whether an
exposed collective should be pipelined or re-balanced. These tests pin the contract:
an int keeps meaning "this one rank" (existing configs unchanged), a list selects each
listed rank and no other, every entry is bounds-checked, and each profiled rank still
writes its own trace file.
"""
# pylint: disable=wrong-import-position

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from tests.common.mark_utils import arg_mark

from hyper_parallel.trainer.callbacks.profiling_callback import ProfilingCallback
from hyper_parallel.trainer.config.training import ProfilingConfig
from hyper_parallel.trainer.runtime import profiling

_MARK = dict(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
             card_mark="allcards", essential_mark="essential")


def _callback(rank, world_size, global_rank, enabled=True):
    """Build a ``ProfilingCallback`` over a trainer stub and return it."""
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            profiling=ProfilingConfig(enabled=enabled, start_step=1, end_step=2, rank=rank)
        ),
        global_rank=global_rank,
        world_size=world_size,
        mesh=None,
    )
    return ProfilingCallback(trainer)


class TestProfilingRankSelection(unittest.TestCase):
    """The rank setting selects exactly the ranks that are meant to record."""

    @arg_mark(**_MARK)
    def test_int_rank_keeps_the_single_rank_contract(self):
        """An int still profiles that one rank, as every existing config assumes."""
        self.assertEqual(ProfilingConfig().rank, 0)
        self.assertTrue(_callback(0, 8, global_rank=0).enabled)
        self.assertFalse(_callback(0, 8, global_rank=1).enabled)

    @arg_mark(**_MARK)
    def test_rank_list_enables_each_listed_rank_only(self):
        """Every listed rank records; the ranks in between stay quiet."""
        for global_rank in (0, 7, 255):
            with self.subTest(global_rank=global_rank):
                self.assertTrue(_callback([0, 7, 255], 256, global_rank=global_rank).enabled)
        for global_rank in (1, 8, 254):
            with self.subTest(global_rank=global_rank):
                self.assertFalse(_callback([0, 7, 255], 256, global_rank=global_rank).enabled)

    @arg_mark(**_MARK)
    def test_disabled_profiling_stays_off_whatever_the_rank_says(self):
        """The master switch wins, and a disabled config is not rank-validated."""
        callback = _callback([9999], 8, global_rank=0, enabled=False)
        self.assertFalse(callback.enabled)
        self.assertIsNone(callback.profiler)


class TestProfilingRankValidation(unittest.TestCase):
    """Bad rank settings fail loudly instead of silently recording nothing."""

    @arg_mark(**_MARK)
    def test_out_of_range_rank_raises(self):
        """An entry outside ``[0, world_size)`` is rejected, int or list."""
        with self.assertRaisesRegex(ValueError, r"profiling\.rank must be in \[0, 8\)"):
            _callback(8, 8, global_rank=0)
        with self.assertRaisesRegex(ValueError, r"profiling\.rank must be in \[0, 8\)"):
            _callback([0, 8], 8, global_rank=0)
        with self.assertRaisesRegex(ValueError, r"profiling\.rank must be in \[0, 8\)"):
            _callback([-1], 8, global_rank=0)

    @arg_mark(**_MARK)
    def test_non_int_rank_raises(self):
        """A malformed entry from YAML raises ValueError, not TypeError."""
        with self.assertRaisesRegex(ValueError, r"must be an int or a list of ints"):
            _callback([0, "1"], 8, global_rank=0)
        with self.assertRaisesRegex(ValueError, r"must be an int or a list of ints"):
            _callback("0", 8, global_rank=0)


class TestProfilingMultiRankOutput(unittest.TestCase):
    """Ranks that record must land in distinct trace files."""

    def setUp(self) -> None:
        """Give every test a scratch trace directory that is removed afterwards."""
        self.trace_dir = tempfile.mkdtemp(prefix="hp_profiling_ranks_ut_")
        self.addCleanup(lambda: shutil.rmtree(self.trace_dir, ignore_errors=True))

    @arg_mark(**_MARK)
    def test_each_rank_writes_its_own_trace_file(self):
        """Two profiled ranks produce two files, named by rank."""
        traced = []
        for global_rank in (0, 255):
            with mock.patch.object(profiling, "IS_NPU_AVAILABLE", False), \
                    mock.patch.object(profiling, "IS_CUDA_AVAILABLE", True):
                with mock.patch.object(profiling.torch.profiler, "profile",
                                       return_value=mock.Mock()) as profile:
                    profiling.create_profiler(
                        start_step=1,
                        end_step=2,
                        trace_dir=self.trace_dir,
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                        with_modules=False,
                        global_rank=global_rank,
                        offline_parse=True,
                    )
                profiler_instance = SimpleNamespace(export_chrome_trace=mock.Mock())
                profile.call_args.kwargs["on_trace_ready"](profiler_instance)
                traced.append(profiler_instance.export_chrome_trace.call_args[0][0])

        self.assertEqual(len(set(traced)), 2, traced)
        for global_rank, path in zip((0, 255), traced):
            self.assertTrue(os.path.basename(path).startswith(f"veomni_rank{global_rank}_"), path)
            self.assertEqual(os.path.dirname(path), self.trace_dir)


if __name__ == "__main__":
    unittest.main()
