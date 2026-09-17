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
"""``profiling.offline_parse`` skips the blocking in-process trace parse.

``torch_npu.profiler.tensorboard_trace_handler`` parses the collected trace from
``on_trace_ready``, i.e. inside ``profiler.step()``: the host process blocks for the whole
parse while its peers wait in a collective, which trips the communication timeout and kills
the run (a 30-minute parse against a 1800 s HCCL timeout). The opt-in honours that the raw
data is already on disk and only the parse is deferred; these tests pin both halves -- the
handler is not invoked when the flag is on, still invoked when it is off -- plus the fact
that the post-handler path must not touch ``p.prof_if`` once the parse is skipped.
"""
# pylint: disable=wrong-import-position

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from tests.common.mark_utils import arg_mark

from hyper_parallel.trainer.callbacks.profiling_callback import ProfilingCallback
from hyper_parallel.trainer.config.training import ProfilingConfig
from hyper_parallel.trainer.runtime import profiling


class _FakeNpuProfiler:
    """Record what ``create_profiler`` asks of ``torch_npu.profiler``.

    ``as_module()`` is a stand-in for the ``torch_npu`` module itself, so the tests never
    need torch_npu to be importable or an NPU to be present.
    """

    def __init__(self) -> None:
        """Initialize the recording fields."""
        self.handler_dir = None
        self.npu_trace_handler = mock.Mock(name="npu_trace_handler")
        self.on_trace_ready = None
        self.profile_kwargs = None

    def _tensorboard_trace_handler(self, dir_name):
        """Return the trace handler and keep the directory it was registered with."""
        self.handler_dir = dir_name
        return self.npu_trace_handler

    def _profile(self, **kwargs):
        """Return a profiler stand-in and capture the trace callback it was given."""
        self.profile_kwargs = kwargs
        self.on_trace_ready = kwargs["on_trace_ready"]
        return SimpleNamespace(**kwargs)

    def as_module(self) -> Any:
        """Build the ``torch_npu`` module stand-in this recorder stands behind."""
        profiler = SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            AiCMetrics=SimpleNamespace(PipeUtilization="pipe_utilization"),
            ProfilerLevel=SimpleNamespace(Level1="level1"),
            _ExperimentalConfig=lambda **kwargs: kwargs,
            schedule=lambda **kwargs: kwargs,
            profile=self._profile,
            tensorboard_trace_handler=self._tensorboard_trace_handler,
        )
        return SimpleNamespace(profiler=profiler)


def _create_profiler(fake, trace_dir, offline_parse, profile_memory=False):
    """Build a profiler (NPU fakes must already be patched in) and return its callback."""
    profiling.create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=trace_dir,
        record_shapes=False,
        profile_memory=profile_memory,
        with_stack=False,
        with_modules=False,
        global_rank=3,
        offline_parse=offline_parse,
    )
    return fake.on_trace_ready


class TestNpuOfflineParse(unittest.TestCase):
    """The NPU trace handler is invoked only when ``offline_parse`` is off."""

    def setUp(self) -> None:
        """Give every test a scratch trace directory that is removed afterwards."""
        self.trace_dir = tempfile.mkdtemp(prefix="hp_profiling_ut_")
        self.addCleanup(lambda: shutil.rmtree(self.trace_dir, ignore_errors=True))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_offline_parse_skips_the_npu_trace_handler(self):
        """With the flag on, the handler is built but never invoked."""
        fake = _FakeNpuProfiler()

        with mock.patch.object(profiling, "IS_NPU_AVAILABLE", True), \
                mock.patch.object(profiling, "torch_npu", fake.as_module(), create=True), \
                self.assertLogs(profiling.logger, level="INFO") as logs:
            on_trace_ready = _create_profiler(fake, self.trace_dir, offline_parse=True)
            # A profiler that never produced a ``prof_if``: the post-handler path must not
            # reach for ``p.prof_if.prof_path`` while the parse is skipped.
            on_trace_ready(SimpleNamespace())

        fake.npu_trace_handler.assert_not_called()
        # The handler's construction is what registers the output directory with torch_npu,
        # so it has to keep happening -- otherwise the raw trace has nowhere to land.
        self.assertEqual(fake.handler_dir, self.trace_dir)
        self.assertTrue(
            any(f"msprof --export=on --output={self.trace_dir}" in line for line in logs.output),
            logs.output,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_still_runs_the_npu_trace_handler(self):
        """Without the flag the in-process parse is unchanged."""
        fake = _FakeNpuProfiler()
        raw_dir = os.path.join(self.trace_dir, "host_1_20260101_ascend_pt")
        profiler_instance = SimpleNamespace(prof_if=SimpleNamespace(prof_path=raw_dir))

        with mock.patch.object(profiling, "IS_NPU_AVAILABLE", True), \
                mock.patch.object(profiling, "torch_npu", fake.as_module(), create=True), \
                self.assertLogs(profiling.logger, level="INFO") as logs:
            on_trace_ready = _create_profiler(fake, self.trace_dir, offline_parse=False)
            on_trace_ready(profiler_instance)

        fake.npu_trace_handler.assert_called_once_with(profiler_instance)
        self.assertTrue(any(f"Profiling result saved at {raw_dir}" in line for line in logs.output),
                        logs.output)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_offline_parse_keeps_the_memory_snapshot(self):
        """The memory snapshot dump is independent of the trace parse."""
        fake = _FakeNpuProfiler()
        device = mock.Mock()

        with mock.patch.object(profiling, "IS_NPU_AVAILABLE", True), \
                mock.patch.object(profiling, "torch_npu", fake.as_module(), create=True), \
                mock.patch.object(profiling, "get_torch_device", return_value=device):
            on_trace_ready = _create_profiler(
                fake, self.trace_dir, offline_parse=True, profile_memory=True
            )
            on_trace_ready(SimpleNamespace())

        snapshot_path = device.memory._dump_snapshot.call_args[0][0]
        self.assertEqual(os.path.dirname(snapshot_path), self.trace_dir)
        self.assertTrue(os.path.basename(snapshot_path).startswith("veomni_rank3_"))
        self.assertTrue(snapshot_path.endswith(".pkl"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cuda_path_is_untouched(self):
        """A CUDA run exports the chrome trace whatever the flag says."""
        chrome_trace = mock.Mock()
        profiler_instance = SimpleNamespace(export_chrome_trace=chrome_trace)

        with mock.patch.object(profiling, "IS_NPU_AVAILABLE", False), \
                mock.patch.object(profiling, "IS_CUDA_AVAILABLE", True):
            with mock.patch.object(profiling.torch.profiler, "profile", return_value=mock.Mock()) as profile:
                profiling.create_profiler(
                    start_step=1,
                    end_step=2,
                    trace_dir=self.trace_dir,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_modules=False,
                    global_rank=0,
                    offline_parse=True,
                )
            profile.call_args.kwargs["on_trace_ready"](profiler_instance)

        chrome_trace.assert_called_once()
        self.assertTrue(chrome_trace.call_args[0][0].startswith(self.trace_dir))


class TestProfilingConfigThreading(unittest.TestCase):
    """``ProfilingConfig.offline_parse`` defaults off and reaches ``create_profiler``."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_is_off(self):
        """An existing config keeps the synchronous parse."""
        self.assertIs(ProfilingConfig().offline_parse, False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_callback_threads_the_flag_to_create_profiler(self):
        """The callback passes its configured value through to the profiler."""
        for flag in (True, False):
            with self.subTest(offline_parse=flag):
                trainer = SimpleNamespace(
                    config=SimpleNamespace(
                        profiling=ProfilingConfig(
                            enabled=True, start_step=1, end_step=2, rank=0, offline_parse=flag
                        )
                    ),
                    global_rank=0,
                    world_size=1,
                    mesh=None,
                )
                with mock.patch(
                    "hyper_parallel.trainer.callbacks.profiling_callback.create_profiler",
                    return_value=mock.Mock(),
                ) as create:
                    callback = ProfilingCallback(trainer)
                    callback.on_train_begin(SimpleNamespace())

                self.assertIs(create.call_args.kwargs["offline_parse"], flag)


if __name__ == "__main__":
    unittest.main()
