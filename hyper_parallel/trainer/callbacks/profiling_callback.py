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
"""Optional PyTorch profiler callback for short training investigations."""

from typing import Any

from hyper_parallel.trainer.runtime.profiling import create_profiler

from .base import Callback, TrainerState


def _resolve_profiling_ranks(rank: Any, world_size: int) -> frozenset:
    """Expand ``profiling.rank`` into the set of ranks that must record a trace.

    Args:
        rank: A single rank index, or an iterable of them.
        world_size: Number of ranks in the job, used to validate every entry.

    Returns:
        The ranks to profile, as a set for O(1) membership test on each rank.

    Raises:
        ValueError: If ``rank`` is neither an int nor an iterable of ints, or if any
            entry falls outside ``[0, world_size)``.
    """
    if isinstance(rank, int):
        ranks = [rank]
    else:
        try:
            ranks = list(rank)
        except TypeError as exc:
            raise ValueError(
                f"profiling.rank must be an int or a list of ints, but got {rank!r}"
            ) from exc
    for entry in ranks:
        if not isinstance(entry, int):
            raise ValueError(
                f"profiling.rank must be an int or a list of ints, but got {entry!r}"
            )
        if entry < 0 or entry >= world_size:
            raise ValueError(
                f"profiling.rank must be in [0, {world_size}), but got {entry}"
            )
    return frozenset(ranks)


class ProfilingCallback(Callback):
    """Record a bounded CPU and accelerator trace on the configured distributed ranks."""

    def __init__(self, trainer: Any) -> None:
        """Initialize the callback from ``TrainerConfig.profiling``.

        Args:
            trainer: Trainer that owns the callback lifecycle.

        Raises:
            ValueError: If an enabled profiling window or rank is invalid.
        """
        super().__init__(trainer)
        config = trainer.config.profiling
        self.config = config
        self.profiler = None
        if not config.enabled:
            self.enabled = False
            return
        if config.start_step < 1:
            raise ValueError("profiling.start_step must be at least 1")
        if config.end_step <= config.start_step:
            raise ValueError("profiling.end_step must be greater than profiling.start_step")
        self.ranks = _resolve_profiling_ranks(config.rank, trainer.world_size)
        self.enabled = trainer.global_rank in self.ranks

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Create and start the profiler on every configured rank.

        Args:
            state: Trainer state at the start of training.
        """
        del state, kwargs
        if not self.enabled:
            return
        self.profiler = create_profiler(
            start_step=self.config.start_step,
            end_step=self.config.end_step,
            trace_dir=self.config.trace_dir,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_modules=self.config.with_modules,
            global_rank=self.trainer.global_rank,
            offline_parse=self.config.offline_parse,
        )
        self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """Advance the profiler schedule after one complete optimizer step."""
        del kwargs
        if self.profiler is not None:
            self.profiler.step()
            if state.global_step >= self.config.end_step:
                self.profiler.stop()
                self.profiler = None

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Stop the profiler and flush any pending trace output.

        Args:
            state: Trainer state at the end of training.
        """
        del state, kwargs
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
