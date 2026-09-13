# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-side FSDP gradient-sync, reshard, and unshard-pipeline policies.

Split out of the former ``auto_models/trainer/base.py`` in stage 7
(05 §15.11 step 3). ``BaseTrainer`` keeps the same-named methods as thin
delegating subclass hooks; the policy itself lives here. The FSDP config is
duck-typed (``reshard_after_backward`` / ``dp_shard_size`` /
``requires_grad_sync`` attributes) so this module does not import Trainer DTOs.
"""

import logging
import os
from typing import Any, Iterator, List

logger = logging.getLogger(__name__)

UNIT_PIPELINE_ENV = "HP_OFFLOAD_UNIT_PIPELINE"
UNSHARD_WINDOW_ENV = "HP_OFFLOAD_UNSHARD_WINDOW"

_reported_keys: set = set()


def model_reshard(
    hsdp_model_parts: List[Any],
    fsdp_config: Any,
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Reshard model after backward pass."""
    if (
            fsdp_config.reshard_after_backward is False
            and num_micro_steps > 1
    ):
        if micro_step == 0:
            for model_part in hsdp_model_parts:
                model_part.set_reshard_after_backward(False)
        elif micro_step == num_micro_steps - 1:
            for model_part in hsdp_model_parts:
                model_part.set_reshard_after_backward(True)


def configure_fsdp_gradient_sync(
    hsdp_model_parts: List[Any],
    fsdp_config: Any,
    dp_replicate_size: int,
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Configure FSDP gradient synchronization for one micro step."""
    if (
            fsdp_config.dp_shard_size > 1
            and num_micro_steps > 1
    ):
        is_last_micro_batch = micro_step == num_micro_steps - 1
        requires_gradient_sync = (
            fsdp_config.requires_grad_sync
            or is_last_micro_batch
        )
        is_hsdp = dp_replicate_size > 1
        for model_part in hsdp_model_parts:
            model_part.set_requires_gradient_sync(requires_gradient_sync)
            model_part.set_is_last_backward(is_last_micro_batch)
            if is_hsdp:
                model_part.set_requires_all_reduce(is_last_micro_batch)


def unit_pipeline_enabled() -> bool:
    """Whether the optimizer/unshard unit pipeline is switched on.

    Returns:
        bool: True only when ``HP_OFFLOAD_UNIT_PIPELINE`` is exactly ``"1"``.
    """
    return os.environ.get(UNIT_PIPELINE_ENV, "0") == "1"


def unshard_window() -> int:
    """Return the unshard window: prefetched units not yet consumed by forward.

    Returns:
        int: The window size. Values below 1 disable prefetching, so the step is
        still split per unit but no unshard overlaps it.

    Raises:
        ValueError: When the configured value is not an integer.
    """
    raw = os.environ.get(UNSHARD_WINDOW_ENV, "1")
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{UNSHARD_WINDOW_ENV} must be an integer, got {raw!r}") from None


def _log_once(key: str, level: int, message: str, *args: Any) -> None:
    """Emit one record per distinct key, so a per-step path cannot flood the log."""
    if key in _reported_keys:
        return
    _reported_keys.add(key)
    logger.log(level, message, *args)


def _supports_step_subset(optimizer: Any) -> bool:
    """Whether ``optimizer`` can be stepped over an explicit parameter subset."""
    supports = getattr(optimizer, "supports_step_subset", None)
    if callable(supports):
        return bool(supports())
    return (
        callable(getattr(optimizer, "step_subset", None))
        and callable(getattr(optimizer, "advance_step_counters", None))
    )


def _optimizer_params(optimizers: List[Any]) -> Iterator[Any]:
    """Yield every parameter owned by the given optimizers."""
    for optimizer in optimizers:
        for group in getattr(optimizer, "param_groups", ()):
            yield from group.get("params", ())


def run_unit_pipelined_step(
    optimizers: List[Any],
    model: Any,
    dispatch_no_skip: Any = None,
) -> bool:
    """Step the optimizers unit by unit, prefetching each unit's unshard.

    The step is split along fully_shard unit boundaries: every unit is updated
    on its own, and right after a unit is updated its unshard (CPU-to-device
    copy plus all-gather) is started asynchronously, so the device-side copies
    and collectives overlap the host-side optimizer work on the remaining
    units. The next forward then finds a prefetched unit already materialized;
    units beyond the window, and units this path skipped, are unsharded
    synchronously by their forward hook exactly as before.

    Gradient clipping must have run before this call: it needs the gradient of
    every parameter, and this call updates parameters.

    Args:
        optimizers (List[Any]): Optimizers to step, in call order.
        model (Any): Module whose fully_shard units define the step boundaries.
        dispatch_no_skip (set, optional): Ops that must still go through DTensor
            dispatch while the optimizer calls run, matching the
            ``SkipDTensorDispatch`` the caller uses for a full ``step()``. The
            default disables dispatch for every op.

    Returns:
        bool: True when the pipelined path ran. False when the pipeline is off
        or unsupported by this optimizer and model, in which case the caller
        must run its ordinary full ``step()``/``zero_grad()`` loop.
    """
    if not unit_pipeline_enabled():
        return False

    # Imported here so this module stays import-light, and because the FSDP API
    # pulls in the distributed and DTensor stacks.
    from hyper_parallel import SkipDTensorDispatch  # pylint: disable=C0415
    from hyper_parallel.core.fully_shard.api import get_fully_shard_units  # pylint: disable=C0415

    if any(not _supports_step_subset(optimizer) for optimizer in optimizers):
        _log_once(
            "unsupported-optimizer",
            logging.WARNING,
            "%s=1 is ignored: an optimizer does not implement step_subset/advance_step_counters",
            UNIT_PIPELINE_ENV,
        )
        return False
    units = get_fully_shard_units(model)
    if not units:
        _log_once(
            "no-units",
            logging.WARNING,
            "%s=1 is ignored: the model has no fully_shard unit",
            UNIT_PIPELINE_ENV,
        )
        return False

    unit_param_ids: set = set()
    for unit in units:
        for parameter in unit.params:
            if id(parameter) in unit_param_ids:
                _log_once(
                    "duplicate-parameter",
                    logging.WARNING,
                    "%s=1 is ignored: parameter %s is owned by more than one fully_shard unit",
                    UNIT_PIPELINE_ENV,
                    getattr(parameter, "model_name", "<unknown>"),
                )
                return False
            unit_param_ids.add(id(parameter))

    # Parameters outside every unit -- excluded from fully_shard, or owned by no
    # unit -- are stepped in one batch, which is what the un-split step does.
    leftover = [
        parameter
        for parameter in _optimizer_params(optimizers)
        if id(parameter) not in unit_param_ids
    ]

    window = unshard_window()
    _log_once(
        "enabled",
        logging.INFO,
        "%s=1: stepping %d fully_shard units with an unshard window of %d (%d leftover params)",
        UNIT_PIPELINE_ENV,
        len(units),
        window,
        len(leftover),
    )
    prefetched = 0
    with SkipDTensorDispatch(no_skip=dispatch_no_skip):
        for optimizer in optimizers:
            optimizer.advance_step_counters()
        if leftover:
            for optimizer in optimizers:
                optimizer.step_subset(leftover)
        for unit in units:
            for optimizer in optimizers:
                optimizer.step_subset(unit.params)
            if unit.is_sharded and 0 < window and prefetched < window:
                unit.prefetch()
                prefetched += 1
    for optimizer in optimizers:
        optimizer.zero_grad()
    return True


__all__ = [
    "configure_fsdp_gradient_sync",
    "model_reshard",
    "run_unit_pipelined_step",
    "unit_pipeline_enabled",
    "unshard_window",
]
