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
"""Load the fused PSM update kernel and apply it to host-offloaded parameter groups.

The kernel in ``csrc/fused_psm.cpp`` is compiled on first use with the host
compiler and cached, so a source checkout works without a build step. It handles the
contiguous float32 and bfloat16 CPU tensors that host-offloaded parameters and
optimizer state use. Everything else is best effort: when the compiler, the platform
or the parameter layout is not what the fused kernel needs, :func:`update` returns
``False`` and the caller falls back to the batched ``torch._foreach_*`` path. Nothing
in this module raises for a missing toolchain, because a training run must not fail
over an optional fast path.

Set ``HP_PSM_FUSED=0`` to disable the fast path (used when A/B measuring it).
``HP_PSM_FUSED_CACHE`` overrides the build cache location.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).resolve().parent / "csrc" / "fused_psm.cpp"
_LIBRARY_NAME = "libhp_fused_psm.so"
_BUILD_TIMEOUT_S = 300
#: Storage types the kernel understands, mapped to the C ``dtype_code`` argument.
_DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1}

_library: Optional[ctypes.CDLL] = None
_unavailable_reason: str = ""
_probed = False
_logged_success = False
_logged_decline = False


def _cache_directory() -> Path:
    """Return the directory holding the compiled kernel."""
    override = os.environ.get("HP_PSM_FUSED_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "hyper_parallel" / "psm_fused"


def _compile(destination: Path) -> None:
    """Compile the kernel to ``destination`` with the host C++ compiler.

    The build is written to a unique temporary file and moved into place, so
    several ranks starting at once cannot observe a half-written library.

    Args:
        destination: Final path of the shared library.

    Raises:
        RuntimeError: If the compiler is missing, fails, or times out.
    """
    compiler = os.environ.get("HP_PSM_CXX") or os.environ.get("CXX") or "g++"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(destination.parent), suffix=".so.tmp")
    os.close(handle)
    command = [
        compiler, "-O3", "-fopenmp", "-shared", "-fPIC", "-std=c++17",
        # Keep the momentum accumulation bit-identical to the two-op torch sequence:
        # with the default -ffp-contract=fast the compiler folds gamma * m + g into an
        # FMA, which is more accurate but no longer matches the fallback bit for bit.
        "-ffp-contract=off",
        str(_SOURCE), "-o", temporary,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        Path(temporary).unlink(missing_ok=True)
        raise RuntimeError(f"cannot run {compiler}: {error}") from error
    if completed.returncode != 0:
        Path(temporary).unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no compiler output"
        raise RuntimeError(f"{compiler} failed (rc={completed.returncode}): {tail}")
    os.replace(temporary, destination)


def _load() -> Optional[ctypes.CDLL]:
    """Build the kernel if needed, load it, and bind the symbols.

    Returns:
        The loaded library, or ``None`` when it cannot be built or loaded.
    """
    global _unavailable_reason
    if os.environ.get("HP_PSM_FUSED", "1") == "0":
        _unavailable_reason = "disabled by HP_PSM_FUSED=0"
        return None
    if not _SOURCE.is_file():
        _unavailable_reason = f"kernel source missing at {_SOURCE}"
        return None
    library_path = _cache_directory() / _LIBRARY_NAME
    try:
        stale = (
            not library_path.is_file()
            or library_path.stat().st_mtime < _SOURCE.stat().st_mtime
        )
        if stale:
            _compile(library_path)
        library = ctypes.CDLL(str(library_path))
        library.hp_psm_update.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_long),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        library.hp_psm_update.restype = ctypes.c_int
    except Exception as error:  # pylint: disable=broad-except - optional fast path
        _unavailable_reason = f"{type(error).__name__}: {error}"
        return None
    logger.info("PSM: using the fused native update path (%s)", library_path)
    return library


def _get_library() -> Optional[ctypes.CDLL]:
    """Return the loaded kernel, attempting the build at most once per process."""
    global _library, _probed
    if not _probed:
        _library = _load()
        _probed = True
        if _library is None:
            logger.warning("PSM: fused native update unavailable (%s)", _unavailable_reason)
    return _library


def is_available() -> bool:
    """Return whether the fused native update can be used on this host."""
    return _get_library() is not None


def unavailable_reason() -> str:
    """Return why the fused native update is unavailable (empty when it works)."""
    _get_library()
    return _unavailable_reason


def set_threads(threads: int) -> bool:
    """Set the OpenMP thread count used by the fused kernel.

    The setting applies to every OpenMP region in the process, which is why it is
    opt-in: by default the update follows ``OMP_NUM_THREADS`` like the rest of the
    host-side work.

    Args:
        threads: Positive thread count; non-positive values are ignored.

    Returns:
        ``True`` when the kernel accepted the request.
    """
    library = _get_library()
    if library is None or threads <= 0:
        return False
    library.hp_psm_set_threads.argtypes = [ctypes.c_int]
    library.hp_psm_set_threads.restype = None
    library.hp_psm_set_threads(int(threads))
    return True


def _report_success(dtype: torch.dtype, count: int, elements: int) -> None:
    """Log the first group the fused kernel handled, so a run proves which path it used."""
    global _logged_success
    if not _logged_success:
        _logged_success = True
        logger.info(
            "PSM: fused native update applied (dtype=%s, tensors=%s, elements=%s)",
            dtype,
            count,
            elements,
        )


def _report_decline(reason: str) -> None:
    """Log the first rejected group once, so a silent fallback cannot go unnoticed."""
    global _logged_decline
    if not _logged_decline:
        _logged_decline = True
        logger.warning("PSM: falling back to the batched update path (%s)", reason)


def _describe(tensor: Optional[torch.Tensor]) -> str:
    """Describe a tensor compactly so a fallback message names the real culprit."""
    if tensor is None:
        return "None"
    return (
        f"{type(tensor).__name__}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, contiguous={tensor.is_contiguous()})"
    )


def _check_tensor(tensor: Optional[torch.Tensor], role: str, dtype: torch.dtype,
                  param: torch.Tensor) -> str:
    """Return an empty string when ``tensor`` suits the kernel, else the reason why not."""
    if tensor is None:
        return f"{role} is None"
    if tensor.dtype != dtype:
        return f"{role} is {_describe(tensor)}, expected {dtype}"
    if tensor.device.type != "cpu":
        return f"{role} is {_describe(tensor)}, expected a cpu tensor"
    if not tensor.is_contiguous():
        return f"{role} is not contiguous: {_describe(tensor)}"
    if tensor.numel() != param.numel():
        return (
            f"{role} has {tensor.numel()} elements but the parameter has {param.numel()}: "
            f"{_describe(tensor)} vs {_describe(param)}"
        )
    return ""


def update(
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    exp_avgs: Sequence[torch.Tensor],
    lr: float,
    gamma: float,
    beta: float,
    weight_decay: float,
) -> bool:
    """Apply one PSM update to ``params`` in place through the fused kernel.

    Every tensor must be a contiguous CPU float32 or bfloat16 tensor, which is the
    layout of host-offloaded parameters and optimizer state. The update is skipped
    without touching any state when that does not hold.

    Args:
        params: Parameters to update in place.
        grads: Gradients matching ``params``.
        exp_avgs: Momentum buffers matching ``params``.
        lr: Learning rate.
        gamma: Momentum factor.
        beta: Power-law exponent applied to the momentum magnitude.
        weight_decay: Decoupled weight-decay coefficient.

    Returns:
        ``True`` when the fused kernel performed the update, ``False`` when the
        caller must use a fallback path instead.
    """
    library = _get_library()
    if library is None:
        return False
    count = len(params)
    if count == 0 or len(grads) != count or len(exp_avgs) != count:
        _report_decline(
            f"group shape mismatch: params={count} grads={len(grads)} exp_avgs={len(exp_avgs)}"
        )
        return False
    for param in params:
        if param is None:
            _report_decline("a parameter is None")
            return False
    dtype = params[0].dtype
    dtype_code = _DTYPE_CODES.get(dtype)
    if dtype_code is None:
        _report_decline(f"unsupported dtype {dtype}")
        return False
    elements = 0
    for param, grad, exp_avg in zip(params, grads, exp_avgs):
        reason = (
            _check_tensor(param, "param", dtype, param)
            or _check_tensor(grad, "grad", dtype, param)
            or _check_tensor(exp_avg, "exp_avg", dtype, param)
        )
        if reason:
            _report_decline(reason)
            return False
        elements += param.numel()

    pointer_array = ctypes.c_void_p * count
    sizes = (ctypes.c_long * count)(*[param.numel() for param in params])
    status = library.hp_psm_update(
        pointer_array(*[param.data_ptr() for param in params]),
        pointer_array(*[grad.data_ptr() for grad in grads]),
        pointer_array(*[exp_avg.data_ptr() for exp_avg in exp_avgs]),
        sizes,
        count,
        dtype_code,
        float(lr),
        float(gamma),
        float(beta),
        float(weight_decay),
    )
    if status != 0:
        _report_decline(f"kernel returned status {status}")
        return False
    _report_success(dtype, count, elements)
    return True
