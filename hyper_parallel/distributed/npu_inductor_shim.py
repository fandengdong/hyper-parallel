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
"""Compatibility shim for a torch_npu inductor codegen defect.

``torch_npu._inductor.codegen.ir.detect_flattened_dims`` walks the sub-expressions of an
addition index and records each of them as if it were an axis variable.  A *composite*
sub-expression such as ``x1 - 1`` -- an axis shifted by a constant, which the MLA
attention's cat/slice/RoPE indexing produces readily -- is then looked up in the kernel's
range-tree dictionaries in the second half of the function:

    parent_axis = kernel.range_tree_nodes_removed[var]      # ir.py:74
    torch._inductor.exc.InductorError: KeyError: 'x1 - 1'

Neither dictionary can contain a compound expression, so codegen aborts and the whole
training run dies in the first backward of a compile-enabled step.  This was measured on
the 2-SN shape: the text attention compiles and fuses (72 graphs, 2784 Triton kernels, 48
of them ``fused_cat``/``fused_slice``) and then step 0 dies with that KeyError.

The shim wraps the detector: when it raises for an index that contains such a composite
expression, that index is reported as having **no flattened dimension**.  Flattening is a
codegen *optimisation* (it re-tiles an axis pair), not a correctness requirement, so
falling back to the unflattened index keeps the generated kernel correct while letting the
run proceed.  It is deliberately opt-in via ``HP_COMPILE_NPU_SHIM=1`` and it logs every
index it defuses, because a silent workaround is exactly how a broken arm comes to look
like a neutral one.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "install_npu_flattened_dims_shim",
    "shim_is_requested",
    "apply_npu_kernel_fallbacks",
    "fallback_spec",
    "rope_eager_is_requested",
    "install_rope_eager_break",
]

_INSTALLED = False


def shim_is_requested() -> bool:
    """Return whether ``HP_COMPILE_NPU_SHIM`` asks for the workaround."""
    return os.environ.get("HP_COMPILE_NPU_SHIM", "0") == "1"


def install_npu_flattened_dims_shim() -> bool:
    """Defuse the composite-expression ``KeyError`` in torch_npu's dim flattening.

    Idempotent, and a no-op when ``torch_npu`` is not importable (GPU / CPU runs) or when
    the function has been fixed upstream.

    Returns:
        ``True`` when the wrapper is installed, ``False`` when nothing was changed.
    """
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return True
    try:
        from torch_npu._inductor.codegen import ir as npu_ir  # pylint: disable=C0415
    except ImportError:
        logger.debug("torch_npu inductor codegen not importable; NPU shim skipped")
        return False

    original = getattr(npu_ir, "detect_flattened_dims", None)
    if original is None or getattr(original, "_hp_composite_guard", False):
        return False

    def detect_flattened_dims(kernel, index):
        """Report no flattened dimension when the detector trips on a composite index."""
        try:
            return original(kernel, index)
        except KeyError as error:
            logger.warning(
                "NPU inductor shim: skipping dim flattening for index %r (torch_npu's "
                "detect_flattened_dims raised KeyError %s on a composite sub-expression; "
                "flattening is an optimisation, the unflattened index is still correct)",
                index,
                error,
            )
            return {}

    # Marking our own wrapper: the attribute is private by naming convention only.
    detect_flattened_dims._hp_composite_guard = True  # type: ignore[attr-defined]  # pylint: disable=protected-access
    npu_ir.detect_flattened_dims = detect_flattened_dims
    _INSTALLED = True
    logger.info("NPU inductor shim installed (HP_COMPILE_NPU_SHIM=1)")
    return True


def fallback_spec() -> str:
    """Return the ``HP_COMPILE_NPU_FALLBACK_KERNELS`` request (empty when unset)."""
    return os.environ.get("HP_COMPILE_NPU_FALLBACK_KERNELS", "").strip()


def apply_npu_kernel_fallbacks(spec: str) -> bool:
    """Send named NPU Triton kernels back to eager FX-graph calls.

    torch_npu raises ``NoTritonConfigsError`` when *no* tiling config works for a
    generated kernel; on the 2-SN text attention exactly one does, the backward glue
    ``triton_unk_fused_add_cat_mul_neg_slice_slice_backward_view_8``, while the eight
    other fused kernels of the same graph (``fused_cat``, the rotary mul/add, the
    ``ones_triu`` mask, the ``clone_transpose``) compile fine.  ``torch_npu`` ships a knob
    for precisely this situation -- ``force_fallback_kernel_id`` -- so one kernel pays the
    eager path instead of the whole run paying nothing.

    Args:
        spec: comma-separated kernel id suffixes (``"8"``), or ``"all"``.

    Returns:
        ``True`` when the fallback list was set, ``False`` when there is nothing to do or
        torch_npu's config is unavailable.
    """
    spec = spec.strip()
    if not spec:
        return False
    try:
        from torch_npu._inductor import config as npu_config  # pylint: disable=C0415
    except ImportError:
        logger.debug("torch_npu inductor config not importable; kernel fallback skipped")
        return False
    if spec == "all":
        npu_config.force_fallback_kernel_id = "all"
        logger.warning("NPU inductor: forcing ALL triton kernels back to eager calls")
        return True
    ids = [int(part) for part in spec.split(",") if part.strip()]
    npu_config.force_fallback_kernel_id = ids
    logger.warning(
        "NPU inductor: forcing kernel ids %s back to eager calls "
        "(NoTritonConfigsError workaround)", ids,
    )
    return True


# RoPE entry points in the vendored modelling files.  Their body is
# ``(q * cos) + (rotate_half(q) * sin)`` with ``rotate_half = cat(-x[half:], x[:half])``,
# which inductor fuses into one kernel whose NPU lowering is broken on both paths: the
# Triton route finds no valid tiling config and the FX fallback route raises
# ``ValueError: Cannot broadcast ... [7, 32, 2], [7, 32, 1]``.
ROPE_MODULES: Tuple[str, ...] = (
    "transformers.models.deepseek_v3.modeling_deepseek_v3",
    "transformers.models.deepseek_v2.modeling_deepseek_v2",
)
ROPE_FUNCTIONS: Tuple[str, ...] = (
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_interleave",
)


def rope_eager_is_requested() -> bool:
    """Return whether ``HP_COMPILE_ROPE_EAGER`` asks for the rotary graph break."""
    return os.environ.get("HP_COMPILE_ROPE_EAGER", "0") == "1"


def install_rope_eager_break() -> int:
    """Run the model's rotary application eagerly so its fusion group cannot form.

    Breaking the graph at the rotary boundary leaves the rest of the attention compiled --
    the ``fused_cat`` q/k/v rebuild, the ``ones_triu`` mask and the ``clone_transpose`` all
    keep their kernels -- while the one region inductor cannot lower correctly stays eager.

    Returns:
        The number of functions wrapped (0 when nothing matched).
    """
    import importlib  # pylint: disable=import-outside-toplevel

    patched = 0
    for module_name in ROPE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for function_name in ROPE_FUNCTIONS:
            function = getattr(module, function_name, None)
            if function is None or getattr(function, "_hp_rope_eager", False):
                continue
            # torch exposes no public equivalent of this tracing control.
            # pylint: disable=protected-access
            wrapped = torch._dynamo.disable(function)
            wrapped._hp_rope_eager = True  # type: ignore[attr-defined]  # pylint: disable=protected-access
            setattr(module, function_name, wrapped)
            logger.warning(
                "NPU inductor: running %s.%s eagerly (its fused backward has no working "
                "NPU lowering; the rest of the attention stays compiled)",
                module_name, function_name,
            )
            patched += 1
    if patched == 0:
        logger.warning(
            "HP_COMPILE_ROPE_EAGER=1 but no rotary entry point was found in %s; "
            "the compile arm may still hit the broken fusion", list(ROPE_MODULES),
        )
    return patched
