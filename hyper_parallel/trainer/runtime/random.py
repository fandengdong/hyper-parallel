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

"""Trainer-side seeding and numerical-determinism controls.

Split out of the former ``auto_models/components/utils/helper.py`` in stage 7
(05 §10.4); function names and signatures are unchanged.
"""

import os
import random

import numpy as np
import torch
from transformers import set_seed as set_seed_func

from hyper_parallel.models.build_options import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu  # noqa: F401


def enable_high_precision_for_bf16() -> None:
    """
    Set high accumulation dtype for matmul and reduction.
    """
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def enable_full_determinism(seed: int) -> None:
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.

    Each numerical-determinism knob is applied with ``os.environ.setdefault``, so
    a value already exported by the launcher wins. That keeps the default
    behaviour unchanged while making the knobs individually switchable for
    profiling, e.g. ``FLASH_ATTENTION_DETERMINISTIC=0``.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    # Ascend wants the literal "true"; CUDA reads "1". Resolve the value once so
    # the NPU spelling is not masked by the earlier default.
    os.environ.setdefault("NCCL_DETERMINISTIC", "true" if IS_NPU_AVAILABLE else "1")
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    if IS_NPU_AVAILABLE:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ.setdefault("CLOSE_MATMUL_K_SHIFT", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("HP_TORCH_DETERMINISTIC", "1") != "0":
        torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if IS_NPU_AVAILABLE:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def set_seed(seed: int | None, full_determinism: bool = False) -> None:
    """
    Sets a manual seed on all devices.
    """
    if seed is None:
        return
    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed_func(seed)


__all__ = [
    "enable_full_determinism",
    "enable_high_precision_for_bf16",
    "set_seed",
]
