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

"""Standalone layers used by the random-weight Qwen MoE benchmark.

The benchmark builds one model twice (common MoE and MegaMoe) and compares
numerics, so it needs layers that are constructible from a shape alone. The
library layers under ``hyper_parallel.components.modules`` are replacement
factories: they require a source module to convert and therefore cannot be
constructed directly here. These definitions are the standalone
``RMSNorm`` / ``SwiGLUMLP`` / ``RotaryEmbedding`` layers the benchmark used
before the module reorganization.
"""

from __future__ import annotations

import os

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import


def _use_npu_kernels() -> bool:
    """True when ``HYPER_USE_V1_KERNELS=1`` and ``torch_npu`` is importable."""
    if os.environ.get("HYPER_USE_V1_KERNELS", "0") != "1":
        return False
    try:
        import torch_npu  # pylint: disable=C0415,W0611
        return True
    except ImportError:
        return False


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize the normalized-dimension weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and apply the learned scale."""
        if _use_npu_kernels() and hidden_states.device.type == "npu":
            import torch_npu  # pylint: disable=C0415

            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]

        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (Llama / Qwen / Mistral convention)."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        """Build the Gate, Up, and Down projections."""
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation."""
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Rotary position embedding matching Llama / Qwen / Mistral conventions."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        """Pre-compute the inverse frequency table."""
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.register_buffer("inv_freq", self._build_inv_freq(), persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        """Return ``theta ** (-2i / dim)`` for every rotary pair."""
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

    def reset_inv_freq(self) -> None:
        """Re-derive ``inv_freq`` on its current device.

        ``inv_freq`` is a non-persistent buffer, so a meta-materialization step
        that zero-initializes buffers silently disables RoPE (cos=1, sin=0
        gives an identity rotation). Call this after such a step.
        """
        self.inv_freq.copy_(self._build_inv_freq().to(self.inv_freq.device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cosine and sine tables for ``position_ids``.

        Args:
            x: Hidden states, used only for the target device and dtype.
            position_ids: Position indices of shape ``(seq_len,)``, ``(batch, seq_len)``,
                or ``(3, batch, seq_len)``. A 3D multi-modal table collapses to its
                first (temporal) dimension, which is plain 1D RoPE.
        """
        del x
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
        flat = position_ids.float().reshape(-1)
        freqs = torch.outer(flat, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Accepts ``cos`` / ``sin`` of shape ``(seq_len, head_dim)`` or
    ``(batch, seq_len, head_dim)``. Q/K are expected to be
    ``(batch, n_heads, seq_len, head_dim)``.
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    else:
        cos = cos.unsqueeze(1).to(q.dtype)
        sin = sin.unsqueeze(1).to(q.dtype)

    if _use_npu_kernels() and q.device.type == "npu":
        import torch_npu  # pylint: disable=C0415

        return torch_npu.npu_rotary_mul(q, cos, sin, rotary_mode="half"), torch_npu.npu_rotary_mul(
            k, cos, sin, rotary_mode="half"
        )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
