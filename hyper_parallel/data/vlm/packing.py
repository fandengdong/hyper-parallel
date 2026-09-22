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
"""VeOmni-style packing for VLM samples: fill an 8K window, pad only the remainder.

Why this exists
---------------
The VLM collator can only *pad* a micro-batch up to ``max_seq_len``
(``pad_to_batch: true``).  Real instruction data is short -- COCO2017 LLaVA-instruct
samples average ~175 tokens -- so padding each one to 8192 throws away 97.7% of the compute:
a measured step carried 48,185 valid tokens against 2,097,150 padded slots, MFU 11-15%
where the same model reaches 28% on data that fills the window.

Packing instead concatenates whole samples until the window is full and pads only the tail,
which is what ``build_vlm_collator(packing=True)`` is reserved for -- it currently raises
``NotImplementedError`` because the text-only ``TextPackingCollator`` knows nothing about the
multimodal fields.

Correctness, not just throughput
--------------------------------
Concatenating tokens is only sound if attention stays *block diagonal*: a token must never
attend across the boundary between two packed samples.  The packed batch therefore carries
``cu_seq_lens`` (the cumulative sub-sequence ends, zero-prefixed, int32) and the consumer must
build a block-diagonal mask from it, exactly as the text packing path does.  Packing without
that mask silently trains a different model, so this module's contract is: the returned windows
always describe their boundaries.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

# Sequence-length fields: concatenated along the token axis when packing.
SEQ_FIELDS: tuple[str, ...] = (
    "input_ids",
    "labels",
    "attention_mask",
    "loss_mask",
    "mm_token_type_ids",
)
# Position-like fields: concatenated *and* restarted at zero for every packed sub-sequence,
# otherwise a sample would inherit the previous sample's positions.
POSITION_FIELDS: tuple[str, ...] = ("position_ids", "text_position_ids")
# Modality fields: concatenated along dim 0, in the order the samples are packed, so the
# vision splice keeps matching the image placeholders inside the packed ``input_ids``.
MODAL_FIELDS: tuple[str, ...] = ("pixel_values", "image_grid_thw")

__all__ = [
    "block_diagonal_mask",
    "SEQ_FIELDS",
    "POSITION_FIELDS",
    "MODAL_FIELDS",
    "VlmPackingCollator",
    "pack_vlm_samples",
    "sample_length",
]


def sample_length(sample: Mapping[str, Any]) -> int:
    """Return the token length of one sample.

    Args:
        sample: One transformed sample carrying ``input_ids``.

    Returns:
        The number of tokens in the sample.

    Raises:
        ValueError: If the sample carries no ``input_ids``.
    """
    input_ids = sample.get("input_ids")
    if input_ids is None:
        raise ValueError("a packable sample must carry input_ids")
    return int(input_ids.shape[-1])


def _pad_to(value: torch.Tensor, target: int, fill: int) -> torch.Tensor:
    """Right-pad ``value`` along its last dim to ``target`` with ``fill``."""
    missing = target - value.shape[-1]
    if missing <= 0:
        return value
    return torch.cat((value, value.new_full((*value.shape[:-1], missing), fill)), dim=-1)


def _reset_positions(value: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Restart a position field at zero inside every packed sub-sequence.

    Args:
        value: Concatenated position tensor of length ``sum(lengths)``.
        lengths: Token length of each packed sample, in order.

    Returns:
        The same tensor with each sub-sequence renumbered from zero.
    """
    pieces = []
    cursor = 0
    for length in lengths:
        piece = value[..., cursor:cursor + length].clone()
        piece = piece - piece[..., :1]
        pieces.append(piece)
        cursor += length
    return torch.cat(pieces, dim=-1)


def _pack_window(samples: Sequence[Mapping[str, Any]], lengths: Sequence[int],
                 pad_to_length: int, ignore_index: int) -> dict[str, Any]:
    """Concatenate one window's samples and describe its sub-sequence boundaries."""
    window: dict[str, Any] = {}
    total = sum(lengths)
    for field in SEQ_FIELDS:
        present = [sample[field] for sample in samples if field in sample]
        if not present:
            continue
        # Concat along the token axis on the ORIGINAL shapes: mRoPE position fields are
        # ``[3, S]`` here, and flattening them (as the first attempt did) broke the model with
        # "too many indices for tensor of dimension 1" on all 256 ranks.
        merged = torch.cat(present, dim=-1)
        # Keep the batch dimension: the padded path gets it from ``default_collate``
        # ([B, S] masks), and HF's ``sdpa_mask`` indexes the mask with two indices -- without
        # this it raises "too many indices for tensor of dimension 1" on every rank.
        window[field] = _pad_to(merged, pad_to_length,
                                ignore_index if field == "labels" else 0).unsqueeze(0)
    for field in POSITION_FIELDS:
        present = [sample[field] for sample in samples if field in sample]
        if not present:
            continue
        merged = torch.cat(present, dim=-1)
        window[field] = _pad_to(
            _reset_positions(merged, lengths), pad_to_length, 0).unsqueeze(0)
    for field in MODAL_FIELDS:
        present = [sample[field] for sample in samples if field in sample]
        if present:
            window[field] = torch.cat(present, dim=0)
    # Boundaries: one entry per packed sub-sequence, plus a synthetic final boundary when the
    # tail was padded, so attention metadata covers every physical token (same convention as
    # ``TextPackingCollator``).  Labels on the tail stay IGNORE_INDEX and are not supervised.
    ends = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
    if pad_to_length > total:
        ends = torch.cat((ends, torch.tensor([pad_to_length], dtype=torch.int64)))
    window["cu_seq_lens"] = torch.cat(
        (torch.zeros(1, dtype=torch.int64), ends)
    ).to(torch.int32)
    return window


def pack_vlm_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    max_seq_len: int,
    pad_to_length: Optional[int] = None,
    ignore_index: int = -100,
) -> list[dict[str, Any]]:
    """Greedily pack samples into windows of at most ``max_seq_len`` tokens.

    Whole samples are appended while they fit, so no sample is ever split across windows and
    only the final window's remainder is padded -- pack short samples to fill the window, pad
    the little that is left over.

    Args:
        samples: Transformed samples, each with ``input_ids`` (and optional labels/masks and
            modality fields).
        max_seq_len: Window budget in tokens (e.g. 8192).
        pad_to_length: Length to pad each window to; defaults to ``max_seq_len``.
        ignore_index: Label value for padded positions.

    Returns:
        One dict per packed window, each carrying ``cu_seq_lens``.

    Raises:
        ValueError: If ``max_seq_len`` is not positive or a sample is longer than the window.
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    target = int(pad_to_length or max_seq_len)

    windows: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    lengths: list[int] = []
    used = 0
    for sample in samples:
        length = sample_length(sample)
        if length > max_seq_len:
            raise ValueError(
                f"sample of {length} tokens exceeds the {max_seq_len}-token window; "
                "truncate it in the transform instead of splitting it here"
            )
        if current and used + length > max_seq_len:
            windows.append(_pack_window(current, lengths, target, ignore_index))
            current, lengths, used = [], [], 0
        current.append(sample)
        lengths.append(length)
        used += length
    if current:
        windows.append(_pack_window(current, lengths, target, ignore_index))
    return windows


class VlmPackingCollator:
    """Pack one step's selected samples into a single ``max_seq_len`` window.

    The contract is "one training step = one packed 8K window": the dynamic batch sampler
    (``DynamicBatchDataLoader`` with ``batch_size=1`` and ``max_seq_len=8192``) already selects
    samples up to that token budget, so this collator only has to concatenate them and pad the
    remainder.  More than one window means the sampler's budget and this window disagree -- a
    configuration error, so it raises loudly instead of silently dropping samples.

    Args:
        max_seq_len: Window budget in tokens (e.g. 8192).
        pad_to_length: Length to pad the window to; defaults to ``max_seq_len``.
        ignore_index: Label value for padded positions.
    """

    def __init__(self, *, max_seq_len: int, pad_to_length: Optional[int] = None,
                 ignore_index: int = -100) -> None:
        """Validate the window size and remember the padding contract."""
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len!r}")
        self.max_seq_len = max_seq_len
        self.pad_to_length = pad_to_length
        self.ignore_index = ignore_index

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return the packed window for one step.

        Args:
            samples: Samples the batch sampler selected for this step.

        Returns:
            One packed batch carrying ``cu_seq_lens``.

        Raises:
            ValueError: If the selection spans more than one window (sampler/config mismatch).
        """
        windows = pack_vlm_samples(
            samples,
            max_seq_len=self.max_seq_len,
            pad_to_length=self.pad_to_length,
            ignore_index=self.ignore_index,
        )
        if not windows:
            raise ValueError("no samples to pack")
        if len(windows) > 1:
            raise ValueError(
                f"the sampler selected {len(windows)} windows of {self.max_seq_len} tokens; "
                "one step must be exactly one packed window -- lower the sampler's token "
                "budget (batch_size * max_seq_len) or raise max_seq_len"
            )
        window = windows[0]
        # Ship the boundary enforcement with the batch: the SDPA path has no idea the window was
        # packed, and without this mask tokens would attend across samples.
        window["block_diagonal_mask"] = block_diagonal_mask(window["cu_seq_lens"])
        return window


def block_diagonal_mask(cu_seq_lens: torch.Tensor, *, device: Any = None) -> torch.Tensor:
    """Return the ``[S, S]`` mask that keeps packed attention block diagonal.

    Packing concatenates whole samples, so a plain causal mask would let a token attend to every
    *earlier* token in the window -- including tokens from a different sample.  That is not a
    detail: it silently trains a different model.  This mask is the boundary enforcement for the
    SDPA path (``attn_implementation: sdpa``): True inside the sub-sequence a position belongs
    to, False everywhere else, so the caller ANDs it with its causal/padding mask.

    The cost is O(S^2) booleans (8192^2 = 67 MB for the 8K window) and the attention kernel still
    visits every position; this buys correctness, not FLOPs.  A varlen kernel
    (``components/functional/npu_fusion_attention.py``, which already consumes ``cu_seq_lens``)
    is the way to also skip the masked blocks.

    Args:
        cu_seq_lens: Cumulative sub-sequence ends, zero-prefixed (as packed batches carry).
        device: Optional device for the returned mask.

    Returns:
        A bool tensor of shape ``[S, S]`` where ``S = cu_seq_lens[-1]``.

    Raises:
        ValueError: If the boundaries are not strictly increasing from zero.
    """
    ends = cu_seq_lens.to(torch.int64)
    if ends.numel() < 2 or int(ends[0]) != 0:
        raise ValueError(f"cu_seq_lens must start at 0 and hold >=2 entries, got {ends.tolist()}")
    if bool((ends[1:] <= ends[:-1]).any()):
        raise ValueError(f"cu_seq_lens must be strictly increasing, got {ends.tolist()}")
    total = int(ends[-1])
    positions = torch.arange(total, device=device or ends.device)
    # bucketize maps every position to its sub-sequence index.  ``right=True`` is required:
    # with the default the position *equal* to a boundary stays in the left bucket, which made
    # the first packed sample one token too long (measured: [0,2,5] produced a 3-token block).
    segment = torch.bucketize(positions, ends[1:-1], right=True)
    return segment[:, None] == segment[None, :]
