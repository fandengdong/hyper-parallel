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
"""Build the VLM micro-batch collator."""

from typing import Any, Optional

import torch
from torch.utils.data import default_collate

from hyper_parallel.data.constants import IGNORE_INDEX

_TEXT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "loss_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
    "mm_token_type_ids",
}


class VLMCollator:
    """Collate text and modality fields into one VLM micro-batch.

    Text fields use default collation. Modality fields such as
    ``pixel_values`` and ``image_grid_thw`` are concatenated along dim 0 so
    variable-length images batch correctly. This temporary implementation does
    not depend on the LLM batching pipeline.

    With ``pad_to_batch`` the collator pads every sequence field up to
    ``max_seq_len`` (rounded up to ``pad_granularity``) instead of requiring the
    transform to pre-pad everything to that length. The transform must be built
    with the same ``max_seq_len`` and a matching ``pad_granularity``, otherwise
    its own padding already made all samples equal length. A fixed padded length
    keeps the shape of every step identical; it also means ``pad_granularity``
    saves no compute, because each micro-batch still pays ``max_seq_len``.
    """

    def __init__(self, *, pad_to_batch: bool = False,
                 pad_granularity: int = 1,
                 max_seq_len: Optional[int] = None) -> None:
        """Store the batch-padding policy.

        Args:
            pad_to_batch: Pad the micro-batch up to ``max_seq_len``.
            pad_granularity: Round the padded length up to a multiple of this
                value (keep it a multiple of the parallel sequence alignment).
            max_seq_len: Hard cap used as the padded length. Required when
                ``pad_to_batch`` is set.

        Raises:
            ValueError: If ``pad_granularity`` or ``max_seq_len`` is not a
                positive integer, or if ``pad_to_batch`` is enabled without
                ``max_seq_len``.
        """
        if isinstance(pad_granularity, bool) or not isinstance(pad_granularity, int) \
                or pad_granularity <= 0:
            raise ValueError(
                f"pad_granularity must be a positive integer, got {pad_granularity!r}"
            )
        if max_seq_len is not None and (
                isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int)
                or max_seq_len <= 0):
            raise ValueError(
                f"max_seq_len must be a positive integer, got {max_seq_len!r}"
            )
        if pad_to_batch and max_seq_len is None:
            raise ValueError(
                "max_seq_len is required when pad_to_batch is enabled; set "
                "collate_fn.max_seq_len to the transform's max_seq_len"
            )
        self.pad_to_batch = pad_to_batch
        self.pad_granularity = pad_granularity
        self.max_seq_len = max_seq_len

    def _batch_target_len(self, samples: list[dict[str, Any]]) -> int:
        """Return the sequence length the whole micro-batch pads up to.

        The target is ``max_seq_len`` rounded up to ``pad_granularity``, not the
        longest member, so every micro-batch in a run has the same shape.

        Raises:
            ValueError: If a sample is longer than the target, which means the
                transform cap and this collator disagree.
        """
        granularity = self.pad_granularity
        target_len = -(-self.max_seq_len // granularity) * granularity
        longest = max(int(sample["input_ids"].shape[0]) for sample in samples)
        if longest > target_len:
            raise ValueError(
                f"sample length {longest} exceeds the padded length {target_len}; "
                "build the transform with the same max_seq_len"
            )
        return target_len

    @staticmethod
    def _pad_sequence_field(value: torch.Tensor, seq_len: int,
                            target_len: int, pad_value: Any) -> torch.Tensor:
        """Pad one sequence-aligned field to ``target_len`` along its seq dim."""
        if value.ndim == 1 and value.shape[0] == seq_len:
            missing = target_len - value.shape[0]
            if missing <= 0:
                return value
            return torch.cat([
                value,
                torch.full((missing,), pad_value, dtype=value.dtype),
            ])
        if value.ndim == 2 and value.shape[-1] == seq_len:
            missing = target_len - value.shape[-1]
            if missing <= 0:
                return value
            return torch.cat([
                value,
                torch.full((value.shape[0], missing), pad_value, dtype=value.dtype),
            ], dim=-1)
        return value

    def __call__(self, samples: Any) -> dict[str, Any]:
        """Collate one micro-batch of VLM samples."""
        if self.pad_to_batch:
            target_len = self._batch_target_len(samples)
            samples = [
                {
                    field: self._pad_sequence_field(
                        value,
                        int(sample["input_ids"].shape[0]),
                        target_len,
                        IGNORE_INDEX if field == "labels" else 0,
                    ) if isinstance(value, torch.Tensor) else value
                    for field, value in sample.items()
                }
                for sample in samples
            ]

        text_samples = [
            {field: value for field, value in sample.items() if field in _TEXT_FIELDS}
            for sample in samples
        ]
        modal_samples = [
            {field: value for field, value in sample.items() if field not in _TEXT_FIELDS}
            for sample in samples
        ]

        batch = default_collate(text_samples)
        if any(modal_samples):
            for field in {field for sample in modal_samples for field in sample}:
                values = [sample[field] for sample in modal_samples if field in sample]
                batch[field] = (
                    torch.cat(values, dim=0)
                    if isinstance(values[0], torch.Tensor)
                    else default_collate(values)
                )
        return batch


def build_vlm_collator(
        *,
        packing: bool = False,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
        pad_to_length: Optional[int] = None,
        pad_to_batch: bool = False,
        pad_granularity: int = 1,
        max_seq_len: Optional[int] = None,
) -> VLMCollator:
    """Build the VLM micro-batch collator.

    Args:
        packing: Reserved switch for VeOmni-style text packing.
        pad_token_id: Reserved padding value for text input IDs.
        ignore_index: Reserved label value excluded from loss computation.
        pad_to_length: Reserved packed text sequence length. The fixed padded
            length used outside packing is ``max_seq_len``.
        pad_to_batch: Pad each micro-batch up to ``max_seq_len``. Pair with a
            data transform built with a matching ``pad_granularity`` and the same
            ``max_seq_len``.
        pad_granularity: Round the batch-padded length up to this multiple.
        max_seq_len: Hard cap used as the padded length. Required when
            ``pad_to_batch`` is set.

    Returns:
        A collator producing one VLM micro-batch dictionary.

    Raises:
        NotImplementedError: If a reserved option is requested.
        ValueError: If ``pad_to_batch`` is set without ``max_seq_len``, or a
            padding option is invalid.
    """
    if packing:
        # One step = one packed window (see ``data/vlm/packing.py``): the dynamic batch sampler
        # selects samples up to the token budget, then this packs them and pads the remainder.
        from hyper_parallel.data.vlm.packing import (  # pylint: disable=import-outside-toplevel
            VlmPackingCollator,
        )

        return VlmPackingCollator(
            max_seq_len=max_seq_len or pad_to_length,
            pad_to_length=pad_to_length,
            ignore_index=ignore_index,
        )
    if pad_token_id != 0 or ignore_index != IGNORE_INDEX or pad_to_length is not None:
        raise NotImplementedError("The temporary VLM collator does not support custom text padding")
    return VLMCollator(pad_to_batch=pad_to_batch, pad_granularity=pad_granularity,
                       max_seq_len=max_seq_len)


__all__ = ["VLMCollator", "build_vlm_collator"]
