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

    With ``pad_to_batch`` the collator pads every sequence field up to the
    longest sample in the micro-batch (rounded up to ``pad_granularity``)
    instead of requiring the transform to pre-pad everything to
    ``max_seq_len``. The transform must be built with a matching
    ``pad_granularity``, otherwise its fixed ``max_seq_len`` padding already
    made all samples equal length and this is a no-op.
    """

    def __init__(self, *, pad_to_batch: bool = False,
                 pad_granularity: int = 1) -> None:
        """Store the batch-padding policy.

        Args:
            pad_to_batch: Pad the micro-batch to its longest sample.
            pad_granularity: Round the padded length up to a multiple of this
                value (keep it a multiple of the parallel sequence alignment).

        Raises:
            ValueError: If ``pad_granularity`` is not a positive integer.
        """
        if isinstance(pad_granularity, bool) or not isinstance(pad_granularity, int) \
                or pad_granularity <= 0:
            raise ValueError(
                f"pad_granularity must be a positive integer, got {pad_granularity!r}"
            )
        self.pad_to_batch = pad_to_batch
        self.pad_granularity = pad_granularity

    def _batch_target_len(self, samples: list[dict[str, Any]]) -> int:
        """Return the sequence length the whole micro-batch pads up to."""
        longest = max(int(sample["input_ids"].shape[0]) for sample in samples)
        granularity = self.pad_granularity
        return -(-longest // granularity) * granularity

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
) -> VLMCollator:
    """Build the VLM micro-batch collator.

    Args:
        packing: Reserved switch for VeOmni-style text packing.
        pad_token_id: Reserved padding value for text input IDs.
        ignore_index: Reserved label value excluded from loss computation.
        pad_to_length: Reserved packed text sequence length.
        pad_to_batch: Pad each micro-batch to its own longest sample. Pair with
            a data transform built with a matching ``pad_granularity``.
        pad_granularity: Round the batch-padded length up to this multiple.

    Returns:
        A collator producing one VLM micro-batch dictionary.
    """
    if packing:
        raise NotImplementedError("The temporary VLM collator does not support packing")
    if pad_token_id != 0 or ignore_index != IGNORE_INDEX or pad_to_length is not None:
        raise NotImplementedError("The temporary VLM collator does not support custom text padding")
    return VLMCollator(pad_to_batch=pad_to_batch, pad_granularity=pad_granularity)


__all__ = ["VLMCollator", "build_vlm_collator"]
