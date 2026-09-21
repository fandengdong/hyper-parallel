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
"""Temporary self-contained VLM batch preparation."""

__all__ = ["VLMBatchProcessor", "VLMGetBatch", "build_vlm_get_batch"]

from collections.abc import Mapping
from typing import Any

import torch


_MODEL_INPUT_FIELDS = {
    "input_ids",
    "labels",
    "attention_mask",
    "position_ids",
    "text_position_ids",
    "router_attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "audio_features",
    "audio_attention_mask",
    "audio_feature_lengths",
    "image_mask",
    "video_mask",
    "audio_mask",
    "image_grid_hw",
    "image_grid_thw",
    "video_grid_thw",
    "video_timestamp",
}
_LOSS_INPUT_FIELDS = {"labels", "loss_mask", "stream_loss_mask"}


class VLMBatchProcessor:
    """Normalize and classify one VLM batch without shared LLM adapters."""

    @staticmethod
    def normalize_source_batch(source_batch: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one collated batch into the temporary VLM contract."""
        batch = dict(source_batch)
        if "input_ids" not in batch:
            raise ValueError("VLM batch must contain 'input_ids'")
        if "labels" not in batch:
            raise ValueError("VLM batch must contain 'labels'")
        if "loss_mask" not in batch:
            batch["loss_mask"] = batch["labels"] >= 0
        return batch

    @staticmethod
    def prepare_batch(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split device-resident VLM fields into model and loss inputs."""
        model_inputs = {field: value for field, value in batch.items() if field in _MODEL_INPUT_FIELDS}
        loss_inputs = {field: value for field, value in batch.items() if field in _LOSS_INPUT_FIELDS}
        return model_inputs, loss_inputs


class VLMGetBatch:
    """Prepare VLM batches for the temporary TP=PP=1 training path.

    Context parallelism is supported with the "replicated vision, sharded
    text" split: the media fields (``input_ids``/``attention_mask``/
    ``pixel_values``/``image_grid_thw``) stay complete on every CP rank so the
    vision tower and the full-sequence media scatter stay consistent, while the
    loss targets (``labels``/``loss_mask``) are sliced to this rank's sequence
    window to match the text tower's sharded output. The matching input slice
    and offset mask are installed by
    ``models/kimi_k25/adapter/distributed/context_parallel.py``.
    """

    def __init__(self, *, mesh_context: Any, device: Any, pp_shared_data: bool = False) -> None:
        """Validate the temporary VLM parallel boundary and store the device.

        Args:
            mesh_context: Trainer mesh exposing TP, CP, and PP sizes.
            device: Destination model device.
            pp_shared_data: Whether pipeline stages share the source batch.

        Raises:
            NotImplementedError: If pipeline parallelism or pipeline batch sharing is enabled.
        """
        self.cp_size = int(getattr(mesh_context, "cp_size", 1))
        self.cp_rank = int(getattr(mesh_context, "cp_rank", 0))
        pp_size = int(getattr(mesh_context, "pp_size", 1))
        if pp_size != 1:
            raise NotImplementedError(
                "The temporary VLM batch path requires PP=1, but got "
                f"pp_size={pp_size}"
            )
        if pp_shared_data:
            raise NotImplementedError("The temporary VLM batch path does not support pp_shared_data")
        self.device = device
        self.processor = VLMBatchProcessor()

    def _shard_targets_for_cp(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Slice loss targets to this CP rank's sequence window."""
        if self.cp_size <= 1:
            return batch
        labels = batch.get("labels")
        if labels is None:
            return batch
        seq_len = labels.shape[-1]
        if seq_len % self.cp_size:
            raise ValueError(
                "VLM CP labels require a sequence length divisible by cp_size: "
                f"seq_len={seq_len}, cp_size={self.cp_size} "
                "(the collator pads to data_transform.max_seq_len)"
            )
        local_len = seq_len // self.cp_size
        start = self.cp_rank * local_len
        window = slice(start, start + local_len)
        batch["labels"] = labels[..., window]
        for field in ("loss_mask", "stream_loss_mask"):
            value = batch.get(field)
            if torch.is_tensor(value) and value.shape[-1] == seq_len:
                batch[field] = value[..., window]
        return batch

    def __call__(
            self,
            data_iterator: Any,
            *,
            external_batch: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read, normalize, transfer, and classify one VLM batch.

        Args:
            data_iterator: Iterator yielding collated VLM batches.
            external_batch: Optional batch supplied without advancing the iterator.

        Returns:
            Model inputs and loss inputs on the configured device.
        """
        source_batch = external_batch if external_batch is not None else next(data_iterator)
        if not isinstance(source_batch, Mapping):
            raise ValueError("VLM DataLoader must yield a mapping batch")
        normalized_batch = self.processor.normalize_source_batch(source_batch)
        normalized_batch = self._shard_targets_for_cp(normalized_batch)
        device_batch = {
            field: value.to(self.device, non_blocking=True)
            if torch.is_tensor(value) else value
            for field, value in normalized_batch.items()
        }
        return self.processor.prepare_batch(device_batch)


def build_vlm_get_batch(
        *,
        mesh_context: Any,
        device: Any,
        pp_shared_data: bool = False,
) -> VLMGetBatch:
    """Build the temporary self-contained VLM batch adapter.

    Args:
        mesh_context: Trainer mesh used to validate VLM parallel sizes.
        device: Destination model device.
        pp_shared_data: Reserved pipeline batch-sharing option.

    Returns:
        Callable VLM batch adapter.
    """
    return VLMGetBatch(mesh_context=mesh_context, device=device, pp_shared_data=pp_shared_data)
