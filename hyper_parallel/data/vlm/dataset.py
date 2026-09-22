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
"""Build the VLM dataset from a LLaVA-style JSON list."""

import json
import os
from collections.abc import Callable
from typing import Any, Optional, TypeAlias

from torch.utils.data import Dataset

from hyper_parallel.data.constants import IGNORE_INDEX

SampleTransform: TypeAlias = Callable[[Any], Any]


class VLMDataset(Dataset):
    """Load a LLaVA-style JSON list of multimodal conversations.

    Each record is ``{"messages": [...], "images": [...]}`` where ``messages``
    content is either a Qwen3-VL content-list or a string carrying ``<image>`` /
    ``<video>`` placeholders. Image paths are resolved relative to the JSON file.
    """

    def __init__(self, data_path: str, **dataset_options: Any) -> None:
        """Load records from the JSON file and resolve media paths."""
        del dataset_options
        with open(data_path, "r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        root = os.path.dirname(os.path.abspath(data_path))
        for record in self.records:
            self._resolve_record_paths(record, root)

    def _resolve_record_paths(self, record: dict[str, Any], root: str) -> None:
        """Resolve media paths in one record against the JSON directory."""
        for key in ("images", "videos"):
            if key in record:
                record[key] = [self._resolve(root, path) for path in record[key]]
        for message in record.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                for key in ("url", "image", "video"):
                    if key in item and isinstance(item[key], str):
                        item[key] = self._resolve(root, item[key])

    @staticmethod
    def _resolve(root: str, path: str) -> str:
        """Resolve a relative media path against the JSON directory."""
        if isinstance(path, str) and not path.startswith(("http://", "https://", "/", "data:")):
            return os.path.join(root, path)
        return path

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the record at the given index."""
        return self.records[index]


class _TransformDataset(Dataset):
    """Apply one Trainer-built transform after source-specific IO.

    With ``filter_trainable`` (the default) records whose assistant turn was
    truncated away (all labels masked) are dropped up front, so each index maps
    to exactly one sample that carries a non-trivial loss. That filter has to
    run the full transform on every record during construction, which for
    media-heavy data is very expensive (it is a pre-training cost paid per
    rank) -- and for a throughput run it is unwanted, because a record that
    happens to have no trainable label is still perfectly valid work.

    Turning it off keeps every record and transforms lazily on access, so
    construction is O(1) instead of O(dataset). A kept record with no trainable
    label simply contributes nothing to the loss.
    """

    def __init__(self, source: Dataset, transform: Optional[SampleTransform],
                 filter_trainable: bool = True) -> None:
        """Build the index of samples, transforming the source once if filtering.

        Args:
            source: Underlying record dataset.
            transform: Per-record transform applied on access.
            filter_trainable: Drop records whose labels are all ``IGNORE_INDEX``.
                Requires transforming every record up front; disable to keep
                every record and pay the transform only on access.
        """
        self.source = source
        self.transform = transform
        self.filter_trainable = filter_trainable
        if not filter_trainable:
            self.indices = list(range(len(source)))
            if not self.indices:
                raise ValueError("VLM dataset is empty")
            return
        self.indices = []
        for index, record in enumerate(source):
            sample = transform(record) if transform is not None else record
            labels = sample.get("labels")
            if labels is None or self._has_trainable(labels):
                self.indices.append(index)
        if not self.indices:
            raise ValueError(
                "VLM dataset contains no samples with trainable labels after "
                "truncation; raise data_transform.max_seq_len so the assistant "
                "turn survives, or set filter_trainable=false to keep every record"
            )

    @staticmethod
    def _has_trainable(labels: Any) -> bool:
        if hasattr(labels, "ne"):
            return bool(labels.ne(IGNORE_INDEX).any())
        return any(value != IGNORE_INDEX for value in labels)

    def __len__(self) -> int:
        """Return the number of trainable samples."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Return the transformed sample for a trainable index."""
        record = self.source[self.indices[index]]
        return self.transform(record) if self.transform is not None else record


def build_vlm_dataset(
        *,
        data_config: dict[str, Any],
        data_path: Optional[str] = None,
        transform: Optional[SampleTransform] = None,
        tokenizer: Any = None,
        mesh_context: Any = None,
        training_config: Any = None,
        filter_trainable: bool = True,
        **dataset_options: Any,
) -> Any:
    """Build a transform-wrapped VLM dataset from an online source.

    Args:
        data_config: Source options; must contain ``source_type``.
        data_path: Path to the LLaVA-style JSON list.
        transform: VLM sample transform applied to each raw record.
        tokenizer: Tokenizer (accepted for the shared Trainer contract).
        mesh_context: Mesh context (accepted for the shared Trainer contract).
        training_config: Training plan (accepted for the shared Trainer contract).
        filter_trainable: Drop records whose labels end up all ``IGNORE_INDEX``
            (assistant turn truncated away). Requires transforming the whole
            source during construction; set ``false`` for throughput runs to
            keep every record and transform lazily.
        **dataset_options: Reserved source-specific options.

    Returns:
        A transform-wrapped map-style dataset.

    Raises:
        ValueError: If ``source_type`` is unsupported or ``data_path`` is missing.
    """
    del tokenizer, mesh_context, training_config, dataset_options
    if data_config.get("source_type") != "online":
        raise ValueError(f"Unsupported VLM source type: {data_config.get('source_type')!r}")
    if data_path is None:
        raise ValueError("online VLM dataset requires data_path")
    dataset = _TransformDataset(VLMDataset(data_path), transform,
                                filter_trainable=filter_trainable)
    repeat = int(data_config.get("repeat", 1) or 1)
    return _RepeatedDataset(dataset, repeat) if repeat > 1 else dataset


class _RepeatedDataset(Dataset):
    """Repeat a map-style dataset over a multiplied index space.

    The dynamic batch loader defines no ``__len__``, so the trainer falls back to
    "one epoch = ``train_iters`` steps" (``TrainerBase._resolve_train_plan``).  A
    source that holds fewer steps than that exhausts mid-schedule, and because ranks
    run out at slightly different steps the job does not stop -- it deadlocks in the
    next collective (observed: COCO's 157,712 records at ~3k samples/step exhausted
    after 51 of 1800 steps, then an ALLREDUCE dispatch timeout).  Repeating the index
    space keeps the source ahead of the schedule.
    """

    def __init__(self, dataset: Dataset, repeat: int) -> None:
        """Wrap ``dataset`` so it can be read ``repeat`` times over.

        Args:
            dataset: Map-style source dataset.
            repeat: Number of passes to expose; must be positive.
        """
        if repeat < 1:
            raise ValueError(f"repeat must be positive, got {repeat}")
        self.dataset = dataset
        self.repeat = int(repeat)

    def __len__(self) -> int:
        """Return the multiplied length of the wrapped dataset."""
        return len(self.dataset) * self.repeat

    def __getitem__(self, index: int) -> Any:
        """Map an index onto the wrapped dataset, wrapping around each pass."""
        length = len(self.dataset)
        if length <= 0:
            raise IndexError("cannot index an empty dataset")
        return self.dataset[index % length]


__all__ = ["VLMDataset", "build_vlm_dataset"]
