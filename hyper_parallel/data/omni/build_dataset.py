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
"""Compose an Online source with the configured Omni transform lifecycle."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Callable, Optional

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import Dataset  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY, ONLINE_SPLIT_COUNT
from hyper_parallel.data.omni.omni_transform import (
    OmniDataTransform,
    _OmniTransformStrategy,
)
from hyper_parallel.data.online import (
    MappingTransformDataset,
    OnlineDataPath,
    build_online_mapping_source,
)

SampleTransform = Callable[[Any], Any]


def build_online_omni_mapping_dataset(
    *,
    data_config: Mapping[str, Any],
    data_path: OnlineDataPath | None = None,
    transform: OmniDataTransform | None = None,
    training_config: Any = None,
) -> Any:
    """Build an Online Mapping source and apply its Omni transform lazily.

    The optional ``data_config`` keys below are applied after the transform
    lifecycle and preserve the historical VLM dataset-builder options:

    * ``repeat``: expose the source ``repeat`` times over a multiplied index
      space, so a short source cannot exhaust in the middle of a long schedule.
    * ``shuffle``: visit the source in a fixed random order (``torch.randperm``)
      instead of file order.
    * ``seed``: permutation seed for ``shuffle`` (default ``1234``).
    * ``filter_trainable``: drop records whose encoded labels are entirely
      ``IGNORE_INDEX`` (the assistant turn was truncated away); requires
      transforming every record during construction, so it defaults to ``False``
      to keep media-heavy sources lazy.

    Args:
        data_config: Online Mapping source options.
        data_path: Optional local source path, ordered paths, or pre-split
            train/valid/test path mapping.
        transform: Omni sample transform selected by the Trainer.
        training_config: Training plan providing the random seed.

    Returns:
        A transformed Online Mapping Dataset or train-valid-test tuple.

    Raises:
        TypeError: If transform is not an OmniDataTransform.
        ValueError: If no Omni transform is configured.
    """
    if transform is None:
        raise ValueError("Online Omni Dataset requires a data_transform")
    if not isinstance(transform, OmniDataTransform):
        raise TypeError("Online Omni Dataset transform must be an OmniDataTransform")

    dataset_config = dict(data_config)
    training_seed = getattr(training_config, "seed", None)
    dataset_config["random_seed"] = 42 if training_seed is None else int(training_seed)

    sample_filter = transform.is_valid_sample
    source_dataset = build_online_mapping_source(
        data_path=data_path,
        data_config=dataset_config,
        sample_filter=sample_filter,
    )
    source_path = _resolve_local_source_path(data_path)
    if source_path is not None:
        source_dataset = _SourcePathDataset.attach(source_dataset, source_path)
    transformed_dataset = _OmniMappingDataset.apply(source_dataset, transform)
    dataset_options = _DatasetOptions.from_config(data_config)
    return dataset_options.apply(transformed_dataset)


def _resolve_local_source_path(data_path: OnlineDataPath | None) -> str | None:
    """Return the absolute path of one local source file, or ``None``."""
    if isinstance(data_path, str) and os.path.isfile(data_path):
        return os.path.abspath(data_path)
    return None


class _SourcePathDataset:
    """Attach the source file path to records that do not already carry it.

    ``_JsonlMappingSource`` sets ``ONLINE_SOURCE_PATH_KEY`` so the Omni transform can
    resolve media paths relative to the source. Online sources loaded through the
    ``datasets`` json builder (a JSON *array* ``.json`` file) do not set it, which
    would leave relative ``images``/content paths unresolved. Annotating the record
    here restores the resolution our VLM dataset performed while loading.
    """

    def __init__(self, dataset: Any, source_path: str) -> None:
        """Store the wrapped source and the file path it was loaded from."""
        self.dataset = dataset
        self.source_path = source_path

    @classmethod
    def attach(cls, dataset: Any, source_path: str) -> Any:
        """Wrap one source or each available train-valid-test split."""
        if isinstance(dataset, tuple) and len(dataset) == ONLINE_SPLIT_COUNT:
            split_datasets = []
            for split_dataset in dataset:
                split_datasets.append(cls._attach_one(split_dataset, source_path))
            source_splits = (split_datasets[0], split_datasets[1], split_datasets[2])
            return source_splits

        return cls._attach_one(dataset, source_path)

    @classmethod
    def _attach_one(cls, dataset: Any, source_path: str) -> Any:
        """Wrap one source, or return it unchanged when absent."""
        if dataset is None:
            return None
        return cls(dataset, source_path)

    def __len__(self) -> int:
        """Return the wrapped source length."""
        dataset_length = len(self.dataset)
        return dataset_length

    def __getitem__(self, index: int) -> Any:
        """Return one record annotated with its source file path."""
        record = self.dataset[index]
        if not isinstance(record, Mapping) or ONLINE_SOURCE_PATH_KEY in record:
            return record

        annotated_record = dict(record)
        annotated_record[ONLINE_SOURCE_PATH_KEY] = self.source_path
        return annotated_record


class _DatasetOptions:
    """Post-transform dataset options shared by every split."""

    def __init__(
            self,
            *,
            repeat: int,
            shuffle: bool,
            seed: int,
            filter_trainable: bool,
    ) -> None:
        """Store the validated post-transform dataset options."""
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed
        self.filter_trainable = filter_trainable

    @classmethod
    def from_config(cls, data_config: Mapping[str, Any]) -> "_DatasetOptions":
        """Read the optional dataset options from one data configuration."""
        dataset_options = cls(
            repeat=int(data_config.get("repeat", 1) or 1),
            shuffle=bool(data_config.get("shuffle", False)),
            seed=int(data_config.get("seed", 1234)),
            filter_trainable=bool(data_config.get("filter_trainable", False)),
        )
        return dataset_options

    def apply(self, dataset: Any) -> Any:
        """Wrap one dataset, or each available split, with the configured options."""
        if isinstance(dataset, tuple) and len(dataset) == ONLINE_SPLIT_COUNT:
            transformed_splits = []
            for split_dataset in dataset:
                transformed_splits.append(self._apply_one(split_dataset))
            dataset_splits = (
                transformed_splits[0],
                transformed_splits[1],
                transformed_splits[2],
            )
            return dataset_splits

        transformed_dataset = self._apply_one(dataset)
        return transformed_dataset

    def _apply_one(self, dataset: Any) -> Any:
        """Wrap one Mapping dataset with trainable filtering and repetition."""
        if dataset is None:
            return None

        transformed_dataset = dataset
        if self.filter_trainable:
            transformed_dataset = _TransformDataset(transformed_dataset, None, filter_trainable=True)
        if self.repeat > 1 or self.shuffle:
            transformed_dataset = _RepeatedDataset(
                transformed_dataset,
                self.repeat,
                shuffle=self.shuffle,
                seed=self.seed,
            )
        return transformed_dataset


class _TransformDataset(Dataset):
    """Apply one transform after source-specific IO.

    With ``filter_trainable`` records whose assistant turn was truncated away
    (all labels masked) are dropped up front, so each index maps to exactly one
    sample that carries a non-trivial loss. That filter has to run the full
    transform on every record during construction, which for media-heavy data is
    very expensive (a pre-training cost paid per rank) -- and for a throughput run
    it is unwanted, because a record that happens to have no trainable label is
    still perfectly valid work.

    Turning it off keeps every record and transforms lazily on access, so
    construction is O(1) instead of O(dataset). A kept record with no trainable
    label simply contributes nothing to the loss.
    """

    def __init__(self, source: Dataset, transform: Optional[SampleTransform],
                 filter_trainable: bool = True) -> None:
        """Build the index of samples, transforming the source once if filtering.

        Args:
            source: Underlying record dataset.
            transform: Per-record transform applied on access, or ``None`` when
                the source already yields encoded model samples.
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
                raise ValueError("Omni dataset is empty")
            return
        self.indices = []
        for index, record in enumerate(source):
            sample = transform(record) if transform is not None else record
            labels = sample.get("labels")
            if labels is None or self._has_trainable(labels):
                self.indices.append(index)
        if not self.indices:
            raise ValueError(
                "Omni dataset contains no samples with trainable labels after "
                "truncation; raise data_transform.max_seq_len so the assistant "
                "turn survives, or set filter_trainable=false to keep every record"
            )

    @staticmethod
    def _has_trainable(labels: Any) -> bool:
        """Return whether a label vector holds at least one supervised token."""
        if hasattr(labels, "ne"):
            return bool(labels.ne(IGNORE_INDEX).any())
        return any(value != IGNORE_INDEX for value in labels)

    def __len__(self) -> int:
        """Return the number of retained samples."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        """Return the transformed sample for a retained index."""
        record = self.source[self.indices[index]]
        return self.transform(record) if self.transform is not None else record

    def __getattr__(self, name: str) -> Any:
        """Forward source-owned interface attributes to the wrapped dataset.

        The Omni packing loader drives the deferred encoding lifecycle through the
        dataset object (``encode_selected_sample`` / ``encode_batch`` and an
        optional ``packing_selector``). Filtering wraps the source, so without this
        delegation packing combined with ``filter_trainable`` fails with
        ``OmniPackingLoader requires encode_selected_sample() and encode_batch()``.
        ``__getattr__`` only runs when normal lookup fails, so it cannot shadow the
        wrapper's own members.
        """
        if name == "source":
            raise AttributeError(name)
        return getattr(self.source, name)


class _RepeatedDataset(Dataset):
    """Repeat a map-style dataset over a multiplied index space.

    A source that holds fewer steps than the schedule exhausts mid-schedule, and
    because ranks run out at slightly different steps the job does not stop -- it
    deadlocks in the next collective. Repeating the index space keeps the source
    ahead of the schedule. ``shuffle`` gives the source a fixed random order,
    because dynamic batching reads the source sequentially and would otherwise
    track the file's ordering.
    """

    def __init__(self, dataset: Dataset, repeat: int, shuffle: bool = False, seed: int = 1234) -> None:
        """Wrap ``dataset`` so it can be read ``repeat`` times over.

        Args:
            dataset: Map-style source dataset.
            repeat: Number of passes to expose; must be positive.
            shuffle: Visit the source in a fixed random order instead of file
                order.
            seed: Permutation seed; fixed so a run stays reproducible.

        Raises:
            ValueError: If ``repeat`` is not positive.
        """
        if repeat < 1:
            raise ValueError(f"repeat must be positive, got {repeat}")
        self.dataset = dataset
        self.repeat = int(repeat)
        self.shuffle = bool(shuffle)
        self.perm = None
        if self.shuffle:
            length = len(dataset)
            generator = torch.Generator().manual_seed(int(seed))
            self.perm = torch.randperm(length, generator=generator).tolist()

    def __len__(self) -> int:
        """Return the multiplied length of the wrapped dataset."""
        return len(self.dataset) * self.repeat

    def __getitem__(self, index: int) -> Any:
        """Map an index onto the wrapped dataset, wrapping around each pass."""
        length = len(self.dataset)
        if length <= 0:
            raise IndexError("cannot index an empty dataset")
        position = index % length
        if self.perm is not None:
            position = self.perm[position]
        return self.dataset[position]

    def __getattr__(self, name: str) -> Any:
        """Forward source-owned interface attributes to the wrapped dataset.

        Harness note: repetition and ``shuffle`` are implemented here, but the Omni
        packing loader still has to reach the deferred encoding hooks
        (``encode_selected_sample`` / ``encode_batch``) and ``packing_selector`` on
        the wrapped ``_OmniMappingDataset``. Hiding them here made packing combined
        with ``shuffle``/``repeat`` fail with
        ``TypeError: OmniPackingLoader requires encode_selected_sample() and
        encode_batch()`` on every rank. ``__getattr__`` is only consulted when
        normal lookup fails, so the wrapper's own members still win.
        """
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class _OmniMappingDataset(MappingTransformDataset):
    """Retain the Omni encoding lifecycle around one Mapping source."""

    def __init__(self, source_dataset: Any, transform: OmniDataTransform) -> None:
        """Build the pre-selection strategy and retain post-selection hooks."""
        self.data_transform = transform
        self._transform_strategy = _OmniTransformStrategy.from_transform(transform)
        super().__init__(source_dataset, self._transform_strategy.prepare_samples)

    @classmethod
    def apply(cls, source_dataset: Any, transform: OmniDataTransform) -> Any:
        """Wrap one Mapping source or each available train-valid-test split."""
        if isinstance(source_dataset, tuple) and len(source_dataset) == ONLINE_SPLIT_COUNT:
            transformed_splits = []
            for split_dataset in source_dataset:
                transformed_dataset = None
                if split_dataset is not None:
                    transformed_dataset = cls(split_dataset, transform)
                transformed_splits.append(transformed_dataset)

            mapping_splits = (
                transformed_splits[0],
                transformed_splits[1],
                transformed_splits[2],
            )
            return mapping_splits

        if source_dataset is None:
            return None

        mapping_dataset = cls(source_dataset, transform)
        return mapping_dataset

    def encode_selected_sample(self, sample: Any) -> Any:
        """Finish encoding one sample after packing selection."""
        encoded_sample = self._transform_strategy.encode_selected_sample(sample)
        return encoded_sample

    def encode_batch(self, batch: Any) -> Any:
        """Apply the transform's final batch encoding hook."""
        encoded_batch = self.data_transform.encode_batch(batch)
        return encoded_batch


__all__ = ["build_online_omni_mapping_dataset"]
