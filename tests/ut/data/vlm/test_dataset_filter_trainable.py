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
"""``filter_trainable`` on the temporary VLM dataset wrapper.

By default the wrapper drops records whose labels are all ``IGNORE_INDEX``
(assistant turn truncated away). Finding those requires transforming every
record during construction, which is a per-rank pre-training cost that scales
with the dataset -- prohibitive for media-heavy data. For a pure throughput run
the filter is also unwanted: a record with no trainable label is still valid
work. ``filter_trainable=False`` keeps every record and transforms lazily.
"""
# pylint: disable=wrong-import-position

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from torch.utils.data import Dataset

from tests.common.mark_utils import arg_mark

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.vlm.dataset import _TransformDataset


class _CountingSource(Dataset):
    """Source dataset that records how many records were transformed."""

    def __init__(self, labels_per_record):
        """Store the label vector each record should transform into."""
        self.labels_per_record = list(labels_per_record)
        self.transformed = []

    def __len__(self):
        """Number of records."""
        return len(self.labels_per_record)

    def __getitem__(self, index):
        """Return a raw record carrying its index.

        Raises:
            IndexError: Out of range, as the ``Dataset`` iteration protocol
                requires -- ``_TransformDataset`` walks the source with
                ``enumerate`` and relies on this to stop.
        """
        if index >= len(self.labels_per_record):
            raise IndexError(index)
        return {"index": index}


def _make_transform(source):
    """Return a transform that counts calls and yields the record's labels."""
    def transform(record):
        source.transformed.append(record["index"])
        return {"labels": source.labels_per_record[record["index"]]}

    return transform


def _labels(*values):
    """Build a label list from 0 (trainable) / -100 (masked) markers."""
    return [0 if value else IGNORE_INDEX for value in values]


class TestFilterTrainable(unittest.TestCase):
    """The wrapper keeps every record when the filter is disabled."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_filter_on_drops_masked_records_and_transforms_eagerly(self):
        """Default behaviour: drop all-masked records, transforming up front."""
        source = _CountingSource([_labels(1, 1), _labels(0, 0), _labels(1, 0)])
        dataset = _TransformDataset(source, _make_transform(source))

        self.assertEqual(list(dataset.indices), [0, 2])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(source.transformed, [0, 1, 2], "every record transformed once")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_filter_off_keeps_every_record_without_transforming_eagerly(self):
        """Disabling the filter makes construction O(1) and keeps all records."""
        source = _CountingSource([_labels(1, 1), _labels(0, 0), _labels(1, 0)])
        dataset = _TransformDataset(source, _make_transform(source),
                                    filter_trainable=False)

        self.assertEqual(list(dataset.indices), [0, 1, 2])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(source.transformed, [], "nothing transformed at construction")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_filter_off_keeps_the_all_masked_record_trainable_path(self):
        """An all-masked record is still returned (its loss contribution is 0)."""
        source = _CountingSource([_labels(0, 0)])
        dataset = _TransformDataset(source, _make_transform(source),
                                    filter_trainable=False)

        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertTrue(all(label == IGNORE_INDEX for label in sample["labels"]))
        self.assertEqual(source.transformed, [0], "transformed exactly once, on access")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_filter_off_does_not_raise_on_a_fully_dropped_dataset(self):
        """A dataset the filter would reject entirely is usable with it off."""
        source = _CountingSource([_labels(0, 0), _labels(0, 0)])

        with self.assertRaisesRegex(ValueError, "no samples with trainable labels"):
            _TransformDataset(source, _make_transform(source))

        dataset = _TransformDataset(source, _make_transform(source),
                                    filter_trainable=False)
        self.assertEqual(len(dataset), 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_actionable_error_mentions_the_switch(self):
        """The failure tells the user how to proceed."""
        source = _CountingSource([_labels(0, 0)])

        with self.assertRaisesRegex(ValueError, "filter_trainable=false"):
            _TransformDataset(source, _make_transform(source))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_dataset_is_rejected(self):
        """An empty source is an error in both modes."""
        for flag in (True, False):
            with self.subTest(filter_trainable=flag):
                with self.assertRaises(ValueError):
                    _TransformDataset(_CountingSource([]), None, filter_trainable=flag)


if __name__ == "__main__":
    unittest.main()
