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
"""``repeat`` on the temporary VLM dataset wrapper.

The dynamic batch loader defines no ``__len__``, so the trainer falls back to
"one epoch = ``train_iters`` steps".  A source that holds fewer steps than that
exhausts mid-schedule, and because ranks run out at slightly different steps the
job does not stop -- it deadlocks in the next collective.  ``data_config.repeat``
multiplies the index space so the source stays ahead of the schedule.

These tests pin the wrapper: it multiplies ``__len__``, serves the source's
records in order on every pass, is a pure index remap (the source object is not
copied or re-instantiated), rejects a non-positive repeat, and refuses to index
an empty source rather than looping.
"""
# pylint: disable=wrong-import-position

import json
import os
import tempfile
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from torch.utils.data import Dataset

from tests.common.mark_utils import arg_mark

from hyper_parallel.data.vlm.dataset import _RepeatedDataset, build_vlm_dataset

_CPU_MARKS = dict(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
                  card_mark="allcards", essential_mark="essential")


class _CountingSource(Dataset):
    """Map-style source that names each record and rejects out-of-range reads."""

    def __init__(self, count: int) -> None:
        """Expose ``count`` records."""
        self.count = count

    def __len__(self) -> int:
        """Number of records in the source."""
        return self.count

    def __getitem__(self, index: int) -> str:
        """Return the record's name.

        Raises:
            IndexError: Out of range, as the ``Dataset`` protocol requires.
        """
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return f"record-{index}"


class TestRepeatedDataset(unittest.TestCase):
    """The wrapper remaps indices; it never copies the source."""

    @arg_mark(**_CPU_MARKS)
    def test_length_is_multiplied(self):
        """Feature: repeat wrapper length.

        Description: Wrap a 3-record source with repeat=4.
        Expectation: The wrapper exposes 12 records.
        """
        self.assertEqual(len(_RepeatedDataset(_CountingSource(3), 4)), 12)

    @arg_mark(**_CPU_MARKS)
    def test_repeat_one_is_the_source(self):
        """Feature: repeat wrapper default.

        Description: Wrap with repeat=1, which ``build_vlm_dataset`` skips entirely.
        Expectation: Length and every record match the source.
        """
        source = _CountingSource(3)
        wrapped = _RepeatedDataset(source, 1)
        self.assertEqual(len(wrapped), len(source))
        self.assertEqual([wrapped[i] for i in range(3)], [source[i] for i in range(3)])

    @arg_mark(**_CPU_MARKS)
    def test_every_pass_serves_the_source_in_order(self):
        """Feature: repeat wrapper ordering.

        Description: Read the whole wrapped index space of a 3-record source with
            repeat=3.
        Expectation: The source's records come back in order on every pass, so a
            repeated epoch differs from the first only by which step it is.
        """
        wrapped = _RepeatedDataset(_CountingSource(3), 3)
        self.assertEqual(
            [wrapped[i] for i in range(9)],
            ["record-0", "record-1", "record-2"] * 3,
        )

    @arg_mark(**_CPU_MARKS)
    def test_wrapper_keeps_the_source_object(self):
        """Feature: repeat wrapper is a pure remap.

        Description: Inspect the wrapped source after construction.
        Expectation: It is the object handed in -- a stateful source (a lazy
            transform, an open file handle) is not rebuilt per pass.
        """
        source = _CountingSource(2)
        self.assertIs(_RepeatedDataset(source, 2).dataset, source)

    @arg_mark(**_CPU_MARKS)
    def test_non_positive_repeat_is_rejected(self):
        """Feature: repeat wrapper validation.

        Description: Construct with zero and with negative repeats.
        Expectation: A ValueError names the argument instead of producing an empty
            or infinite index space.
        """
        for repeat in (0, -1, -5):
            with self.subTest(repeat=repeat):
                with self.assertRaisesRegex(ValueError, "repeat must be positive"):
                    _RepeatedDataset(_CountingSource(2), repeat)

    @arg_mark(**_CPU_MARKS)
    def test_an_empty_source_cannot_be_indexed(self):
        """Feature: repeat wrapper on an empty source.

        Description: Wrap a source with no records and read index 0.
        Expectation: Length is 0 and indexing raises IndexError rather than looping
            over the empty source.
        """
        wrapped = _RepeatedDataset(_CountingSource(0), 3)
        self.assertEqual(len(wrapped), 0)
        with self.assertRaises(IndexError):
            _ = wrapped[0]


class TestBuildVlmDatasetRepeat(unittest.TestCase):
    """``data_config['repeat']`` reaches the wrapper through the builder."""

    @staticmethod
    def _write_list(path: str, count: int) -> None:
        """Write a LLaVA-style JSON list of ``count`` empty records."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"messages": []} for _ in range(count)], handle)

    def _build(self, data_path: str, data_config: dict) -> object:
        """Build a lazy dataset (no transform, no trainable filter)."""
        return build_vlm_dataset(
            data_config=data_config,
            data_path=data_path,
            transform=None,
            filter_trainable=False,
        )

    @arg_mark(**_CPU_MARKS)
    def test_repeat_config_multiplies_the_built_dataset(self):
        """Feature: repeat wiring.

        Description: Build the same 3-record source with and without
            ``data_config['repeat'] = 3``.
        Expectation: The repeated dataset is three times as long and index ``i + 3``
            is the same record as index ``i``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "data.json")
            self._write_list(data_path, 3)
            plain = self._build(data_path, {"source_type": "online"})
            repeated = self._build(data_path, {"source_type": "online", "repeat": 3})
            self.assertEqual(len(plain), 3)
            self.assertEqual(len(repeated), 9)
            self.assertEqual(repeated[3], repeated[0])

    @arg_mark(**_CPU_MARKS)
    def test_repeat_one_leaves_the_source_unwrapped(self):
        """Feature: repeat wiring default.

        Description: Build with ``repeat`` absent and with ``repeat`` 1.
        Expectation: Neither is wrapped, so the default path is unchanged and pays
            no extra indirection.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "data.json")
            self._write_list(data_path, 2)
            for config in ({"source_type": "online"}, {"source_type": "online", "repeat": 1}):
                with self.subTest(config=config):
                    built = self._build(data_path, config)
                    self.assertNotIsInstance(built, _RepeatedDataset)
                    self.assertEqual(len(built), 2)


if __name__ == "__main__":
    unittest.main()
