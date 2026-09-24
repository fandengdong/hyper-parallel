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
from typing import Any

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from torch.utils.data import Dataset

from tests.common.mark_utils import arg_mark

from hyper_parallel.data.omni import OmniDataTransform
from hyper_parallel.data.omni.build_dataset import _RepeatedDataset, build_online_omni_mapping_dataset

_CPU_MARKS = {"plat_marks": ["cpu_linux", "cpu_macos"], "level_mark": "level0",
              "card_mark": "allcards", "essential_mark": "essential"}


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


class TestRepeatedDatasetShuffle(unittest.TestCase):
    """``shuffle`` visits the source in a fixed random order."""

    _BIG = 64

    @arg_mark(**_CPU_MARKS)
    def test_shuffle_defaults_off(self):
        """Feature: shuffle default.

        Description: Wrap a source without asking for shuffle.
        Expectation: The natural order is preserved and ``shuffle`` reads back False,
            so the default path is unchanged.
        """
        wrapped = _RepeatedDataset(_CountingSource(8), 1)
        self.assertFalse(wrapped.shuffle)
        self.assertEqual(
            [wrapped[i] for i in range(8)], [f"record-{i}" for i in range(8)])

    @arg_mark(**_CPU_MARKS)
    def test_shuffle_visits_every_record_exactly_once(self):
        """Feature: shuffle is a permutation.

        Description: Read the whole shuffled index space of a 64-record source.
        Expectation: Every record appears exactly once and the order is no longer the
            natural one -- shuffling reorders an epoch, it never drops or duplicates.
        """
        wrapped = _RepeatedDataset(_CountingSource(self._BIG), 1, shuffle=True)
        visited = [wrapped[i] for i in range(self._BIG)]
        self.assertEqual(
            sorted(visited), sorted(f"record-{i}" for i in range(self._BIG)))
        self.assertNotEqual(visited, [f"record-{i}" for i in range(self._BIG)])

    @arg_mark(**_CPU_MARKS)
    def test_shuffle_order_is_reproducible_for_a_seed(self):
        """Feature: shuffle reproducibility.

        Description: Build the order twice with the same seed, then with another seed.
        Expectation: The same seed replays the order exactly (a resumed run sees the
            same batches); a different seed does not.
        """
        def _order(seed):
            """Return the served record order for one shuffle seed."""
            wrapped = _RepeatedDataset(
                _CountingSource(self._BIG), 1, shuffle=True, seed=seed)
            return [wrapped[i] for i in range(self._BIG)]

        self.assertEqual(_order(1234), _order(1234))
        self.assertNotEqual(_order(1234), _order(4321))

    @arg_mark(**_CPU_MARKS)
    def test_shuffle_repeats_the_same_order_on_every_pass(self):
        """Feature: shuffle with repeat.

        Description: Wrap with ``repeat=2`` and shuffle, then read both passes.
        Expectation: The second pass repeats the first pass's order -- repeating
            multiplies the epoch, it does not reshuffle between passes.
        """
        wrapped = _RepeatedDataset(_CountingSource(8), 2, shuffle=True)
        first = [wrapped[i] for i in range(8)]
        second = [wrapped[8 + i] for i in range(8)]
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted(f"record-{i}" for i in range(8)))
        self.assertEqual(len(wrapped), 16)

    @arg_mark(**_CPU_MARKS)
    def test_shuffle_still_rejects_a_non_positive_repeat(self):
        """Feature: shuffle validation.

        Description: Construct a shuffling wrapper with ``repeat`` 0.
        Expectation: The repeat check still fires before any permutation is built.
        """
        with self.assertRaisesRegex(ValueError, "repeat must be positive"):
            _RepeatedDataset(_CountingSource(4), 0, shuffle=True)


class _IdentityOmniTransform(OmniDataTransform):
    """Return each Omni source record unchanged (test-only transform)."""

    def __init__(self) -> None:
        """Bind a placeholder processor and a tiny sequence limit."""
        super().__init__(max_seq_len=8, processor=object())

    def encode_sample(self, sample: dict) -> dict:
        """Return the raw record unchanged."""
        return sample


class TestBuildVlmDatasetRepeat(unittest.TestCase):
    """``data_config['repeat']`` reaches the wrapper through the builder."""

    @staticmethod
    def _write_list(path: str, count: int) -> None:
        """Write ``count`` JSONL records, each with messages and a distinct id."""
        with open(path, "w", encoding="utf-8") as handle:
            for index in range(count):
                record = {"messages": [{"role": "user", "content": "hi"}], "id": index}
                handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _ids(dataset: object, count: int) -> list:
        """Return the ids of the first ``count`` records served by ``dataset``."""
        return [dataset[index]["id"] for index in range(count)]

    def _build(self, data_path: str, data_config: dict) -> object:
        """Build a lazy Omni mapping dataset with an identity transform."""
        return build_online_omni_mapping_dataset(
            data_config=data_config,
            data_path=data_path,
            transform=_IdentityOmniTransform(),
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
            data_path = os.path.join(tmp, "data.jsonl")
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
            data_path = os.path.join(tmp, "data.jsonl")
            self._write_list(data_path, 2)
            for config in ({"source_type": "online"}, {"source_type": "online", "repeat": 1}):
                with self.subTest(config=config):
                    built = self._build(data_path, config)
                    self.assertNotIsInstance(built, _RepeatedDataset)
                    self.assertEqual(len(built), 2)


    @arg_mark(**_CPU_MARKS)
    def test_shuffle_config_wraps_even_without_repeat(self):
        """Feature: shuffle wiring.

        Description: Build with ``shuffle`` set and ``repeat`` left at its default.
        Expectation: The dataset is wrapped -- shuffling needs the wrapper even for a
            single pass -- and every record id is still served exactly once.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "data.jsonl")
            self._write_list(data_path, 8)
            built = self._build(data_path, {"source_type": "online", "shuffle": True})
            self.assertIsInstance(built, _RepeatedDataset)
            self.assertEqual(len(built), 8)
            served = self._ids(built, 8)
            self.assertEqual(sorted(served), list(range(8)))
            self.assertNotEqual(served, list(range(8)))

    @arg_mark(**_CPU_MARKS)
    def test_shuffle_absent_keeps_the_source_order(self):
        """Feature: shuffle wiring default.

        Description: Build with ``shuffle`` false and with it absent.
        Expectation: Neither is wrapped, so both serve the same source order --
            the Omni Mapping source owns its own deterministic per-epoch order.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "data.jsonl")
            self._write_list(data_path, 4)
            built_default = self._build(data_path, {"source_type": "online"})
            built_false = self._build(data_path, {"source_type": "online", "shuffle": False})
            for config, built in (("default", built_default), ("false", built_false)):
                with self.subTest(config=config):
                    self.assertNotIsInstance(built, _RepeatedDataset)
                    self.assertEqual(sorted(self._ids(built, 4)), list(range(4)))
            self.assertEqual(self._ids(built_default, 4), self._ids(built_false, 4))


class _PackingAwareSource(_CountingSource):
    """Map-style source that also exposes the Omni packing loader's interface."""

    def __init__(self, count: int) -> None:
        """Expose ``count`` records plus the deferred-encoding hooks."""
        super().__init__(count)
        self.packing_selector = "selector-sentinel"

    def encode_selected_sample(self, sample: Any) -> Any:
        """Encode one selected sample (identity stand-in)."""
        return sample

    def encode_batch(self, batch: Any) -> Any:
        """Encode one packed batch (identity stand-in)."""
        return batch


class TestWrapperKeepsPackingInterface(unittest.TestCase):
    """Wrapping a source must not hide the hooks the Omni packing loader reads.

    Regression: packing combined with ``shuffle``/``repeat`` failed on every rank of a
    2-SN run with ``TypeError: OmniPackingLoader requires encode_selected_sample() and
    encode_batch()``, because ``_RepeatedDataset``/``_TransformDataset`` exposed only
    ``__len__``/``__getitem__`` while the loader reads those hooks off the dataset
    object it is handed.
    """

    @arg_mark(**_CPU_MARKS)
    def test_repeat_and_shuffle_forward_the_packing_hooks(self):
        """Feature: repeat+shuffle keeps the packing interface.

        Description: Wrap a hook-bearing source with ``repeat`` and ``shuffle``.
        Expectation: Both hooks and the selector stay reachable, and the wrapper's own
            ``__len__`` still wins over the delegated one.
        """
        wrapped = _RepeatedDataset(_PackingAwareSource(4), 2, shuffle=True, seed=1)
        self.assertTrue(callable(getattr(wrapped, "encode_selected_sample", None)))
        self.assertTrue(callable(getattr(wrapped, "encode_batch", None)))
        self.assertEqual(wrapped.packing_selector, "selector-sentinel")
        self.assertEqual(len(wrapped), 8)

    @arg_mark(**_CPU_MARKS)
    def test_repeat_alone_forwards_the_packing_hooks(self):
        """Feature: repeat without shuffle.

        Description: Wrap with ``repeat`` only.
        Expectation: The deferred-encoding hooks are still reachable.
        """
        wrapped = _RepeatedDataset(_PackingAwareSource(3), 2)
        self.assertTrue(callable(getattr(wrapped, "encode_selected_sample", None)))
        self.assertTrue(callable(getattr(wrapped, "encode_batch", None)))

    @arg_mark(**_CPU_MARKS)
    def test_wrapper_does_not_invent_missing_hooks(self):
        """Feature: a source without the hooks stays hook-free.

        Description: Wrap a plain source that has no packing interface.
        Expectation: The wrapper does not fabricate one, so the packing loader still
            rejects it instead of failing later.
        """
        wrapped = _RepeatedDataset(_CountingSource(3), 2)
        self.assertFalse(hasattr(wrapped, "encode_selected_sample"))
        self.assertFalse(hasattr(wrapped, "encode_batch"))


if __name__ == "__main__":
    unittest.main()
