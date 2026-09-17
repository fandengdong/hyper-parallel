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
"""Distinct-sample accounting under context parallelism.

Data is sharded by ``dp_rank`` while the CP peers of one DP group hold
replicas of the *same* sample, so ``data/step_samples`` must reduce over the
DP-only domain. Text tokens are unaffected: the CP batch path slices the loss
targets per rank, so ``data/step_tokens`` already partitions across CP.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from tests.common.mark_utils import arg_mark

from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.trainer.callbacks.environ_meter_callback import (
    EnvironMeterCallback,
)


class _FakeDeviceMesh:
    """Mesh stand-in that records which dimensions a lookup selected."""

    def __init__(self, dim_names):
        """Store the advertised dimension names."""
        self.mesh_dim_names = dim_names
        self.selected = []

    def __getitem__(self, key):
        """Return a sentinel tagged with the requested dimension selection."""
        self.selected.append(key)
        return _FakeSubMesh(key, ndim=2 if isinstance(key, tuple) else 1)


class _FakeSubMesh:
    """Flattened sub-mesh stand-in exposing a process group."""

    def __init__(self, key, ndim=1):
        """Store the dimension key this sub-mesh stands for."""
        self.key = key
        self.ndim = ndim

    def flatten(self, name):
        """Return a 1-D sub-mesh labelled with the flattened dimension name."""
        return _FakeSubMesh((self.key, name))

    def get_group(self):
        """Return the dimension key as the group identity."""
        return self.key


def _build_callback(dp_cp_group, dp_group):
    """Build a callback whose trainer mesh exposes both reduction domains."""
    trainer = SimpleNamespace(
        config=SimpleNamespace(training=SimpleNamespace(peak_tflops=400.0)),
        mesh=SimpleNamespace(
            dp_cp_mesh=SimpleNamespace(get_group=lambda: dp_cp_group),
            dp_mesh=SimpleNamespace(get_group=lambda: dp_group),
        ),
    )
    trainer.step_train_metrics = {}
    trainer.step_env_metrics = {}
    return EnvironMeterCallback(trainer)


class TestMeshContextDpMesh(unittest.TestCase):
    """``MeshContext.dp_mesh`` selects the DP domain and excludes CP."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dp_dimension_is_selected_without_cp(self):
        """A (dp, cp, tp) mesh resolves dp_mesh to the dp sub-mesh."""
        device_mesh = _FakeDeviceMesh(("dp", "cp", "tp"))
        mesh = MeshContext(device_mesh=device_mesh, dp_size=64, cp_size=2)

        self.assertEqual(mesh.dp_mesh.key, "dp")
        self.assertNotIn(("dp", "cp"), device_mesh.selected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dp_mesh_falls_back_to_shard_dimensions(self):
        """A shard-based mesh uses dp_replicate/dp_shard when dp is absent."""
        device_mesh = _FakeDeviceMesh(("dp_replicate", "dp_shard", "cp"))
        mesh = MeshContext(device_mesh=device_mesh, dp_size=64, cp_size=2)

        self.assertEqual(mesh.dp_mesh.key, (("dp_replicate", "dp_shard"), "dp"))
        self.assertEqual(device_mesh.selected, [("dp_replicate", "dp_shard")])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dp_mesh_is_none_without_a_device_mesh(self):
        """No DeviceMesh means no DP domain to reduce over."""
        self.assertIsNone(MeshContext(device_mesh=None).dp_mesh)


class TestEnvironMeterSampleGroup(unittest.TestCase):
    """The meter reduces sample counts over DP, tokens over DP+CP."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sample_group_is_dp_only(self):
        """Samples use the DP group so CP replicas are not double counted."""
        callback = _build_callback(dp_cp_group="dp_cp", dp_group="dp")

        self.assertEqual(callback._metric_group(), "dp_cp")
        self.assertEqual(callback._sample_group(), "dp")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sample_group_falls_back_to_dp_cp(self):
        """A mesh without a DP domain keeps the previous behaviour."""
        callback = _build_callback(dp_cp_group="dp_cp", dp_group=None)
        callback.trainer.mesh = SimpleNamespace(
            dp_cp_mesh=SimpleNamespace(get_group=lambda: "dp_cp"),
        )

        self.assertEqual(callback._sample_group(), "dp_cp")


if __name__ == "__main__":
    unittest.main()
