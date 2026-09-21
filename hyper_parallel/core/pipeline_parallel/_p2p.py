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
"""Pipeline edge groups and batched P2P communicator initialization."""
import time

import torch.distributed as dist

_P2P_MULTI_STREAM_GROUPS = {}


def _build_p2p_edge_rank_lists(pp_rank_list: list[int], include_wrap: bool = False) -> list[tuple[int, int]]:
    """Build normalized two-rank groups for adjacent pipeline ranks."""
    if not isinstance(pp_rank_list, (list, tuple)):
        raise ValueError(
            f"pp_rank_list must be a list or tuple of integer ranks, but got {type(pp_rank_list)}."
        )
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in pp_rank_list):
        raise ValueError(f"pp_rank_list must contain only integer ranks, but got {pp_rank_list}.")
    if len(set(pp_rank_list)) != len(pp_rank_list):
        raise ValueError(f"pp_rank_list must not contain duplicate ranks, but got {pp_rank_list}.")
    if len(pp_rank_list) < 2:
        return []

    edge_rank_lists = {
        tuple(sorted((src_rank, dst_rank)))
        for src_rank, dst_rank in zip(pp_rank_list, pp_rank_list[1:])
    }
    if include_wrap and len(pp_rank_list) > 2:
        edge_rank_lists.add(tuple(sorted((pp_rank_list[-1], pp_rank_list[0]))))
    return sorted(edge_rank_lists)


def create_p2p_multi_stream_groups(
        pp_rank_list: list[int],
        include_wrap: bool = False,
) -> dict[int, dist.ProcessGroup]:
    """Create adjacent two-rank PP groups for independent communication streams.

    Args:
        pp_rank_list: Ordered global ranks in one pipeline-parallel group.
        include_wrap: Whether to include the last-to-first interleaved edge.

    Returns:
        A mapping from adjacent peer rank to its process-group handle.
    """
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered_pp_rank_lists = [None] * world_size
    dist.all_gather_object(gathered_pp_rank_lists, list(pp_rank_list))
    edge_rank_lists = sorted({
        edge_ranks
        for ranks in gathered_pp_rank_lists
        for edge_ranks in _build_p2p_edge_rank_lists(ranks, include_wrap)
    })

    # Overlapping groups can deadlock with local synchronization when
    # neighboring ranks enter different edge creations first. Every rank
    # therefore creates the global edge set in the same order.
    local_groups = {}
    world_rank_list = tuple(range(world_size))
    for edge_ranks in edge_rank_lists:
        group_key = str(edge_ranks)
        group = _P2P_MULTI_STREAM_GROUPS.get(group_key)
        if group is None:
            group = (
                dist.group.WORLD
                if edge_ranks == world_rank_list
                else dist.new_group(ranks=list(edge_ranks))
            )
            _P2P_MULTI_STREAM_GROUPS[group_key] = group
        if current_rank not in edge_ranks:
            continue
        peer_rank = edge_ranks[0] if edge_ranks[1] == current_rank else edge_ranks[1]
        local_groups[peer_rank] = group
    return local_groups


def prepare_batch_p2p_group(group: dist.ProcessGroup = None) -> None:
    """Synchronize a group before its first subset batched P2P call.

    PyTorch requires every rank in a process group to participate when
    ``batch_isend_irecv`` is the first collective on that group. A barrier
    at the common pipeline run boundary initializes the communicator
    before ranks reach peer operations at different times. Group members
    first rendezvous through the CPU store, then group roots serialize the
    one-time HCCL initialization through a job-local store lock. This
    avoids host-port collisions when independent pipeline domains prepare
    different communicators concurrently.

    Args:
        group: The process group used by the batched P2P operations.
            ``None`` uses the default group.
    """
    group = group or dist.group.WORLD
    group_store = group.get_group_store()
    group_size = dist.get_world_size(group=group)
    group_name = group.group_name
    ready_key = f"hyper_parallel_p2p_group_init_ready:{group_name}"
    ready_count = group_store.add(ready_key, 1)
    ready_done_key = f"{ready_key}_done"
    if ready_count == group_size:
        group_store.set(ready_done_key, "1")
    group_store.wait([ready_done_key])

    root_ready_key = f"hyper_parallel_p2p_group_init_root_ready:{group_name}"
    if dist.get_rank(group=group) != 0:
        group_store.wait([root_ready_key])
        dist.barrier(group=group)
        return

    default_store = dist.group.WORLD.get_group_store()
    lock_key = "hyper_parallel_p2p_group_init_lock"
    lock_token = group_name
    deadline = time.monotonic() + 300
    while default_store.compare_set(lock_key, "", lock_token).decode() != lock_token:
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting to initialize a batched P2P process group.")
        time.sleep(0.01)
    try:
        group_store.set(root_ready_key, "1")
        dist.barrier(group=group)
    finally:
        default_store.compare_set(lock_key, lock_token, "")
