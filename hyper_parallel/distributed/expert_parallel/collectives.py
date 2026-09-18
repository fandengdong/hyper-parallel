# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""expert_parallel.collectives: backend-dispatched EP all_to_all.

NCCL/HCCL use the ragged a2a (``_EPAllToAllUneven``, zero-padding); gloo and
other backends that do not support ragged a2a use pad-to-max +
``all_to_all_single`` (``_EPAllToAllPadded``). Both paths are numerically
equivalent (padding only adds filler rows that do not participate in
computation).

When every per-peer count is equal -- which is exactly what a balanced routing
plan produces -- the split sizes carry no information, so ``HP_EP_EQUAL_A2A=1``
swaps the ragged exchange for the plain equal-length ``all_to_all_single``
(``_EPAllToAllEqual``), whose backend kernel is cheaper than alltoallv. The
rows, their order and the values are identical on both paths.

Split out of components/distributed/ep_utils.py in stage 4e.
"""

import os
from typing import Any, Callable, Optional
import torch
import torch.distributed as dist

from hyper_parallel.platform import get_platform


platform = get_platform()

_UNEVEN_A2A_BACKENDS = ("nccl", "hccl")

# A ragged exchange reaches NCCL/HCCL as alltoallv, which takes a per-peer count
# vector and runs a count-driven kernel.  Under a balanced routing plan every
# per-peer count is the same, so the counts add nothing and the plain
# equal-length all_to_all_single (alltoall) can be issued instead.  Opt-in while
# it is being measured (HP_EP_EQUAL_A2A=1).
_EQUAL_A2A_RAW = os.environ.get("HP_EP_EQUAL_A2A", "0")


def _backend_supports_uneven_a2a(group) -> bool:
    return dist.get_backend(group) in _UNEVEN_A2A_BACKENDS


def _equal_a2a_enabled() -> bool:
    """Resolve ``HP_EP_EQUAL_A2A`` (``"0"`` = off, ``"1"`` = on).

    Returns:
        Whether the equal-length fast path may be taken.

    Raises:
        ValueError: If the knob is neither ``"0"`` nor ``"1"``; a typo must fail
            loudly instead of silently leaving the fast path off.
    """
    raw = _EQUAL_A2A_RAW.strip()
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"HP_EP_EQUAL_A2A must be '0' or '1', but got {raw!r}")


class _EPAllToAllUneven(torch.autograd.Function):  # pylint: disable=abstract-method
    """Ragged all_to_all (NCCL/HCCL production path): split by send/recv counts.

    forward:  split(x, send_counts) -> dist.all_to_all(out_list, in_list) -> cat
    backward: swap send/recv counts and run the ragged all_to_all again
              (a2a is self-inverse).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Run the ragged all_to_all and retain the counts for backward."""
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
        dist.all_to_all(list(out.split(recv_counts)),
                        list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Swap send/recv counts and re-run the self-inverse ragged all_to_all."""
        grad = _EPAllToAllUneven.apply(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


class _EPAllToAllEqual(torch.autograd.Function):  # pylint: disable=abstract-method
    """Equal-length ``all_to_all_single`` (NCCL/HCCL path when counts are uniform).

    Without split sizes every peer gets ``rows_per_peer`` rows -- the backend
    derives the chunking from the tensor shape alone -- and the received chunks
    are concatenated in peer order. That is exactly the layout
    :class:`_EPAllToAllUneven` produces when all counts are equal, so this path
    is interchangeable with it (see :func:`_equal_a2a_rows` for when it applies).

    forward:  all_to_all_single(no splits) -> [ep_size * rows_per_peer, ...]
    backward: the same exchange; an equal-length a2a is its own inverse, and
              with a uniform plan the reverse chunks are the same size.
    """

    @staticmethod
    def _exchange(
        x: torch.Tensor,
        rows_per_peer: int,
        ep_size: int,
        group: Any,
    ) -> torch.Tensor:
        """Move ``rows_per_peer`` rows to each peer and return the peer-major result."""
        out = x.new_empty((rows_per_peer * ep_size,) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x, group=group)
        return out

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        rows_per_peer: int,
        ep_size: int,
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Run the equal-length all_to_all and retain its geometry for backward.

        Args:
            ctx: Autograd context of this exchange.
            x: Payload to exchange; it holds ``ep_size * rows_per_peer`` rows.
            rows_per_peer: Rows handed to each peer.
            ep_size: Number of ranks in ``group``.
            group: Process group to exchange over.

        Returns:
            The received rows, concatenated in peer order.
        """
        ctx.rows_per_peer = rows_per_peer
        ctx.ep_size = ep_size
        ctx.group = group
        return _EPAllToAllEqual._exchange(x, rows_per_peer, ep_size, group)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Re-run the self-inverse equal-length exchange on the output gradient.

        Args:
            ctx: Autograd context of the forward exchange.
            grad_output: Output gradient, one row per exchanged row.

        Returns:
            The input gradient, followed by ``None`` for the non-tensor arguments.
        """
        grad = _EPAllToAllEqual._exchange(
            grad_output.contiguous(), ctx.rows_per_peer, ctx.ep_size, ctx.group)
        return grad, None, None, None


class _EPAllToAllPadded(torch.autograd.Function):  # pylint: disable=abstract-method
    """pad-to-max + all_to_all_single (gloo test path).

    forward:  pad each dest chunk to the global max(counts) (a2a_single
              requires equal-length chunks per rank -> pad_to must be
              globally consistent, obtained via all_reduce MAX)
              -> a2a_single -> unpad by recv_counts;
    backward: pad by recv_counts -> a2a_single (equal-length self-inverse)
              -> unpad by send_counts.
    """

    @staticmethod
    def _pad_and_exchange(x, counts, pad_to, group):
        """Pad each chunk to pad_to, run equal-length a2a_single, return [ep*pad_to, ...]."""
        chunks = []
        for chunk, n in zip(x.split(counts), counts):
            if n < pad_to:
                pad = x.new_zeros((pad_to - n,) + tuple(x.shape[1:]))
                chunk = torch.cat([chunk, pad])
            chunks.append(chunk)
        send = torch.cat(chunks).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv

    @staticmethod
    def _unpad(recv, counts, pad_to):
        """Take the valid rows from the equal-length buffer per counts and cat."""
        pieces = []
        for i, n in enumerate(counts):
            if n > 0:
                pieces.append(recv[i * pad_to: i * pad_to + n])
        if not pieces:
            return recv.new_zeros((0,) + tuple(recv.shape[1:]))
        return torch.cat(pieces)

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Exchange padded expert-token chunks and retain counts for backward."""
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        local_max = max([*send_counts, *recv_counts, 1])
        pad_to = x.new_tensor([local_max], dtype=torch.int64)
        dist.all_reduce(pad_to, op=dist.ReduceOp.MAX, group=group)
        ctx.pad_to = pad_to = int(pad_to.item())
        recv = _EPAllToAllPadded._pad_and_exchange(x, send_counts, pad_to, group)
        return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Reverse the exchange: pad by recv_counts, a2a_single, unpad by send_counts."""
        # backward = reversed a2a: pad by recv_counts -> a2a_single -> unpad by send_counts
        recv = _EPAllToAllPadded._pad_and_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.pad_to, ctx.group)
        grad = _EPAllToAllPadded._unpad(recv, ctx.send_counts, ctx.pad_to)
        return grad, None, None, None


def _equal_a2a_rows(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> Optional[int]:
    """Rows per peer when the exchange may take the equal-length path, else None.

    The equal-length path is only equivalent to the ragged one when the plan is
    uniform *and* the payload really holds one such chunk per peer: the split
    sizes then carry no information, and the plain all-to-all -- which derives
    its chunking from the tensor shape alone -- moves exactly the rows the
    ragged call would.  Anything else (unequal counts, a payload whose row count
    disagrees with the counts, a non-contiguous payload, the knob off, a backend
    without a ragged all-to-all) yields ``None`` so the caller keeps the path it
    runs today.

    Args:
        x: Payload to exchange, split along dim 0.
        send_counts: Rows sent to each EP rank.
        recv_counts: Rows received from each EP rank.
        group: EP process group.

    Returns:
        The uniform per-peer row count, or ``None`` when the fast path does not
        apply.
    """
    if not _equal_a2a_enabled() or not _backend_supports_uneven_a2a(group):
        return None
    ep_size = len(send_counts)
    if ep_size == 0 or len(recv_counts) != ep_size or not x.is_contiguous():
        return None
    rows_per_peer = send_counts[0]
    # The split-free call takes the chunk size from the tensor shape and needs
    # the receive buffer to be the input's size, so a uniform plan is only
    # equivalent when both count lists hold that same count.
    uniform = (all(count == rows_per_peer for count in send_counts)
               and all(count == rows_per_peer for count in recv_counts))
    if not uniform or x.shape[0] != rows_per_peer * ep_size:
        return None
    return rows_per_peer


def ep_all_to_all(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """Unified entry for EP token exchange (autograd-differentiable).

    send_counts/recv_counts: list[int], length ep_size, row counts per dest/src rank.
    NCCL/HCCL -> ragged a2a (zero-padding); other backends (gloo test path) -> pad-to-max.
    With ``HP_EP_EQUAL_A2A=1`` a uniform plan is exchanged with the equal-length
    a2a instead (same rows, same order, cheaper kernel).
    """
    if not _backend_supports_uneven_a2a(group):
        return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
    rows_per_peer = _equal_a2a_rows(x, send_counts, recv_counts, group)
    if rows_per_peer is not None:
        return _EPAllToAllEqual.apply(x, rows_per_peer, len(send_counts), group)
    return _EPAllToAllUneven.apply(x, send_counts, recv_counts, group)


def ep_all_to_all_async(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """Non-blocking variant of :func:`ep_all_to_all` (lazy wait).

    On backends whose ``all_to_all_single`` accepts unequal splits the exchange
    is issued through ``differentiable_all_to_all_single_async``, which returns
    an ``AsyncCollectiveTensor``: the ``wait_tensor`` op is only enqueued when a
    non-view op first consumes the result.  Issuing independent work between the
    exchange and that first read therefore overlaps with the in-flight transfer
    (forward and backward alike).

    Backends without that support fall back to the blocking path, so the result
    always carries the same values — only the schedule differs.

    With ``HP_EP_EQUAL_A2A=1`` and a uniform plan the exchange goes through
    :class:`_EPAllToAllEqual` instead, i.e. the plain equal-length
    ``all_to_all_single``.  That call reaches the backend without split sizes,
    which is the cheaper kernel, but it is a ``dist`` collective rather than a
    functional one: the current stream is ordered on the collective when it is
    issued, so the returned tensor is not an ``AsyncCollectiveTensor`` and
    independent work issued right after it no longer overlaps the transfer.
    The knob therefore trades the lazy wait for the kernel; with it off, the
    lazy ragged exchange below is exactly the one this entry runs today.

    Args:
        x: Input tensor, split along dim 0 by ``send_counts``.
        send_counts: Rows sent to each EP rank.
        recv_counts: Rows received from each EP rank.
        group: EP process group.

    Returns:
        The exchanged rows, materialized lazily on the async path.
    """
    if not _backend_supports_uneven_a2a(group):
        return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
    rows_per_peer = _equal_a2a_rows(x, send_counts, recv_counts, group)
    if rows_per_peer is not None:
        return _EPAllToAllEqual.apply(x, rows_per_peer, len(send_counts), group)
    return platform.differentiable_all_to_all_single_async(
        x, send_counts, recv_counts, group,
    )
