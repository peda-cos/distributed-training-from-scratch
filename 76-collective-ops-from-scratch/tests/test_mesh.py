"""Transport-level tests that do not spawn worker processes."""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest
import torch

from collective_ops import main as collective

CONTEXT = mp.get_context("spawn")


@pytest.fixture
def make_queue():
    created = []

    def factory():
        q = CONTEXT.Queue()
        created.append(q)
        return q

    yield factory
    for q in created:
        q.cancel_join_thread()
        q.close()


def test_mesh_preserves_order_and_demuxes_sources(make_queue) -> None:
    inboxes = [make_queue() for _ in range(3)]
    mesh0 = collective.Mesh(0, 3, inboxes[0], {1: inboxes[1], 2: inboxes[2]})
    mesh1 = collective.Mesh(1, 3, inboxes[1], {0: inboxes[0], 2: inboxes[2]})
    mesh2 = collective.Mesh(2, 3, inboxes[2], {0: inboxes[0], 1: inboxes[1]})

    later = torch.tensor([222], dtype=torch.int64)
    earlier = torch.tensor([111], dtype=torch.int64)
    # rank 2 puts first, so rank 0 must buffer it while waiting for rank 1.
    mesh2.send(0, later, tag="two")
    mesh1.send(0, earlier, tag="one")

    first = mesh0.recv(1, tag="one")
    second = mesh0.recv(2, tag="two")
    assert torch.equal(first, earlier)
    assert torch.equal(second, later)
    assert mesh0.stats.messages_recv == 2
    assert mesh0.stats.bytes_recv == 16
    assert mesh1.stats.bytes_sent == 8
    assert mesh2.stats.bytes_sent == 8


def test_recv_timeout_surfaces_comm_timeout(make_queue) -> None:
    inbox = make_queue()
    other = make_queue()
    mesh = collective.Mesh(0, 2, inbox, {1: other}, recv_timeout_s=0.2)

    started = time.monotonic()
    with pytest.raises(collective.CommTimeout) as excinfo:
        mesh.recv(1)
    elapsed = time.monotonic() - started

    assert excinfo.value.rank == 0
    assert excinfo.value.src == 1
    assert elapsed >= 0.15
    assert elapsed < 5.0


def test_protocol_tag_mismatch_raises(make_queue) -> None:
    inbox0 = make_queue()
    inbox1 = make_queue()
    mesh0 = collective.Mesh(0, 2, inbox0, {1: inbox1})
    mesh1 = collective.Mesh(1, 2, inbox1, {0: inbox0})

    mesh1.send(0, torch.tensor([7], dtype=torch.int64), tag="actual")
    with pytest.raises(collective.ProtocolError):
        mesh0.recv(1, tag="expected")
