"""End-to-end tests: spawned queue mesh versus a real gloo process group."""

from __future__ import annotations

import math

import pytest
import torch

from collective_ops import main as collective


def _manual_sum(inputs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(inputs).sum(dim=0)


def _chunk_sizes(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def test_ring_allreduce_int64_matches_gloo() -> None:
    world_size = 4
    total = 1024
    inputs = [
        torch.arange(total, dtype=torch.int64) + 17 * rank
        for rank in range(world_size)
    ]
    result = collective.run_collective("allreduce", world_size, inputs)
    expected = _manual_sum(inputs)

    for output in result.outputs:
        assert torch.equal(output, expected)
    reference = collective._gloo_reference("allreduce", world_size, inputs)
    for output, gloo_output in zip(result.outputs, reference):
        assert torch.equal(output, gloo_output)

    expected_bytes = 2 * total * 8 * (world_size - 1) // world_size
    assert all(stat.bytes_sent == expected_bytes for stat in result.stats)
    assert all(stat.rounds == 2 * (world_size - 1) for stat in result.stats)


def test_ring_allreduce_float32_matches_gloo() -> None:
    world_size = 4
    total = 4096
    generator = torch.Generator().manual_seed(0)
    inputs = [
        torch.randn(total, generator=generator) + rank
        for rank in range(world_size)
    ]
    result = collective.run_collective("allreduce", world_size, inputs)
    reference = collective._gloo_reference("allreduce", world_size, inputs)

    for output, gloo_output in zip(result.outputs, reference):
        torch.testing.assert_close(output, gloo_output, atol=1e-5, rtol=1e-5)
    for output in result.outputs[1:]:
        assert torch.equal(result.outputs[0], output)


def test_ring_allreduce_uneven_chunks() -> None:
    world_size = 4
    total = 1001
    inputs = [
        torch.arange(total, dtype=torch.int64) + 1000 * rank
        for rank in range(world_size)
    ]
    result = collective.run_collective("allreduce", world_size, inputs)
    expected = _manual_sum(inputs)
    sizes = _chunk_sizes(total, world_size)

    for rank, output in enumerate(result.outputs):
        assert torch.equal(output, expected)
        # Reduce-scatter sends every chunk except rank r; allgather sends every
        # chunk except rank r+1. Equal chunk sizes collapse this to 2T(N-1)/N.
        sent_elements = (total - sizes[rank]) + (
            total - sizes[(rank + 1) % world_size]
        )
        assert result.stats[rank].bytes_sent == 8 * sent_elements
    assert all(stat.rounds == 2 * (world_size - 1) for stat in result.stats)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_allreduce_byte_scaling(world_size: int) -> None:
    total = 2048
    base = torch.arange(total, dtype=torch.float32)
    inputs = [base + rank for rank in range(world_size)]
    result = collective.run_collective("allreduce", world_size, inputs)

    expected_bytes = 2 * total * 4 * (world_size - 1) // world_size
    for rank, stat in enumerate(result.stats):
        assert stat.bytes_sent == expected_bytes
        assert stat.rounds == 2 * (world_size - 1)
        assert stat.targets == {(rank, (rank + 1) % world_size)}


def test_allgather_matches_gloo() -> None:
    world_size = 4
    shard_len = 128
    inputs = [
        torch.arange(shard_len, dtype=torch.int64) + rank * shard_len
        for rank in range(world_size)
    ]
    result = collective.run_collective("allgather", world_size, inputs)
    expected = torch.cat(inputs)

    for output in result.outputs:
        assert torch.equal(output, expected)
    reference = collective._gloo_reference("allgather", world_size, inputs)
    for output, gloo_output in zip(result.outputs, reference):
        assert torch.equal(output, gloo_output)

    expected_bytes = (world_size - 1) * shard_len * 8
    assert all(stat.bytes_sent == expected_bytes for stat in result.stats)
    assert all(stat.rounds == world_size - 1 for stat in result.stats)


def test_allgather_ragged_shards() -> None:
    world_size = 3
    inputs = [
        torch.arange(length, dtype=torch.int64) + 10 * rank
        for rank, length in enumerate((2, 3, 4))
    ]
    result = collective.run_collective("allgather", world_size, inputs)
    expected = torch.cat(inputs)

    for output in result.outputs:
        assert torch.equal(output, expected)
    assert all(stat.rounds == world_size - 1 for stat in result.stats)


def test_reduce_scatter_matches_gloo() -> None:
    world_size = 4
    total = 1024
    shard_len = total // world_size
    inputs = [
        torch.arange(total, dtype=torch.int64) + 31 * rank
        for rank in range(world_size)
    ]
    result = collective.run_collective("reduce_scatter", world_size, inputs)
    expected = _manual_sum(inputs)
    reference = collective._gloo_reference("reduce_scatter", world_size, inputs)

    for rank, output in enumerate(result.outputs):
        expected_shard = expected[rank * shard_len:(rank + 1) * shard_len]
        assert output.shape == (shard_len,)
        assert torch.equal(output, expected_shard)
        assert torch.equal(output, reference[rank])
        assert result.stats[rank].bytes_sent == (world_size - 1) * shard_len * 8
        assert result.stats[rank].rounds == world_size - 1


@pytest.mark.parametrize("src", [0, 2])
def test_broadcast_matches_gloo(src: int) -> None:
    world_size = 4
    total = 64
    inputs = [
        torch.full((total,), rank, dtype=torch.int64)
        for rank in range(world_size)
    ]
    result = collective.run_collective(
        "broadcast", world_size, inputs, src=src
    )
    reference = collective._gloo_reference("broadcast", world_size, inputs, src=src)

    for output, gloo_output in zip(result.outputs, reference):
        assert torch.equal(output, inputs[src])
        assert torch.equal(output, gloo_output)
    assert sum(stat.bytes_sent for stat in result.stats) == total * 8 * (world_size - 1)
    assert sum(stat.messages_sent for stat in result.stats) == world_size - 1
    assert all(stat.rounds == math.ceil(math.log2(world_size)) for stat in result.stats)


def test_broadcast_tree_rounds_and_bytes() -> None:
    world_size = 5
    total = 100
    base = torch.arange(total, dtype=torch.float32)
    inputs = [base + rank for rank in range(world_size)]
    result = collective.run_collective("broadcast", world_size, inputs)
    expected = inputs[0]

    for output in result.outputs:
        assert torch.equal(output, expected)
    assert all(stat.rounds == 3 for stat in result.stats)
    assert sum(stat.bytes_sent for stat in result.stats) == total * 4 * (world_size - 1)
    assert sum(stat.messages_sent for stat in result.stats) == world_size - 1
    for rank, stat in enumerate(result.stats):
        assert all(target[0] == rank and target[1] != rank for target in stat.targets)


def test_single_rank_all_ops() -> None:
    ctx = collective.mp.get_context("spawn")
    inbox = ctx.Queue()
    mesh = collective.Mesh(0, 1, inbox, {})
    tensor = torch.arange(8, dtype=torch.int64)

    try:
        operations = [
            (collective.ring_allreduce, {}),
            (collective.broadcast, {"src": 0}),
            (collective.allgather, {}),
            (collective.reduce_scatter, {}),
        ]
        for operation, kwargs in operations:
            output = operation(mesh, 0, 1, tensor, **kwargs)
            assert torch.equal(output, tensor)
            assert mesh.stats.bytes_sent == 0
            assert mesh.stats.bytes_recv == 0
            assert mesh.stats.rounds == 0
    finally:
        inbox.cancel_join_thread()
        inbox.close()


def test_ring_allreduce_more_ranks_than_elements() -> None:
    world_size = 4
    inputs = [
        torch.tensor([rank, rank + 10], dtype=torch.int64)
        for rank in range(world_size)
    ]
    result = collective.run_collective("allreduce", world_size, inputs)
    expected = _manual_sum(inputs)
    sizes = _chunk_sizes(2, world_size)

    for rank, output in enumerate(result.outputs):
        assert torch.equal(output, expected)
        sent_elements = (2 - sizes[rank]) + (
            2 - sizes[(rank + 1) % world_size]
        )
        assert result.stats[rank].bytes_sent == 8 * sent_elements
    assert all(stat.rounds == 6 for stat in result.stats)
