"""Unit tests for flat parameters, shard planning, and the sharded optimizer."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zero_sharding import main as zero


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _assert_parameters_equal(left: nn.Module, right: nn.Module) -> None:
    for first, second in zip(left.parameters(), right.parameters()):
        assert torch.equal(first, second)


def test_flatten_unflatten_round_trip() -> None:
    source = zero.make_model(7, 7, torch.float32)
    target = zero.make_model(11, 7, torch.float32)
    flat = zero.flatten_params(source)
    total = _parameter_count(source)

    assert flat.dim() == 1
    assert flat.numel() == total
    assert flat.dtype == torch.float32
    assert not flat.requires_grad

    identities = [id(parameter) for parameter in target.parameters()]
    zero.unflatten_into(target, flat)

    assert [id(parameter) for parameter in target.parameters()] == identities
    _assert_parameters_equal(source, target)
    for parameter in target.parameters():
        assert parameter.is_leaf
        assert parameter.requires_grad

    loss = (target(torch.randn(5, zero.INPUT_DIM)) ** 2).mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in target.parameters())


def test_flatten_unflatten_reject_bad_inputs() -> None:
    mixed = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    mixed[-1].to(torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        zero.flatten_params(mixed)

    target = zero.make_model(7, 5, torch.float32)
    total = _parameter_count(target)
    before = [parameter.detach().clone() for parameter in target.parameters()]
    with pytest.raises(ValueError):
        zero.unflatten_into(target, torch.zeros(total + 1))
    with pytest.raises(ValueError, match="dtype"):
        zero.unflatten_into(target, torch.zeros(total, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="one-dimensional"):
        zero.unflatten_into(target, torch.zeros(1, total))
    for parameter, original in zip(target.parameters(), before):
        assert torch.equal(parameter, original)

    with pytest.raises(ValueError, match="no parameters"):
        zero.flatten_params(nn.Sequential())

    grads = zero.flatten_grads(target)
    assert grads.numel() == total
    assert torch.count_nonzero(grads) == 0


def test_shard_spec_covers_uneven_params_and_extra_ranks() -> None:
    specs = [zero.shard_spec(10, 3, rank) for rank in range(3)]
    assert [(spec.start, spec.end) for spec in specs] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert {spec.shard_numel for spec in specs} == {4}
    assert {spec.padded_numel for spec in specs} == {12}

    covered = [0] * 10
    for spec in specs:
        for index in range(spec.start, spec.end):
            covered[index] += 1
    assert covered == [1] * 10

    extra = [zero.shard_spec(2, 4, rank) for rank in range(4)]
    assert [(spec.start, spec.end) for spec in extra] == [
        (0, 1),
        (1, 2),
        (2, 2),
        (2, 2),
    ]
    assert extra[2].owns_any is False
    assert extra[3].owns_any is False

    with pytest.raises(ValueError):
        zero.shard_spec(10, 0, 0)
    with pytest.raises(ValueError):
        zero.shard_spec(10, 3, 3)
    with pytest.raises(ValueError):
        zero.shard_spec(0, 1, 0)


def test_pad_flat_zero_fills_only_the_tail() -> None:
    spec = zero.shard_spec(10, 3, 1)
    flat = torch.arange(10, dtype=torch.float32)
    padded = zero.pad_flat(flat, spec)

    assert padded.numel() == 12
    assert padded.is_contiguous()
    assert torch.equal(padded[:10], flat)
    assert torch.count_nonzero(padded[10:]) == 0
    assert torch.equal(flat, torch.arange(10, dtype=torch.float32))

    with pytest.raises(ValueError):
        zero.pad_flat(torch.arange(9, dtype=torch.float32), spec)


def test_zero_optimizer_keeps_only_local_state() -> None:
    model = zero.make_model(7, 6, torch.float32)
    assert _parameter_count(model) == 151
    optimizer = zero.ZeroOptimizer(model, 4, 2, lr=1e-3)

    assert optimizer.spec.shard_numel == 38
    assert optimizer.spec.padded_numel == 152
    assert (optimizer.spec.start, optimizer.spec.end) == (76, 114)
    assert optimizer.state_nbytes() == 38 * 12
    for state in (optimizer.master, optimizer.exp_avg, optimizer.exp_avg_sq):
        assert state.dtype == torch.float32
        assert state.numel() == 38
        assert state.dim() == 1
    local = zero.flatten_params(model).float()[76:114]
    assert torch.equal(optimizer.master, local)

    last = zero.ZeroOptimizer(model, 4, 3, lr=1e-3)
    assert (last.spec.start, last.spec.end) == (114, 151)
    assert last.spec.local_numel == 37

    ranges = [
        (zero.shard_spec(151, 4, rank).start, zero.shard_spec(151, 4, rank).end)
        for rank in range(4)
    ]
    for left, right in zip(ranges, ranges[1:]):
        assert left[1] == right[0]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 151


def test_single_process_zero_matches_torch_adam() -> None:
    torch.set_num_threads(1)
    zero_model = zero.make_model(7, 6, torch.float32)
    reference_model = zero.make_model(7, 6, torch.float32)
    zero_optimizer = zero.ZeroOptimizer(zero_model, 1, 0, lr=0.01)
    reference_optimizer = torch.optim.Adam(
        reference_model.parameters(), lr=0.01, foreach=False
    )
    x, y = zero.make_dataset(2026)

    for step in range(5):
        indices = zero.batch_indices(
            step, 32, x.shape[0], zero.SHUFFLE_SEED
        )
        xb, yb = x[indices], y[indices]
        for model, optimizer in (
            (zero_model, zero_optimizer),
            (reference_model, reference_optimizer),
        ):
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        delta = max(
            (left - right).abs().max().item()
            for left, right in zip(
                zero_model.parameters(), reference_model.parameters()
            )
        )
        assert delta <= 1e-6, f"step {step}: {delta}"

    assert zero_optimizer.step_count == 5


def test_memory_budget_matches_published_formulas() -> None:
    total = 7_000_000_000
    shard_8 = total // 8
    ddp = zero.memory_budget(total, 8, 0)
    stage1 = zero.memory_budget(total, 8, 1)
    stage2 = zero.memory_budget(total, 8, 2)
    stage3 = zero.memory_budget(total, 8, 3)

    assert ddp.total_bytes == 16 * total
    assert stage1.total_bytes == 4 * total + 12 * shard_8
    assert stage2.total_bytes == 2 * total + 14 * shard_8
    assert stage3.total_bytes == 16 * shard_8
    assert stage1.optimizer_bytes == 12 * shard_8

    for budget in (ddp, stage1, stage2, stage3):
        assert budget.total_bytes == (
            budget.params_bytes
            + budget.grads_bytes
            + budget.master_bytes
            + budget.moments_bytes
        )
        assert budget.optimizer_bytes == budget.master_bytes + budget.moments_bytes

    assert math.isclose(
        1.0 - stage1.total_bytes / (16 * total), 0.65625, rel_tol=0, abs_tol=1e-12
    )
    stage1_64 = zero.memory_budget(total, 64, 1)
    assert math.isclose(
        1.0 - stage1_64.total_bytes / (16 * total),
        0.73828125,
        rel_tol=0,
        abs_tol=1e-12,
    )

    with pytest.raises(ValueError):
        zero.memory_budget(total, 8, 4)
    with pytest.raises(ValueError):
        zero.memory_budget(-1, 8, 1)
    with pytest.raises(ValueError):
        zero.memory_budget(total, 0, 1)
    with pytest.raises(ValueError):
        zero.memory_budget(total, 8, 1, param_bytes=0)
    with pytest.raises(ValueError):
        zero.memory_budget(total, 8, 1, master_bytes=-1)


def test_communication_accounting_matches_ddp_bandwidth() -> None:
    total = 1633
    world_size = 4
    param_bytes = 2
    padded = zero.ceil_div(total, world_size) * world_size
    assert padded == 1636

    reduce_scatter, allgather = zero.zero_payload_bytes(
        total, world_size, param_bytes
    )
    assert (reduce_scatter, allgather) == (3272, 3272)
    assert reduce_scatter + allgather == zero.ddp_payload_bytes(
        padded, param_bytes
    )
    assert zero.ddp_payload_bytes(total, param_bytes) == 6532
    assert (
        reduce_scatter + allgather - zero.ddp_payload_bytes(total, param_bytes)
        == 2 * (padded - total) * param_bytes
    )
    assert reduce_scatter + allgather - 6532 == 12

    exact_rs, exact_ag = zero.zero_payload_bytes(1636, 4, 2)
    assert exact_rs + exact_ag == zero.ddp_payload_bytes(1636, 2)

    expected_ring = 2 * total * param_bytes * (world_size - 1) / world_size
    assert math.isclose(
        zero.ring_bandwidth_bytes(total, world_size, param_bytes),
        expected_ring,
    )
    assert zero.ring_bandwidth_bytes(total, 1, param_bytes) == 0.0

    with pytest.raises(ValueError):
        zero.zero_payload_bytes(total, 0, 2)
    with pytest.raises(ValueError):
        zero.zero_payload_bytes(total, 4, 0)
    with pytest.raises(ValueError):
        zero.ddp_payload_bytes(-1, 2)
    with pytest.raises(ValueError):
        zero.ring_bandwidth_bytes(total, 0, 2)


def test_zero_optimizer_requires_initialized_group() -> None:
    model = zero.make_model(7, 5, torch.float32)
    optimizer = zero.ZeroOptimizer(model, 2, 0, lr=1e-3)
    before = optimizer.master.clone()

    with pytest.raises(RuntimeError, match="torch.distributed"):
        optimizer.step()

    assert torch.equal(optimizer.master, before)
    assert optimizer.step_count == 0


def test_run_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        zero.RunConfig(world_size=3, batch_size=10)
    with pytest.raises(ValueError):
        zero.RunConfig(steps=-1)
    with pytest.raises(ValueError):
        zero.RunConfig(world_size=0)
    with pytest.raises(ValueError):
        zero.RunConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="dtype"):
        zero.RunConfig(dtype="fp16")
    with pytest.raises(ValueError, match="nccl"):
        zero.RunConfig(device="cpu", backend="nccl")
    with pytest.raises(ValueError, match="cuda"):
        zero.RunConfig(device="cuda", backend="gloo")

    config = zero.RunConfig()
    assert config.torch_dtype is torch.bfloat16
    assert zero.RunConfig(dtype="fp32").torch_dtype is torch.float32
    assert zero.RunConfig(world_size=2, batch_size=16).batch_size == 16
