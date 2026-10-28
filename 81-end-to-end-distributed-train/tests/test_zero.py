"""Flat sharding, memory accounting, and Adam parity tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from end_to_end_distributed import main as demo


def _full_batch_loss(model: torch.nn.Module) -> torch.Tensor:
    inputs, targets = demo.make_batch(demo.make_corpus(11, 512))
    logits = model(inputs)
    return F.cross_entropy(
        logits.reshape(-1, demo.VOCAB_SIZE), targets.reshape(-1)
    )


def test_shard_spec_covers_every_parameter() -> None:
    total = demo.PARAM_COUNT
    for world_size in (1, 2, 4):
        lengths = demo.partition_lengths(total, world_size)
        assert sum(lengths) == total
        assert lengths == [
            spec.local_numel
            for spec in (
                demo.shard_spec(total, world_size, rank)
                for rank in range(world_size)
            )
        ]
        specs = [
            demo.shard_spec(total, world_size, rank)
            for rank in range(world_size)
        ]
        assert specs[0].start == 0
        assert specs[-1].end == total
        for left, right in zip(specs, specs[1:]):
            assert left.end == right.start
        for spec in specs:
            assert spec.padded_numel % world_size == 0
            assert spec.padded_numel >= total
            assert spec.owns_any


def test_optimizer_state_matches_zero1_formula() -> None:
    model = demo.make_model(demo.DEFAULT_SEED)
    optimizer = demo.Zero1Optimizer(model, 0, 4)
    _full_batch_loss(model).backward()
    memory = demo.measure_memory(model, optimizer)

    assert optimizer.state_nbytes() == 3 * 4 * (demo.PARAM_COUNT // 4)
    assert optimizer.state_nbytes() == 90432 == 12 * demo.PARAM_COUNT // 4
    assert memory.params_bytes == 60288
    assert memory.grads_bytes == 60288
    assert memory.optimizer_bytes == 90432
    assert memory.total_bytes == 211008
    assert memory.formula_bytes == 90432
    assert memory.formula_ok


def test_world_one_adam_step_matches_torch_optimizer() -> None:
    model = demo.make_model(demo.DEFAULT_SEED)
    inputs, targets = demo.make_batch(demo.make_corpus(11, 512))
    loss = F.cross_entropy(
        model(inputs).reshape(-1, demo.VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()
    initial = demo.flatten_params(model).float().clone()
    grad = demo.flatten_grads(model).float().clone()
    optimizer = demo.Zero1Optimizer(model, 0, 1, lr=0.01)
    optimizer.step()

    reference_master = torch.nn.Parameter(initial.clone())
    reference = torch.optim.Adam(
        [reference_master], lr=0.01, betas=(0.9, 0.999), eps=1e-8,
        foreach=False,
    )
    reference_master.grad = grad.clone()
    reference.step()

    assert torch.allclose(
        optimizer.master, reference_master.detach(), atol=1e-6, rtol=0
    )
    assert optimizer.step_count == 1


def test_world_one_optimizer_skips_collectives() -> None:
    model = demo.make_model(demo.DEFAULT_SEED)
    optimizer = demo.Zero1Optimizer(model, 0, 1, lr=0.01)
    before = float(_full_batch_loss(model).detach())
    optimizer.zero_grad()
    loss = _full_batch_loss(model)
    loss.backward()
    optimizer.step()
    after = float(_full_batch_loss(model).detach())

    assert optimizer.comm.calls == 0
    assert optimizer.comm.payload_bytes == 0
    assert optimizer.step_count == 1
    assert after < before
    assert optimizer.master.numel() == demo.PARAM_COUNT
