"""Pure tests: model determinism, data sharding, buckets, hashes, config."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_parallel_ddp import main as ddp


def test_model_is_three_layer_mlp_and_seed_deterministic() -> None:
    model = ddp.make_model(7)
    assert sum(isinstance(module, nn.Linear) for module in model.modules()) == 3
    assert model(torch.randn(5, ddp.INPUT_DIM, dtype=torch.float64)).shape == (5, 1)

    first = ddp.make_model(7)
    second = ddp.make_model(7)
    other = ddp.make_model(8)
    assert ddp.state_hash(first) == ddp.state_hash(second)
    assert ddp.state_hash(first) != ddp.state_hash(other)

    # The unused-parameter branch still runs three linear layers' worth of
    # parameter registration; it just skips fc2 in the forward.
    assert model(torch.randn(4, ddp.INPUT_DIM, dtype=torch.float64), use_hidden=False).shape == (4, 1)


def test_global_batch_shards_are_disjoint_and_cover_batch() -> None:
    indices = ddp.batch_indices(
        step=3, batch_size=96, dataset_size=ddp.DATASET_SIZE, shuffle_seed=99
    )
    micro_batches = ddp.micro_batches(indices, 3)
    assert len(micro_batches) == 3
    assert all(batch.numel() == 32 for batch in micro_batches)

    seen: set[int] = set()
    for batch in micro_batches:
        chunks = [
            ddp.rank_slice(batch, rank, 4) for rank in range(4)
        ]
        assert all(chunk.numel() == 8 for chunk in chunks)
        reconstructed = torch.cat(chunks)
        assert torch.equal(reconstructed, batch)
        for chunk in chunks:
            values = set(chunk.tolist())
            assert seen.isdisjoint(values)
            seen.update(values)
    assert len(seen) == 96


def test_plan_buckets_covers_all_parameters_and_respects_cap() -> None:
    model = ddp.make_model(7)
    parameters = list(model.parameters())
    buckets = ddp.plan_buckets(parameters, cap_bytes=4194)

    assert len(buckets) >= 2
    flattened = [parameter for bucket in buckets for parameter in bucket]
    assert [id(parameter) for parameter in flattened] == [
        id(parameter) for parameter in reversed(parameters)
    ]

    for bucket in buckets:
        bucket_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in bucket
        )
        if len(bucket) > 1:
            assert bucket_bytes <= 4194
        else:
            assert bucket_bytes > 0
    assert sum(
        parameter.numel() * parameter.element_size()
        for parameter in flattened
    ) == sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
    )


def test_state_hash_is_sensitive_to_parameter_values() -> None:
    model = ddp.make_model(7)
    first = ddp.state_hash(model)
    assert ddp.state_hash(model) == first
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert ddp.state_hash(model) != first


def test_config_rejects_invalid_splits() -> None:
    with pytest.raises(ValueError):
        ddp.RunConfig(world_size=0)
    with pytest.raises(ValueError):
        ddp.RunConfig(world_size=3, batch_size=64)
    with pytest.raises(ValueError):
        ddp.RunConfig(world_size=2, batch_size=64, no_sync_microbatches=3)
    with pytest.raises(ValueError):
        ddp.RunConfig(bucket_cap_mb=0)
    with pytest.raises(ValueError):
        ddp.RunConfig(dtype="float16")


def test_world_size_one_wrapper_matches_plain_sgd() -> None:
    torch.manual_seed(7)
    wrapped_model = ddp.make_model(7)
    plain_model = ddp.make_model(7)
    wrapped = ddp.DistributedDataParallel(wrapped_model, world_size=1)
    wrapped_optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    plain_optimizer = torch.optim.SGD(plain_model.parameters(), lr=0.1)

    x, y = ddp.make_dataset(ddp.DATA_SEED, size=64)
    indices = ddp.batch_indices(0, 8, 64, ddp.SHUFFLE_SEED)

    wrapped_loss = F.mse_loss(wrapped(x[indices]), y[indices])
    wrapped_loss.backward()
    wrapped.sync_grads()
    wrapped_optimizer.step()

    plain_loss = F.mse_loss(plain_model(x[indices]), y[indices])
    plain_loss.backward()
    plain_optimizer.step()

    assert torch.allclose(
        torch.tensor(wrapped_loss.item()),
        torch.tensor(plain_loss.item()),
        atol=0,
        rtol=0,
    )
    for wrapped_param, plain_param in zip(
        wrapped.parameters(), plain_model.parameters()
    ):
        assert torch.equal(wrapped_param, plain_param)


def test_missing_grad_guard_lists_skipped_layer() -> None:
    x, y = ddp.make_dataset(ddp.DATA_SEED, size=64)
    indices = ddp.batch_indices(0, 8, 64, ddp.SHUFFLE_SEED)

    skipped_model = ddp.make_model(7)
    skipped_loss = F.mse_loss(
        skipped_model(x[indices], use_hidden=False), y[indices]
    )
    skipped_loss.backward()
    assert set(ddp._missing_grad_params(skipped_model)) == {
        "fc2.weight",
        "fc2.bias",
    }

    full_model = ddp.make_model(7)
    full_loss = F.mse_loss(full_model(x[indices]), y[indices])
    full_loss.backward()
    assert ddp._missing_grad_params(full_model) == []
