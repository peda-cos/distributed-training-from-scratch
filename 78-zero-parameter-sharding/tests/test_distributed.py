"""Integration tests: real gloo ranks, exact shard bytes, and the CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from zero_sharding import main as zero

ROOT = Path(__file__).resolve().parents[1]


def _replicas_equal(run: zero.ZeroRun, step: int) -> bool:
    reference = run.params_by_rank[0][step]
    return all(
        torch.equal(run.params_by_rank[rank][step], reference)
        for rank in range(run.config.world_size)
    )


def test_three_rank_zero_matches_single_process_adam() -> None:
    config = zero.RunConfig(
        world_size=3,
        steps=4,
        batch_size=48,
        hidden_dim=8,
        lr=0.005,
        dtype="fp32",
    )
    reference = zero.train_reference(config)
    run = zero.run_zero(config)
    comparison = zero.compare_runs(run, reference)

    assert comparison.passed
    assert comparison.max_loss_delta <= 1e-6
    assert comparison.max_param_delta <= 1e-5
    assert all(comparison.replica_equal)
    for step in range(1, config.steps + 1):
        assert _replicas_equal(run, step)

    comm = run.comm
    assert comm["reduce_scatter_calls"] == config.steps
    assert comm["allgather_calls"] == config.steps
    assert comm["reduce_scatter_bytes"] == config.steps * 219 * 4
    assert comm["allgather_bytes"] == config.steps * 219 * 4
    assert comm["payload_bytes"] == config.steps * 2 * 219 * 4

    ranges = [
        (payload["spec"]["start"], payload["spec"]["end"])
        for payload in run.rank_payloads
    ]
    assert ranges == [(0, 73), (73, 146), (146, 217)]
    assert run.spec["shard_numel"] == 73
    assert run.spec["padded_numel"] == 219
    for payload in run.rank_payloads:
        assert payload["memory"]["optimizer_bytes"] == 3 * 4 * 73 == 876


def test_bf16_mixed_precision_matches_reference() -> None:
    config = zero.RunConfig(
        world_size=4,
        steps=3,
        batch_size=32,
        hidden_dim=9,
        dtype="bf16",
    )
    reference = zero.train_reference(config)
    run = zero.run_zero(config)
    comparison = zero.compare_runs(run, reference)

    assert comparison.passed
    assert comparison.max_loss_delta <= 1e-2
    assert comparison.max_param_delta <= 5e-2
    assert all(comparison.replica_equal)
    for step in range(1, config.steps + 1):
        assert _replicas_equal(run, step)

    total = 253
    memory = run.memory
    assert run.spec["total_numel"] == total
    assert memory["params_bytes"] == total * 2
    assert memory["grads_bytes"] == total * 2
    assert memory["optimizer_bytes"] == 3 * 4 * 64 == 768
    ddp_total = zero.memory_budget(total, 4, 0).total_bytes
    assert memory["total_bytes"] < ddp_total
    assert 1.0 - memory["total_bytes"] / ddp_total > 0.5


def test_world_size_one_skips_collectives() -> None:
    config = zero.RunConfig(
        world_size=1,
        steps=3,
        batch_size=16,
        hidden_dim=5,
        dtype="fp32",
    )
    reference = zero.train_reference(config)
    run = zero.run_zero(config)
    comparison = zero.compare_runs(run, reference)

    assert comparison.passed
    assert comparison.max_param_delta <= 1e-5
    assert run.comm == {
        "reduce_scatter_calls": 0,
        "allgather_calls": 0,
        "reduce_scatter_bytes": 0,
        "allgather_bytes": 0,
        "calls": 0,
        "payload_bytes": 0,
    }
    total = 121
    assert run.memory["optimizer_bytes"] == 3 * 4 * total
    assert (run.spec["start"], run.spec["end"]) == (0, total)


def test_demo_cli_prints_memory_table_and_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "code/main.py",
            "--steps",
            "2",
            "--world-size",
            "2",
            "--hidden",
            "6",
            "--batch-size",
            "16",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout
    for needle in ("ALL CHECKS PASSED", "ZeRO-1", "reduce_scatter", "total", "PASS"):
        assert needle in output
    assert "CHECKS FAILED" not in output
