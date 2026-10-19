"""End-to-end tests: real gloo ranks, file rendezvous, and spawned workers."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from data_parallel_ddp import main as ddp

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> ddp.RunConfig:
    return replace(
        ddp.RunConfig(world_size=2, steps=3),
        **overrides,
    )


def test_broadcast_overwrites_rank_specific_init() -> None:
    config = ddp.RunConfig(world_size=2, steps=0, init_jitter=True)
    run = ddp.run_ddp(config)

    pre_hashes = {
        payload["pre_broadcast_hash"] for payload in run.rank_payloads
    }
    post_hashes = {
        payload["post_broadcast_hash"] for payload in run.rank_payloads
    }
    assert len(pre_hashes) > 1
    assert len(post_hashes) == 1
    assert post_hashes == {ddp.state_hash(ddp.make_model(config.seed))}
    assert run.rank_payloads[0]["stats"]["broadcast_bytes"] > 0
    assert run.params_by_rank[0].shape[0] == 0


def test_ddp_training_matches_reference_each_step() -> None:
    config = _config(world_size=4, steps=6)
    reference = ddp.run_reference(config)
    run = ddp.run_ddp(config)
    comparison = ddp.compare_runs(run, reference)

    assert comparison.passed
    assert all(comparison.replica_equal)
    assert comparison.max_loss_delta <= 1e-12
    assert comparison.max_param_delta <= 1e-12

    losses = [
        ddp.mean_rank_loss(run, step) for step in range(config.steps)
    ]
    assert losses[-1] < losses[0]

    payload = run.rank_payloads[0]
    assert payload["bucket_count"] == 1
    assert payload["stats"]["allreduce_calls"] == (
        config.steps * payload["bucket_count"]
    )


def test_mean_gradient_is_world_size_invariant() -> None:
    config_two = _config(world_size=2, steps=3)
    reference = ddp.run_reference(config_two)
    run_two = ddp.run_ddp(config_two)
    comparison_two = ddp.compare_runs(run_two, reference)
    assert comparison_two.passed

    config_four = replace(config_two, world_size=4)
    run_four = ddp.run_ddp(config_four)
    comparison_four = ddp.compare_runs(run_four, reference)
    assert comparison_four.passed

    final_two = run_two.params_by_rank[0][-1]
    final_four = run_four.params_by_rank[0][-1]
    assert torch.max(torch.abs(final_two - final_four)).item() <= 1e-12


def test_buckets_fuse_reductions_and_launch_during_backward() -> None:
    config = _config(world_size=2, steps=4, bucket_cap_mb=0.004)
    reference = ddp.run_reference(config)
    run = ddp.run_ddp(config)
    comparison = ddp.compare_runs(run, reference)

    assert comparison.passed
    assert all(comparison.replica_equal)
    payload = run.rank_payloads[0]
    bucket_count = payload["bucket_count"]
    param_count = payload["param_count"]
    assert bucket_count >= 2
    assert payload["stats"]["allreduce_calls"] == config.steps * bucket_count
    assert payload["stats"]["overlapped_buckets"] == (
        config.steps * bucket_count
    )
    assert payload["stats"]["allreduce_calls"] < config.steps * param_count


def test_naive_and_bucketed_modes_agree() -> None:
    config = _config(world_size=2, steps=3, bucket_cap_mb=0.004)
    reference = ddp.run_reference(config)
    naive = ddp.run_ddp(replace(config, sync_mode="naive"))
    bucketed = ddp.run_ddp(replace(config, sync_mode="bucketed"))
    naive_comparison = ddp.compare_runs(naive, reference)
    bucketed_comparison = ddp.compare_runs(bucketed, reference)

    assert naive_comparison.passed
    assert bucketed_comparison.passed
    for step in range(config.steps):
        assert torch.allclose(
            naive.params_by_rank[0][step],
            bucketed.params_by_rank[0][step],
            atol=1e-12,
            rtol=0,
        )

    naive_payload = naive.rank_payloads[0]
    bucketed_payload = bucketed.rank_payloads[0]
    assert naive_payload["bucket_count"] == 0
    assert naive_payload["stats"]["allreduce_calls"] == (
        config.steps * naive_payload["param_count"]
    )
    assert naive_payload["stats"]["overlapped_buckets"] == 0
    assert bucketed_payload["bucket_count"] >= 2
    assert bucketed_payload["stats"]["allreduce_calls"] == (
        config.steps * bucketed_payload["bucket_count"]
    )
    assert bucketed_payload["stats"]["overlapped_buckets"] == (
        config.steps * bucketed_payload["bucket_count"]
    )


def test_no_sync_accumulation_matches_single_process() -> None:
    config = _config(
        world_size=2,
        steps=4,
        batch_size=96,
        no_sync_microbatches=3,
    )
    reference = ddp.run_reference(config)
    run = ddp.run_ddp(config)
    comparison = ddp.compare_runs(run, reference)

    assert comparison.passed
    assert all(comparison.replica_equal)
    payload = run.rank_payloads[0]
    assert payload["stats"]["allreduce_calls"] == (
        config.steps * payload["bucket_count"]
    )


def test_find_unused_parameters_handles_branching_forward() -> None:
    config = _config(
        world_size=2,
        steps=3,
        use_hidden=False,
        find_unused_parameters=True,
    )
    reference = ddp.run_reference(config)
    run = ddp.run_ddp(config)
    comparison = ddp.compare_runs(run, reference)

    assert comparison.passed
    for payload in run.rank_payloads:
        assert set(payload["none_grad_names"]) == {
            "fc2.weight",
            "fc2.bias",
        }
        assert payload["stats"]["overlapped_buckets"] == 0


def test_cli_demo_reports_pass() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "code/main.py",
            "--steps",
            "3",
            "--world-size",
            "2",
            "--bucket-cap-mb",
            "0.004",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ALL CHECKS PASSED" in completed.stdout
    assert "replicas" in completed.stdout
    assert "PASS" in completed.stdout


def test_cli_benchmark_reports_both_modes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "code/main.py",
            "--benchmark",
            "--steps",
            "2",
            "--world-size",
            "2",
            "--bucket-cap-mb",
            "0.004",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "naive" in completed.stdout
    assert "bucketed" in completed.stdout
    assert "ALL CHECKS PASSED" in completed.stdout
