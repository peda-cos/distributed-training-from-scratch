"""End-to-end tests: spawned gloo ranks, replica identity, and the CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from end_to_end_distributed import main as demo

ROOT = Path(__file__).resolve().parents[1]


def _losses(payloads: list[dict]) -> list[float]:
    return [float(record["loss"]) for record in payloads[0]["records"]]


def test_init_broadcast_overwrites_rank_jitter(tmp_path: Path) -> None:
    config = demo.RunConfig(world_size=2, steps=1, checkpoint_step=1)
    payloads = demo.run_distributed(config, tmp_path / "broadcast")

    pre_hashes = {payload["pre_init_hash"] for payload in payloads}
    post_hashes = {payload["post_init_hash"] for payload in payloads}
    expected = demo.state_digest(
        demo.flatten_params(demo.make_model(config.seed))
    )

    assert len(payloads) == 2
    assert len(pre_hashes) > 1
    assert post_hashes == {expected}
    assert all(payload["resume_verified"] for payload in payloads)


def test_world_sizes_agree_on_shared_full_batch(tmp_path: Path) -> None:
    one = demo.RunConfig(world_size=1, steps=5, checkpoint_step=3)
    four = demo.RunConfig(world_size=4, steps=5, checkpoint_step=3)
    one_root = tmp_path / "one"
    four_root = tmp_path / "four"
    one_payloads = demo.run_distributed(one, one_root)
    four_payloads = demo.run_distributed(four, four_root)

    assert demo.validate_run(one, one_payloads, one_root / "checkpoint") == []
    assert demo.validate_run(four, four_payloads, four_root / "checkpoint") == []
    one_losses = _losses(one_payloads)
    four_losses = _losses(four_payloads)
    assert abs(one_losses[-1] - four_losses[-1]) <= 1e-2
    for losses in (one_losses, four_losses):
        assert all(
            later <= earlier + demo.LOSS_NOISE_TOLERANCE
            for earlier, later in zip(losses, losses[1:])
        )
    assert all(payload["resume_verified"] for payload in one_payloads)
    assert all(payload["resume_verified"] for payload in four_payloads)


def test_demo_prints_20_loss_rows_and_verifies_resume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "code/main.py",
            "--world-size",
            "4",
            "--steps",
            "20",
            "--checkpoint-step",
            "10",
            "--checkpoint-root",
            str(root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout

    loss_section = output.split("LOSS CURVE (global mean, 20 steps)", 1)[1]
    loss_section = loss_section.split("MEMORY PROFILE", 1)[0]
    loss_rows = []
    for line in loss_section.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            loss_rows.append((int(parts[0]), float(parts[1])))
    assert [step for step, _ in loss_rows] == list(range(1, 21))
    losses = [loss for _, loss in loss_rows]
    assert all(
        later <= earlier + demo.LOSS_NOISE_TOLERANCE
        for earlier, later in zip(losses, losses[1:])
    )
    assert losses[-1] < losses[0]

    memory_section = output.split("MEMORY PROFILE", 1)[1].split(
        "CHECKPOINT MANIFEST", 1
    )[0]
    memory_rows = [
        line
        for line in memory_section.splitlines()
        if line.split() and line.split()[0].isdigit()
    ]
    assert len(memory_rows) == 4
    assert "RESUME VERIFIED (4/4 rank shards byte-equal)" in output
    assert "ALL CHECKS PASSED" in output

    run_dirs = list(root.glob("run-*"))
    assert len(run_dirs) == 1
    step_dir = run_dirs[0] / "checkpoint" / "step-00000010"
    assert len(list(step_dir.glob("rank*.bin"))) == 4
    loaded = demo.load_checkpoint(run_dirs[0] / "checkpoint", 4)
    assert loaded.step == 10
    assert loaded.manifest.param_numel == demo.PARAM_COUNT
    assert loaded.manifest.adam_step == 10


def test_rank_payloads_have_identical_norms_and_hashes_every_step(
    tmp_path: Path,
) -> None:
    config = demo.RunConfig(world_size=4, steps=5, checkpoint_step=3)
    run_root = tmp_path / "ranks"
    payloads = demo.run_distributed(config, run_root)

    assert demo.validate_run(config, payloads, run_root / "checkpoint") == []
    for index in range(config.steps):
        records = [payload["records"][index] for payload in payloads]
        assert len({record["param_hash"] for record in records}) == 1
        norms = [float(record["param_norm"]) for record in records]
        assert max(norms) - min(norms) <= 1e-12
        assert all(record["replica_equal"] for record in records)
    for payload in payloads:
        assert payload["memory"]["optimizer_bytes"] == 90432
        assert payload["resume_verified"] is True
