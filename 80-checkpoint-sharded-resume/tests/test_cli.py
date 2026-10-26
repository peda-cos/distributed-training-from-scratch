"""End-to-end test of the spec's direct script command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sharded_checkpoint import main as ckpt

ROOT = Path(__file__).resolve().parents[1]


def test_demo_cli_writes_shards_and_verifies_resume(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "demo"
    result = subprocess.run(
        [
            sys.executable,
            "code/main.py",
            "--dir",
            str(checkpoint_root),
            "--world-size",
            "4",
            "--total-numel",
            "257",
            "--checkpoint-step",
            "2",
            "--resume-steps",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "byte-equal shards: PASS" in result.stdout
    assert "resume exact: PASS (max |delta|=0.0)" in result.stdout
    assert "ALL CHECKS PASSED" in result.stdout

    step_dir = checkpoint_root / "step-00000002"
    assert sorted(path.name for path in step_dir.glob("rank*.bin")) == [
        "rank0.bin",
        "rank1.bin",
        "rank2.bin",
        "rank3.bin",
    ]
    assert (step_dir / "manifest.json").is_file()
    loaded = ckpt.load_sharded(checkpoint_root, 4)
    assert loaded.step == 2
    assert loaded.manifest.world_size == 4
