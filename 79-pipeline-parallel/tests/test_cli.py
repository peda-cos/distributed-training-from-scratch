"""End-to-end CLI test for the four-stage demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_demo_runs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "code/main.py",
            "--microbatches",
            "4",
            "--batch-size",
            "16",
            "--stages",
            "4",
            "--hidden",
            "8",
            "--no-gantt",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0
    assert "ALL CHECKS PASSED" in completed.stdout
    assert "gpipe" in completed.stdout
    assert "1f1b" in completed.stdout
    assert "Traceback" not in completed.stderr
