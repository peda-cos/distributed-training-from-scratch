"""Crash-before-commit and retry tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sharded_checkpoint import main as ckpt
from tests.support import make_rank_states, save_checkpoint


def test_crash_before_manifest_rename_keeps_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoint"
    first_states = make_rank_states(adam_step=3)
    save_checkpoint(root, first_states, step=1)
    first_manifest = (root / "step-00000001" / "manifest.json").read_bytes()

    implementation = ckpt.save_sharded.__globals__
    original_replace = implementation["_replace"]

    def crash_on_manifest(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == "manifest.json":
            raise OSError("injected crash before manifest commit")
        original_replace(source, destination)

    monkeypatch.setitem(implementation, "_replace", crash_on_manifest)
    with pytest.raises(OSError, match="injected crash"):
        save_checkpoint(root, make_rank_states(adam_step=4), step=2)

    step_two = root / "step-00000002"
    assert not (step_two / "manifest.json").exists()
    assert (step_two / "manifest.json.tmp").is_file()
    assert len(list(step_two.glob("rank*.bin"))) == 4
    loaded = ckpt.load_sharded(root, 4)
    assert loaded.step == 1
    assert loaded.flat_param_fp16() == b"".join(
        state.param_fp16 for state in first_states
    )
    assert (root / "step-00000001" / "manifest.json").read_bytes() == first_manifest


def test_retry_after_crash_cleans_and_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoint"
    save_checkpoint(root, make_rank_states(adam_step=3), step=1)
    implementation = ckpt.save_sharded.__globals__
    original_replace = implementation["_replace"]

    def crash_on_manifest(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == "manifest.json":
            raise OSError("injected crash before manifest commit")
        original_replace(source, destination)

    monkeypatch.setitem(implementation, "_replace", crash_on_manifest)
    with pytest.raises(OSError):
        save_checkpoint(root, make_rank_states(adam_step=4), step=2)
    monkeypatch.setitem(implementation, "_replace", original_replace)

    step_two = save_checkpoint(root, make_rank_states(adam_step=5), step=2)
    assert (step_two / "manifest.json").is_file()
    assert not list(step_two.glob("*.tmp"))
    loaded = ckpt.load_sharded(root, 4)
    assert loaded.step == 2
    assert len(list(step_two.glob("rank*.bin"))) == 4


def test_uncommitted_step_directory_is_ignored_and_untouched(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    save_checkpoint(root, make_rank_states(), step=1)
    junk = root / "step-00000099"
    junk.mkdir()
    (junk / "rank0.bin").write_bytes(b"not a real shard")

    assert ckpt.load_sharded(root, 4).step == 1
    save_checkpoint(root, make_rank_states(adam_step=4), step=2)

    assert (junk / "rank0.bin").read_bytes() == b"not a real shard"
    assert ckpt.load_sharded(root, 4).step == 2
