"""World-size, shard-count, corruption, and manifest-path defenses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sharded_checkpoint import main as ckpt
from tests.support import make_rank_states, save_checkpoint


def _step_dir(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoint"
    return save_checkpoint(root, make_rank_states(), step=3)


def test_world_size_mismatch_is_loud(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    save_checkpoint(root, make_rank_states(), step=3)
    with pytest.raises(ckpt.WorldSizeMismatchError, match="world_size=4"):
        ckpt.load_sharded(root, 8)


def test_manifest_shard_count_mismatch_fails_before_shard_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, make_rank_states(), step=3)
    payload = json.loads((step_dir / "manifest.json").read_text())
    payload["world_size"] = 8
    (step_dir / "manifest.json").write_text(json.dumps(payload))
    for path in step_dir.glob("rank*.bin"):
        path.unlink()

    with pytest.raises(ckpt.ManifestError, match="shards for world_size"):
        ckpt.load_sharded(root, 4)


def test_missing_and_extra_shards_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, make_rank_states(), step=3)
    (step_dir / "rank2.bin").unlink()
    with pytest.raises(ckpt.ShardIntegrityError, match="rank2.bin"):
        ckpt.load_sharded(root, 4)

    other_root = tmp_path / "extra"
    other_step = save_checkpoint(other_root, make_rank_states(), step=3)
    (other_step / "rank99.bin").write_bytes(b"stale")
    with pytest.raises(ckpt.ShardIntegrityError, match="rank99.bin"):
        ckpt.load_sharded(other_root, 4)


def test_truncated_and_bit_flipped_shards_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, make_rank_states(), step=3)
    (step_dir / "rank1.bin").write_bytes(b"truncated")
    with pytest.raises(ckpt.ShardIntegrityError):
        ckpt.load_sharded(root, 4)

    other_root = tmp_path / "corrupt"
    other_step = save_checkpoint(other_root, make_rank_states(), step=3)
    raw = bytearray((other_step / "rank0.bin").read_bytes())
    raw[-1] ^= 0x01
    (other_step / "rank0.bin").write_bytes(raw)
    with pytest.raises(ckpt.ShardIntegrityError, match="sha256"):
        ckpt.load_sharded(other_root, 4)


def test_header_must_agree_with_manifest_even_if_hash_is_updated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, make_rank_states(), step=3)
    copied = (step_dir / "rank0.bin").read_bytes()
    (step_dir / "rank1.bin").write_bytes(copied)
    payload = json.loads((step_dir / "manifest.json").read_text())
    payload["shards"][1]["sha256"] = hashlib.sha256(copied).hexdigest()
    (step_dir / "manifest.json").write_text(json.dumps(payload))

    with pytest.raises(ckpt.ShardIntegrityError, match="header rank"):
        ckpt.load_sharded(root, 4)


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, make_rank_states(), step=3)
    payload = json.loads((step_dir / "manifest.json").read_text())
    payload["shards"][0]["path"] = "../rank0.bin"
    (step_dir / "manifest.json").write_text(json.dumps(payload))
    for path in step_dir.glob("rank*.bin"):
        path.unlink()

    with pytest.raises(ckpt.ManifestError, match="path"):
        ckpt.load_sharded(root, 4)
