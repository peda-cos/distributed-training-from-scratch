"""Shard layout, byte-equal round-trip, resume, and defense tests."""

from __future__ import annotations

import hashlib
import json
from array import array
from pathlib import Path

import pytest

from end_to_end_distributed import main as demo
from tests.support import make_rank_states, save_checkpoint


def test_checkpoint_round_trip_is_byte_equal(tmp_path: Path) -> None:
    states = make_rank_states()
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, states, step=10)
    loaded = demo.load_checkpoint(root, 4)

    assert step_dir == root / "step-00000010"
    assert loaded.step == 10
    assert loaded.manifest.world_size == 4
    assert len(loaded.ranks) == 4
    for saved, restored in zip(states, loaded.ranks):
        assert restored.payload_bytes() == saved.payload_bytes()
        assert restored.adam_step == saved.adam_step == 10
    assert loaded.flat_master_fp32().tobytes() == b"".join(
        state.master_fp32.tobytes() for state in states
    )


def test_manifest_offsets_hashes_and_adam_step(tmp_path: Path) -> None:
    states = make_rank_states()
    step_dir = save_checkpoint(tmp_path / "checkpoint", states, step=10)
    manifest = demo.ShardManifest.from_json(
        (step_dir / "manifest.json").read_bytes()
    )

    assert [entry.param_shard_offset for entry in manifest.shards] == [
        0,
        7536,
        15072,
        22608,
    ]
    assert [entry.param_shard_numel for entry in manifest.shards] == [
        7536,
        7536,
        7536,
        7536,
    ]
    assert sum(entry.param_shard_numel for entry in manifest.shards) == 30144
    assert manifest.optimizer_bytes_per_rank == 90432
    assert manifest.adam_step == 10
    for entry in manifest.shards:
        raw = (step_dir / entry.path).read_bytes()
        header_length = int.from_bytes(raw[8:12], "little")
        assert len(raw) == 12 + header_length + 14 * entry.param_shard_numel
        assert hashlib.sha256(raw).hexdigest() == entry.sha256
        header = json.loads(raw[12 : 12 + header_length])
        assert header["rank"] == entry.rank
        assert header["step"] == 10
        assert header["adam_step"] == 10


def test_verify_resume_rejects_altered_master(tmp_path: Path) -> None:
    states = make_rank_states()
    root = tmp_path / "checkpoint"
    save_checkpoint(root, states, step=10)

    assert demo.verify_resume(root, states[2], 4).step == 10

    altered = list(states[2].master_fp32)
    altered[0] += 0.5
    changed = demo.RankState(
        rank=states[2].rank,
        param_fp16=states[2].param_fp16,
        master_fp32=array("f", altered),
        exp_avg_fp32=states[2].exp_avg_fp32,
        exp_avg_sq_fp32=states[2].exp_avg_sq_fp32,
        adam_step=states[2].adam_step,
    )
    with pytest.raises(demo.ResumeMismatchError, match="rank 2"):
        demo.verify_resume(root, changed, 4)


def test_corruption_and_world_size_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, make_rank_states(), step=10)

    with pytest.raises(demo.WorldSizeMismatchError):
        demo.load_checkpoint(root, 8)

    raw = bytearray((step_dir / "rank1.bin").read_bytes())
    raw[-1] ^= 0x01
    (step_dir / "rank1.bin").write_bytes(raw)
    with pytest.raises(demo.ShardIntegrityError, match="sha256"):
        demo.load_checkpoint(root, 4)

    other_root = tmp_path / "truncated"
    other_step = save_checkpoint(other_root, make_rank_states(), step=10)
    (other_step / "rank1.bin").write_bytes(b"truncated")
    with pytest.raises(demo.ShardIntegrityError):
        demo.load_checkpoint(other_root, 4)


def test_uncommitted_step_is_ignored_and_missing_shard_fails(
    tmp_path: Path,
) -> None:
    states = make_rank_states()
    uncommitted = tmp_path / "uncommitted"
    demo.write_rank_shard(
        uncommitted, 10, states[0], demo.PARAM_COUNT, 4
    )
    with pytest.raises(demo.CheckpointError, match="no committed checkpoint"):
        demo.load_checkpoint(uncommitted, 4)

    root = tmp_path / "committed"
    step_dir = save_checkpoint(root, states, step=10)
    (step_dir / "rank2.bin").unlink()
    with pytest.raises(demo.ShardIntegrityError, match="rank2.bin"):
        demo.load_checkpoint(root, 4)


def test_manifest_rejects_noncanonical_paths_and_headers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "path"
    step_dir = save_checkpoint(root, make_rank_states(), step=10)
    payload = json.loads((step_dir / "manifest.json").read_text())
    payload["shards"][0]["path"] = "../rank0.bin"
    (step_dir / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(demo.ManifestError, match="rank0.bin"):
        demo.load_checkpoint(root, 4)

    other_root = tmp_path / "header"
    other_step = save_checkpoint(other_root, make_rank_states(), step=10)
    copied = (other_step / "rank0.bin").read_bytes()
    (other_step / "rank1.bin").write_bytes(copied)
    payload = json.loads((other_step / "manifest.json").read_text())
    payload["shards"][1]["sha256"] = hashlib.sha256(copied).hexdigest()
    (other_step / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(demo.ShardIntegrityError, match="header rank"):
        demo.load_checkpoint(other_root, 4)
