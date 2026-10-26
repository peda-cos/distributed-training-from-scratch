"""Shard layout, round-trip, overwrite, and rotation tests."""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import replace
from pathlib import Path

import pytest

from sharded_checkpoint import main as ckpt
from tests.support import make_rank_states, save_checkpoint


def test_round_trip_is_byte_equal_for_every_rank(tmp_path: Path) -> None:
    states = make_rank_states()
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, states, step=3)
    loaded = ckpt.load_sharded(root, 4)

    assert loaded.step == 3
    assert loaded.manifest.world_size == 4
    assert loaded.flat_param_fp16() == b"".join(
        state.param_fp16 for state in states
    )
    assert loaded.flat_master_fp32().tobytes() == b"".join(
        state.master_fp32.tobytes() for state in states
    )
    for saved, restored in zip(states, loaded.ranks):
        assert restored.param_fp16 == saved.param_fp16
        assert restored.master_fp32.tobytes() == saved.master_fp32.tobytes()
        assert restored.exp_avg_fp32.tobytes() == saved.exp_avg_fp32.tobytes()
        assert restored.exp_avg_sq_fp32.tobytes() == saved.exp_avg_sq_fp32.tobytes()
        assert restored.adam_step == saved.adam_step
    assert step_dir == root / "step-00000003"
    assert len(loaded.state_dicts()) == 4


def test_manifest_offsets_sizes_and_hashes(tmp_path: Path) -> None:
    states = make_rank_states()
    step_dir = save_checkpoint(tmp_path / "checkpoint", states, step=3)
    manifest = ckpt.ShardManifest.from_json(
        (step_dir / "manifest.json").read_bytes()
    )

    assert [entry.param_shard_offset for entry in manifest.shards] == [
        0,
        258,
        516,
        774,
    ]
    assert [entry.param_shard_numel for entry in manifest.shards] == [
        258,
        258,
        258,
        257,
    ]
    for entry in manifest.shards:
        raw = (step_dir / entry.path).read_bytes()
        header_length = int.from_bytes(raw[8:12], "little")
        expected_size = 12 + header_length + 14 * entry.param_shard_numel
        assert len(raw) == expected_size
        assert hashlib.sha256(raw).hexdigest() == entry.sha256


def test_save_is_deterministic_and_loads_from_root_or_step_dir(
    tmp_path: Path,
) -> None:
    states = make_rank_states()
    first = save_checkpoint(tmp_path / "first", states, step=3)
    second = save_checkpoint(tmp_path / "second", states, step=3)

    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    for rank in range(4):
        name = f"rank{rank}.bin"
        assert (first / name).read_bytes() == (second / name).read_bytes()

    assert ckpt.load_sharded(tmp_path / "first", 4).step == 3
    assert ckpt.load_sharded(tmp_path / "second", 4).step == 3
    assert ckpt.load_sharded(first, 4).step == 3


def test_committed_step_cannot_be_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    states = make_rank_states()
    step_dir = save_checkpoint(root, states, step=3)
    manifest_before = (step_dir / "manifest.json").read_bytes()
    shard_before = (step_dir / "rank0.bin").read_bytes()

    with pytest.raises(ckpt.CheckpointExistsError):
        save_checkpoint(root, states, step=3)

    assert (step_dir / "manifest.json").read_bytes() == manifest_before
    assert (step_dir / "rank0.bin").read_bytes() == shard_before


def test_save_rejects_malformed_state_layouts(tmp_path: Path) -> None:
    states = make_rank_states()
    wrong_lengths = [
        ckpt.RankState(
            param_fp16=ckpt.encode_fp16(array("f", [0.1] * 300)),
            master_fp32=array("f", [0.1] * 300),
            exp_avg_fp32=array("f", [0.0] * 300),
            exp_avg_sq_fp32=array("f", [0.0] * 300),
            adam_step=3,
        ),
        ckpt.RankState(
            param_fp16=ckpt.encode_fp16(array("f", [0.2] * 731)),
            master_fp32=array("f", [0.2] * 731),
            exp_avg_fp32=array("f", [0.0] * 731),
            exp_avg_sq_fp32=array("f", [0.0] * 731),
            adam_step=3,
        ),
    ]
    with pytest.raises(ValueError, match="canonical split"):
        ckpt.save_sharded(wrong_lengths, tmp_path / "wrong", 1)
    with pytest.raises(ValueError, match="Adam step"):
        ckpt.save_sharded(
            [replace(states[0], adam_step=3), replace(states[1], adam_step=4)],
            tmp_path / "adam",
            1,
        )
    with pytest.raises(ValueError, match="array"):
        ckpt.RankState(
            param_fp16=states[0].param_fp16,
            master_fp32=array("d", [0.0] * states[0].numel),
            exp_avg_fp32=states[0].exp_avg_fp32,
            exp_avg_sq_fp32=states[0].exp_avg_sq_fp32,
        )


def test_rotation_keeps_last_three_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    for step in range(1, 6):
        save_checkpoint(
            root,
            make_rank_states(adam_step=step),
            step=step,
            keep_last=3,
        )

    committed = sorted(
        int(path.name.removeprefix("step-"))
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    assert committed == [3, 4, 5]
    assert ckpt.load_sharded(root, 4).step == 5

    with pytest.raises(ValueError, match="keep_last"):
        save_checkpoint(root, make_rank_states(), step=6, keep_last=0)
    assert not (root / "step-00000006").exists()
