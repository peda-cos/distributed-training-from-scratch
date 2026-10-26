"""fp16 codec and exact optimizer-resume tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sharded_checkpoint import main as ckpt
from tests.support import save_checkpoint


def test_fp16_binary16_round_trip_and_quantization() -> None:
    encoded = ckpt.encode_fp16([1.5, -2.25, 0.1])
    assert encoded == bytes.fromhex("003e80c0662e")
    decoded = ckpt.decode_fp16(encoded)
    assert list(decoded) == [1.5, -2.25, 0.0999755859375]

    with pytest.raises(ValueError, match="multiple of 2"):
        ckpt.decode_fp16(b"\x00")
    with pytest.raises(ValueError, match="binary16"):
        ckpt.encode_fp16([70000.0])


def test_resume_continues_adam_exactly(tmp_path: Path) -> None:
    initial = ckpt.make_training_state(1031)
    uninterrupted = ckpt.train_steps(initial, 8)
    before_save = ckpt.train_steps(initial, 3)
    states = ckpt.shard_training_state(before_save, 4)
    root = tmp_path / "checkpoint"
    save_checkpoint(
        root,
        states,
        step=3,
        wall_clock_seconds=17,
        keep_last=3,
    )

    loaded = ckpt.load_sharded(root, 4)
    resumed = ckpt.train_steps(ckpt.unshard_states(loaded.ranks), 5)

    assert loaded.manifest.wall_clock_seconds == 17
    assert resumed.step == uninterrupted.step == 8
    assert resumed.param_fp16 == uninterrupted.param_fp16
    assert resumed.master_fp32.tobytes() == uninterrupted.master_fp32.tobytes()
    assert resumed.exp_avg_fp32.tobytes() == uninterrupted.exp_avg_fp32.tobytes()
    assert resumed.exp_avg_sq_fp32.tobytes() == uninterrupted.exp_avg_sq_fp32.tobytes()
    max_delta = max(
        (
            abs(left - right)
            for left, right in zip(
                resumed.master_fp32, uninterrupted.master_fp32
            )
        ),
        default=0.0,
    )
    assert max_delta == 0.0


def test_loaded_state_is_fresh_on_every_load(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    step_dir = save_checkpoint(root, ckpt.shard_training_state(
        ckpt.train_steps(ckpt.make_training_state(1031), 3), 4
    ), step=3)
    loaded = ckpt.load_sharded(root, 4)
    original = loaded.ranks[0].master_fp32[0]
    shard_before = (step_dir / "rank0.bin").read_bytes()

    loaded.ranks[0].master_fp32[0] = original + 1.0

    reloaded = ckpt.load_sharded(root, 4)
    assert reloaded.ranks[0].master_fp32[0] == original
    assert (step_dir / "rank0.bin").read_bytes() == shard_before
