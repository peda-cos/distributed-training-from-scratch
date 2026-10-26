"""Deterministic state builders shared by the checkpoint tests."""

from __future__ import annotations

import math
from array import array
from pathlib import Path

from sharded_checkpoint import main as ckpt


def make_rank_states(
    total_numel: int = 1031,
    world_size: int = 4,
    adam_step: int = 3,
) -> list[ckpt.RankState]:
    lengths = ckpt.partition_lengths(total_numel, world_size)
    states: list[ckpt.RankState] = []
    offset = 0
    for rank, length in enumerate(lengths):
        master = array(
            "f",
            (
                0.02 * math.sin(0.017 * (offset + index + 1))
                + 0.001 * math.cos(0.003 * (offset + index + rank))
                for index in range(length)
            ),
        )
        exp_avg = array(
            "f",
            (
                0.0003 * math.sin(0.011 * (offset + index) + 0.4)
                for index in range(length)
            ),
        )
        exp_avg_sq = array(
            "f",
            (
                0.0002 + 0.0001 * abs(math.cos(0.013 * (offset + index)))
                for index in range(length)
            ),
        )
        states.append(
            ckpt.RankState(
                param_fp16=ckpt.encode_fp16(master),
                master_fp32=master,
                exp_avg_fp32=exp_avg,
                exp_avg_sq_fp32=exp_avg_sq,
                adam_step=adam_step,
            )
        )
        offset += length
    return states


def save_checkpoint(
    root: str | Path,
    states: list[ckpt.RankState] | None = None,
    *,
    step: int = 3,
    wall_clock_seconds: int = 3600,
    keep_last: int | None = None,
) -> Path:
    if states is None:
        states = make_rank_states()
    return ckpt.save_sharded(
        states,
        root,
        step,
        wall_clock_seconds=wall_clock_seconds,
        keep_last=keep_last,
    )
