"""Deterministic checkpoint states shared by the checkpoint tests."""

from __future__ import annotations

import math
from array import array
from pathlib import Path

from end_to_end_distributed import main as demo


def make_rank_states(
    total_numel: int = demo.PARAM_COUNT,
    world_size: int = 4,
    adam_step: int = 10,
) -> list[demo.RankState]:
    lengths = demo.partition_lengths(total_numel, world_size)
    states: list[demo.RankState] = []
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
            demo.RankState(
                rank=rank,
                param_fp16=demo.encode_fp16(master),
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
    states: list[demo.RankState] | None = None,
    *,
    step: int = 10,
) -> Path:
    if states is None:
        states = make_rank_states()
    return demo.save_sharded(states, root, step)
