"""Schedule-order, bubble, activation-memory, and Gantt tests."""

from __future__ import annotations

from collections import Counter

import pytest

from pipeline_parallel import main as pipeline


def _by_key(trace: pipeline.ScheduleTrace) -> dict[tuple[int, int, str], pipeline.Op]:
    return {(op.stage, op.microbatch, op.kind): op for op in trace.ops}


def test_bubble_fraction_closed_form() -> None:
    assert pipeline.bubble_fraction(4, 8) == pytest.approx(3 / 11)
    assert pipeline.bubble_fraction(4, 64) == pytest.approx(3 / 67)
    assert pipeline.bubble_fraction(1, 5) == 0.0
    assert pipeline.bubble_fraction(4, 8) > pipeline.bubble_fraction(4, 64)

    with pytest.raises(ValueError):
        pipeline.bubble_fraction(0, 8)
    with pytest.raises(ValueError):
        pipeline.bubble_fraction(4, 0)


def test_gpipe_trace_matches_closed_form() -> None:
    trace = pipeline.build_trace("gpipe", 4, 8)

    assert trace.makespan == pytest.approx(22.0)
    assert trace.idle_fraction == pytest.approx(pipeline.bubble_fraction(4, 8))
    assert len(trace.ops) == 2 * 4 * 8
    assert trace.busy == pytest.approx((16.0, 16.0, 16.0, 16.0))
    assert trace.stage_cycles == (1.0, 1.0, 1.0, 1.0)

    for stage in range(4):
        counts = Counter(
            (op.kind, op.microbatch) for op in trace.ops if op.stage == stage
        )
        assert len(counts) == 16
        assert sum(count == 1 for count in counts.values()) == 16
        assert sum(op.kind == "forward" for op in trace.ops if op.stage == stage) == 8
        assert sum(op.kind == "backward" for op in trace.ops if op.stage == stage) == 8


def test_1f1b_same_bubble_bounded_activations() -> None:
    gpipe = pipeline.build_trace("gpipe", 4, 8)
    onef1b = pipeline.build_trace("1f1b", 4, 8)

    assert onef1b.makespan == pytest.approx(gpipe.makespan)
    assert onef1b.idle_fraction == pytest.approx(gpipe.idle_fraction)
    assert gpipe.activation_high_watermark == (8, 8, 8, 8)
    assert onef1b.activation_high_watermark == (4, 3, 2, 1)
    assert onef1b.peak_activation_microbatches == 4

    long_gpipe = pipeline.build_trace("gpipe", 4, 64)
    long_onef1b = pipeline.build_trace("1f1b", 4, 64)
    assert long_gpipe.peak_activation_microbatches == 64
    assert long_onef1b.peak_activation_microbatches == 4

    shallow = pipeline.build_trace("1f1b", 4, 2)
    assert shallow.peak_activation_microbatches <= 2


def test_ops_respect_dependencies_and_stage_exclusivity() -> None:
    for schedule in ("gpipe", "1f1b"):
        trace = pipeline.build_trace(schedule, 4, 8)
        by_key = _by_key(trace)
        assert len(by_key) == 4 * 8 * 2

        for op in trace.ops:
            if op.kind == "forward" and op.stage > 0:
                dependency = by_key[(op.stage - 1, op.microbatch, "forward")]
                assert op.start >= dependency.end - 1e-9
            if op.kind == "backward":
                dependency = by_key[(op.stage, op.microbatch, "forward")]
                assert op.start >= dependency.end - 1e-9
                if op.stage < trace.num_stages - 1:
                    dependency = by_key[(op.stage + 1, op.microbatch, "backward")]
                    assert op.start >= dependency.end - 1e-9

        for stage in range(trace.num_stages):
            stage_ops = sorted(
                (op for op in trace.ops if op.stage == stage),
                key=lambda op: op.start,
            )
            assert len(stage_ops) == 16
            for previous, following in zip(stage_ops, stage_ops[1:]):
                assert following.start >= previous.end - 1e-9


def test_gantt_rows() -> None:
    trace = pipeline.build_trace("gpipe", 4, 8)
    rows = pipeline.gantt_rows(trace)

    assert len(rows) == 4
    assert len(rows[0]) == 2 * 22
    assert rows[0].startswith("F0F1")
    for row in rows:
        assert len(row) == 2 * 22
        assert row.count("F") == 8
        assert row.count("B") == 8
        assert row.count(".") == 12
    assert "stage 0 |" in pipeline.render_gantt(trace)
