"""Stage-balance partitions and deadlock-free handoff tests."""

from __future__ import annotations

import itertools

import pytest

from pipeline_parallel import main as pipeline


def _brute_force_squares(values: list[int], parts: int) -> int:
    best = None
    for cuts in itertools.combinations(range(1, len(values)), parts - 1):
        bounds = (0, *cuts, len(values))
        total = sum(
            sum(values[start:end]) ** 2
            for start, end in zip(bounds, bounds[1:])
        )
        best = total if best is None else min(best, total)
    assert best is not None
    return best


def test_partition_expected_ranges_and_utilization() -> None:
    costs = pipeline.transformer_layer_costs()
    params = pipeline.partition_even(costs, 4, "params")
    flops = pipeline.partition_even(costs, 4, "flops")

    assert params.ranges == ((0, 1), (1, 5), (5, 9), (9, 14))
    assert flops.ranges == ((0, 4), (4, 7), (7, 10), (10, 14))
    assert params.layer_counts == (1, 4, 4, 5)
    assert flops.layer_counts == (4, 3, 3, 4)
    assert params.total_params == flops.total_params == 13_697_024
    assert params.total_flops == flops.total_flops == 1_216_348_160
    assert params.compute_utilization == pytest.approx(
        1_216_348_160 / (4 * 411_041_792)
    )
    assert flops.compute_utilization == pytest.approx(
        1_216_348_160 / (4 * 310_378_496)
    )

    for assignment in (params, flops):
        covered = [0] * len(costs)
        for start, stop in assignment.ranges:
            for index in range(start, stop):
                covered[index] += 1
        assert covered == [1] * len(costs)
        assert sum(assignment.layer_counts) == len(costs)
        assert sum(assignment.params) == assignment.total_params
        assert sum(assignment.flops) == assignment.total_flops


def test_partition_is_optimal_by_brute_force() -> None:
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    costs = [
        pipeline.LayerCost(f"layer{index}", value, value)
        for index, value in enumerate(values)
    ]

    for axis in ("params", "flops"):
        assignment = pipeline.partition_even(costs, 3, axis)
        best = _brute_force_squares(values, 3)
        assert sum(value * value for value in assignment.flops) == best

    single = pipeline.partition_even(costs, 1, "params")
    assert single.ranges == ((0, len(values)),)
    assert single.layer_counts == (len(values),)

    every = pipeline.partition_even(costs, len(values), "flops")
    assert every.ranges == tuple((index, index + 1) for index in range(len(values)))
    assert every.layer_counts == (1,) * len(values)

    with pytest.raises(ValueError):
        pipeline.partition_even(costs, 0, "params")
    with pytest.raises(ValueError):
        pipeline.partition_even(costs, len(values) + 1, "params")
    with pytest.raises(ValueError):
        pipeline.partition_even(costs, 2, "weights")
    with pytest.raises(ValueError):
        pipeline.partition_even([], 1, "params")


def test_compute_balanced_schedule_has_shorter_makespan() -> None:
    costs = pipeline.transformer_layer_costs()
    params = pipeline.partition_even(costs, 4, "params")
    flops = pipeline.partition_even(costs, 4, "flops")
    block_flops = costs[1].flops

    params_trace = pipeline.build_trace(
        "gpipe", 4, 8, [value / block_flops for value in params.flops]
    )
    flops_trace = pipeline.build_trace(
        "gpipe", 4, 8, [value / block_flops for value in flops.flops]
    )

    assert flops_trace.makespan <= 0.9 * params_trace.makespan
    assert params_trace.makespan == pytest.approx(81.333333, abs=1e-5)
    assert flops_trace.makespan == pytest.approx(67.333333, abs=1e-5)
    assert len(flops_trace.ops) == len(params_trace.ops) == 2 * 4 * 8


def test_handoff_order_avoids_deadlock() -> None:
    assert pipeline.exchange_order(0) == ("send", "recv")
    assert pipeline.exchange_order(1) == ("recv", "send")
    assert pipeline.exchange_order(2) == ("send", "recv")
    assert pipeline.exchange_order(3) == ("recv", "send")

    alternating = pipeline.simulate_exchange(
        (pipeline.exchange_order(0), pipeline.exchange_order(1))
    )
    assert alternating.completed is True
    assert alternating.steps == 2

    send_first = pipeline.simulate_exchange(
        (("send", "recv"), ("send", "recv"))
    )
    assert send_first.completed is False
    assert send_first.steps == 0

    recv_first = pipeline.simulate_exchange(
        (("recv", "send"), ("recv", "send"))
    )
    assert recv_first.completed is False
    assert recv_first.steps == 0

    with pytest.raises(ValueError):
        pipeline.simulate_exchange((("send", "recv"),))
    with pytest.raises(ValueError):
        pipeline.simulate_exchange((("send", "send"), ("recv", "recv")))
    with pytest.raises(ValueError):
        pipeline.exchange_order(-1)
