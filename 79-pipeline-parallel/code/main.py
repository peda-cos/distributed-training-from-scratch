"""GPipe and 1F1B pipeline schedules with real stagewise PyTorch autograd.

The schedule simulator is a deterministic event model: each stage runs one
operation at a time, forwards depend on the previous stage's forward for the
same microbatch, and backwards depend on the same stage's forward plus the next
stage's backward for the same microbatch.  The executor runs the resulting
operations in start-time order, detaching activations at stage boundaries so
each backward is genuinely local to one stage.  Parameter gradients accumulate
across microbatches and are compared with a full-batch reference.
"""

from __future__ import annotations

import argparse
import copy
import math
import platform
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy:.*")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Tensor: TypeAlias = torch.Tensor
ScheduleName = Literal["gpipe", "1f1b"]

HIDDEN_DIM = 16
NUM_CLASSES = 4
NUM_LAYERS = 8
DEFAULT_BATCH = 32
DEFAULT_MICROBATCHES = 8
DEFAULT_STAGES = 4
DEFAULT_SEED = 7


# ---------------------------------------------------------------------------
# Schedule simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Op:
    """One scheduled stage operation."""

    stage: int
    microbatch: int
    kind: Literal["forward", "backward"]
    start: float
    end: float


@dataclass(frozen=True)
class ScheduleTrace:
    """Completed schedule with timing, utilization, and activation accounting."""

    ops: tuple[Op, ...]
    num_stages: int
    num_microbatches: int
    stage_cycles: tuple[float, ...]
    makespan: float
    busy: tuple[float, ...]
    idle_fraction: float
    activation_high_watermark: tuple[int, ...]
    peak_activation_microbatches: int


def _validate_schedule(schedule: str, num_stages: int, num_microbatches: int) -> None:
    if schedule not in ("gpipe", "1f1b"):
        raise ValueError(f"unknown schedule {schedule!r}; expected 'gpipe' or '1f1b'")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")


def build_orders(
    schedule: ScheduleName,
    num_stages: int,
    num_microbatches: int,
) -> list[list[tuple[str, int]]]:
    """Return each stage's fixed forward/backward program.

    GPipe performs every forward before any backward.  1F1B gives stage ``s``
    ``min(N - s, M)`` warm-up forwards, then alternates backward/forward, then
    drains the remaining backwards.  Both programs contain exactly M forwards
    and M backwards per stage.
    """

    _validate_schedule(schedule, num_stages, num_microbatches)
    m = num_microbatches
    orders: list[list[tuple[str, int]]] = []
    for stage in range(num_stages):
        if schedule == "gpipe":
            program = [("forward", microbatch) for microbatch in range(m)]
            program += [("backward", microbatch) for microbatch in reversed(range(m))]
        else:
            warmup = min(num_stages - stage, m)
            program = [("forward", microbatch) for microbatch in range(warmup)]
            for microbatch in range(m - warmup):
                program.append(("backward", microbatch))
                program.append(("forward", warmup + microbatch))
            program += [
                ("backward", microbatch) for microbatch in range(m - warmup, m)
            ]
        orders.append(program)
    return orders


def simulate_schedule(
    orders: Sequence[Sequence[tuple[str, int]]],
    stage_cycles: Sequence[float],
) -> ScheduleTrace:
    """Simulate a pipeline schedule with one operation per stage at a time.

    The list scheduler always picks the ready operation with the smallest
    ``(start, stage, forward-first, microbatch)`` key.  This is deterministic
    and, with equal cycles, reproduces the closed-form bubble.
    """

    if not orders:
        raise ValueError("orders must contain at least one stage")
    num_stages = len(orders)
    if len(stage_cycles) != num_stages:
        raise ValueError("stage_cycles length must match the number of stages")
    cycles = tuple(float(value) for value in stage_cycles)
    if any(not math.isfinite(value) or value < 0.0 for value in cycles):
        raise ValueError("stage_cycles must be finite and non-negative")

    num_microbatches = len(orders[0]) // 2
    if num_microbatches <= 0:
        raise ValueError("each stage must contain at least one forward/backward pair")
    expected = {("forward", index) for index in range(num_microbatches)}
    expected |= {("backward", index) for index in range(num_microbatches)}
    for stage, program in enumerate(orders):
        if len(program) != 2 * num_microbatches or set(program) != expected:
            raise ValueError(
                f"stage {stage} must contain each forward and backward exactly once"
            )

    indices = [0] * num_stages
    free = [0.0] * num_stages
    done: dict[tuple[int, int, str], tuple[float, float]] = {}
    ops: list[Op] = []
    total_ops = 2 * num_microbatches * num_stages

    while len(ops) < total_ops:
        best: tuple[tuple[float, int, int, int], int, str, int] | None = None
        for stage in range(num_stages):
            if indices[stage] >= len(orders[stage]):
                continue
            kind, microbatch = orders[stage][indices[stage]]
            earliest = free[stage]
            if kind == "forward":
                if stage > 0:
                    dependency = done.get((stage - 1, microbatch, "forward"))
                    if dependency is None:
                        continue
                    earliest = max(earliest, dependency[1])
            else:
                dependency = done.get((stage, microbatch, "forward"))
                if dependency is None:
                    continue
                earliest = max(earliest, dependency[1])
                if stage < num_stages - 1:
                    dependency = done.get((stage + 1, microbatch, "backward"))
                    if dependency is None:
                        continue
                    earliest = max(earliest, dependency[1])
            candidate = (earliest, stage, 0 if kind == "forward" else 1, microbatch)
            if best is None or candidate < best[0]:
                best = (candidate, stage, kind, microbatch)

        if best is None:
            raise RuntimeError("pipeline schedule deadlocked")
        start, stage, kind, microbatch = best[0][0], best[1], best[2], best[3]
        end = start + cycles[stage]
        free[stage] = end
        done[(stage, microbatch, kind)] = (start, end)
        ops.append(
            Op(
                stage,
                microbatch,
                cast(Literal["forward", "backward"], kind),
                start,
                end,
            )
        )
        indices[stage] += 1

    makespan = max(op.end for op in ops)
    busy = tuple(
        sum(op.end - op.start for op in ops if op.stage == stage)
        for stage in range(num_stages)
    )
    idle_fraction = (
        0.0
        if makespan == 0.0
        else 1.0 - sum(busy) / (num_stages * makespan)
    )
    idle_fraction = min(1.0, max(0.0, idle_fraction))

    high_watermark: list[int] = []
    for stage in range(num_stages):
        events: list[tuple[float, int]] = []
        for microbatch in range(num_microbatches):
            events.append((done[(stage, microbatch, "forward")][1], 1))
            events.append((done[(stage, microbatch, "backward")][1], -1))
        events.sort(key=lambda event: (event[0], event[1]))
        current = 0
        peak = 0
        for _, delta in events:
            current = max(0, current + delta)
            peak = max(peak, current)
        high_watermark.append(peak)

    return ScheduleTrace(
        ops=tuple(ops),
        num_stages=num_stages,
        num_microbatches=num_microbatches,
        stage_cycles=cycles,
        makespan=makespan,
        busy=busy,
        idle_fraction=idle_fraction,
        activation_high_watermark=tuple(high_watermark),
        peak_activation_microbatches=max(high_watermark),
    )


def build_trace(
    schedule: ScheduleName,
    num_stages: int,
    num_microbatches: int,
    stage_cycles: Sequence[float] | None = None,
) -> ScheduleTrace:
    """Build a schedule and simulate it with unit or supplied stage cycles."""

    orders = build_orders(schedule, num_stages, num_microbatches)
    cycles = [1.0] * num_stages if stage_cycles is None else stage_cycles
    return simulate_schedule(orders, cycles)


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """Closed-form per-stage bubble fraction from the spec.

    This is idle / (useful + idle) under equal unit stage costs, i.e.
    ``(N - 1) / (M + N - 1)``.  Megatron reports idle / ideal compute instead,
    which is ``(N - 1) / M``.
    """

    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


def _gantt_cell_width(num_microbatches: int) -> int:
    return 1 + len(str(num_microbatches - 1))


def gantt_rows(trace: ScheduleTrace) -> tuple[str, ...]:
    """Render one fixed-width string per stage; idle cells use dots."""

    width = _gantt_cell_width(trace.num_microbatches)
    cells = max(0, math.ceil(trace.makespan - 1e-9))
    idle = "." * width
    rows: list[str] = []
    for stage in range(trace.num_stages):
        stage_ops = [op for op in trace.ops if op.stage == stage]
        cells_out: list[str] = []
        for cycle in range(cells):
            chosen: Op | None = None
            for op in stage_ops:
                if op.start < cycle + 1 - 1e-9 and op.end > cycle + 1e-9:
                    if chosen is None or op.start < chosen.start:
                        chosen = op
            if chosen is None:
                cells_out.append(idle)
            else:
                label = "F" if chosen.kind == "forward" else "B"
                cells_out.append(f"{label}{chosen.microbatch:0{width - 1}d}")
        rows.append("".join(cells_out))
    return tuple(rows)


def render_gantt(trace: ScheduleTrace) -> str:
    """Render a stage-by-cycle Gantt chart with a digit ruler."""

    width = _gantt_cell_width(trace.num_microbatches)
    cells = max(0, math.ceil(trace.makespan - 1e-9))
    ruler = "".join(str(cycle % 10) * width for cycle in range(cells))
    lines = [f"{'':<10}{ruler}"]
    for stage, row in enumerate(gantt_rows(trace)):
        lines.append(f"stage {stage} | {row}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model, execution, and gradient reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    batch_size: int = DEFAULT_BATCH
    num_microbatches: int = DEFAULT_MICROBATCHES
    num_stages: int = DEFAULT_STAGES
    hidden_dim: int = HIDDEN_DIM
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")
        if self.batch_size % self.num_microbatches != 0:
            raise ValueError("batch_size must be divisible by num_microbatches")
        if self.num_stages not in (1, 2, 4, 8):
            raise ValueError("num_stages must be one of 1, 2, 4, or 8")
        if NUM_LAYERS % self.num_stages != 0:
            raise ValueError("NUM_LAYERS must be divisible by num_stages")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


class PipelineStage(nn.Module):
    """One pipeline stage: a small module whose input and output are activations."""

    def __init__(self, layers: Sequence[nn.Module], simulated_cycles: float = 1.0):
        super().__init__()
        if not layers:
            raise ValueError("a pipeline stage needs at least one layer")
        if not math.isfinite(simulated_cycles) or simulated_cycles <= 0.0:
            raise ValueError("simulated_cycles must be finite and positive")
        self.body = nn.Sequential(*layers)
        self.simulated_cycles = float(simulated_cycles)

    def forward(self, activation: Tensor) -> Tensor:
        return self.body(activation)


@dataclass(frozen=True)
class StepResult:
    losses: tuple[float, ...]
    loss: float
    trace: ScheduleTrace
    measured_activation_high_watermark: tuple[int, ...]


class Pipeline(nn.Module):
    """Execute microbatches through stages using the selected schedule."""

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        num_microbatches: int,
        schedule: ScheduleName = "gpipe",
        stage_cycles: Sequence[float] | None = None,
    ):
        super().__init__()
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        if num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")
        _validate_schedule(schedule, len(stages), num_microbatches)
        self.stages = nn.ModuleList(stages)
        self.num_microbatches = num_microbatches
        self.schedule: ScheduleName = schedule
        if stage_cycles is None:
            self.stage_cycles = tuple(stage.simulated_cycles for stage in stages)
        else:
            cycles = tuple(float(value) for value in stage_cycles)
            if len(cycles) != len(stages):
                raise ValueError("stage_cycles length must match stages")
            if any(not math.isfinite(value) or value <= 0.0 for value in cycles):
                raise ValueError("stage_cycles must be finite and positive")
            self.stage_cycles = cycles

    def forward(self, inputs: Tensor) -> Tensor:
        """Full-batch inference through all stages; training uses ``step``."""

        activation = inputs
        for stage in self.stages:
            activation = stage(activation)
        return activation

    def full_model(self) -> nn.Sequential:
        """A sequential view over the stage bodies, sharing parameters."""

        units = [unit for stage in self.stages for unit in stage.body]
        return nn.Sequential(*units)

    def step(self, inputs: Tensor, targets: Tensor) -> StepResult:
        """Run one pipeline step and accumulate parameter gradients."""

        if inputs.ndim != 2:
            raise ValueError("inputs must be two-dimensional")
        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same batch size")
        batch = inputs.shape[0]
        if batch <= 0:
            raise ValueError("batch must be non-empty")
        if batch % self.num_microbatches != 0:
            raise ValueError("batch_size must be divisible by num_microbatches")

        self.zero_grad(set_to_none=True)
        trace = build_trace(
            self.schedule,
            len(self.stages),
            self.num_microbatches,
            self.stage_cycles,
        )
        x_micro = list(inputs.chunk(self.num_microbatches))
        y_micro = list(targets.chunk(self.num_microbatches))
        microbatch_losses = [0.0] * self.num_microbatches
        loss_seen = [False] * self.num_microbatches
        forward_store: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        grad_store: dict[tuple[int, int], Tensor] = {}
        pending = [0] * len(self.stages)
        peak_pending = [0] * len(self.stages)

        ordered_ops = sorted(
            trace.ops,
            key=lambda op: (
                op.start,
                op.stage,
                0 if op.kind == "forward" else 1,
                op.microbatch,
            ),
        )
        for op in ordered_ops:
            stage_index = op.stage
            microbatch = op.microbatch
            if op.kind == "forward":
                if stage_index == 0:
                    source = x_micro[microbatch]
                else:
                    source = forward_store[(stage_index - 1, microbatch)][1]
                leaf = source.detach().requires_grad_(True)
                output = self.stages[stage_index](leaf)
                forward_store[(stage_index, microbatch)] = (leaf, output)
                pending[stage_index] += 1
                peak_pending[stage_index] = max(
                    peak_pending[stage_index], pending[stage_index]
                )
                continue

            leaf, output = forward_store[(stage_index, microbatch)]
            if stage_index == len(self.stages) - 1:
                raw_loss = F.cross_entropy(output, y_micro[microbatch])
                microbatch_losses[microbatch] = raw_loss.item()
                loss_seen[microbatch] = True
                torch.autograd.backward(raw_loss / self.num_microbatches)
                incoming = leaf.grad
            else:
                incoming_grad = grad_store.pop((stage_index + 1, microbatch))
                torch.autograd.backward(output, grad_tensors=incoming_grad)
                incoming = leaf.grad
            if incoming is None:
                raise RuntimeError("stage backward produced no input gradient")
            if stage_index > 0:
                grad_store[(stage_index, microbatch)] = incoming
            forward_store.pop((stage_index, microbatch))
            pending[stage_index] -= 1

        if forward_store or grad_store or any(pending):
            raise RuntimeError("incomplete pipeline execution")
        if not all(loss_seen):
            raise RuntimeError("incomplete pipeline execution")
        return StepResult(
            losses=tuple(microbatch_losses),
            loss=sum(microbatch_losses) / self.num_microbatches,
            trace=trace,
            measured_activation_high_watermark=tuple(peak_pending),
        )


def make_layer_stack(hidden_dim: int, dtype: torch.dtype = torch.float32) -> nn.Sequential:
    """Eight sequential units: seven linear+ReLU blocks and one classifier."""

    units: list[nn.Module] = []
    for _ in range(NUM_LAYERS - 1):
        units.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
    units.append(nn.Linear(hidden_dim, NUM_CLASSES))
    return nn.Sequential(*units).to(dtype=dtype)


def build_pipeline(
    config: RunConfig,
    schedule: ScheduleName = "gpipe",
    dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build a deterministic 8-unit model split into equal contiguous stages."""

    torch.manual_seed(config.seed)
    units = list(make_layer_stack(config.hidden_dim, dtype))
    chunk = NUM_LAYERS // config.num_stages
    stages = [
        PipelineStage(units[index * chunk : (index + 1) * chunk])
        for index in range(config.num_stages)
    ]
    return Pipeline(stages, config.num_microbatches, schedule)


def make_batch(
    config: RunConfig,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """A fixed random classification batch with an independent generator."""

    generator = torch.Generator().manual_seed(config.seed + 1)
    inputs = torch.randn(
        config.batch_size,
        config.hidden_dim,
        generator=generator,
        dtype=dtype,
    )
    targets = torch.randint(
        0,
        NUM_CLASSES,
        (config.batch_size,),
        generator=generator,
    )
    return inputs, targets


def reference_gradients(
    pipeline: Pipeline,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, list[Tensor]]:
    """Full-batch loss and parameter gradients from an independent deep copy."""

    model = copy.deepcopy(pipeline.full_model())
    model = model.to(device=inputs.device, dtype=inputs.dtype)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    grads: list[Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("reference parameter has no gradient")
        grads.append(parameter.grad.detach().clone())
    return loss.item(), grads


def max_grad_delta(pipeline: Pipeline, reference: Sequence[Tensor]) -> float:
    """Largest absolute difference between pipeline and reference gradients."""

    parameters = list(pipeline.parameters())
    if len(parameters) != len(reference):
        raise ValueError("reference gradient count mismatch")
    largest = 0.0
    for parameter, expected in zip(parameters, reference):
        if parameter.grad is None:
            raise RuntimeError("pipeline parameter has no gradient")
        largest = max(largest, (parameter.grad - expected).abs().max().item())
    return largest


# ---------------------------------------------------------------------------
# Stage balance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerCost:
    name: str
    params: int
    flops: int

    def __post_init__(self) -> None:
        if self.params < 0 or self.flops < 0:
            raise ValueError("layer costs must be non-negative")


@dataclass(frozen=True)
class StageAssignment:
    axis: Literal["params", "flops"]
    ranges: tuple[tuple[int, int], ...]
    layer_counts: tuple[int, ...]
    params: tuple[int, ...]
    flops: tuple[int, ...]

    @property
    def total_params(self) -> int:
        return sum(self.params)

    @property
    def total_flops(self) -> int:
        return sum(self.flops)

    @property
    def compute_utilization(self) -> float:
        largest = max(self.flops, default=0)
        if largest == 0:
            return 0.0
        return self.total_flops / (len(self.flops) * largest)


def transformer_layer_costs(
    num_blocks: int = 12,
    d_model: int = 256,
    seq_len: int = 64,
    vocab_size: int = 16384,
    num_classes: int = 256,
) -> list[LayerCost]:
    """Analytic parameter and FLOP costs for a toy transformer.

    Linear work uses ``2 * in_features * out_features * tokens``.  The
    embedding is a parameter-heavy lookup with zero modeled FLOPs; each
    transformer block uses 12*d^2 weights and 24*d^2*seq forward FLOPs; the
    classifier head is small.
    """

    for name, value in (
        ("num_blocks", num_blocks),
        ("d_model", d_model),
        ("seq_len", seq_len),
        ("vocab_size", vocab_size),
        ("num_classes", num_classes),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    block_params = 12 * d_model * d_model
    block_flops = 24 * d_model * d_model * seq_len
    costs = [LayerCost("embed", vocab_size * d_model, 0)]
    for index in range(num_blocks):
        costs.append(LayerCost(f"block{index:02d}", block_params, block_flops))
    costs.append(
        LayerCost(
            "head",
            d_model * num_classes,
            2 * d_model * num_classes * seq_len,
        )
    )
    return costs


def partition_even(
    costs: Sequence[LayerCost],
    num_stages: int,
    axis: Literal["params", "flops"],
) -> StageAssignment:
    """Contiguous least-squares partition of layers into stages.

    Minimizing the sum of squared stage totals pushes stage totals together
    without the tie ambiguity of a bare minimax objective.
    """

    if axis not in ("params", "flops"):
        raise ValueError("axis must be 'params' or 'flops'")
    if not costs:
        raise ValueError("costs must not be empty")
    if num_stages <= 0 or num_stages > len(costs):
        raise ValueError("num_stages must be between 1 and the layer count")

    values = [int(getattr(layer, axis)) for layer in costs]
    if any(value < 0 for value in values):
        raise ValueError("layer costs must be non-negative")
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)

    count = len(values)
    infinity = float("inf")
    dp = [[infinity] * (count + 1) for _ in range(num_stages + 1)]
    cuts = [[-1] * (count + 1) for _ in range(num_stages + 1)]
    dp[0][0] = 0.0
    for parts in range(1, num_stages + 1):
        for end in range(parts, count + 1):
            for start in range(parts - 1, end):
                stage_total = prefix[end] - prefix[start]
                candidate = dp[parts - 1][start] + stage_total * stage_total
                if candidate < dp[parts][end]:
                    dp[parts][end] = candidate
                    cuts[parts][end] = start

    ranges: list[tuple[int, int]] = []
    end = count
    for parts in range(num_stages, 0, -1):
        start = cuts[parts][end]
        if start < 0:
            raise RuntimeError("partition reconstruction failed")
        ranges.append((start, end))
        end = start
    ranges.reverse()

    return StageAssignment(
        axis=axis,
        ranges=tuple(ranges),
        layer_counts=tuple(stop - start for start, stop in ranges),
        params=tuple(sum(layer.params for layer in costs[start:stop]) for start, stop in ranges),
        flops=tuple(sum(layer.flops for layer in costs[start:stop]) for start, stop in ranges),
    )


# ---------------------------------------------------------------------------
# Deadlock-free handoff
# ---------------------------------------------------------------------------


def exchange_order(rank: int) -> tuple[str, str]:
    """Even ranks send first; odd ranks receive first."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    return ("send", "recv") if rank % 2 == 0 else ("recv", "send")


@dataclass(frozen=True)
class ExchangeResult:
    completed: bool
    steps: int


def simulate_exchange(orders: Sequence[Sequence[str]]) -> ExchangeResult:
    """Simulate a zero-buffer rendezvous between two adjacent ranks.

    A send and receive only complete when both ranks offer the opposite
    operation.  Two send-first programs therefore deadlock; an even/odd
    alternating pair completes both handoffs.
    """

    if len(orders) != 2:
        raise ValueError("exactly two rank programs are required")
    programs = [tuple(order) for order in orders]
    for program in programs:
        if len(program) != 2 or sorted(program) != ["recv", "send"]:
            raise ValueError("each program must be one send and one recv")
    positions = [0, 0]
    steps = 0
    while positions[0] < 2 or positions[1] < 2:
        if positions[0] >= 2 or positions[1] >= 2:
            return ExchangeResult(False, steps)
        first = programs[0][positions[0]]
        second = programs[1][positions[1]]
        if {first, second} == {"send", "recv"}:
            positions[0] += 1
            positions[1] += 1
            steps += 1
        else:
            return ExchangeResult(False, steps)
    return ExchangeResult(True, steps)


# ---------------------------------------------------------------------------
# Reporting and CLI
# ---------------------------------------------------------------------------


def _print_pipeline_result(
    schedule: str,
    result: StepResult,
    reference_loss: float,
    grad_delta: float,
) -> bool:
    loss_delta = abs(result.loss - reference_loss)
    passed = loss_delta <= 1e-5 and grad_delta <= 1e-4
    status = "PASS" if passed else "FAIL"
    print(
        f"{schedule:<6} loss={result.loss:.6f} ref={reference_loss:.6f} "
        f"dloss={loss_delta:.2e} max|dw|={grad_delta:.2e} {status}"
    )
    return passed


def _print_bubble_table() -> bool:
    print("bubble fraction, N=4 (idle / (useful + idle))")
    print("M      closed      GPipe      1F1B")
    print("------------------------------------")
    passed = True
    for microbatches in (2, 4, 8, 16, 32, 64):
        predicted = bubble_fraction(4, microbatches)
        gpipe = build_trace("gpipe", 4, microbatches)
        onef1b = build_trace("1f1b", 4, microbatches)
        error = max(
            abs(gpipe.idle_fraction - predicted),
            abs(onef1b.idle_fraction - predicted),
        )
        passed = passed and error <= 1e-9
        print(
            f"{microbatches:<6} {predicted * 100:8.2f}% "
            f"{gpipe.idle_fraction * 100:10.2f}% {onef1b.idle_fraction * 100:10.2f}%"
        )
    return passed


def _print_activation_table() -> bool:
    gpipe = build_trace("gpipe", 4, 8)
    onef1b = build_trace("1f1b", 4, 8)
    print("activation high-watermark, M=8, microbatches per stage")
    print("stage    GPipe    1F1B")
    print("-----------------------")
    for stage in range(4):
        print(
            f"{stage:<8} {gpipe.activation_high_watermark[stage]:6d} "
            f"{onef1b.activation_high_watermark[stage]:7d}"
        )
    print(
        f"peak     {gpipe.peak_activation_microbatches:6d} "
        f"{onef1b.peak_activation_microbatches:7d}"
    )
    return (
        gpipe.peak_activation_microbatches == 8
        and onef1b.peak_activation_microbatches <= 4
    )


def _print_balance_table() -> bool:
    costs = transformer_layer_costs()
    params_assignment = partition_even(costs, 4, "params")
    flops_assignment = partition_even(costs, 4, "flops")
    block_flops = costs[1].flops
    params_cycles = [flops / block_flops for flops in params_assignment.flops]
    flops_cycles = [flops / block_flops for flops in flops_assignment.flops]
    params_trace = build_trace("gpipe", 4, 8, params_cycles)
    flops_trace = build_trace("gpipe", 4, 8, flops_cycles)

    print("stage balance, 4 stages, 12 blocks + embedding + head")
    print("axis   stage layers       params        flops  flops/max")
    print("-----------------------------------------------------------")
    for assignment in (params_assignment, flops_assignment):
        for stage, (layers, params, flops) in enumerate(
            zip(
                assignment.layer_counts,
                assignment.params,
                assignment.flops,
            )
        ):
            print(
                f"{assignment.axis:<6} {stage:<5} {layers:<6} "
                f"{params:>12,} {flops:>13,} "
                f"{flops / max(assignment.flops):9.2%}"
            )
        print(
            f"{assignment.axis:<6} total  {sum(assignment.layer_counts):<6} "
            f"{assignment.total_params:>12,} {assignment.total_flops:>13,} "
            f"compute utilization {assignment.compute_utilization:.2%}"
        )
    ratio = params_trace.makespan / flops_trace.makespan
    print(
        "simulated makespan, M=8, cycles in block-forward units: "
        f"params={params_trace.makespan:.2f} "
        f"flops={flops_trace.makespan:.2f} ({ratio:.2f}x)"
    )
    return flops_trace.makespan < params_trace.makespan


def _print_handoff() -> bool:
    alternating = simulate_exchange(
        (exchange_order(0), exchange_order(1))
    )
    deadlocked = simulate_exchange(
        (("send", "recv"), ("send", "recv"))
    )
    print(
        "handoff: even ranks send-first, odd ranks recv-first; "
        f"alternating completed={alternating.completed} "
        f"steps={alternating.steps}; "
        f"both-send-first completed={deadlocked.completed} "
        f"steps={deadlocked.steps}"
    )
    return alternating.completed and alternating.steps == 2 and not deadlocked.completed


def run_demo(config: RunConfig, show_gantt: bool = True) -> int:
    """Run correctness checks and print the schedule analysis."""

    torch.set_num_threads(1)
    print(
        f"torch={torch.__version__} python={platform.python_version()} "
        f"cpus={torch.get_num_threads()} platform={platform.platform()}"
    )
    inputs, targets = make_batch(config)
    passed = True
    traces: dict[str, StepResult] = {}

    for schedule in ("gpipe", "1f1b"):
        pipeline = build_pipeline(config, schedule)
        reference_loss, reference_grads = reference_gradients(
            pipeline, inputs, targets
        )
        result = pipeline.step(inputs, targets)
        grad_delta = max_grad_delta(pipeline, reference_grads)
        passed = _print_pipeline_result(
            schedule, result, reference_loss, grad_delta
        ) and passed
        traces[schedule] = result

    if show_gantt:
        for schedule in ("gpipe", "1f1b"):
            print(f"\n{schedule} Gantt: F<mb> forward, B<mb> backward, . idle")
            print(render_gantt(traces[schedule].trace))

    print()
    passed = _print_bubble_table() and passed
    print()
    passed = _print_activation_table() and passed
    print()
    passed = _print_balance_table() and passed
    print()
    passed = _print_handoff() and passed

    if passed:
        print("\nALL CHECKS PASSED")
        return 0
    print("\nCHECK FAILED")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPipe and 1F1B pipeline schedule analysis"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--microbatches", type=int, default=DEFAULT_MICROBATCHES)
    parser.add_argument("--stages", type=int, choices=(1, 2, 4, 8), default=DEFAULT_STAGES)
    parser.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-gantt", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RunConfig(
        batch_size=args.batch_size,
        num_microbatches=args.microbatches,
        num_stages=args.stages,
        hidden_dim=args.hidden,
        seed=args.seed,
    )
    return run_demo(config, show_gantt=not args.no_gantt)


if __name__ == "__main__":
    raise SystemExit(main())
