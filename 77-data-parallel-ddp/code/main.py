"""Distributed data parallel training from broadcast, allreduce, and hooks.

The wrapper in this module is deliberately shaped like
``torch.nn.parallel.DistributedDataParallel``: it broadcasts the initial
parameters (and buffers) from rank 0, installs a post-accumulate gradient hook
on every trainable parameter, fuses gradients into reverse-order buckets, and
issues an asynchronous allreduce as soon as a bucket is ready.  ``sync_grads``
waits for those collectives, divides by ``world_size``, and scatters the mean
gradient back into ``param.grad``.

The module also owns the deterministic demo/reference path used by the tests:
four CPU ranks over gloo with a file rendezvous, and a sequential single-process
loop over the same global batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Generator, Sequence

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy:.*")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Tensor = torch.Tensor

INPUT_DIM = 8
DATASET_SIZE = 4096
DEFAULT_HIDDEN = 32
DEFAULT_BATCH = 64
DEFAULT_STEPS = 20
DEFAULT_LR = 0.1
DEFAULT_SEED = 7
DATA_SEED = 2026
SHUFFLE_SEED = 99
MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# Deterministic model, data, and hashing helpers
# ---------------------------------------------------------------------------


class MiniMLP(nn.Module):
    """Three-layer MLP with a real, broadcastable buffer.

    ``use_hidden=False`` skips the middle layer, which gives the unused
    parameter path a genuine branch to detect.
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, hidden_dim, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, 1, dtype=dtype)
        # A buffer, not a parameter: it exercises the constructor broadcast.
        self.register_buffer(
            "output_scale", 0.9 + 0.2 * torch.rand((), dtype=dtype)
        )

    def forward(self, x: Tensor, use_hidden: bool = True) -> Tensor:
        hidden = F.relu(self.fc1(x))
        if use_hidden:
            hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden) * self.output_scale


def make_model(
    seed: int,
    *,
    hidden_dim: int = DEFAULT_HIDDEN,
    dtype: torch.dtype = torch.float64,
) -> MiniMLP:
    """Build a model from one explicit seed.

    Every rank calls this with the shared parameter-init seed; the broadcast in
    ``DistributedDataParallel`` then overwrites the local state.  The jitter test
    deliberately calls it with ``seed + rank`` to prove the broadcast is doing
    real work.
    """

    torch.manual_seed(seed)
    return MiniMLP(hidden_dim=hidden_dim, dtype=dtype)


def make_dataset(
    seed: int,
    *,
    size: int = DATASET_SIZE,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Deterministic nonlinear regression data, identical on every rank."""

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(size, INPUT_DIM, generator=generator, dtype=dtype)
    weights = torch.tensor(
        [0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.7, 0.4], dtype=dtype
    )
    y = (torch.tanh(x @ weights) + 0.2 * x[:, 0] * x[:, 1]).unsqueeze(1)
    return x, y


def batch_indices(
    step: int, batch_size: int, dataset_size: int, shuffle_seed: int
) -> Tensor:
    """Return one deterministic global batch of dataset indices."""

    generator = torch.Generator().manual_seed(shuffle_seed + step)
    return torch.randperm(dataset_size, generator=generator)[:batch_size]


def micro_batches(indices: Tensor, micro_steps: int) -> list[Tensor]:
    return list(indices.chunk(micro_steps))


def rank_slice(micro_indices: Tensor, rank: int, world_size: int) -> Tensor:
    return micro_indices.chunk(world_size)[rank]


def _tensor_bytes(tensor: Tensor) -> bytes:
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    return bytes(flat.view(torch.uint8).tolist())


def state_hash(module: nn.Module) -> str:
    """SHA-256 of names and raw bytes for every parameter and buffer."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\0")
    return digest.hexdigest()


def state_checksum(module: nn.Module) -> float:
    return math.fsum(
        float(parameter.detach().double().sum().item())
        for parameter in module.parameters()
    )


def flatten_params(module: nn.Module) -> Tensor:
    tensors = [
        parameter.detach().to("cpu").reshape(-1)
        for parameter in module.parameters()
    ]
    if not tensors:
        return torch.empty(0)
    return torch.cat(tensors)


def _missing_grad_params(module: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


# ---------------------------------------------------------------------------
# Configuration and statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    world_size: int = 4
    steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH
    hidden_dim: int = DEFAULT_HIDDEN
    lr: float = DEFAULT_LR
    seed: int = DEFAULT_SEED
    data_seed: int = DATA_SEED
    shuffle_seed: int = SHUFFLE_SEED
    bucket_cap_mb: float = 25.0
    sync_mode: str = "bucketed"
    find_unused_parameters: bool = False
    no_sync_microbatches: int = 1
    use_hidden: bool = True
    init_jitter: bool = False
    dtype: str = "float64"
    backend: str = "gloo"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not 0 < self.batch_size <= DATASET_SIZE:
            raise ValueError(f"batch_size must be in [1, {DATASET_SIZE}]")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be positive")
        if self.sync_mode not in {"bucketed", "naive"}:
            raise ValueError("sync_mode must be 'bucketed' or 'naive'")
        if self.no_sync_microbatches < 1:
            raise ValueError("no_sync_microbatches must be at least 1")
        divisor = self.world_size * self.no_sync_microbatches
        if self.batch_size % divisor:
            raise ValueError(
                "batch_size must be divisible by "
                f"world_size * no_sync_microbatches ({divisor})"
            )
        if self.dtype not in {"float64", "float32"}:
            raise ValueError("dtype must be 'float64' or 'float32'")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


@dataclass
class SyncStats:
    broadcast_bytes: int = 0
    allreduce_calls: int = 0
    allreduce_bytes: int = 0
    buckets_launched: int = 0
    overlapped_buckets: int = 0
    sync_seconds: float = 0.0


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    slots: list[tuple[int, int]]
    flat: Tensor
    ready: int = 0
    launched: bool = False
    handle: Any = None
    reduce_started: float | None = None


# ---------------------------------------------------------------------------
# Bucket construction
# ---------------------------------------------------------------------------


def plan_buckets(
    params: Sequence[nn.Parameter], cap_bytes: int
) -> list[list[nn.Parameter]]:
    """Group parameters into reverse-order buckets no larger than the cap.

    A single parameter larger than the cap receives its own bucket.  Reverse
    ``Model.parameters()`` order approximates backward readiness, so the first
    bucket can be reduced while earlier layers are still computing.
    """

    if cap_bytes <= 0:
        raise ValueError("cap_bytes must be positive")

    buckets: list[list[nn.Parameter]] = []
    current: list[nn.Parameter] = []
    current_bytes = 0

    for parameter in reversed(list(params)):
        parameter_bytes = parameter.numel() * parameter.element_size()
        if current and current_bytes + parameter_bytes > cap_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(parameter)
        current_bytes += parameter_bytes
        if current_bytes >= cap_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0

    if current:
        buckets.append(current)
    return buckets


# ---------------------------------------------------------------------------
# The DDP-shaped wrapper
# ---------------------------------------------------------------------------


class DistributedDataParallel(nn.Module):
    """Broadcast initial state, allreduce gradients, divide by world size.

    ``sync_mode="bucketed"`` installs post-accumulate hooks and launches async
    bucket allreduces during backward.  ``sync_mode="naive"`` issues one
    synchronous allreduce per parameter after backward and is the benchmark
    baseline.  ``find_unused_parameters`` switches to a safe post-backward scan
    of ``.grad`` fields; it keeps collectives ordered without an autograd graph
    walk, at the cost of overlap.
    """

    def __init__(
        self,
        module: nn.Module,
        world_size: int,
        *,
        bucket_cap_mb: float = 25.0,
        sync_mode: str = "bucketed",
        find_unused_parameters: bool = False,
    ) -> None:
        super().__init__()
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if sync_mode not in {"bucketed", "naive"}:
            raise ValueError("sync_mode must be 'bucketed' or 'naive'")
        if bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be positive")
        if world_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before constructing "
                    "DistributedDataParallel with world_size > 1"
                )
            if dist.get_world_size() != world_size:
                raise RuntimeError(
                    f"world_size={world_size} does not match the process group "
                    f"size {dist.get_world_size()}"
                )

        self.module = module
        self.world_size = world_size
        self.sync_mode = sync_mode
        self.find_unused_parameters = find_unused_parameters
        self.stats = SyncStats()
        self._no_sync_depth = 0
        self._next_bucket = 0
        self.buckets: list[_Bucket] = []
        self._param_slots: dict[int, tuple[int, int]] = {}
        self._hook_handles: list[Any] = []

        trainable = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if trainable:
            dtypes = {parameter.dtype for parameter in trainable}
            devices = {parameter.device for parameter in trainable}
            if len(dtypes) != 1 or len(devices) != 1:
                raise ValueError(
                    "all trainable parameters must share one dtype and device"
                )

        self._broadcast_initial_state()

        cap_bytes = max(1, int(bucket_cap_mb * MIB))
        planned = (
            plan_buckets(trainable, cap_bytes)
            if sync_mode == "bucketed"
            else []
        )
        for bucket_index, params in enumerate(planned):
            flat = torch.empty(
                sum(parameter.numel() for parameter in params),
                dtype=params[0].dtype,
                device=params[0].device,
            )
            slots: list[tuple[int, int]] = []
            start = 0
            for parameter in params:
                end = start + parameter.numel()
                slots.append((start, end))
                self._param_slots[id(parameter)] = (bucket_index, start)
                start = end
            self.buckets.append(_Bucket(params=params, slots=slots, flat=flat))

        if sync_mode == "bucketed" and not find_unused_parameters:
            for parameter in trainable:
                self._hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(
                        self._make_hook()
                    )
                )

    # -- initial state -----------------------------------------------------

    def _broadcast_initial_state(self) -> None:
        self.pre_broadcast_hash = state_hash(self.module)
        if self.world_size > 1:
            for tensor in list(self.module.parameters()) + list(
                self.module.buffers()
            ):
                dist.broadcast(tensor.data, src=0)
                self.stats.broadcast_bytes += (
                    tensor.numel() * tensor.element_size()
                )
        self.post_broadcast_hash = state_hash(self.module)

    # -- forward and no_sync ----------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    @property
    def _sync_disabled(self) -> bool:
        return self._no_sync_depth > 0

    @contextmanager
    def no_sync(self) -> Generator[DistributedDataParallel, None, None]:
        """Accumulate local gradients without starting bucket allreduces."""

        self._no_sync_depth += 1
        try:
            yield self
        finally:
            self._no_sync_depth -= 1

    # -- backward hooks ----------------------------------------------------

    def _make_hook(self) -> Any:
        def hook(param: Tensor) -> None:
            if self._sync_disabled:
                return
            slot = self._param_slots.get(id(param))
            if slot is None:
                return
            bucket_index, start = slot
            bucket = self.buckets[bucket_index]
            if bucket.launched:
                raise RuntimeError(
                    "a gradient bucket was already launched for this step; "
                    "call sync_grads() before the next backward pass"
                )
            if param.grad is None:
                return
            end = start + param.numel()
            bucket.flat[start:end].copy_(param.grad.detach().reshape(-1))
            bucket.ready += 1
            self._drain_ready()

        return hook

    def _drain_ready(self) -> None:
        # Bucket index order is mandatory: every rank must issue collectives in
        # the same sequence even if one rank's compute runs ahead.
        while self._next_bucket < len(self.buckets):
            bucket = self.buckets[self._next_bucket]
            if bucket.ready < len(bucket.params):
                break
            self._launch(bucket)
            self._next_bucket += 1

    def _launch(self, bucket: _Bucket) -> None:
        bucket.launched = True
        bucket.reduce_started = time.perf_counter()
        self.stats.buckets_launched += 1
        if self.world_size > 1:
            bucket.handle = dist.all_reduce(
                bucket.flat, op=dist.ReduceOp.SUM, async_op=True
            )
            self.stats.allreduce_calls += 1
            self.stats.allreduce_bytes += (
                bucket.flat.numel() * bucket.flat.element_size()
            )
            self.stats.overlapped_buckets += 1

    # -- synchronization ----------------------------------------------------

    def _wait_launched_handles(self) -> None:
        for bucket in self.buckets:
            if bucket.handle is not None:
                bucket.handle.wait()

    def _reset_buckets(self) -> None:
        for bucket in self.buckets:
            bucket.ready = 0
            bucket.launched = False
            bucket.handle = None
            bucket.reduce_started = None
        self._next_bucket = 0

    def sync_grads(self) -> None:
        """Wait for gradient reductions and install mean gradients.

        Call immediately after ``backward()`` and before ``optimizer.step()``.
        """

        if self._sync_disabled:
            raise RuntimeError("sync_grads() cannot be called inside no_sync()")

        started = time.perf_counter()
        missing = _missing_grad_params(self.module)
        if missing and not self.find_unused_parameters:
            self._wait_launched_handles()
            self._reset_buckets()
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"{len(missing)} trainable parameters received no gradient "
                f"({preview}). Set find_unused_parameters=True for a branchy "
                "forward or run the full model graph."
            )

        if self.sync_mode == "naive":
            self._sync_naive()
        elif self.find_unused_parameters:
            self._sync_find_unused()
        else:
            self._sync_buckets()

        self.stats.sync_seconds += time.perf_counter() - started

    def _sync_naive(self) -> None:
        for parameter in self.module.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if self.world_size > 1:
                dist.all_reduce(
                    parameter.grad, op=dist.ReduceOp.SUM, async_op=False
                )
                self.stats.allreduce_calls += 1
                self.stats.allreduce_bytes += (
                    parameter.grad.numel() * parameter.grad.element_size()
                )
            parameter.grad.div_(self.world_size)
        self._reset_buckets()

    def _sync_buckets(self) -> None:
        if self._next_bucket != len(self.buckets):
            self._wait_launched_handles()
            self._reset_buckets()
            raise RuntimeError(
                "not all gradient buckets were launched; run backward() after "
                "the last optimizer.zero_grad() and before sync_grads()"
            )
        for bucket in self.buckets:
            if bucket.handle is not None:
                bucket.handle.wait()
            bucket.flat.div_(self.world_size)
            for parameter, (start, end) in zip(bucket.params, bucket.slots):
                if parameter.grad is not None:
                    parameter.grad.copy_(
                        bucket.flat[start:end].view_as(parameter.grad)
                    )
        self._reset_buckets()

    def _sync_find_unused(self) -> None:
        for bucket in self.buckets:
            bucket.flat.zero_()
            for parameter, (start, end) in zip(bucket.params, bucket.slots):
                if parameter.grad is not None:
                    bucket.flat[start:end].copy_(
                        parameter.grad.detach().reshape(-1)
                    )
            if self.world_size > 1:
                dist.all_reduce(
                    bucket.flat, op=dist.ReduceOp.SUM, async_op=False
                )
                self.stats.allreduce_calls += 1
                self.stats.allreduce_bytes += (
                    bucket.flat.numel() * bucket.flat.element_size()
                )
            bucket.flat.div_(self.world_size)
            for parameter, (start, end) in zip(bucket.params, bucket.slots):
                if parameter.grad is not None:
                    parameter.grad.copy_(
                        bucket.flat[start:end].view_as(parameter.grad)
                    )
        self._reset_buckets()


# ---------------------------------------------------------------------------
# Sequential reference and spawned DDP run
# ---------------------------------------------------------------------------


@dataclass
class ReferenceRun:
    config: RunConfig
    records: list[dict[str, Any]]
    params: Tensor
    elapsed_s: float


@dataclass
class DDPRun:
    config: RunConfig
    rank_payloads: list[dict[str, Any]]
    params_by_rank: list[Tensor]
    elapsed_s: float
    train_seconds: float


@dataclass
class Comparison:
    replica_equal: list[bool]
    loss_deltas: list[float]
    param_deltas: list[float]
    broadcast_ok: bool
    pre_hashes_differ: bool
    passed: bool
    max_loss_delta: float
    max_param_delta: float


def _reference_single_process_loop(
    config: RunConfig,
) -> tuple[list[dict[str, Any]], Tensor]:
    """Train the same model and batches sequentially on one process."""

    dtype = config.torch_dtype
    model = make_model(
        config.seed, hidden_dim=config.hidden_dim, dtype=dtype
    )
    x, y = make_dataset(config.data_seed, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    records: list[dict[str, Any]] = []
    history: list[Tensor] = []

    for step in range(config.steps):
        indices = batch_indices(
            step, config.batch_size, x.shape[0], config.shuffle_seed
        )
        batches = micro_batches(indices, config.no_sync_microbatches)
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        for batch in batches:
            loss = F.mse_loss(
                model(x[batch], use_hidden=config.use_hidden), y[batch]
            )
            loss.backward()
            losses.append(loss.detach())
        optimizer.step()

        records.append(
            {
                "step": step,
                "loss": float(torch.stack(losses).mean().item()),
                "checksum": state_checksum(model),
                "hash": state_hash(model),
            }
        )
        history.append(flatten_params(model).clone())

    total = sum(parameter.numel() for parameter in model.parameters())
    params = (
        torch.stack(history)
        if history
        else torch.empty((0, total), dtype=dtype)
    )
    return records, params


def run_reference(config: RunConfig) -> ReferenceRun:
    started = time.perf_counter()
    records, params = _reference_single_process_loop(config)
    return ReferenceRun(
        config=config,
        records=records,
        params=params,
        elapsed_s=time.perf_counter() - started,
    )


def worker(rank: int, config_dict: dict[str, Any], out_dir: str) -> None:
    """One gloo rank: init, broadcast, train, and write artifacts."""

    torch.set_num_threads(1)
    config = RunConfig(**config_dict)
    dtype = config.torch_dtype
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    if config.device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={config.device!r} requested but CUDA is unavailable"
            )
        if ":" not in config.device:
            device = torch.device(
                f"{config.device}:{rank % torch.cuda.device_count()}"
            )
        else:
            device = torch.device(config.device)
        torch.cuda.set_device(device)

    rendezvous = output / "rendezvous"
    dist.init_process_group(
        backend=config.backend,
        init_method=f"file://{rendezvous.as_posix()}",
        rank=rank,
        world_size=config.world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        init_seed = config.seed + rank if config.init_jitter else config.seed
        model = make_model(
            init_seed, hidden_dim=config.hidden_dim, dtype=dtype
        ).to(device)
        x, y = make_dataset(config.data_seed, dtype=dtype)
        ddp = DistributedDataParallel(
            model,
            config.world_size,
            bucket_cap_mb=config.bucket_cap_mb,
            sync_mode=config.sync_mode,
            find_unused_parameters=config.find_unused_parameters,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=config.lr)

        records: list[dict[str, Any]] = []
        history: list[Tensor] = []
        none_grad_names: list[str] = []
        train_started = time.perf_counter()

        for step in range(config.steps):
            indices = batch_indices(
                step, config.batch_size, x.shape[0], config.shuffle_seed
            )
            batches = micro_batches(indices, config.no_sync_microbatches)
            optimizer.zero_grad(set_to_none=True)
            losses: list[Tensor] = []

            for micro_index, batch in enumerate(batches):
                local = rank_slice(batch, rank, config.world_size)
                xb = x[local].to(device)
                yb = y[local].to(device)
                loss = F.mse_loss(
                    ddp(xb, use_hidden=config.use_hidden), yb
                )
                if (
                    config.no_sync_microbatches > 1
                    and micro_index < len(batches) - 1
                ):
                    with ddp.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                losses.append(loss.detach())

            none_grad_names = _missing_grad_params(ddp.module)
            ddp.sync_grads()
            optimizer.step()

            records.append(
                {
                    "step": step,
                    "loss": float(torch.stack(losses).mean().item()),
                    "checksum": state_checksum(ddp.module),
                    "hash": state_hash(ddp.module),
                }
            )
            history.append(flatten_params(ddp.module).clone())

        train_seconds = time.perf_counter() - train_started
        total = sum(parameter.numel() for parameter in model.parameters())
        params = (
            torch.stack(history)
            if history
            else torch.empty((0, total), dtype=dtype)
        )
        torch.save({"params": params}, output / f"rank-{rank}-params.pt")

        payload = {
            "rank": rank,
            "world_size": config.world_size,
            "config": asdict(config),
            "pre_broadcast_hash": ddp.pre_broadcast_hash,
            "post_broadcast_hash": ddp.post_broadcast_hash,
            "records": records,
            "none_grad_names": none_grad_names,
            "stats": asdict(ddp.stats),
            "bucket_count": len(ddp.buckets),
            "param_count": sum(
                1
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "param_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "train_seconds": train_seconds,
        }
        temporary = output / f"rank-{rank}.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, output / f"rank-{rank}.json")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_ddp(config: RunConfig) -> DDPRun:
    with tempfile.TemporaryDirectory(prefix="ddp77-") as temporary:
        output = Path(temporary)
        started = time.perf_counter()
        mp.spawn(
            worker,
            args=(asdict(config), str(output)),
            nprocs=config.world_size,
            join=True,
        )
        elapsed = time.perf_counter() - started

        payloads: list[dict[str, Any]] = []
        histories: list[Tensor] = []
        for rank in range(config.world_size):
            payloads.append(
                json.loads((output / f"rank-{rank}.json").read_text())
            )
            loaded = torch.load(
                output / f"rank-{rank}-params.pt",
                map_location="cpu",
                weights_only=True,
            )
            histories.append(loaded["params"])

    for rank, payload in enumerate(payloads):
        if payload["rank"] != rank:
            raise RuntimeError(
                f"rank artifact mismatch: expected {rank}, got {payload['rank']}"
            )
    shapes = {tuple(history.shape) for history in histories}
    if len(shapes) != 1:
        raise RuntimeError(f"rank parameter histories have different shapes: {shapes}")

    return DDPRun(
        config=config,
        rank_payloads=payloads,
        params_by_rank=histories,
        elapsed_s=elapsed,
        train_seconds=sum(
            payload["train_seconds"] for payload in payloads
        )
        / config.world_size,
    )


def mean_rank_loss(run: DDPRun, step: int) -> float:
    return sum(
        payload["records"][step]["loss"] for payload in run.rank_payloads
    ) / run.config.world_size


def compare_runs(run: DDPRun, reference: ReferenceRun) -> Comparison:
    config = run.config
    if config.steps != len(reference.records):
        raise RuntimeError("reference and DDP step counts differ")

    tolerance = 1e-12 if config.dtype == "float64" else 1e-6
    replica_equal: list[bool] = []
    loss_deltas: list[float] = []
    param_deltas: list[float] = []

    for step in range(config.steps):
        loss_deltas.append(
            abs(mean_rank_loss(run, step) - reference.records[step]["loss"])
        )
        reference_params = reference.params[step]
        param_deltas.append(
            max(
                torch.max(
                    torch.abs(
                        run.params_by_rank[rank][step] - reference_params
                    )
                ).item()
                for rank in range(config.world_size)
            )
        )
        replica_equal.append(
            all(
                torch.equal(
                    run.params_by_rank[0][step],
                    run.params_by_rank[rank][step],
                )
                for rank in range(1, config.world_size)
            )
        )

    broadcast_ok = (
        len(
            {
                payload["post_broadcast_hash"]
                for payload in run.rank_payloads
            }
        )
        == 1
    )
    pre_hashes_differ = (
        len(
            {
                payload["pre_broadcast_hash"]
                for payload in run.rank_payloads
            }
        )
        > 1
    )
    passed = (
        broadcast_ok
        and all(replica_equal)
        and all(delta <= tolerance for delta in loss_deltas)
        and all(delta <= tolerance for delta in param_deltas)
    )
    return Comparison(
        replica_equal=replica_equal,
        loss_deltas=loss_deltas,
        param_deltas=param_deltas,
        broadcast_ok=broadcast_ok,
        pre_hashes_differ=pre_hashes_differ,
        passed=passed,
        max_loss_delta=max(loss_deltas, default=0.0),
        max_param_delta=max(param_deltas, default=0.0),
    )


# ---------------------------------------------------------------------------
# CLI, demo, and benchmark
# ---------------------------------------------------------------------------


def _environment() -> str:
    return (
        f"torch={torch.__version__} python={platform.python_version()} "
        f"cpus={os.cpu_count()} platform={platform.platform()}"
    )


def _print_config(config: RunConfig) -> None:
    print(
        "config: "
        f"world_size={config.world_size} steps={config.steps} "
        f"batch={config.batch_size} hidden={config.hidden_dim} "
        f"lr={config.lr:g} dtype={config.dtype} "
        f"sync={config.sync_mode} bucket_cap_mb={config.bucket_cap_mb:g} "
        f"no_sync_microbatches={config.no_sync_microbatches} "
        f"find_unused={config.find_unused_parameters} "
        f"use_hidden={config.use_hidden} init_jitter={config.init_jitter}"
    )


def _print_broadcast(run: DDPRun) -> None:
    pre = [payload["pre_broadcast_hash"][:12] for payload in run.rank_payloads]
    post = [payload["post_broadcast_hash"][:12] for payload in run.rank_payloads]
    bytes_per_rank = run.rank_payloads[0]["stats"]["broadcast_bytes"]
    print(f"broadcast pre hashes:  {pre}")
    print(f"broadcast post hashes: {post}")
    print(f"broadcast bytes/rank:  {bytes_per_rank}")
    if len({*pre}) > 1:
        print("broadcast proof: rank-specific initial state differed before broadcast")
    else:
        print("broadcast proof: ranks shared the initialization seed")


def run_demo(config: RunConfig) -> int:
    print("Data Parallel DDP From Scratch: broadcast, allreduce, buckets, overlap")
    print(_environment())
    _print_config(config)

    reference = run_reference(config)
    run = run_ddp(config)
    comparison = compare_runs(run, reference)
    _print_broadcast(run)

    losses: list[float] = []
    if config.steps:
        print()
        print(
            f"{'step':>4} {'ref_loss':>10} {'ddp_loss':>10} {'dloss':>9} "
            f"{'ref_sum':>13} {'ddp_sum':>13} {'dsum':>9} "
            f"{'max|dw|':>9} {'replicas':>8} result"
        )
        print("-" * 112)
        for step in range(config.steps):
            ref_record = reference.records[step]
            rank_zero = run.rank_payloads[0]["records"][step]
            ddp_loss = mean_rank_loss(run, step)
            losses.append(ddp_loss)
            loss_delta = abs(ddp_loss - ref_record["loss"])
            checksum_delta = abs(
                rank_zero["checksum"] - ref_record["checksum"]
            )
            param_delta = comparison.param_deltas[step]
            replicas = "exact" if comparison.replica_equal[step] else "differ"
            tolerance = 1e-12 if config.dtype == "float64" else 1e-6
            step_ok = (
                loss_delta <= tolerance
                and param_delta <= tolerance
                and comparison.replica_equal[step]
            )
            print(
                f"{step:>4} {ref_record['loss']:>10.6f} "
                f"{ddp_loss:>10.6f} {loss_delta:>9.2e} "
                f"{ref_record['checksum']:>13.8f} "
                f"{rank_zero['checksum']:>13.8f} "
                f"{checksum_delta:>9.2e} {param_delta:>9.2e} "
                f"{replicas:>8} {'PASS' if step_ok else 'FAIL'}"
            )

    stats = run.rank_payloads[0]["stats"]
    steps = max(1, config.steps)
    print()
    print(
        f"buckets={run.rank_payloads[0]['bucket_count']} "
        f"params={run.rank_payloads[0]['param_count']} "
        f"param_bytes/rank={run.rank_payloads[0]['param_bytes']}"
    )
    print(
        f"allreduce calls/step={stats['allreduce_calls'] / steps:.2f} "
        f"bytes/step={stats['allreduce_bytes'] / steps:,.0f} "
        f"overlapped launches/step={stats['overlapped_buckets'] / steps:.2f}"
    )
    print(
        f"worker train_s={run.train_seconds:.3f} "
        f"wall_s={run.elapsed_s:.3f} "
        f"sync_s={stats['sync_seconds']:.3f}"
    )
    print(
        f"max loss delta={comparison.max_loss_delta:.3e} "
        f"max param delta={comparison.max_param_delta:.3e} "
        f"broadcast_ok={comparison.broadcast_ok} "
        f"pre_hashes_differ={comparison.pre_hashes_differ}"
    )

    loss_decreased = True
    if len(losses) >= 2:
        loss_decreased = losses[-1] < losses[0]
        print(f"loss first={losses[0]:.6f} last={losses[-1]:.6f}")
    passed = comparison.passed and loss_decreased
    print("ALL CHECKS PASSED" if passed else "CHECKS FAILED")
    return 0 if passed else 1


def _mode_stats(run: DDPRun) -> dict[str, float]:
    payload = run.rank_payloads[0]
    stats = payload["stats"]
    steps = max(1, run.config.steps)
    return {
        "calls_per_step": stats["allreduce_calls"] / steps,
        "bytes_per_step": stats["allreduce_bytes"] / steps,
        "buckets": payload["bucket_count"],
        "overlap_per_step": stats["overlapped_buckets"] / steps,
        "train_s": run.train_seconds,
    }


def run_benchmark(config: RunConfig) -> int:
    if config.steps < 1:
        raise ValueError("benchmark requires at least one step")

    print("Data Parallel DDP From Scratch: naive versus bucketed gradient sync")
    print(_environment())
    _print_config(config)

    reference = run_reference(config)
    naive = run_ddp(replace(config, sync_mode="naive"))
    bucketed = run_ddp(replace(config, sync_mode="bucketed"))
    naive_comparison = compare_runs(naive, reference)
    bucketed_comparison = compare_runs(bucketed, reference)

    mode_delta = max(
        torch.max(
            torch.abs(
                naive.params_by_rank[0][step]
                - bucketed.params_by_rank[0][step]
            )
        ).item()
        for step in range(config.steps)
    )
    tolerance = 1e-12 if config.dtype == "float64" else 1e-6

    print()
    print(
        f"{'mode':<10} {'allreduce/step':>15} {'bytes/step':>12} "
        f"{'buckets':>8} {'overlap/step':>13} {'train_s':>9} "
        f"{'max|dw|':>9} result"
    )
    print("-" * 88)
    rows = [
        ("naive", naive, naive_comparison),
        ("bucketed", bucketed, bucketed_comparison),
    ]
    for name, run, comparison in rows:
        stats = _mode_stats(run)
        row_ok = comparison.passed
        print(
            f"{name:<10} {stats['calls_per_step']:>15.2f} "
            f"{stats['bytes_per_step']:>12,.0f} {stats['buckets']:>8d} "
            f"{stats['overlap_per_step']:>13.2f} {stats['train_s']:>9.3f} "
            f"{comparison.max_param_delta:>9.2e} "
            f"{'PASS' if row_ok else 'FAIL'}"
        )

    passed = (
        naive_comparison.passed
        and bucketed_comparison.passed
        and mode_delta <= tolerance
    )
    print()
    print(
        f"naive vs bucketed max|dw|={mode_delta:.3e} "
        f"max loss delta (naive)={naive_comparison.max_loss_delta:.3e} "
        f"max loss delta (bucketed)={bucketed_comparison.max_loss_delta:.3e}"
    )
    print("ALL CHECKS PASSED" if passed else "CHECKS FAILED")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a MiniMLP with a from-scratch broadcast+allreduce DDP "
            "wrapper over CPU gloo."
        )
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--bucket-cap-mb", type=float, default=25.0)
    parser.add_argument(
        "--sync-mode", choices=("bucketed", "naive"), default="bucketed"
    )
    parser.add_argument("--no-sync-microbatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument(
        "--dtype", choices=("float64", "float32"), default="float64"
    )
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-jitter", action="store_true")
    parser.add_argument("--find-unused", action="store_true")
    parser.add_argument(
        "--use-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip the middle MLP layer with --no-use-hidden",
    )
    parser.add_argument("--benchmark", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = RunConfig(
            world_size=args.world_size,
            steps=args.steps,
            batch_size=args.batch_size,
            hidden_dim=args.hidden,
            lr=args.lr,
            seed=args.seed,
            data_seed=args.data_seed,
            bucket_cap_mb=args.bucket_cap_mb,
            sync_mode=args.sync_mode,
            find_unused_parameters=args.find_unused,
            no_sync_microbatches=args.no_sync_microbatches,
            use_hidden=args.use_hidden,
            init_jitter=args.init_jitter,
            dtype=args.dtype,
            backend=args.backend,
            device=args.device,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.benchmark:
        return run_benchmark(config)
    return run_demo(config)


if __name__ == "__main__":
    raise SystemExit(main())
