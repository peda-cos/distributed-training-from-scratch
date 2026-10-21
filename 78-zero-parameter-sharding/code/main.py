"""ZeRO stage 1 optimizer-state sharding over real CPU gloo ranks.

The model still runs forward and backward in full on every rank.  The flat
gradient is reduce-scattered so rank ``r`` receives only the summed gradient
for its contiguous parameter shard.  Rank ``r`` owns the matching fp32 master
weights and Adam moments, applies Adam locally, then all-gathers its updated
compute-dtype parameter shard.  Every rank reconstructs the full model before
the next forward pass.

The implementation is intentionally small enough to inspect: no buckets, no
overlap, no stage 2/3, and no checkpoint sharding.  The collectives and the
optimizer math are real; only the hardware is a local CPU.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy:.*")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Tensor = torch.Tensor

INPUT_DIM = 16
DATASET_SIZE = 4096
DEFAULT_HIDDEN = 32
DEFAULT_BATCH = 64
DEFAULT_STEPS = 20
DEFAULT_LR = 0.01
DEFAULT_SEED = 7
DATA_SEED = 2026
SHUFFLE_SEED = 99
DEFAULT_WORLD_SIZE = 4
COLLECTIVE_TIMEOUT_S = 120

PARAM_TOLERANCE = {torch.float32: 1e-5, torch.bfloat16: 5e-2}
LOSS_TOLERANCE = {torch.float32: 1e-6, torch.bfloat16: 1e-2}


# ---------------------------------------------------------------------------
# Configuration, deterministic model, and data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    world_size: int = DEFAULT_WORLD_SIZE
    steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH
    hidden_dim: int = DEFAULT_HIDDEN
    lr: float = DEFAULT_LR
    seed: int = DEFAULT_SEED
    data_seed: int = DATA_SEED
    dtype: str = "bf16"
    backend: str = "gloo"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if self.steps < 0:
            raise ValueError("steps must not be negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.lr < 0:
            raise ValueError("lr must not be negative")
        if self.batch_size % self.world_size:
            raise ValueError("batch_size must be divisible by world_size")
        if self.dtype not in ("bf16", "fp32"):
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.backend not in ("gloo", "nccl"):
            raise ValueError(f"unsupported backend {self.backend!r}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"unsupported device {self.device!r}")
        if self.device == "cpu" and self.backend == "nccl":
            raise ValueError("the nccl backend requires a CUDA device")
        if self.device == "cuda" and self.backend != "nccl":
            raise ValueError("the cuda device requires the nccl backend")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bf16" else torch.float32


class ThreeLayerMLP(nn.Module):
    """Three linear layers with ReLU activations, small enough for CPU."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, hidden_dim, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, 1, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def make_model(seed: int, hidden_dim: int = DEFAULT_HIDDEN, dtype: torch.dtype = torch.float32) -> ThreeLayerMLP:
    """Create the MLP deterministically.

    ``torch.manual_seed`` seeds the global CPU RNG; every rank and the
    single-process reference call this with the same seed.
    """

    torch.manual_seed(seed)
    return ThreeLayerMLP(hidden_dim=hidden_dim, dtype=dtype)


def make_dataset(seed: int, size: int = DATASET_SIZE) -> tuple[Tensor, Tensor]:
    """A fixed linear-regression dataset with independent generators."""

    generator_x = torch.Generator().manual_seed(seed)
    x = torch.randn(size, INPUT_DIM, generator=generator_x)
    teacher = torch.randn(INPUT_DIM, 1, generator=torch.Generator().manual_seed(seed + 1))
    teacher = teacher / math.sqrt(INPUT_DIM)
    noise = 0.05 * torch.randn(size, 1, generator=torch.Generator().manual_seed(seed + 2))
    return x, x @ teacher + noise


def batch_indices(step: int, batch_size: int, dataset_size: int, shuffle_seed: int) -> Tensor:
    """Return a deterministic global batch for one step."""

    if step < 0:
        raise ValueError("step must not be negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size > dataset_size:
        raise ValueError("batch_size must not exceed dataset_size")
    generator = torch.Generator().manual_seed(shuffle_seed + step)
    permutation = torch.randperm(dataset_size, generator=generator)
    return permutation[:batch_size]


# ---------------------------------------------------------------------------
# Flat parameters and rank shards
# ---------------------------------------------------------------------------


def _parameter_list(module: nn.Module) -> list[nn.Parameter]:
    params = list(module.parameters())
    if not params:
        raise ValueError("model has no parameters")
    dtypes = {parameter.dtype for parameter in params}
    if len(dtypes) != 1:
        raise ValueError(f"parameters must share one dtype, got {sorted(map(str, dtypes))}")
    devices = {parameter.device for parameter in params}
    if len(devices) != 1:
        raise ValueError(f"parameters must share one device, got {sorted(map(str, devices))}")
    return params


def flatten_params(module: nn.Module) -> Tensor:
    """Pack every parameter into one contiguous 1-D detached copy."""

    params = _parameter_list(module)
    return torch.cat([parameter.detach().reshape(-1) for parameter in params])


def unflatten_into(module: nn.Module, flat: Tensor) -> None:
    """Copy a flat vector back into the module's existing Parameter objects."""

    params = _parameter_list(module)
    if flat.dim() != 1:
        raise ValueError("flat must be one-dimensional")
    total = sum(parameter.numel() for parameter in params)
    if flat.numel() != total:
        raise ValueError(f"flat has {flat.numel()} elements, model needs {total}")
    for parameter in params:
        if parameter.dtype != flat.dtype:
            raise ValueError(
                f"flat dtype {flat.dtype} does not match parameter dtype {parameter.dtype}"
            )
        if parameter.device != flat.device:
            raise ValueError(
                f"flat device {flat.device} does not match parameter device {parameter.device}"
            )

    offset = 0
    with torch.no_grad():
        for parameter in params:
            count = parameter.numel()
            parameter.copy_(flat.narrow(0, offset, count).view_as(parameter))
            offset += count


def flatten_grads(module: nn.Module) -> Tensor:
    """Pack gradients into one 1-D copy; unused parameters contribute zeros."""

    params = _parameter_list(module)
    pieces: list[Tensor] = []
    for parameter in params:
        grad = parameter.grad
        if grad is None:
            pieces.append(torch.zeros_like(parameter).detach().reshape(-1))
            continue
        if grad.dtype != parameter.dtype:
            raise ValueError(
                f"gradient dtype {grad.dtype} does not match parameter dtype {parameter.dtype}"
            )
        if grad.device != parameter.device:
            raise ValueError(
                f"gradient device {grad.device} does not match parameter device {parameter.device}"
            )
        pieces.append(grad.detach().reshape(-1))
    return torch.cat(pieces)


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


@dataclass(frozen=True)
class ShardSpec:
    total_numel: int
    world_size: int
    rank: int
    shard_numel: int
    padded_numel: int
    start: int
    end: int

    @property
    def local_numel(self) -> int:
        return self.end - self.start

    @property
    def owns_any(self) -> bool:
        return self.start < self.end

    def as_dict(self) -> dict[str, int]:
        return {
            "total_numel": self.total_numel,
            "world_size": self.world_size,
            "rank": self.rank,
            "shard_numel": self.shard_numel,
            "padded_numel": self.padded_numel,
            "start": self.start,
            "end": self.end,
            "local_numel": self.local_numel,
        }


def shard_spec(total_numel: int, world_size: int, rank: int) -> ShardSpec:
    """Plan a padded, rank-major partition of a flat parameter vector."""

    if total_numel < 1:
        raise ValueError("total_numel must be at least 1")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    shard_numel = ceil_div(total_numel, world_size)
    padded_numel = shard_numel * world_size
    start = min(rank * shard_numel, total_numel)
    end = min(start + shard_numel, total_numel)
    return ShardSpec(
        total_numel=total_numel,
        world_size=world_size,
        rank=rank,
        shard_numel=shard_numel,
        padded_numel=padded_numel,
        start=start,
        end=end,
    )


def pad_flat(flat: Tensor, spec: ShardSpec) -> Tensor:
    """Pad a flat vector to a whole number of equal rank shards."""

    if flat.dim() != 1:
        raise ValueError("flat must be one-dimensional")
    if flat.numel() != spec.total_numel:
        raise ValueError(
            f"flat has {flat.numel()} elements, expected {spec.total_numel}"
        )
    if flat.numel() == spec.padded_numel:
        return flat.contiguous()
    padded = flat.new_zeros(spec.padded_numel)
    padded[: flat.numel()].copy_(flat)
    return padded


# ---------------------------------------------------------------------------
# Memory and communication accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryBudget:
    params_bytes: int
    grads_bytes: int
    master_bytes: int
    moments_bytes: int

    @property
    def optimizer_bytes(self) -> int:
        return self.master_bytes + self.moments_bytes

    @property
    def total_bytes(self) -> int:
        return self.params_bytes + self.grads_bytes + self.optimizer_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "params_bytes": self.params_bytes,
            "grads_bytes": self.grads_bytes,
            "master_bytes": self.master_bytes,
            "moments_bytes": self.moments_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class MeasuredMemory:
    params_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    total_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "params_bytes": self.params_bytes,
            "grads_bytes": self.grads_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass
class CommStats:
    reduce_scatter_calls: int = 0
    allgather_calls: int = 0
    reduce_scatter_bytes: int = 0
    allgather_bytes: int = 0

    @property
    def calls(self) -> int:
        return self.reduce_scatter_calls + self.allgather_calls

    @property
    def payload_bytes(self) -> int:
        return self.reduce_scatter_bytes + self.allgather_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "reduce_scatter_calls": self.reduce_scatter_calls,
            "allgather_calls": self.allgather_calls,
            "reduce_scatter_bytes": self.reduce_scatter_bytes,
            "allgather_bytes": self.allgather_bytes,
            "calls": self.calls,
            "payload_bytes": self.payload_bytes,
        }


def memory_budget(
    num_params: int,
    world_size: int,
    stage: int,
    param_bytes: int = 2,
    master_bytes: int = 4,
) -> MemoryBudget:
    """Formula budget per rank for DDP and ZeRO stages 1-3.

    The model is mixed precision: ``param_bytes`` is the compute-dtype size
    (2 for bf16/fp16, 4 for fp32) and ``master_bytes`` is the separate fp32
    master copy (0 when the parameters themselves are the master).
    """

    if num_params < 0:
        raise ValueError("num_params must not be negative")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be one of 0, 1, 2, 3")
    if param_bytes <= 0:
        raise ValueError("param_bytes must be positive")
    if master_bytes < 0:
        raise ValueError("master_bytes must not be negative")

    shard = ceil_div(num_params, world_size) if num_params else 0
    params_bytes = num_params * param_bytes
    grads_bytes = num_params * param_bytes
    master = num_params * master_bytes
    moments = 8 * num_params

    if stage >= 2:
        grads_bytes = shard * param_bytes
    if stage >= 1:
        master = shard * master_bytes
        moments = 8 * shard
    if stage >= 3:
        params_bytes = shard * param_bytes

    return MemoryBudget(
        params_bytes=params_bytes,
        grads_bytes=grads_bytes,
        master_bytes=master,
        moments_bytes=moments,
    )


def zero_payload_bytes(
    num_params: int, world_size: int, param_bytes: int = 2
) -> tuple[int, int]:
    """Ideal per-rank reduce-scatter input and all-gather output bytes."""

    if num_params < 0:
        raise ValueError("num_params must not be negative")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if param_bytes <= 0:
        raise ValueError("param_bytes must be positive")
    padded = ceil_div(num_params, world_size) * world_size
    payload = padded * param_bytes
    return payload, payload


def ddp_payload_bytes(num_params: int, param_bytes: int = 2) -> int:
    """One all-reduce input plus output, in bytes."""

    if num_params < 0:
        raise ValueError("num_params must not be negative")
    if param_bytes <= 0:
        raise ValueError("param_bytes must be positive")
    return 2 * num_params * param_bytes


def ring_bandwidth_bytes(num_params: int, world_size: int, param_bytes: int = 2) -> float:
    """Ring all-reduce / reduce-scatter+all-gather bytes moved per rank."""

    if num_params < 0:
        raise ValueError("num_params must not be negative")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if param_bytes <= 0:
        raise ValueError("param_bytes must be positive")
    return 2.0 * num_params * param_bytes * (world_size - 1) / world_size


def format_bytes(value: int) -> str:
    return f"{value:,}"


def format_gib(value: int | float) -> str:
    return f"{value / (1024 ** 3):.2f}"


# ---------------------------------------------------------------------------
# ZeRO stage 1 optimizer and single-process reference
# ---------------------------------------------------------------------------


class ZeroOptimizer:
    """Adam with rank-sharded fp32 master weights and moments.

    Construction does not need a process group: the shard is defined by the
    explicit ``world_size``/``rank``.  ``step()`` requires an initialized
    default process group when ``world_size > 1``.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int,
        rank: int,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must not be negative")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must not be negative")

        params = _parameter_list(model)
        frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if frozen:
            raise ValueError(f"ZeroOptimizer requires trainable parameters, frozen: {frozen}")

        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.betas = (float(beta1), float(beta2))
        self.eps = eps
        self.weight_decay = weight_decay

        flat = flatten_params(model)
        self.compute_dtype = flat.dtype
        self.device = flat.device
        self.total_numel = flat.numel()
        self.spec = shard_spec(self.total_numel, world_size, rank)
        self._param_count = len(params)

        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size() != world_size:
                raise ValueError(
                    f"process group world size {dist.get_world_size()} != {world_size}"
                )
            if dist.get_rank() != rank:
                raise ValueError(f"process group rank {dist.get_rank()} != {rank}")

        self.master = torch.zeros(
            self.spec.shard_numel, dtype=torch.float32, device=self.device
        )
        self.exp_avg = torch.zeros_like(self.master)
        self.exp_avg_sq = torch.zeros_like(self.master)
        local = flat[self.spec.start : self.spec.end].float()
        self.master[: self.spec.local_numel].copy_(local)

        self.step_count = 0
        self.comm = CommStats()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.model.zero_grad(set_to_none=set_to_none)

    def _validate_model(self) -> None:
        params = list(self.model.parameters())
        if len(params) != self._param_count:
            raise RuntimeError("model parameter count changed after optimizer construction")
        total = sum(parameter.numel() for parameter in params)
        if total != self.total_numel:
            raise RuntimeError("model parameter size changed after optimizer construction")
        for parameter in params:
            if parameter.dtype != self.compute_dtype:
                raise RuntimeError("model parameter dtype changed after optimizer construction")
            if parameter.device != self.device:
                raise RuntimeError("model parameter device changed after optimizer construction")
            if not parameter.requires_grad:
                raise RuntimeError("model parameter became frozen after optimizer construction")

    def _require_group(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ZeroOptimizer with world_size > 1 requires torch.distributed "
                "to be initialized; call init_process_group before step()"
            )
        if dist.get_world_size() != self.world_size:
            raise RuntimeError(
                f"process group world size {dist.get_world_size()} != {self.world_size}"
            )
        if dist.get_rank() != self.rank:
            raise RuntimeError(f"process group rank {dist.get_rank()} != {self.rank}")

    def step(self) -> None:
        self._validate_model()
        if self.world_size > 1:
            self._require_group()

        grad_flat = flatten_grads(self.model)
        grad_padded = pad_flat(grad_flat, self.spec)

        if self.world_size == 1:
            grad_shard = grad_padded.narrow(0, 0, self.spec.shard_numel)
        else:
            grad_shard = torch.empty(
                self.spec.shard_numel,
                dtype=self.compute_dtype,
                device=self.device,
            )
            dist.reduce_scatter_tensor(
                grad_shard, grad_padded, op=dist.ReduceOp.SUM
            )
            self.comm.reduce_scatter_calls += 1
            self.comm.reduce_scatter_bytes += (
                grad_padded.numel() * grad_padded.element_size()
            )

        grad_fp32 = grad_shard.to(torch.float32)
        if self.world_size > 1:
            grad_fp32.div_(self.world_size)

        self.step_count += 1
        beta1, beta2 = self.betas
        bias_correction1 = 1.0 - beta1 ** self.step_count
        bias_correction2 = 1.0 - beta2 ** self.step_count
        if self.weight_decay:
            grad_fp32 = grad_fp32.add(self.master, alpha=self.weight_decay)
        self.exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(
            grad_fp32, grad_fp32, value=1.0 - beta2
        )
        denom = (
            self.exp_avg_sq.sqrt()
            .div_(math.sqrt(bias_correction2))
            .add_(self.eps)
        )
        self.master.addcdiv_(
            self.exp_avg, denom, value=-self.lr / bias_correction1
        )

        param_shard = self.master.to(self.compute_dtype)
        if self.world_size == 1:
            full = param_shard
        else:
            full = torch.empty(
                self.spec.padded_numel,
                dtype=self.compute_dtype,
                device=self.device,
            )
            dist.all_gather_into_tensor(full, param_shard)
            self.comm.allgather_calls += 1
            self.comm.allgather_bytes += full.numel() * full.element_size()

        unflatten_into(self.model, full.narrow(0, 0, self.total_numel))

    def state_nbytes(self) -> int:
        return (
            self.master.nbytes
            + self.exp_avg.nbytes
            + self.exp_avg_sq.nbytes
        )


def measure_memory(model: nn.Module, optimizer: ZeroOptimizer) -> MeasuredMemory:
    """Logical resident bytes at the post-backward step boundary."""

    params_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    grads_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer_bytes = optimizer.state_nbytes()
    return MeasuredMemory(
        params_bytes=params_bytes,
        grads_bytes=grads_bytes,
        optimizer_bytes=optimizer_bytes,
        total_bytes=params_bytes + grads_bytes + optimizer_bytes,
    )


@dataclass
class ReferenceRun:
    config: RunConfig
    records: list[dict[str, Any]]
    params: Tensor
    elapsed_s: float


def train_reference(config: RunConfig) -> ReferenceRun:
    """Train the same per-rank microbatches with a full fp32 Adam master."""

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.device == "cuda" else "cpu")
    compute_dtype = config.torch_dtype
    model = make_model(config.seed, config.hidden_dim, compute_dtype).to(device)
    x, y = make_dataset(config.data_seed)
    master = flatten_params(model).float().clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [master],
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )

    history = [flatten_params(model).float().cpu()]
    records: list[dict[str, Any]] = []
    local_batch = config.batch_size // config.world_size
    started = time.perf_counter()

    for step in range(config.steps):
        unflatten_into(model, master.detach().to(compute_dtype))
        optimizer.zero_grad(set_to_none=True)
        accumulator = torch.zeros_like(master)
        losses: list[float] = []
        indices = batch_indices(
            step, config.batch_size, x.shape[0], SHUFFLE_SEED
        )
        for rank in range(config.world_size):
            selected = indices[
                rank * local_batch : (rank + 1) * local_batch
            ]
            xb = x[selected].to(device=device, dtype=compute_dtype)
            yb = y[selected].to(device=device, dtype=compute_dtype)
            loss = F.mse_loss(model(xb), yb)
            losses.append(float(loss.detach()))
            loss.backward()
            accumulator += flatten_grads(model).float()
            model.zero_grad(set_to_none=True)

        master.grad = accumulator / config.world_size
        optimizer.step()
        unflatten_into(model, master.detach().to(compute_dtype))
        flat = flatten_params(model).float().cpu()
        history.append(flat)
        records.append(
            {
                "step": step,
                "loss": sum(losses) / config.world_size,
                "param_sum": float(flat.double().sum()),
            }
        )

    return ReferenceRun(
        config=config,
        records=records,
        params=torch.stack(history),
        elapsed_s=time.perf_counter() - started,
    )
