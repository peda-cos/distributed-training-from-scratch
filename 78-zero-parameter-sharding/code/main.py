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
