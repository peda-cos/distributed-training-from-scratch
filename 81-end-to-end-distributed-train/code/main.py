"""End-to-end distributed training: MiniGPT + DDP init + ZeRO-1 + checkpoints.

Four spawned CPU processes join a gloo process group through a local file
rendezvous.  Rank 0's initial parameters are broadcast; every step then
reduce-scatters the flat bf16 gradient, applies Adam to a rank-local fp32
optimizer shard, and all-gathers the updated bf16 parameter shard.  At step 10
each rank writes its own checkpoint shard and rank 0 commits the manifest; the
run reloads that checkpoint after step 20 and compares every byte.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
import time
import warnings
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy:.*")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Tensor = torch.Tensor

VOCAB_SIZE = 64
SEQ_LEN = 16
BATCH_SIZE = 4
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
MLP_HIDDEN = 128
PARAM_COUNT = 30144
DEFAULT_WORLD_SIZE = 4
DEFAULT_STEPS = 20
DEFAULT_CHECKPOINT_STEP = 10
DEFAULT_LR = 0.01
DEFAULT_SEED = 7
DEFAULT_DATA_SEED = 11
CORPUS_TOKENS = 8192
BATCH_PHASE_STRIDE = 16
LOSS_NOISE_TOLERANCE = 1e-6
COLLECTIVE_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# MiniGPT
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, dtype=dtype)
        self.proj = nn.Linear(D_MODEL, D_MODEL, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        head_dim = D_MODEL // N_HEADS
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, seq, N_HEADS, head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, seq, D_MODEL)
        return self.proj(attended)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with a GELU MLP."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL, dtype=dtype)
        self.attn = CausalSelfAttention(dtype=dtype)
        self.ln2 = nn.LayerNorm(D_MODEL, dtype=dtype)
        self.fc1 = nn.Linear(D_MODEL, MLP_HIDDEN, dtype=dtype)
        self.fc2 = nn.Linear(MLP_HIDDEN, D_MODEL, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class MiniGPT(nn.Module):
    """Two transformer blocks, four heads, and an untied LM head."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, dtype=dtype)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL, dtype=dtype)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dtype=dtype) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL, dtype=dtype)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, dtype=dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor) -> Tensor:
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_model(
    seed: int, *, dtype: torch.dtype = torch.bfloat16
) -> MiniGPT:
    """Construct the model deterministically from one seed."""

    torch.manual_seed(seed)
    return MiniGPT(dtype=dtype)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


# ---------------------------------------------------------------------------
# Deterministic next-token data
# ---------------------------------------------------------------------------


def make_corpus(seed: int, total_tokens: int) -> Tensor:
    """An arithmetic next-token stream: token i predicts token i+1."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(total_tokens) is not int or total_tokens < SEQ_LEN + 1:
        raise ValueError(f"total_tokens must be at least {SEQ_LEN + 1}")
    positions = torch.arange(total_tokens, dtype=torch.long)
    return (positions + seed) % VOCAB_SIZE


def make_batch(
    corpus: Tensor, batch_size: int = BATCH_SIZE
) -> tuple[Tensor, Tensor]:
    """Return fixed, phase-shifted next-token windows from the corpus."""

    if corpus.dim() != 1 or corpus.dtype != torch.long:
        raise ValueError("corpus must be a one-dimensional long tensor")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    starts = torch.arange(batch_size, dtype=torch.long) * BATCH_PHASE_STRIDE
    if int(starts[-1]) + SEQ_LEN >= corpus.numel():
        raise ValueError(
            f"corpus has {corpus.numel()} tokens, need at least "
            f"{int(starts[-1]) + SEQ_LEN + 1}"
        )
    inputs = torch.stack([corpus[start : start + SEQ_LEN] for start in starts])
    targets = torch.stack(
        [corpus[start + 1 : start + SEQ_LEN + 1] for start in starts]
    )
    return inputs, targets


# ---------------------------------------------------------------------------
# Flat parameter helpers and byte encodings
# ---------------------------------------------------------------------------


def _parameter_list(module: nn.Module) -> list[nn.Parameter]:
    parameters = list(module.parameters())
    if not parameters:
        raise ValueError("model has no parameters")
    dtypes = {parameter.dtype for parameter in parameters}
    if len(dtypes) != 1:
        raise ValueError(f"parameters must share one dtype, got {sorted(map(str, dtypes))}")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError(f"parameters must share one device, got {sorted(map(str, devices))}")
    return parameters


def flatten_params(module: nn.Module) -> Tensor:
    parameters = _parameter_list(module)
    return torch.cat(
        [parameter.detach().reshape(-1) for parameter in parameters]
    )


def flatten_grads(module: nn.Module) -> Tensor:
    pieces: list[Tensor] = []
    for parameter in _parameter_list(module):
        grad = parameter.grad
        if grad is None:
            pieces.append(torch.zeros_like(parameter).detach().reshape(-1))
            continue
        if grad.dtype != parameter.dtype:
            raise ValueError(
                f"gradient dtype {grad.dtype} does not match {parameter.dtype}"
            )
        if grad.device != parameter.device:
            raise ValueError(
                f"gradient device {grad.device} does not match {parameter.device}"
            )
        pieces.append(grad.detach().reshape(-1))
    return torch.cat(pieces)


def unflatten_into(module: nn.Module, flat: Tensor) -> None:
    parameters = _parameter_list(module)
    if flat.dim() != 1:
        raise ValueError("flat must be one-dimensional")
    total = sum(parameter.numel() for parameter in parameters)
    if flat.numel() != total:
        raise ValueError(f"flat has {flat.numel()} elements, model needs {total}")
    for parameter in parameters:
        if parameter.dtype != flat.dtype:
            raise ValueError(
                f"flat dtype {flat.dtype} does not match {parameter.dtype}"
            )
        if parameter.device != flat.device:
            raise ValueError(
                f"flat device {flat.device} does not match {parameter.device}"
            )
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(flat.narrow(0, offset, count).view_as(parameter))
            offset += count


def encode_fp32_le(values: Iterable[float]) -> bytes:
    encoded = array("f", values)
    if sys.byteorder == "big":
        encoded.byteswap()
    return encoded.tobytes()


def decode_fp32_le(data: bytes) -> array:
    if len(data) % 4:
        raise ValueError("fp32 payload length must be a multiple of 4")
    decoded = array("f")
    decoded.frombytes(data)
    if sys.byteorder == "big":
        decoded.byteswap()
    return decoded


def state_digest(flat: Tensor) -> str:
    values = flat.detach().float().cpu().reshape(-1).tolist()
    return hashlib.sha256(encode_fp32_le(values)).hexdigest()


# ---------------------------------------------------------------------------
# Run configuration and flat-shard planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    world_size: int = DEFAULT_WORLD_SIZE
    steps: int = DEFAULT_STEPS
    checkpoint_step: int = DEFAULT_CHECKPOINT_STEP
    batch_size: int = BATCH_SIZE
    lr: float = DEFAULT_LR
    seed: int = DEFAULT_SEED
    data_seed: int = DEFAULT_DATA_SEED
    backend: str = "gloo"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if (
            type(self.checkpoint_step) is not int
            or not 1 <= self.checkpoint_step <= self.steps
        ):
            raise ValueError("checkpoint_step must be in [1, steps]")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.batch_size % self.world_size:
            raise ValueError("batch_size must be divisible by world_size")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.data_seed) is not int or self.data_seed < 0:
            raise ValueError("data_seed must be a non-negative integer")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.backend not in ("gloo", "nccl"):
            raise ValueError(f"unsupported backend {self.backend!r}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"unsupported device {self.device!r}")
        if self.device == "cpu" and self.backend == "nccl":
            raise ValueError("the nccl backend requires a CUDA device")
        if self.device == "cuda" and self.backend != "nccl":
            raise ValueError("the cuda device requires the nccl backend")


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def partition_lengths(total_numel: int, world_size: int) -> list[int]:
    if type(total_numel) is not int or total_numel < 1:
        raise ValueError("total_numel must be a positive integer")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if total_numel < world_size:
        raise ValueError("total_numel must be at least world_size")
    shard_numel = ceil_div(total_numel, world_size)
    return [shard_numel] * (world_size - 1) + [
        total_numel - shard_numel * (world_size - 1)
    ]


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
    if type(total_numel) is not int or total_numel < 1:
        raise ValueError("total_numel must be a positive integer")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
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


# ---------------------------------------------------------------------------
# DDP init broadcast and scalar reduction
# ---------------------------------------------------------------------------


class DDPBroadcast:
    """The DDP init-sync portion used by the ZeRO-1 composition."""

    def __init__(self, model: nn.Module, rank: int, world_size: int) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DDPBroadcast requires an initialized process group")
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError(
                f"process group rank/world ({dist.get_rank()}/{dist.get_world_size()}) "
                f"does not match ({rank}/{world_size})"
            )
        self.rank = rank
        self.world_size = world_size
        self.device = next(model.parameters()).device
        if world_size > 1:
            with torch.no_grad():
                for tensor in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(tensor.data, src=0)

    def mean(self, value: float) -> float:
        if self.world_size == 1:
            return float(value)
        tensor = torch.tensor(float(value), dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item()) / self.world_size


# ---------------------------------------------------------------------------
# ZeRO-1 optimizer: reduce-scatter grads, Adam on a local shard, all-gather
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryProfile:
    params_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    total_bytes: int
    formula_bytes: int
    formula_ok: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "params_bytes": self.params_bytes,
            "grads_bytes": self.grads_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "total_bytes": self.total_bytes,
            "formula_bytes": self.formula_bytes,
            "formula_ok": self.formula_ok,
        }


class Zero1Optimizer:
    """Adam over a sharded fp32 master with real collectives."""

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.params = _parameter_list(model)
        self.compute_dtype = self.params[0].dtype
        self.device = self.params[0].device
        self.total_numel = sum(parameter.numel() for parameter in self.params)
        self.spec = shard_spec(self.total_numel, world_size, rank)
        self.master = torch.zeros(
            self.spec.shard_numel, dtype=torch.float32, device=self.device
        )
        self.exp_avg = torch.zeros_like(self.master)
        self.exp_avg_sq = torch.zeros_like(self.master)
        flat = flatten_params(model)
        self.master[: self.spec.local_numel].copy_(
            flat[self.spec.start : self.spec.end].float()
        )
        self.step_count = 0
        self.comm = CommStats()
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            if dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise RuntimeError(
                    "process group rank/world does not match the optimizer"
                )

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def _require_group(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Zero1Optimizer with world_size > 1 requires torch.distributed"
            )
        if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
            raise RuntimeError(
                "process group rank/world does not match the optimizer"
            )

    def step(self) -> None:
        if self.world_size > 1:
            self._require_group()
        grad_flat = flatten_grads(self.model)
        grad_padded = pad_flat(grad_flat, self.spec)
        if self.world_size > 1:
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
        else:
            grad_shard = grad_padded.narrow(0, 0, self.spec.shard_numel)

        grad_fp32 = grad_shard.to(torch.float32)
        if self.world_size > 1:
            grad_fp32.div_(self.world_size)

        self.step_count += 1
        beta1, beta2 = self.betas
        bias_correction1 = 1.0 - beta1**self.step_count
        bias_correction2 = 1.0 - beta2**self.step_count
        if self.weight_decay:
            grad_fp32 = grad_fp32.add(
                self.master, alpha=self.weight_decay
            )
        self.exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(
            grad_fp32, grad_fp32, value=1.0 - beta2
        )
        denominator = (
            self.exp_avg_sq.sqrt()
            .div_(math.sqrt(bias_correction2))
            .add_(self.eps)
        )
        self.master.addcdiv_(
            self.exp_avg,
            denominator,
            value=-self.lr / bias_correction1,
        )

        param_shard = self.master.to(self.compute_dtype)
        if self.world_size > 1:
            full = torch.empty(
                self.spec.padded_numel,
                dtype=self.compute_dtype,
                device=self.device,
            )
            dist.all_gather_into_tensor(full, param_shard)
            self.comm.allgather_calls += 1
            self.comm.allgather_bytes += full.numel() * full.element_size()
        else:
            full = param_shard
        unflatten_into(
            self.model, full.narrow(0, 0, self.total_numel)
        )

    def state_nbytes(self) -> int:
        return (
            self.master.nbytes
            + self.exp_avg.nbytes
            + self.exp_avg_sq.nbytes
        )


def measure_memory(
    model: nn.Module, optimizer: Zero1Optimizer
) -> MemoryProfile:
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
    formula_bytes = 12 * optimizer.total_numel // optimizer.world_size
    return MemoryProfile(
        params_bytes=params_bytes,
        grads_bytes=grads_bytes,
        optimizer_bytes=optimizer_bytes,
        total_bytes=params_bytes + grads_bytes + optimizer_bytes,
        formula_bytes=formula_bytes,
        formula_ok=optimizer_bytes == formula_bytes,
    )
