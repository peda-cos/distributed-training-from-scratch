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
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
import warnings
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import batched
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

    def local_state(self, flat_params: Tensor) -> RankState:
        """Snapshot this rank's checkpoint payload after an update."""
        if flat_params.numel() != self.total_numel:
            raise ValueError(
                f"flat params have {flat_params.numel()} elements, "
                f"expected {self.total_numel}"
            )
        param_slice = (
            flat_params.detach()[self.spec.start : self.spec.end]
            .cpu()
            .to(torch.float16)
            .reshape(-1)
            .tolist()
        )
        offset = self.spec.local_numel
        return RankState(
            rank=self.rank,
            param_fp16=encode_fp16(param_slice),
            master_fp32=array(
                "f", self.master[:offset].detach().cpu().tolist()
            ),
            exp_avg_fp32=array(
                "f", self.exp_avg[:offset].detach().cpu().tolist()
            ),
            exp_avg_sq_fp32=array(
                "f", self.exp_avg_sq[:offset].detach().cpu().tolist()
            ),
            adam_step=self.step_count,
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


# ---------------------------------------------------------------------------
# Sharded checkpoint: wire format, atomic commit, load, and resume verify
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
SHARD_FORMAT_VERSION = 1
MAGIC = b"SHARD001"
MANIFEST_NAME = "manifest.json"
STEP_DIR_RE = re.compile(r"^step-(\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 1 << 20
MAX_SHARD_HEADER_BYTES = 4096
PAYLOAD_BYTES_PER_ELEMENT = 14


class CheckpointError(Exception):
    """Base class for checkpoint failures."""


class ManifestError(CheckpointError, ValueError):
    """The manifest or a shard header violates the schema."""


class WorldSizeMismatchError(CheckpointError):
    """The checkpoint's world size differs from the resume request."""


class ShardIntegrityError(CheckpointError):
    """A shard is missing, corrupted, truncated, or mislabeled."""


class ResumeMismatchError(CheckpointError):
    """A loaded shard does not equal the in-memory snapshot."""


class CheckpointExistsError(CheckpointError):
    """A committed checkpoint already exists for the requested step."""


def _json_int(value: object, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _fsync_dir(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def encode_fp16(values: Iterable[float]) -> bytes:
    encoded = bytearray()
    for chunk in batched(values, 1024):
        try:
            encoded.extend(struct.pack(f"<{len(chunk)}e", *chunk))
        except (OverflowError, struct.error, TypeError) as exc:
            raise ValueError(
                "value cannot be encoded as IEEE-754 binary16"
            ) from exc
    return bytes(encoded)


def decode_fp16(data: bytes) -> array:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("fp16 data must be bytes-like")
    if len(data) % 2:
        raise ValueError("fp16 payload length must be a multiple of 2")
    values = array("f")
    for offset in range(0, len(data), 1024 * 2):
        count = min(1024, (len(data) - offset) // 2)
        values.extend(struct.unpack_from(f"<{count}e", data, offset))
    return values


@dataclass(frozen=True)
class RankState:
    rank: int
    param_fp16: bytes
    master_fp32: array
    exp_avg_fp32: array
    exp_avg_sq_fp32: array
    adam_step: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if not isinstance(self.param_fp16, bytes):
            raise ValueError("param_fp16 must be bytes")
        if self.param_fp16 and len(self.param_fp16) % 2:
            raise ValueError("param_fp16 length must be a multiple of 2")
        for name in ("master_fp32", "exp_avg_fp32", "exp_avg_sq_fp32"):
            values = getattr(self, name)
            if not isinstance(values, array) or values.typecode != "f":
                raise ValueError(f"{name} must be an array('f')")
        numel = len(self.master_fp32)
        if numel < 1:
            raise ValueError("rank state must not be empty")
        if len(self.param_fp16) != 2 * numel:
            raise ValueError("param_fp16 length does not match numel")
        if len(self.exp_avg_fp32) != numel or len(self.exp_avg_sq_fp32) != numel:
            raise ValueError("optimizer moment lengths do not match numel")
        if type(self.adam_step) is not int or self.adam_step < 1:
            raise ValueError("adam_step must be a positive integer")

    @property
    def numel(self) -> int:
        return len(self.master_fp32)

    def payload_bytes(self) -> bytes:
        return (
            self.param_fp16
            + encode_fp32_le(self.master_fp32)
            + encode_fp32_le(self.exp_avg_fp32)
            + encode_fp32_le(self.exp_avg_sq_fp32)
        )


@dataclass(frozen=True)
class ShardEntry:
    rank: int
    path: str
    sha256: str
    param_shard_offset: int
    param_shard_numel: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ManifestError("shard rank must be a non-negative integer")
        if self.path != f"rank{self.rank}.bin":
            raise ManifestError(
                f"shard path {self.path!r} must be rank{self.rank}.bin"
            )
        if not isinstance(self.sha256, str) or not SHA256_RE.fullmatch(
            self.sha256
        ):
            raise ManifestError(
                "sha256 must be 64 lowercase hexadecimal characters"
            )
        if type(self.param_shard_offset) is not int or self.param_shard_offset < 0:
            raise ManifestError("param_shard_offset must be a non-negative integer")
        if type(self.param_shard_numel) is not int or self.param_shard_numel < 1:
            raise ManifestError("param_shard_numel must be a positive integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "path": self.path,
            "sha256": self.sha256,
            "param_shard_offset": self.param_shard_offset,
            "param_shard_numel": self.param_shard_numel,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ShardEntry:
        if not isinstance(payload, dict):
            raise ManifestError("manifest shard entry must be an object")
        required = {
            "rank",
            "path",
            "sha256",
            "param_shard_offset",
            "param_shard_numel",
        }
        if set(payload) != required:
            raise ManifestError(
                f"shard entry fields must be {sorted(required)}"
            )
        return cls(
            rank=_json_int(payload["rank"], "rank", 0),
            path=payload["path"] if isinstance(payload["path"], str) else "",
            sha256=payload["sha256"] if isinstance(payload["sha256"], str) else "",
            param_shard_offset=_json_int(
                payload["param_shard_offset"], "param_shard_offset", 0
            ),
            param_shard_numel=_json_int(
                payload["param_shard_numel"], "param_shard_numel", 1
            ),
        )


@dataclass(frozen=True)
class ShardManifest:
    world_size: int
    step: int
    param_numel: int
    shard_numel: int
    optimizer_bytes_per_rank: int
    adam_step: int
    shards: tuple[ShardEntry, ...]

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise ManifestError("world_size must be a positive integer")
        if type(self.step) is not int or self.step < 0:
            raise ManifestError("step must be a non-negative integer")
        if type(self.param_numel) is not int or self.param_numel < 1:
            raise ManifestError("param_numel must be a positive integer")
        if self.param_numel < self.world_size:
            raise ManifestError("param_numel must be at least world_size")
        expected_shard = ceil_div(self.param_numel, self.world_size)
        if self.shard_numel != expected_shard:
            raise ManifestError(
                f"shard_numel must be {expected_shard}, got {self.shard_numel}"
            )
        if self.optimizer_bytes_per_rank != 12 * self.shard_numel:
            raise ManifestError(
                "optimizer_bytes_per_rank must be 12 * shard_numel"
            )
        if type(self.adam_step) is not int or self.adam_step < 1:
            raise ManifestError("adam_step must be a positive integer")
        entries = tuple(self.shards)
        if len(entries) != self.world_size:
            raise ManifestError(
                f"manifest needs {self.world_size} shards, got {len(entries)}"
            )
        if [entry.rank for entry in entries] != list(range(self.world_size)):
            raise ManifestError("manifest ranks must be contiguous from 0")
        lengths = partition_lengths(self.param_numel, self.world_size)
        offset = 0
        for entry, expected_length in zip(entries, lengths):
            if entry.param_shard_offset != offset:
                raise ManifestError(
                    f"rank {entry.rank} offset {entry.param_shard_offset} "
                    f"does not match contiguous offset {offset}"
                )
            if entry.param_shard_numel != expected_length:
                raise ManifestError(
                    f"rank {entry.rank} numel {entry.param_shard_numel} "
                    f"does not match canonical split {expected_length}"
                )
            offset += expected_length
        if offset != self.param_numel:
            raise ManifestError("shard lengths do not cover param_numel")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "world_size": self.world_size,
            "step": self.step,
            "param_numel": self.param_numel,
            "shard_numel": self.shard_numel,
            "optimizer_bytes_per_rank": self.optimizer_bytes_per_rank,
            "adam_step": self.adam_step,
            "shards": [entry.as_dict() for entry in self.shards],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, raw: bytes) -> ShardManifest:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"invalid manifest JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ManifestError("manifest must be a JSON object")
        required = {
            "schema_version",
            "world_size",
            "step",
            "param_numel",
            "shard_numel",
            "optimizer_bytes_per_rank",
            "adam_step",
            "shards",
        }
        if set(payload) != required:
            raise ManifestError(
                f"manifest fields must be {sorted(required)}"
            )
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ManifestError("unsupported manifest schema_version")
        shards_payload = payload["shards"]
        if not isinstance(shards_payload, list):
            raise ManifestError("manifest shards must be a JSON array")
        return cls(
            world_size=_json_int(payload["world_size"], "world_size", 1),
            step=_json_int(payload["step"], "step", 0),
            param_numel=_json_int(payload["param_numel"], "param_numel", 1),
            shard_numel=_json_int(payload["shard_numel"], "shard_numel", 1),
            optimizer_bytes_per_rank=_json_int(
                payload["optimizer_bytes_per_rank"],
                "optimizer_bytes_per_rank",
                1,
            ),
            adam_step=_json_int(payload["adam_step"], "adam_step", 1),
            shards=tuple(ShardEntry.from_dict(item) for item in shards_payload),
        )


@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: ShardManifest
    ranks: tuple[RankState, ...]
    step_dir: Path

    def __post_init__(self) -> None:
        if len(self.ranks) != self.manifest.world_size:
            raise ValueError("loaded ranks do not match manifest world_size")

    @property
    def step(self) -> int:
        return self.manifest.step

    def flat_master_fp32(self) -> array:
        values = array("f")
        for rank in self.ranks:
            values.extend(rank.master_fp32)
        return values


def _step_dir(root: Path, step: int) -> Path:
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    return root / f"step-{step:08d}"


def _shard_header(
    state: RankState, step: int, offset: int
) -> dict[str, int]:
    return {
        "format_version": SHARD_FORMAT_VERSION,
        "rank": state.rank,
        "step": step,
        "param_shard_offset": offset,
        "param_shard_numel": state.numel,
        "adam_step": state.adam_step,
    }


def _serialize_shard(state: RankState, step: int, offset: int) -> bytes:
    header = json.dumps(
        _shard_header(state, step, offset), sort_keys=True
    ).encode("utf-8")
    if len(header) > MAX_SHARD_HEADER_BYTES:
        raise CheckpointError("shard header is too large")
    return (
        MAGIC
        + struct.pack("<I", len(header))
        + header
        + state.payload_bytes()
    )


def _clear_shard_files(directory: Path) -> None:
    for path in directory.glob("rank*.bin*"):
        path.unlink()


def write_rank_shard(
    directory: str | Path,
    step: int,
    state: RankState,
    total_numel: int,
    world_size: int,
) -> ShardEntry:
    """Write one rank-owned shard and fsync it; the manifest is not yet live."""

    if not isinstance(state, RankState):
        raise TypeError("state must be a RankState")
    if type(total_numel) is not int or total_numel < 1:
        raise ValueError("total_numel must be a positive integer")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    lengths = partition_lengths(total_numel, world_size)
    if state.rank >= world_size:
        raise ValueError("state rank is outside the world")
    if state.numel != lengths[state.rank]:
        raise ValueError(
            f"rank {state.rank} has {state.numel} elements, "
            f"expected {lengths[state.rank]}"
        )
    root = Path(directory)
    step_dir = _step_dir(root, step)
    if (step_dir / MANIFEST_NAME).exists():
        raise CheckpointExistsError(
            f"checkpoint already committed for step {step} at {step_dir}"
        )
    step_dir.mkdir(parents=True, exist_ok=True)
    offset = sum(lengths[: state.rank])
    final_path = step_dir / f"rank{state.rank}.bin"
    temporary = step_dir / f"rank{state.rank}.bin.tmp"
    # Only this rank's own leftovers are cleaned; another worker may own the
    # other files in the same step directory.
    if final_path.exists():
        final_path.unlink()
    if temporary.exists():
        temporary.unlink()
    blob = _serialize_shard(state, step, offset)
    with open(temporary, "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final_path)
    _fsync_dir(step_dir)
    return ShardEntry(
        rank=state.rank,
        path=final_path.name,
        sha256=hashlib.sha256(blob).hexdigest(),
        param_shard_offset=offset,
        param_shard_numel=state.numel,
    )


def commit_manifest(
    directory: str | Path,
    step: int,
    entries: Sequence[ShardEntry],
    total_numel: int,
    world_size: int,
    adam_step: int,
) -> Path:
    """Atomically publish the manifest; this is the checkpoint commit point."""

    root = Path(directory)
    step_dir = _step_dir(root, step)
    if (step_dir / MANIFEST_NAME).exists():
        raise CheckpointExistsError(
            f"checkpoint already committed for step {step} at {step_dir}"
        )
    if not step_dir.is_dir():
        raise CheckpointError(f"step directory does not exist: {step_dir}")
    if type(adam_step) is not int or adam_step < 1:
        raise ValueError("adam_step must be a positive integer")
    ordered = tuple(sorted(entries, key=lambda entry: entry.rank))
    if len(ordered) != world_size:
        raise ManifestError(
            f"manifest needs {world_size} entries, got {len(ordered)}"
        )
    lengths = partition_lengths(total_numel, world_size)
    offset = 0
    for entry, expected_length in zip(ordered, lengths):
        if entry.rank >= world_size:
            raise ManifestError("entry rank is outside the world")
        if entry.param_shard_offset != offset:
            raise ManifestError("entry offsets are not contiguous")
        if entry.param_shard_numel != expected_length:
            raise ManifestError("entry lengths do not match the canonical split")
        offset += expected_length
    manifest = ShardManifest(
        world_size=world_size,
        step=step,
        param_numel=total_numel,
        shard_numel=ceil_div(total_numel, world_size),
        optimizer_bytes_per_rank=12 * ceil_div(total_numel, world_size),
        adam_step=adam_step,
        shards=ordered,
    )
    manifest_bytes = manifest.to_json().encode("utf-8")
    temporary = step_dir / (MANIFEST_NAME + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, step_dir / MANIFEST_NAME)
    _fsync_dir(step_dir)
    return step_dir


def save_sharded(
    states: Sequence[RankState],
    directory: str | Path,
    step: int,
) -> Path:
    """Single-process convenience used by tests; rank 0 commits the manifest."""

    try:
        ordered = tuple(sorted(states, key=lambda state: state.rank))
    except (TypeError, AttributeError) as exc:
        raise ValueError("states must be a sequence of RankState") from exc
    if not ordered:
        raise ValueError("states must not be empty")
    if any(not isinstance(state, RankState) for state in ordered):
        raise ValueError("every state must be a RankState")
    world_size = len(ordered)
    if [state.rank for state in ordered] != list(range(world_size)):
        raise ValueError("state ranks must be contiguous from 0")
    total_numel = sum(state.numel for state in ordered)
    lengths = partition_lengths(total_numel, world_size)
    for state, expected in zip(ordered, lengths):
        if state.numel != expected:
            raise ValueError(
                f"rank {state.rank} has {state.numel} elements, "
                f"expected {expected}"
            )
    adam_steps = {state.adam_step for state in ordered}
    if len(adam_steps) != 1:
        raise ValueError("all ranks must have the same Adam step")

    root = Path(directory)
    step_dir = _step_dir(root, step)
    if (step_dir / MANIFEST_NAME).exists():
        raise CheckpointExistsError(
            f"checkpoint already committed for step {step} at {step_dir}"
        )
    step_dir.mkdir(parents=True, exist_ok=True)
    _clear_shard_files(step_dir)

    entries = [
        write_rank_shard(root, step, state, total_numel, world_size)
        for state in ordered
    ]
    return commit_manifest(
        root,
        step,
        entries,
        total_numel,
        world_size,
        next(iter(adam_steps)),
    )


def _read_manifest(step_dir: Path) -> ShardManifest:
    path = step_dir / MANIFEST_NAME
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CheckpointError(f"cannot stat manifest {path}: {exc}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"manifest is larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"cannot read manifest {path}: {exc}") from exc
    return ShardManifest.from_json(raw)


def _resolve_checkpoint_dir(directory: str | Path) -> Path:
    root = Path(directory)
    if not root.exists():
        raise CheckpointError(f"checkpoint path does not exist: {root}")
    if root.is_file():
        raise CheckpointError(f"checkpoint path is not a directory: {root}")
    if (root / MANIFEST_NAME).is_file():
        return root
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = STEP_DIR_RE.fullmatch(child.name)
        if match and child.is_dir() and (child / MANIFEST_NAME).is_file():
            candidates.append((int(match.group(1)), child))
    if not candidates:
        raise CheckpointError(f"no committed checkpoint under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def _header_int(header: dict[str, object], key: str, path: Path) -> int:
    value = header.get(key)
    if type(value) is not int:
        raise ShardIntegrityError(f"{path.name}: header {key} must be an integer")
    return value


def _read_shard(
    path: Path, entry: ShardEntry, manifest: ShardManifest
) -> RankState:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ShardIntegrityError(f"cannot read {path.name}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.sha256:
        raise ShardIntegrityError(
            f"{path.name}: sha256 mismatch (expected {entry.sha256}, got {digest})"
        )
    if len(raw) < 12:
        raise ShardIntegrityError(f"{path.name}: shard is truncated")
    if raw[:8] != MAGIC:
        raise ShardIntegrityError(f"{path.name}: bad shard magic")
    header_length = struct.unpack_from("<I", raw, 8)[0]
    if not 0 < header_length <= MAX_SHARD_HEADER_BYTES:
        raise ShardIntegrityError(f"{path.name}: invalid header length")
    if len(raw) < 12 + header_length:
        raise ShardIntegrityError(f"{path.name}: shard header is truncated")
    try:
        header = json.loads(raw[12 : 12 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardIntegrityError(f"{path.name}: invalid shard header") from exc
    if not isinstance(header, dict):
        raise ShardIntegrityError(f"{path.name}: shard header is not an object")
    if _header_int(header, "format_version", path) != SHARD_FORMAT_VERSION:
        raise ShardIntegrityError(f"{path.name}: unsupported shard format")
    if _header_int(header, "rank", path) != entry.rank:
        raise ShardIntegrityError(
            f"{path.name}: header rank does not match manifest rank {entry.rank}"
        )
    if _header_int(header, "step", path) != manifest.step:
        raise ShardIntegrityError(f"{path.name}: header step does not match")
    if _header_int(header, "param_shard_offset", path) != entry.param_shard_offset:
        raise ShardIntegrityError(f"{path.name}: header offset does not match")
    if _header_int(header, "param_shard_numel", path) != entry.param_shard_numel:
        raise ShardIntegrityError(f"{path.name}: header numel does not match")
    if _header_int(header, "adam_step", path) != manifest.adam_step:
        raise ShardIntegrityError(f"{path.name}: header Adam step does not match")

    payload = raw[12 + header_length :]
    expected_size = PAYLOAD_BYTES_PER_ELEMENT * entry.param_shard_numel
    if len(payload) != expected_size:
        raise ShardIntegrityError(
            f"{path.name}: payload has {len(payload)} bytes, "
            f"expected {expected_size}"
        )
    numel = entry.param_shard_numel
    param = payload[: 2 * numel]
    master_start = 2 * numel
    master = payload[master_start : master_start + 4 * numel]
    avg = payload[master_start + 4 * numel : master_start + 8 * numel]
    avg_sq = payload[master_start + 8 * numel :]
    return RankState(
        rank=entry.rank,
        param_fp16=param,
        master_fp32=decode_fp32_le(master),
        exp_avg_fp32=decode_fp32_le(avg),
        exp_avg_sq_fp32=decode_fp32_le(avg_sq),
        adam_step=manifest.adam_step,
    )


def load_checkpoint(
    directory: str | Path, expected_world_size: int
) -> LoadedCheckpoint:
    """Load the newest committed checkpoint and verify every shard."""

    if type(expected_world_size) is not int or expected_world_size < 1:
        raise ValueError("expected_world_size must be a positive integer")
    step_dir = _resolve_checkpoint_dir(directory)
    manifest = _read_manifest(step_dir)
    if manifest.world_size != expected_world_size:
        raise WorldSizeMismatchError(
            f"checkpoint world_size={manifest.world_size} does not match "
            f"expected_world_size={expected_world_size}"
        )
    expected_paths = {entry.path for entry in manifest.shards}
    on_disk = {
        path.name
        for path in step_dir.iterdir()
        if path.is_file()
        and path.name.startswith("rank")
        and path.name.endswith(".bin")
    }
    if on_disk != expected_paths:
        missing = sorted(expected_paths - on_disk)
        unexpected = sorted(on_disk - expected_paths)
        raise ShardIntegrityError(
            f"shard set mismatch in {step_dir}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    ranks = tuple(
        _read_shard(step_dir / entry.path, entry, manifest)
        for entry in manifest.shards
    )
    if len({state.adam_step for state in ranks}) != 1:
        raise ShardIntegrityError("shards disagree on the Adam step")
    return LoadedCheckpoint(manifest=manifest, ranks=ranks, step_dir=step_dir)


def verify_resume(
    directory: str | Path,
    expected: RankState,
    expected_world_size: int,
) -> LoadedCheckpoint:
    """Reload a checkpoint and compare one rank's payload byte-for-byte."""

    if not isinstance(expected, RankState):
        raise TypeError("expected must be a RankState")
    loaded = load_checkpoint(directory, expected_world_size)
    if expected.rank >= len(loaded.ranks):
        raise ResumeMismatchError(f"rank {expected.rank} is not in the checkpoint")
    actual = loaded.ranks[expected.rank]
    if actual.adam_step != expected.adam_step:
        raise ResumeMismatchError(
            f"rank {expected.rank}: Adam step {actual.adam_step} != {expected.adam_step}"
        )
    if actual.payload_bytes() != expected.payload_bytes():
        raise ResumeMismatchError(
            f"rank {expected.rank}: checkpoint payload differs from snapshot"
        )
    return loaded
