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
