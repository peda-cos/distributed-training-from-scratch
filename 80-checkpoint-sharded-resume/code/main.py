"""Sharded checkpoints with a manifest commit point and exact resume.

Each rank owns a contiguous slice of the flat fp16 parameter vector and the
same slice of the ZeRO/Adam fp32 state.  A checkpoint is one binary shard per
rank plus a JSON manifest that records ownership, byte offsets, and the
SHA-256 of every shard.

The save path writes every shard to a temporary name and fsyncs it, writes the
manifest to a temporary name and fsyncs it, renames the shards, and renames the
manifest last.  The manifest rename is the single commit point.  Checkpoints
live in per-step directories, so a crash before the commit leaves the previous
step as the newest committed checkpoint.

The module is intentionally stdlib-only.  Python's ``array`` has no fp16
typecode, so fp16 parameters are encoded as IEEE-754 binary16 bytes with
``struct`` format ``e`` and stored little-endian.  Optimizer state and fp32
master weights use ``array('f')``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from array import array
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SHARD_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
STEP_DIR_RE = re.compile(r"^step-(\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAGIC = b"SHARD001"
FP16_BYTES = 2
FP32_BYTES = 4
PAYLOAD_BYTES_PER_ELEMENT = FP16_BYTES + 3 * FP32_BYTES
MAX_MANIFEST_BYTES = 1 << 20
MAX_SHARD_HEADER_BYTES = 4096
FP16_CHUNK = 1024
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


class CheckpointError(Exception):
    """Base class for checkpoint failures."""


class ManifestError(CheckpointError, ValueError):
    """The manifest is malformed or violates the schema."""


class WorldSizeMismatchError(CheckpointError):
    """The manifest's world size differs from the resume request."""


class ShardIntegrityError(CheckpointError):
    """A shard is missing, truncated, corrupted, or semantically mislabeled."""


class CheckpointExistsError(CheckpointError):
    """A committed checkpoint already exists for the requested step."""


def _manifest_int(value: object, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestError(f"{field} must be an integer >= {minimum}")
    return value


def encode_fp16(values: Iterable[float]) -> bytes:
    """Encode values as little-endian IEEE-754 binary16 bytes."""
    encoded = bytearray()
    for chunk in batched(values, FP16_CHUNK):
        try:
            encoded.extend(struct.pack(f"<{len(chunk)}e", *chunk))
        except (OverflowError, struct.error, TypeError) as exc:
            raise ValueError("value cannot be encoded as IEEE-754 binary16") from exc
    return bytes(encoded)


def decode_fp16(data: bytes) -> array:
    """Decode little-endian IEEE-754 binary16 bytes into fp32 values."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("fp16 data must be bytes-like")
    if len(data) % FP16_BYTES:
        raise ValueError("fp16 payload length must be a multiple of 2")
    values = array("f")
    for offset in range(0, len(data), FP16_CHUNK * FP16_BYTES):
        count = min(FP16_CHUNK, (len(data) - offset) // FP16_BYTES)
        values.extend(struct.unpack_from(f"<{count}e", data, offset))
    return values


def _encode_fp32(values: array) -> bytes:
    if not isinstance(values, array) or values.typecode != "f":
        raise ValueError("fp32 state must be an array('f')")
    if sys.byteorder == "little":
        return values.tobytes()
    copy = array("f", values)
    copy.byteswap()
    return copy.tobytes()


def _decode_fp32(data: bytes) -> array:
    if len(data) % FP32_BYTES:
        raise ValueError("fp32 payload length must be a multiple of 4")
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


@dataclass(frozen=True)
class ShardEntry:
    rank: int
    path: str
    sha256: str
    param_shard_offset: int
    param_shard_numel: int

    def to_dict(self) -> dict[str, int | str]:
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
            raise ManifestError("shard entry must be a JSON object")
        required = (
            "rank",
            "path",
            "sha256",
            "param_shard_offset",
            "param_shard_numel",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ManifestError(f"shard entry is missing fields: {missing}")

        rank = _manifest_int(payload["rank"], "rank", 0)
        offset = _manifest_int(
            payload["param_shard_offset"], "param_shard_offset", 0
        )
        numel = _manifest_int(payload["param_shard_numel"], "param_shard_numel", 1)
        path = payload["path"]
        digest = payload["sha256"]
        if not isinstance(path, str) or not path:
            raise ManifestError("path must be a non-empty string")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ManifestError("sha256 must be 64 lowercase hexadecimal characters")
        return cls(
            rank=rank,
            path=path,
            sha256=digest,
            param_shard_offset=offset,
            param_shard_numel=numel,
        )


@dataclass(frozen=True)
class ShardManifest:
    world_size: int
    step: int
    wall_clock_seconds: int
    shards: tuple[ShardEntry, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            shards = tuple(self.shards)
        except TypeError as exc:
            raise ManifestError("shards must be an iterable of ShardEntry") from exc
        object.__setattr__(self, "shards", shards)
        self.validate()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        if type(self.world_size) is not int or self.world_size < 1:
            raise ManifestError("world_size must be an integer >= 1")
        if type(self.step) is not int or self.step < 0:
            raise ManifestError("step must be an integer >= 0")
        if type(self.wall_clock_seconds) is not int or self.wall_clock_seconds < 0:
            raise ManifestError("wall_clock_seconds must be an integer >= 0")
        if len(self.shards) != self.world_size:
            raise ManifestError(
                "manifest declares "
                f"{len(self.shards)} shards for world_size={self.world_size}"
            )

        offset = 0
        for expected_rank, entry in enumerate(self.shards):
            if not isinstance(entry, ShardEntry):
                raise ManifestError(f"shards[{expected_rank}] is not a ShardEntry")
            if type(entry.rank) is not int or entry.rank != expected_rank:
                raise ManifestError(
                    f"shards[{expected_rank}] has rank {entry.rank!r}; "
                    "ranks must be contiguous from 0"
                )
            if entry.path != f"rank{entry.rank}.bin":
                raise ManifestError(
                    f"rank {entry.rank} path must be rank{entry.rank}.bin, "
                    f"got {entry.path!r}"
                )
            if not isinstance(entry.sha256, str) or not SHA256_RE.fullmatch(
                entry.sha256
            ):
                raise ManifestError(
                    f"rank {entry.rank} sha256 is not 64 lowercase hexadecimal characters"
                )
            if type(entry.param_shard_numel) is not int or entry.param_shard_numel < 1:
                raise ManifestError(
                    f"rank {entry.rank} param_shard_numel must be an integer >= 1"
                )
            if (
                type(entry.param_shard_offset) is not int
                or entry.param_shard_offset != offset
            ):
                raise ManifestError(
                    f"rank {entry.rank} param_shard_offset must be {offset}, "
                    f"got {entry.param_shard_offset!r}"
                )
            offset += entry.param_shard_numel

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "step": self.step,
            "wall_clock_seconds": self.wall_clock_seconds,
            "shards": [entry.to_dict() for entry in self.shards],
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str | bytes) -> ShardManifest:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifestError(f"invalid manifest JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ManifestError("manifest must be a JSON object")

        required = (
            "world_size",
            "step",
            "wall_clock_seconds",
            "shards",
            "schema_version",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ManifestError(f"manifest is missing fields: {missing}")
        raw_shards = payload["shards"]
        if not isinstance(raw_shards, list):
            raise ManifestError("manifest shards must be a JSON array")
        shards = tuple(ShardEntry.from_dict(item) for item in raw_shards)
        return cls(
            world_size=payload["world_size"],
            step=payload["step"],
            wall_clock_seconds=payload["wall_clock_seconds"],
            shards=shards,
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class RankState:
    """One rank's owned slice of the flat fp16 parameter and fp32 Adam state."""

    param_fp16: bytes
    master_fp32: array
    exp_avg_fp32: array
    exp_avg_sq_fp32: array
    adam_step: int = 0

    def __post_init__(self) -> None:
        if type(self.param_fp16) is not bytes:
            raise ValueError("param_fp16 must be bytes")
        for name in ("master_fp32", "exp_avg_fp32", "exp_avg_sq_fp32"):
            values = getattr(self, name)
            if not isinstance(values, array) or values.typecode != "f":
                raise ValueError(f"{name} must be an array('f')")
            if len(values) == 0:
                raise ValueError(f"{name} must not be empty")
        if not (
            len(self.master_fp32)
            == len(self.exp_avg_fp32)
            == len(self.exp_avg_sq_fp32)
        ):
            raise ValueError("fp32 state arrays must have equal lengths")
        if len(self.param_fp16) != FP16_BYTES * len(self.master_fp32):
            raise ValueError("param_fp16 length must be 2 * numel")
        if type(self.adam_step) is not int or self.adam_step < 0:
            raise ValueError("adam_step must be an integer >= 0")

    @property
    def numel(self) -> int:
        return len(self.master_fp32)


def partition_lengths(total_numel: int, world_size: int) -> list[int]:
    """Split a flat vector into contiguous, non-empty rank shards."""
    if type(total_numel) is not int or type(world_size) is not int:
        raise ValueError("total_numel and world_size must be integers")
    if total_numel < 1 or world_size < 1 or total_numel < world_size:
        raise ValueError(
            "total_numel and world_size must be >= 1 and total_numel >= world_size"
        )
    base, remainder = divmod(total_numel, world_size)
    return [
        base + 1 if rank < remainder else base for rank in range(world_size)
    ]


def _shard_header(
    rank: int, offset: int, state: RankState, step: int
) -> dict[str, int | str]:
    return {
        "adam_step": state.adam_step,
        "format": "checkpoint-shard",
        "format_version": SHARD_FORMAT_VERSION,
        "param_shard_numel": state.numel,
        "param_shard_offset": offset,
        "rank": rank,
        "step": step,
    }


def _serialize_shard(rank: int, offset: int, state: RankState, step: int) -> bytes:
    header = json.dumps(
        _shard_header(rank, offset, state, step),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > MAX_SHARD_HEADER_BYTES:
        raise CheckpointError(f"rank {rank} shard header is too large")

    payload = (
        state.param_fp16
        + _encode_fp32(state.master_fp32)
        + _encode_fp32(state.exp_avg_fp32)
        + _encode_fp32(state.exp_avg_sq_fp32)
    )
    expected_payload = PAYLOAD_BYTES_PER_ELEMENT * state.numel
    if len(payload) != expected_payload:
        raise CheckpointError(
            f"rank {rank} payload is {len(payload)} bytes, expected {expected_payload}"
        )
    return MAGIC + struct.pack("<I", len(header)) + header + payload


def _write_shard_tmp(
    step_dir: Path, rank: int, offset: int, state: RankState, step: int
) -> ShardEntry:
    blob = _serialize_shard(rank, offset, state, step)
    tmp_path = step_dir / f"rank{rank}.bin.tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    return ShardEntry(
        rank=rank,
        path=f"rank{rank}.bin",
        sha256=hashlib.sha256(blob).hexdigest(),
        param_shard_offset=offset,
        param_shard_numel=state.numel,
    )


def _write_manifest_tmp(step_dir: Path, manifest: ShardManifest) -> Path:
    tmp_path = step_dir / (MANIFEST_NAME + ".tmp")
    data = manifest.to_json().encode("utf-8")
    with open(tmp_path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return tmp_path


def _fsync_dir(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clean_uncommitted(step_dir: Path) -> None:
    for pattern in ("rank*.bin", "*.tmp"):
        for path in step_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()


def _committed_step_dirs(root: Path) -> list[tuple[int, Path]]:
    committed: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = STEP_DIR_RE.fullmatch(child.name)
        if match is None or child.is_symlink() or not child.is_dir():
            continue
        if (child / MANIFEST_NAME).is_file():
            committed.append((int(match.group(1)), child))
    committed.sort(key=lambda item: (item[0], item[1].name))
    return committed


def _prune(root: Path, keep_last: int) -> None:
    committed = _committed_step_dirs(root)
    if len(committed) <= keep_last:
        return
    for _, step_dir in committed[:-keep_last]:
        shutil.rmtree(step_dir)
    _fsync_dir(root)


def _resolve_checkpoint_dir(directory: str | Path) -> Path:
    path = Path(directory)
    if not path.exists():
        raise CheckpointError(f"checkpoint directory does not exist: {path}")
    if not path.is_dir():
        raise CheckpointError(f"checkpoint path is not a directory: {path}")
    if (path / MANIFEST_NAME).is_file():
        return path
    committed = _committed_step_dirs(path)
    if not committed:
        raise CheckpointError(f"no committed checkpoint found under {path}")
    return committed[-1][1]


def _read_manifest(checkpoint_dir: Path) -> ShardManifest:
    manifest_path = checkpoint_dir / MANIFEST_NAME
    try:
        size = manifest_path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ManifestError(f"manifest is larger than {MAX_MANIFEST_BYTES} bytes")
        raw = manifest_path.read_bytes()
    except ManifestError:
        raise
    except OSError as exc:
        raise CheckpointError(f"cannot read manifest {manifest_path}: {exc}") from exc
    return ShardManifest.from_json(raw)


def _header_int(
    header: dict[str, Any], field: str, path: Path, minimum: int = 0
) -> int:
    value = header.get(field)
    if type(value) is not int or value < minimum:
        raise ShardIntegrityError(
            f"{path.name}: header field {field!r} must be an integer >= {minimum}"
        )
    return value


def _read_shard(path: Path, entry: ShardEntry, manifest: ShardManifest) -> RankState:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ShardIntegrityError(f"cannot read shard {path.name}: {exc}") from exc
    if len(raw) < 12:
        raise ShardIntegrityError(f"{path.name}: file is truncated")
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != entry.sha256:
        raise ShardIntegrityError(
            f"{path.name}: sha256 mismatch "
            f"(expected {entry.sha256}, got {actual_digest})"
        )

    if raw[: len(MAGIC)] != MAGIC:
        raise ShardIntegrityError(f"{path.name}: invalid shard magic")
    header_length = struct.unpack_from("<I", raw, len(MAGIC))[0]
    if header_length > MAX_SHARD_HEADER_BYTES:
        raise ShardIntegrityError(
            f"{path.name}: shard header exceeds {MAX_SHARD_HEADER_BYTES} bytes"
        )
    header_end = len(MAGIC) + 4 + header_length
    if header_end > len(raw):
        raise ShardIntegrityError(f"{path.name}: shard header is truncated")
    try:
        header = json.loads(raw[len(MAGIC) + 4 : header_end])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ShardIntegrityError(f"{path.name}: invalid shard header JSON") from exc
    if not isinstance(header, dict):
        raise ShardIntegrityError(f"{path.name}: shard header must be an object")
    if header.get("format") != "checkpoint-shard":
        raise ShardIntegrityError(f"{path.name}: unknown shard format")
    if _header_int(header, "format_version", path) != SHARD_FORMAT_VERSION:
        raise ShardIntegrityError(f"{path.name}: unsupported shard format version")
    if _header_int(header, "rank", path) != entry.rank:
        raise ShardIntegrityError(
            f"{path.name}: header rank does not match manifest rank {entry.rank}"
        )
    if _header_int(header, "step", path) != manifest.step:
        raise ShardIntegrityError(
            f"{path.name}: header step does not match manifest step {manifest.step}"
        )
    if _header_int(header, "param_shard_offset", path) != entry.param_shard_offset:
        raise ShardIntegrityError(
            f"{path.name}: header offset does not match manifest offset "
            f"{entry.param_shard_offset}"
        )
    if _header_int(header, "param_shard_numel", path, minimum=1) != (
        entry.param_shard_numel
    ):
        raise ShardIntegrityError(
            f"{path.name}: header numel does not match manifest numel "
            f"{entry.param_shard_numel}"
        )
    adam_step = _header_int(header, "adam_step", path)

    numel = entry.param_shard_numel
    expected_size = header_end + PAYLOAD_BYTES_PER_ELEMENT * numel
    if len(raw) != expected_size:
        raise ShardIntegrityError(
            f"{path.name}: size {len(raw)} != expected {expected_size} "
            "(truncated or trailing bytes)"
        )
    payload = raw[header_end:]
    fp16_end = FP16_BYTES * numel
    master_end = fp16_end + FP32_BYTES * numel
    exp_avg_end = master_end + FP32_BYTES * numel
    exp_avg_sq_end = exp_avg_end + FP32_BYTES * numel
    try:
        return RankState(
            param_fp16=payload[:fp16_end],
            master_fp32=_decode_fp32(payload[fp16_end:master_end]),
            exp_avg_fp32=_decode_fp32(payload[master_end:exp_avg_end]),
            exp_avg_sq_fp32=_decode_fp32(payload[exp_avg_end:exp_avg_sq_end]),
            adam_step=adam_step,
        )
    except ValueError as exc:
        raise ShardIntegrityError(f"{path.name}: invalid shard payload: {exc}") from exc


@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: ShardManifest
    ranks: tuple[RankState, ...]

    def __post_init__(self) -> None:
        if len(self.ranks) != self.manifest.world_size:
            raise ValueError("loaded ranks do not match manifest world_size")

    @property
    def step(self) -> int:
        return self.manifest.step

    def flat_param_fp16(self) -> bytes:
        return b"".join(rank.param_fp16 for rank in self.ranks)

    def flat_master_fp32(self) -> array:
        values = array("f")
        for rank in self.ranks:
            values.extend(rank.master_fp32)
        return values

    def state_dicts(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "param_fp16": rank.param_fp16,
                "master_fp32": rank.master_fp32,
                "exp_avg_fp32": rank.exp_avg_fp32,
                "exp_avg_sq_fp32": rank.exp_avg_sq_fp32,
                "adam_step": rank.adam_step,
            }
            for rank in self.ranks
        )


def save_sharded(
    state_per_rank: Sequence[RankState],
    directory: str | Path,
    step: int,
    *,
    wall_clock_seconds: int = 0,
    keep_last: int | None = None,
) -> Path:
    """Write one shard per rank and commit the manifest last."""
    try:
        states = tuple(state_per_rank)
    except TypeError as exc:
        raise ValueError("state_per_rank must be a sequence of RankState") from exc
    if not states:
        raise ValueError("state_per_rank must not be empty")
    if any(not isinstance(state, RankState) for state in states):
        raise ValueError("every entry in state_per_rank must be a RankState")
    if type(step) is not int or step < 0:
        raise ValueError("step must be an integer >= 0")
    if type(wall_clock_seconds) is not int or wall_clock_seconds < 0:
        raise ValueError("wall_clock_seconds must be an integer >= 0")
    if keep_last is not None and (
        type(keep_last) is not int or keep_last < 1
    ):
        raise ValueError("keep_last must be None or an integer >= 1")

    total_numel = sum(state.numel for state in states)
    expected_lengths = partition_lengths(total_numel, len(states))
    actual_lengths = [state.numel for state in states]
    if actual_lengths != expected_lengths:
        raise ValueError(
            f"shard lengths {actual_lengths} do not match canonical split "
            f"{expected_lengths}"
        )
    adam_steps = {state.adam_step for state in states}
    if len(adam_steps) != 1:
        raise ValueError("all ranks must have the same Adam step")

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _fsync_dir(root)

    step_dir = root / f"step-{step:08d}"
    if (step_dir / MANIFEST_NAME).exists():
        raise CheckpointExistsError(
            f"checkpoint already committed for step {step} at {step_dir}"
        )
    if step_dir.exists() and not step_dir.is_dir():
        raise CheckpointError(f"step path is not a directory: {step_dir}")
    step_dir.mkdir(parents=True, exist_ok=True)
    _clean_uncommitted(step_dir)
    _fsync_dir(root)

    entries: list[ShardEntry] = []
    offset = 0
    with ThreadPoolExecutor(
        max_workers=len(states), thread_name_prefix="ckpt-shard"
    ) as pool:
        futures = []
        for rank, state in enumerate(states):
            futures.append(
                pool.submit(_write_shard_tmp, step_dir, rank, offset, state, step)
            )
            offset += state.numel
        entries = [future.result() for future in futures]
    entries.sort(key=lambda entry: entry.rank)

    manifest = ShardManifest(
        world_size=len(states),
        step=step,
        wall_clock_seconds=wall_clock_seconds,
        shards=tuple(entries),
    )
    manifest_tmp = _write_manifest_tmp(step_dir, manifest)

    for entry in entries:
        _replace(
            step_dir / (entry.path + ".tmp"),
            step_dir / entry.path,
        )
    _fsync_dir(step_dir)
    _replace(manifest_tmp, step_dir / MANIFEST_NAME)
    _fsync_dir(step_dir)

    if keep_last is not None:
        try:
            _prune(root, keep_last)
        except OSError as exc:
            raise CheckpointError(
                f"checkpoint {step_dir.name} committed but rotation failed: {exc}"
            ) from exc
    return step_dir


_replace = os.replace


def load_sharded(directory: str | Path, expected_world_size: int) -> LoadedCheckpoint:
    """Load and fully verify the newest committed checkpoint."""
    if type(expected_world_size) is not int or expected_world_size < 1:
        raise ValueError("expected_world_size must be an integer >= 1")
    checkpoint_dir = _resolve_checkpoint_dir(directory)
    manifest = _read_manifest(checkpoint_dir)
    if manifest.world_size != expected_world_size:
        raise WorldSizeMismatchError(
            f"checkpoint world_size={manifest.world_size} does not match "
            f"expected_world_size={expected_world_size}"
        )

    expected_paths = {entry.path for entry in manifest.shards}
    on_disk = {
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file()
        and path.name.startswith("rank")
        and path.name.endswith(".bin")
    }
    if on_disk != expected_paths:
        missing = sorted(expected_paths - on_disk)
        unexpected = sorted(on_disk - expected_paths)
        raise ShardIntegrityError(
            f"shard set mismatch in {checkpoint_dir}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    ranks: list[RankState] = []
    for entry in manifest.shards:
        ranks.append(
            _read_shard(checkpoint_dir / entry.path, entry, manifest)
        )
    adam_steps = {rank.adam_step for rank in ranks}
    if len(adam_steps) != 1:
        raise ShardIntegrityError(
            f"shards disagree on Adam step: {sorted(adam_steps)}"
        )
    return LoadedCheckpoint(manifest=manifest, ranks=tuple(ranks))


@dataclass(frozen=True)
class TrainingState:
    """A tiny full-model state used to exercise the resume contract locally."""

    param_fp16: bytes
    master_fp32: array
    exp_avg_fp32: array
    exp_avg_sq_fp32: array
    step: int


def make_training_state(total_numel: int) -> TrainingState:
    if type(total_numel) is not int or total_numel < 1:
        raise ValueError("total_numel must be an integer >= 1")
    master = array(
        "f",
        (
            0.02 * math.sin(0.17 * (index + 1))
            + 0.003 * math.cos(0.031 * index)
            for index in range(total_numel)
        ),
    )
    exp_avg = array(
        "f",
        (
            0.0003 * math.sin(0.07 * index + 0.2)
            for index in range(total_numel)
        ),
    )
    exp_avg_sq = array(
        "f",
        (
            0.0004 + 0.0001 * abs(math.cos(0.05 * index))
            for index in range(total_numel)
        ),
    )
    return TrainingState(
        param_fp16=encode_fp16(master),
        master_fp32=master,
        exp_avg_fp32=exp_avg,
        exp_avg_sq_fp32=exp_avg_sq,
        step=0,
    )


def train_steps(state: TrainingState, steps: int, *, lr: float = 0.01) -> TrainingState:
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be an integer >= 0")
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a positive finite number")
    master = array("f", state.master_fp32)
    exp_avg = array("f", state.exp_avg_fp32)
    exp_avg_sq = array("f", state.exp_avg_sq_fp32)
    step = state.step
    for _ in range(steps):
        step += 1
        bias_correction1 = 1.0 - ADAM_BETA1**step
        bias_correction2 = 1.0 - ADAM_BETA2**step
        for index in range(len(master)):
            gradient = 0.02 * math.sin(
                0.013 * (index + 1) + 0.29 * step
            ) + 0.0005 * math.cos(0.007 * index + 0.11 * step)
            first_moment = ADAM_BETA1 * exp_avg[index] + (1.0 - ADAM_BETA1) * gradient
            second_moment = (
                ADAM_BETA2 * exp_avg_sq[index]
                + (1.0 - ADAM_BETA2) * gradient * gradient
            )
            exp_avg[index] = first_moment
            exp_avg_sq[index] = second_moment
            master[index] = master[index] - lr * (
                first_moment / bias_correction1
            ) / (math.sqrt(second_moment / bias_correction2) + ADAM_EPS)
    return TrainingState(
        param_fp16=encode_fp16(master),
        master_fp32=master,
        exp_avg_fp32=exp_avg,
        exp_avg_sq_fp32=exp_avg_sq,
        step=step,
    )


def shard_training_state(state: TrainingState, world_size: int) -> list[RankState]:
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be an integer >= 1")
    lengths = partition_lengths(len(state.master_fp32), world_size)
    shards: list[RankState] = []
    offset = 0
    for length in lengths:
        shards.append(
            RankState(
                param_fp16=state.param_fp16[
                    FP16_BYTES * offset : FP16_BYTES * (offset + length)
                ],
                master_fp32=array(
                    "f", state.master_fp32[offset : offset + length]
                ),
                exp_avg_fp32=array(
                    "f", state.exp_avg_fp32[offset : offset + length]
                ),
                exp_avg_sq_fp32=array(
                    "f", state.exp_avg_sq_fp32[offset : offset + length]
                ),
                adam_step=state.step,
            )
        )
        offset += length
    return shards


def unshard_states(states: Sequence[RankState]) -> TrainingState:
    state_list = tuple(states)
    if not state_list:
        raise ValueError("states must not be empty")
    adam_steps = {state.adam_step for state in state_list}
    if len(adam_steps) != 1:
        raise ValueError("all ranks must have the same Adam step")
    param_fp16 = b"".join(state.param_fp16 for state in state_list)
    master = array("f")
    exp_avg = array("f")
    exp_avg_sq = array("f")
    for state in state_list:
        master.extend(state.master_fp32)
        exp_avg.extend(state.exp_avg_fp32)
        exp_avg_sq.extend(state.exp_avg_sq_fp32)
    return TrainingState(
        param_fp16=param_fp16,
        master_fp32=master,
        exp_avg_fp32=exp_avg,
        exp_avg_sq_fp32=exp_avg_sq,
        step=next(iter(adam_steps)),
    )


def run_demo(args: argparse.Namespace) -> int:
    if type(args.world_size) is not int or args.world_size < 1:
        raise ValueError("world_size must be an integer >= 1")
    if type(args.total_numel) is not int or args.total_numel < args.world_size:
        raise ValueError("total_numel must be an integer >= world_size")
    if type(args.checkpoint_step) is not int or args.checkpoint_step < 0:
        raise ValueError("checkpoint_step must be an integer >= 0")
    if type(args.resume_steps) is not int or args.resume_steps < 0:
        raise ValueError("resume_steps must be an integer >= 0")

    root = (
        Path(args.dir)
        if args.dir is not None
        else Path(tempfile.mkdtemp(prefix="sharded-checkpoint-demo-"))
    )
    started = time.perf_counter()
    initial = make_training_state(args.total_numel)
    before_save = train_steps(initial, args.checkpoint_step)
    states = shard_training_state(before_save, args.world_size)
    wall_clock_seconds = int(time.perf_counter() - started)
    step_dir = save_sharded(
        states,
        root,
        args.checkpoint_step,
        wall_clock_seconds=wall_clock_seconds,
        keep_last=3,
    )
    loaded = load_sharded(root, args.world_size)

    if loaded.step != args.checkpoint_step:
        raise CheckpointError(
            f"loaded step {loaded.step} != requested {args.checkpoint_step}"
        )
    if loaded.flat_param_fp16() != before_save.param_fp16:
        raise CheckpointError("flattened fp16 parameters changed across the round trip")
    if loaded.flat_master_fp32().tobytes() != before_save.master_fp32.tobytes():
        raise CheckpointError("flattened fp32 master weights changed across the round trip")
    for rank, (saved, restored) in enumerate(zip(states, loaded.ranks)):
        if (
            restored.param_fp16 != saved.param_fp16
            or restored.master_fp32.tobytes() != saved.master_fp32.tobytes()
            or restored.exp_avg_fp32.tobytes() != saved.exp_avg_fp32.tobytes()
            or restored.exp_avg_sq_fp32.tobytes()
            != saved.exp_avg_sq_fp32.tobytes()
            or restored.adam_step != saved.adam_step
        ):
            raise CheckpointError(f"rank {rank} state changed across the round trip")

    reference = train_steps(
        initial, args.checkpoint_step + args.resume_steps
    )
    resumed = train_steps(
        unshard_states(loaded.ranks), args.resume_steps
    )
    max_delta = max(
        (
            abs(left - right)
            for left, right in zip(
                reference.master_fp32, resumed.master_fp32
            )
        ),
        default=0.0,
    )
    if (
        reference.param_fp16 != resumed.param_fp16
        or reference.master_fp32.tobytes() != resumed.master_fp32.tobytes()
        or reference.exp_avg_fp32.tobytes() != resumed.exp_avg_fp32.tobytes()
        or reference.exp_avg_sq_fp32.tobytes()
        != resumed.exp_avg_sq_fp32.tobytes()
        or reference.step != resumed.step
        or max_delta != 0.0
    ):
        raise CheckpointError(
            f"resumed state differs from the uninterrupted run (max delta {max_delta})"
        )

    print(f"checkpoint root: {root}")
    print(f"step directory: {step_dir}")
    print("rank  path        numel  file_bytes  sha256[:12]")
    for entry in loaded.manifest.shards:
        shard_bytes = (step_dir / entry.path).stat().st_size
        print(
            f"{entry.rank:>4}  {entry.path:<10} {entry.param_shard_numel:>5} "
            f"{shard_bytes:>11}  {entry.sha256[:12]}"
        )
    print(
        f"manifest: {step_dir / MANIFEST_NAME} "
        f"(world_size={loaded.manifest.world_size}, step={loaded.manifest.step}, "
        f"shards={len(loaded.manifest.shards)}, "
        f"schema_version={loaded.manifest.schema_version})"
    )
    print(f"reloaded {len(loaded.ranks)} ranks; "
          f"flat fp16 params: {len(loaded.flat_param_fp16())} bytes")
    print("byte-equal shards: PASS")
    print(f"resume exact: PASS (max |delta|={max_delta})")
    print("ALL CHECKS PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save and resume a sharded checkpoint with an atomic manifest commit."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="checkpoint root (default: a fresh temporary directory)",
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--total-numel", type=int, default=1031)
    parser.add_argument("--checkpoint-step", type=int, default=3)
    parser.add_argument("--resume-steps", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_demo(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
