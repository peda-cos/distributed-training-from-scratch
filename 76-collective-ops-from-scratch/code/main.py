"""Collective operations built on a multiprocessing.Queue mailbox mesh.

The four primitives -- ring allreduce, broadcast, allgather, reduce_scatter --
move real torch tensors between real processes. Every primitive is checked in
the test suite against torch.distributed with the gloo backend.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import platform
import queue
import shutil
import tempfile
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Sequence

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy:.*")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

Tensor = torch.Tensor
# Tensors cross queues as bytes, not as shared-memory handles. A sender may
# exit before a peer unpickles its last message; torch's FD reduction would
# then fail with ConnectionResetError.
Message = tuple[int, str, bytes]


def _encode_tensor(tensor: Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def _decode_tensor(blob: bytes) -> Tensor:
    if not isinstance(blob, (bytes, bytearray)):
        raise ProtocolError(f"expected tensor bytes, got {type(blob)!r}")
    try:
        loaded = torch.load(
            io.BytesIO(bytes(blob)), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ProtocolError(f"could not decode tensor payload: {exc}") from exc
    if not isinstance(loaded, torch.Tensor):
        raise ProtocolError(f"payload decoded to {type(loaded)!r}, not a tensor")
    return loaded


class CommError(RuntimeError):
    """Base class for communication failures."""


class CommTimeout(CommError):
    """Raised when a rank waits past its receive deadline."""

    def __init__(self, message: str, *, rank: int, src: int) -> None:
        super().__init__(message)
        self.rank = rank
        self.src = src


class ProtocolError(CommError):
    """Raised when a peer violates the expected message protocol."""


@dataclass
class MeshStats:
    rank: int
    bytes_sent: int = 0
    bytes_recv: int = 0
    messages_sent: int = 0
    messages_recv: int = 0
    rounds: int = 0
    targets: set[tuple[int, int]] = field(default_factory=set)


class Mesh:
    """Point-to-point transport over one multiprocessing.Queue per destination.

    A rank owns its inbox and writes to every other rank's inbox. Messages carry
    their source so ``recv(src)`` can demultiplex when a faster rank sends a
    later message before the expected one.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        inbox: Any,
        outboxes: dict[int, Any],
        *,
        recv_timeout_s: float = 30.0,
    ) -> None:
        _validate_world(rank, world_size)
        if inbox is None:
            raise ValueError("inbox queue is required")
        if recv_timeout_s <= 0:
            raise ValueError("recv_timeout_s must be positive")
        expected = set(range(world_size)) - {rank}
        if set(outboxes) != expected:
            raise ValueError(
                f"outboxes must target every rank except {rank}; "
                f"got {sorted(outboxes)}"
            )
        for dst in outboxes:
            if not isinstance(dst, int):
                raise TypeError(f"destination must be int, got {type(dst)!r}")
        self.rank = rank
        self.world_size = world_size
        self.recv_timeout_s = float(recv_timeout_s)
        self._inbox = inbox
        self._outboxes = dict(outboxes)
        self._pending: dict[int, deque[Message]] = {
            src: deque() for src in expected
        }
        self.stats = MeshStats(rank=rank)

    def send(self, dst: int, tensor: Tensor, tag: str = "") -> None:
        if not isinstance(tag, str):
            raise TypeError("tag must be a string")
        if dst not in self._outboxes:
            raise ValueError(f"rank {self.rank} has no channel to rank {dst}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        self._outboxes[dst].put((self.rank, tag, _encode_tensor(tensor)))
        self.stats.bytes_sent += tensor.numel() * tensor.element_size()
        self.stats.messages_sent += 1
        self.stats.targets.add((self.rank, dst))

    def recv(self, src: int, tag: str | None = None) -> Tensor:
        if not isinstance(src, int) or not 0 <= src < self.world_size:
            raise ValueError(f"invalid source rank {src!r}")
        if src == self.rank:
            raise ValueError("a rank cannot receive from itself")
        if tag is not None and not isinstance(tag, str):
            raise TypeError("tag must be a string or None")

        deadline = time.monotonic() + self.recv_timeout_s
        while True:
            buffered = self._pending[src]
            if buffered:
                return self._accept(buffered.popleft(), src, tag)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout(src)
            try:
                message = self._inbox.get(timeout=remaining)
            except queue.Empty:
                raise self._timeout(src) from None

            if not isinstance(message, tuple) or len(message) != 3:
                raise ProtocolError(
                    f"rank {self.rank} got malformed message from queue: {message!r}"
                )
            msg_src, msg_tag, msg_blob = message
            if not isinstance(msg_src, int) or not 0 <= msg_src < self.world_size:
                raise ProtocolError(
                    f"rank {self.rank} got invalid source {msg_src!r}"
                )
            if msg_src == self.rank:
                raise ProtocolError(f"rank {self.rank} received its own message")
            if not isinstance(msg_tag, str) or not isinstance(
                msg_blob, (bytes, bytearray)
            ):
                raise ProtocolError(
                    f"rank {self.rank} got malformed payload from rank {msg_src}"
                )
            if msg_src != src:
                self._pending[msg_src].append(message)
                continue
            return self._accept(message, src, tag)

    def _accept(self, message: Message, src: int, tag: str | None) -> Tensor:
        msg_src, msg_tag, msg_blob = message
        if msg_src != src:
            raise ProtocolError(
                f"rank {self.rank} expected rank {src}, got rank {msg_src}"
            )
        if tag is not None and msg_tag != tag:
            raise ProtocolError(
                f"rank {self.rank} expected tag {tag!r} from rank {src}, "
                f"got {msg_tag!r}"
            )
        tensor = _decode_tensor(msg_blob)
        self.stats.bytes_recv += tensor.numel() * tensor.element_size()
        self.stats.messages_recv += 1
        return tensor

    def _timeout(self, src: int) -> CommTimeout:
        return CommTimeout(
            f"rank {self.rank} timed out after {self.recv_timeout_s:.3f}s "
            f"waiting for rank {src}",
            rank=self.rank,
            src=src,
        )


@dataclass
class RunResult:
    outputs: list[Tensor]
    stats: list[MeshStats]
    elapsed_s: float


def _validate_world(rank: int, world_size: int) -> None:
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError("world_size must be an int")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an int")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")


def _validate_tensor(tensor: Tensor, name: str = "tensor") -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(tensor.shape)}")


def _validate_src(src: int, world_size: int) -> None:
    if isinstance(src, bool) or not isinstance(src, int):
        raise TypeError("src must be an int")
    if not 0 <= src < world_size:
        raise ValueError(f"src must be in [0, {world_size}), got {src}")


def _validate_mesh(mesh: Mesh, rank: int, world_size: int) -> None:
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh")
    if mesh.rank != rank or mesh.world_size != world_size:
        raise ValueError(
            f"mesh is rank {mesh.rank}/{mesh.world_size}, "
            f"not rank {rank}/{world_size}"
        )


def _chunk_sizes(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _chunk_bounds(total: int, parts: int) -> list[int]:
    bounds = [0]
    for size in _chunk_sizes(total, parts):
        bounds.append(bounds[-1] + size)
    return bounds


def _reduce_scatter_steps(
    mesh: Mesh,
    rank: int,
    world_size: int,
    buf: Tensor,
    bounds: list[int],
    tag_prefix: str,
) -> None:
    if world_size == 1:
        return
    nxt = (rank + 1) % world_size
    prev = (rank - 1) % world_size
    for step in range(world_size - 1):
        send_i = (rank - 1 - step) % world_size
        recv_i = (rank - 2 - step) % world_size
        send_lo, send_hi = bounds[send_i], bounds[send_i + 1]
        recv_lo, recv_hi = bounds[recv_i], bounds[recv_i + 1]
        mesh.send(
            nxt,
            buf[send_lo:send_hi].clone(),
            tag=f"{tag_prefix}:{step}:{send_i}",
        )
        got = mesh.recv(prev, tag=f"{tag_prefix}:{step}:{recv_i}")
        if got.shape != buf[recv_lo:recv_hi].shape:
            raise ProtocolError(
                f"rank {rank} got chunk {tuple(got.shape)}, "
                f"expected {tuple(buf[recv_lo:recv_hi].shape)}"
            )
        buf[recv_lo:recv_hi].add_(got)
        mesh.stats.rounds += 1


def _allgather_steps(
    mesh: Mesh,
    rank: int,
    world_size: int,
    buf: Tensor,
    bounds: list[int],
    tag_prefix: str,
) -> None:
    if world_size == 1:
        return
    nxt = (rank + 1) % world_size
    prev = (rank - 1) % world_size
    for step in range(world_size - 1):
        send_i = (rank - step) % world_size
        recv_i = (rank - step - 1) % world_size
        send_lo, send_hi = bounds[send_i], bounds[send_i + 1]
        recv_lo, recv_hi = bounds[recv_i], bounds[recv_i + 1]
        mesh.send(
            nxt,
            buf[send_lo:send_hi].clone(),
            tag=f"{tag_prefix}:{step}:{send_i}",
        )
        got = mesh.recv(prev, tag=f"{tag_prefix}:{step}:{recv_i}")
        if got.shape != buf[recv_lo:recv_hi].shape:
            raise ProtocolError(
                f"rank {rank} got chunk {tuple(got.shape)}, "
                f"expected {tuple(buf[recv_lo:recv_hi].shape)}"
            )
        buf[recv_lo:recv_hi] = got
        mesh.stats.rounds += 1


def ring_allreduce(mesh: Mesh, rank: int, world_size: int, tensor: Tensor) -> Tensor:
    """Two-pass ring allreduce. Every rank returns the full sum."""
    _validate_mesh(mesh, rank, world_size)
    _validate_tensor(tensor)
    buf = tensor.clone()
    bounds = _chunk_bounds(tensor.numel(), world_size)
    _reduce_scatter_steps(mesh, rank, world_size, buf, bounds, "ar-rs")
    _allgather_steps(mesh, rank, world_size, buf, bounds, "ar-ag")
    return buf


def reduce_scatter(mesh: Mesh, rank: int, world_size: int, tensor: Tensor) -> Tensor:
    """Reduce then scatter. Rank r returns the fully reduced chunk r."""
    _validate_mesh(mesh, rank, world_size)
    _validate_tensor(tensor)
    buf = tensor.clone()
    bounds = _chunk_bounds(tensor.numel(), world_size)
    _reduce_scatter_steps(mesh, rank, world_size, buf, bounds, "rs")
    return buf[bounds[rank]:bounds[rank + 1]].clone()


def allgather(mesh: Mesh, rank: int, world_size: int, tensor: Tensor) -> Tensor:
    """Rotate each rank's shard until every rank holds all shards."""
    _validate_mesh(mesh, rank, world_size)
    _validate_tensor(tensor)
    shards: list[Tensor | None] = [None] * world_size
    shards[rank] = tensor
    nxt = (rank + 1) % world_size
    prev = (rank - 1) % world_size
    for step in range(world_size - 1):
        send_i = (rank - step) % world_size
        recv_i = (rank - step - 1) % world_size
        shard = shards[send_i]
        if shard is None:
            raise ProtocolError(f"rank {rank} has no shard {send_i} to send")
        mesh.send(nxt, shard, tag=f"ag:{step}:{send_i}")
        shards[recv_i] = mesh.recv(prev, tag=f"ag:{step}:{recv_i}")
        mesh.stats.rounds += 1
    if any(shard is None for shard in shards):
        raise ProtocolError(f"rank {rank} finished allgather with a missing shard")
    return torch.cat([shard for shard in shards if shard is not None])


def broadcast(
    mesh: Mesh,
    rank: int,
    world_size: int,
    tensor: Tensor,
    src: int = 0,
) -> Tensor:
    """Binomial-tree broadcast from src with ceil(log2(N)) rounds."""
    _validate_mesh(mesh, rank, world_size)
    _validate_tensor(tensor)
    _validate_src(src, world_size)
    out = tensor.clone()
    offset = (rank - src) % world_size
    mask = 1
    while mask < world_size:
        if mask <= offset < 2 * mask:
            got = mesh.recv((rank - mask) % world_size, tag=f"bc:{mask}")
            if got.shape != out.shape:
                raise ProtocolError(
                    f"rank {rank} broadcast shape {tuple(got.shape)}, "
                    f"expected {tuple(out.shape)}"
                )
            out = got
        elif offset < mask and offset + mask < world_size:
            mesh.send((rank + mask) % world_size, out, tag=f"bc:{mask}")
        mesh.stats.rounds += 1
        mask <<= 1
    return out


_OP_FUNCS = {
    "allreduce": ring_allreduce,
    "broadcast": broadcast,
    "allgather": allgather,
    "reduce_scatter": reduce_scatter,
}


def _validate_inputs(inputs: Sequence[Tensor], world_size: int) -> None:
    if isinstance(inputs, torch.Tensor):
        raise ValueError("inputs must be a sequence with one tensor per rank")
    if len(inputs) != world_size:
        raise ValueError(
            f"expected {world_size} input tensors, got {len(inputs)}"
        )
    for index, tensor in enumerate(inputs):
        _validate_tensor(tensor, f"inputs[{index}]")


def _put_result(result_q: Any, payload: dict[str, Any]) -> None:
    try:
        result_q.put(payload)
    finally:
        try:
            result_q.close()
            result_q.join_thread()
        except (OSError, ValueError):
            pass


def _collective_worker(
    rank: int,
    world_size: int,
    inbox: Any,
    outboxes: dict[int, Any],
    op: str,
    tensor: Tensor,
    src: int,
    result_q: Any,
    recv_timeout_s: float,
) -> None:
    torch.set_num_threads(1)
    try:
        mesh = Mesh(
            rank, world_size, inbox, outboxes, recv_timeout_s=recv_timeout_s
        )
        if op == "broadcast":
            output = broadcast(mesh, rank, world_size, tensor, src=src)
        else:
            output = _OP_FUNCS[op](mesh, rank, world_size, tensor)
        payload: dict[str, Any] = {
            "rank": rank,
            "ok": True,
            "output": _encode_tensor(output),
            "stats": mesh.stats,
        }
    except BaseException:
        payload = {"rank": rank, "ok": False, "error": traceback.format_exc()}
    _put_result(result_q, payload)


def _terminate_processes(processes: Sequence[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _dispose_queues(queues: Sequence[Any], *, abort: bool) -> None:
    for q in queues:
        if abort:
            try:
                q.cancel_join_thread()
            except (OSError, ValueError):
                pass
        try:
            q.close()
        except (OSError, ValueError):
            pass
        if not abort:
            try:
                q.join_thread()
            except (OSError, ValueError):
                pass


def _collect_mesh_results(
    result_q: Any, world_size: int, deadline: float, op: str
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    while len(results) < world_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CommTimeout(
                f"{op} did not finish within the process deadline",
                rank=-1,
                src=-1,
            )
        try:
            payload = result_q.get(timeout=remaining)
        except queue.Empty:
            raise CommTimeout(
                f"{op} did not finish within the process deadline",
                rank=-1,
                src=-1,
            ) from None
        rank = payload["rank"]
        if not payload["ok"]:
            raise RuntimeError(f"rank {rank} failed:\n{payload['error']}")
        payload["output"] = _decode_tensor(payload["output"])
        results[rank] = payload
    return results


def run_collective(
    op: str,
    world_size: int,
    inputs: Sequence[Tensor],
    *,
    src: int = 0,
    recv_timeout_s: float = 30.0,
    timeout_s: float = 120.0,
) -> RunResult:
    """Run one collective on a freshly spawned queue mesh."""
    if op not in _OP_FUNCS:
        raise ValueError(f"unknown collective {op!r}")
    _validate_world(0, world_size)
    _validate_inputs(inputs, world_size)
    if op == "broadcast":
        _validate_src(src, world_size)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    ctx = mp.get_context("spawn")
    inboxes = [ctx.Queue() for _ in range(world_size)]
    result_q = ctx.Queue()
    all_queues = inboxes + [result_q]
    processes = [
        ctx.Process(
            target=_collective_worker,
            args=(
                rank,
                world_size,
                inboxes[rank],
                {
                    dst: inboxes[dst]
                    for dst in range(world_size)
                    if dst != rank
                },
                op,
                inputs[rank],
                src,
                result_q,
                recv_timeout_s,
            ),
            name=f"collective-rank-{rank}",
        )
        for rank in range(world_size)
    ]

    started_at = time.monotonic()
    deadline = started_at + timeout_s
    try:
        for process in processes:
            process.start()
        results = _collect_mesh_results(result_q, world_size, deadline, op)
        for process in processes:
            process.join(timeout=10)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            raise RuntimeError(
                f"{op} ranks did not exit: "
                f"{[process.name for process in alive]}"
            )
        failed = [
            (process.name, process.exitcode)
            for process in processes
            if process.exitcode != 0
        ]
        if failed:
            raise RuntimeError(f"{op} workers exited with errors: {failed}")
    except BaseException:
        _terminate_processes(processes)
        _dispose_queues(all_queues, abort=True)
        raise
    else:
        _dispose_queues(all_queues, abort=False)
        return RunResult(
            outputs=[results[rank]["output"] for rank in range(world_size)],
            stats=[results[rank]["stats"] for rank in range(world_size)],
            elapsed_s=time.monotonic() - started_at,
        )


def _gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
    op: str,
    payload: Tensor,
    src: int,
    result_q: Any,
) -> None:
    torch.set_num_threads(1)
    initialized = False
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        initialized = True
        if op == "allreduce":
            output = payload.clone()
            dist.all_reduce(output)
        elif op == "broadcast":
            output = payload.clone()
            dist.broadcast(output, src)
        elif op == "allgather":
            gathered = [torch.empty_like(payload) for _ in range(world_size)]
            dist.all_gather(gathered, payload)
            output = torch.cat(gathered)
        elif op == "reduce_scatter":
            output = torch.empty(
                payload.numel() // world_size,
                dtype=payload.dtype,
                device=payload.device,
            )
            dist.reduce_scatter_tensor(output, payload)
        else:
            raise ValueError(f"unknown gloo collective {op!r}")
        result = {"rank": rank, "ok": True, "output": _encode_tensor(output)}
    except BaseException:
        result = {"rank": rank, "ok": False, "error": traceback.format_exc()}
    finally:
        if initialized:
            try:
                dist.destroy_process_group()
            except BaseException:
                pass
    _put_result(result_q, result)


def _normalise_gloo_inputs(
    op: str, world_size: int, tensor: Tensor | Sequence[Tensor]
) -> list[Tensor]:
    if isinstance(tensor, torch.Tensor):
        if op == "allgather":
            raise ValueError("allgather requires one shard per rank")
        payloads = [tensor] * world_size
    else:
        payloads = list(tensor)
        if len(payloads) != world_size:
            raise ValueError(
                f"expected {world_size} tensors, got {len(payloads)}"
            )
    for index, payload in enumerate(payloads):
        _validate_tensor(payload, f"tensor[{index}]")
    shapes = {tuple(payload.shape) for payload in payloads}
    if len(shapes) != 1:
        raise ValueError(f"gloo requires equal shapes, got {sorted(shapes)}")
    if op == "reduce_scatter" and payloads[0].numel() % world_size != 0:
        raise ValueError("reduce_scatter input must be divisible by world_size")
    return payloads


def _gloo_reference(
    op: str,
    world_size: int,
    tensor: Tensor | Sequence[Tensor],
    *,
    src: int = 0,
) -> list[Tensor]:
    """Run the same collective through torch.distributed with gloo."""
    if op not in _OP_FUNCS:
        raise ValueError(f"unknown collective {op!r}")
    _validate_world(0, world_size)
    if op == "broadcast":
        _validate_src(src, world_size)
    payloads = _normalise_gloo_inputs(op, world_size, tensor)
    if op == "reduce_scatter" and payloads[0].numel() % world_size != 0:
        raise ValueError("reduce_scatter input must be divisible by world_size")

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    tmpdir = tempfile.mkdtemp(prefix="gloo-")
    init_file = os.path.join(tmpdir, "store")
    processes = [
        ctx.Process(
            target=_gloo_worker,
            args=(rank, world_size, init_file, op, payloads[rank], src, result_q),
            name=f"gloo-rank-{rank}",
        )
        for rank in range(world_size)
    ]
    deadline = time.monotonic() + 120.0
    try:
        for process in processes:
            process.start()
        results: dict[int, Tensor] = {}
        while len(results) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"gloo {op} timed out")
            try:
                payload = result_q.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError(f"gloo {op} timed out") from None
            if not payload["ok"]:
                raise RuntimeError(
                    f"gloo rank {payload['rank']} failed:\n{payload['error']}"
                )
            results[payload["rank"]] = _decode_tensor(payload["output"])
        for process in processes:
            process.join(timeout=10)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            raise RuntimeError(
                f"gloo ranks did not exit: {[p.name for p in alive]}"
            )
        failed = [
            (process.name, process.exitcode)
            for process in processes
            if process.exitcode != 0
        ]
        if failed:
            raise RuntimeError(f"gloo workers exited with errors: {failed}")
    except BaseException:
        _terminate_processes(processes)
        _dispose_queues([result_q], abort=True)
        raise
    else:
        _dispose_queues([result_q], abort=False)
        return [results[rank] for rank in range(world_size)]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _max_abs_diff(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape:
        raise AssertionError(f"shape mismatch: {left.shape} vs {right.shape}")
    if left.numel() == 0:
        return 0.0
    return float((left.to(torch.float64) - right.to(torch.float64)).abs().max())


def _fmt_bytes(value: int) -> str:
    return f"{value:,}"


def _verification_row(
    op: str,
    world_size: int,
    inputs: list[Tensor],
    *,
    src: int = 0,
    exact: bool,
) -> tuple[str, bool, RunResult]:
    mesh_result = run_collective(
        op, world_size, inputs, src=src, timeout_s=180.0
    )
    reference = _gloo_reference(op, world_size, inputs, src=src)
    diff = max(
        _max_abs_diff(mesh_out, gloo_out)
        for mesh_out, gloo_out in zip(mesh_result.outputs, reference)
    )
    if exact:
        ok = diff == 0.0 and all(
            torch.equal(mesh_out, gloo_out)
            for mesh_out, gloo_out in zip(mesh_result.outputs, reference)
        )
    else:
        ok = diff <= 1e-5
    steps = mesh_result.stats[0].rounds
    row = (
        f"{op:<15} {world_size:>2}  {diff:>11.3e}  "
        f"{'yes' if diff == 0.0 else 'no':>5}  {steps:>5}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return row, ok, mesh_result


def main() -> int:
    torch.manual_seed(0)
    world_size = 4
    total = 4096
    gen = torch.Generator().manual_seed(0)
    print("Collective Ops From Scratch: queue mesh vs torch.distributed gloo")
    print(
        f"torch={torch.__version__} gloo={dist.is_gloo_available()} "
        f"python={platform.python_version()} cpus={os.cpu_count()} "
        f"platform={platform.platform()}"
    )
    print()
    print(
        f"{'primitive':<15} {'N':>2}  {'max|mesh-gloo|':>13}  "
        f"{'exact':>5}  {'steps':>5}  result"
    )
    print("-" * 65)

    float_inputs = [
        torch.randn(total, generator=gen) + rank for rank in range(world_size)
    ]
    int_inputs = [
        torch.arange(total, dtype=torch.int64) + rank
        for rank in range(world_size)
    ]
    shard_len = total // world_size
    shard_inputs = [
        torch.arange(shard_len, dtype=torch.int64) + rank * shard_len
        for rank in range(world_size)
    ]
    broadcast_inputs = [
        torch.full((total,), rank, dtype=torch.int64)
        for rank in range(world_size)
    ]

    all_ok = True
    broadcast_result: RunResult | None = None
    rows = [
        ("allreduce", float_inputs, 0, False),
        ("allreduce", int_inputs, 0, True),
        ("broadcast", broadcast_inputs, 1, True),
        ("allgather", shard_inputs, 0, True),
        ("reduce_scatter", int_inputs, 0, True),
    ]
    for op, inputs, src, exact in rows:
        row, ok, result = _verification_row(
            op, world_size, inputs, src=src, exact=exact
        )
        print(row)
        all_ok = all_ok and ok
        if op == "broadcast":
            broadcast_result = result

    print()
    print("Allreduce byte accounting (float32, T=2048 elements)")
    print(
        f"{'N':>2}  {'T bytes':>9}  {'per-rank bytes':>14}  "
        f"{'formula':>9}  {'bytes/T':>8}  {'steps':>5}  result"
    )
    print("-" * 76)
    scale_total = 2048
    for scale_world in (2, 4, 8):
        base = torch.arange(scale_total, dtype=torch.float32)
        scale_inputs = [base + rank for rank in range(scale_world)]
        result = run_collective(
            "allreduce", scale_world, scale_inputs, timeout_s=180.0
        )
        per_rank = {stat.bytes_sent for stat in result.stats}
        t_bytes = scale_total * 4
        formula = 2 * t_bytes * (scale_world - 1) // scale_world
        steps = result.stats[0].rounds
        ok = (
            per_rank == {formula}
            and steps == 2 * (scale_world - 1)
        )
        all_ok = all_ok and ok
        measured = next(iter(per_rank)) if len(per_rank) == 1 else -1
        print(
            f"{scale_world:>2}  {t_bytes:>9,}  {measured:>14,}  "
            f"{formula:>9,}  {measured / t_bytes:>8.3f}  {steps:>5}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    if broadcast_result is not None:
        wire = sum(stat.bytes_sent for stat in broadcast_result.stats)
        expected_wire = total * 8 * (world_size - 1)
        rounds = broadcast_result.stats[0].rounds
        ok = wire == expected_wire and rounds == 2
        all_ok = all_ok and ok
        print()
        print(
            f"broadcast N={world_size}: wire bytes={_fmt_bytes(wire)} "
            f"(expected {_fmt_bytes(expected_wire)}), rounds={rounds} "
            f"(ceil(log2 N)=2)  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print("ALL CHECKS PASSED" if all_ok else "CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
