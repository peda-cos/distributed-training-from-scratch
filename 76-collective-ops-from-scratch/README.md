# Collective Ops From Scratch

Four collective operations implemented over a `multiprocessing.Queue` mailbox
mesh and verified against a real `torch.distributed` gloo process group:

- `ring_allreduce` — two-pass ring allreduce (reduce-scatter then allgather)
- `broadcast` — binomial-tree broadcast from any source rank
- `allgather` — N-1 shard rotations
- `reduce_scatter` — the reduce-scatter half of ring allreduce

Every primitive runs in separate OS processes, moves real `torch.Tensor`
payloads, and is checked against gloo on the same inputs. The demo also counts
payload bytes per rank and proves the ring scaling
`2T(N-1)/N`, where `T` is the tensor payload in bytes.

## Architecture

### Queue mesh: N mailboxes

The transport is one `multiprocessing.Queue` per destination rank, N queues in
total. A rank's queue is its inbox; `Mesh.send(dst, tensor)` writes to `dst`'s
inbox and `Mesh.recv(src)` reads the caller's own inbox. Messages are
`(source_rank, tag, serialized_tensor)`.

Two details make this safe:

- **Tags + per-source demux.** A faster rank can enqueue a message from a
  different source before the expected one. `recv(src, tag=...)` buffers
  foreign-source messages in per-source deques and validates the expected tag;
  a wrong tag is a `ProtocolError`, not silent corruption.
- **Bytes, not shared-memory handles.** Torch tensors placed directly on a
  stdlib queue use FD sharing; if the sender exits before the receiver
  unpickles, the receiver gets `ConnectionResetError`. Every payload is
  `torch.save`d to bytes and decoded on receipt. The byte counter reports
  logical tensor payload only, excluding serialization framing.

Ring primitives address only the successor, so they traverse exactly the N ring
edges; the tree broadcast addresses its children directly. `MeshStats.targets`
records every `(src, dst)` pair so tests can prove a ring operation never took a
shortcut.

`code/main.py` is the spec's implementation and CLI. Tests and spawned workers
load it through `collective_ops/main.py` under that non-conflicting module name,
because a top-level package named `code` shadows Python's stdlib `code` module
(and breaks `pdb`, which pytest imports).

### Ring allreduce

Split the buffer into N chunks of `T/N` (uneven chunks are supported), then:

1. **Reduce-scatter**, N-1 steps. At step `s`, rank `r` sends chunk
   `(r-1-s) mod N` and receives chunk `(r-2-s) mod N`, adding the received
   chunk into its local copy.
2. **Allgather**, N-1 steps. At step `s`, rank `r` sends chunk `(r-s) mod N`
   and overwrites chunk `(r-s-1) mod N` with the received full sum.

The `-1` shift in reduce-scatter is load-bearing: it leaves the fully reduced
chunk `r` on rank `r`, which is also the ownership convention of gloo's
`reduce_scatter_tensor`. The schedule without the shift leaves chunk `r+1`
on rank `r`.

Per rank this sends each chunk once per pass except for one chunk per pass, so
equal chunks give `2T(N-1)/N` bytes and `2(N-1)` rounds. The byte-accounting
table below measures it. `reduce_scatter` exposes the same reduce-scatter pass
and returns rank `r`'s reduced chunk.

### Broadcast tree

Broadcast uses a binomial tree over direct mailbox sends. With
`offset = (r - src) mod N`, each rank executes rounds `mask = 1, 2, 4, ...`:

- receive if `mask <= offset < 2*mask`, from `(r - mask) mod N`;
- send if `offset < mask` and `offset + mask < N`, to `(r + mask) mod N`.

Every non-source rank receives exactly once, at `mask = highest power of two
<= offset`; depth is `ceil(log2 N)`, and N-1 messages cross the wire. The tree
works for non-power-of-two N (tested with N=5 and N=8).

`allgather` keeps its local shard and rotates it, receiving predecessor shards
from `(r - s - 1) mod N` until it holds all N shards in rank order.

## Quickstart

Requires Linux, Python 3.14, and [uv](https://docs.astral.sh/uv/). PyTorch is
pinned to the CPU build so the setup does not download CUDA packages.

```bash
cd 76-collective-ops-from-scratch
uv sync --frozen
uv run python code/main.py
```

Expected: a verification table with all `PASS`, followed by the allreduce byte
table and a broadcast byte/round check. Run the full gate with:

```bash
./verify.sh
```

`verify.sh` runs `uv sync --frozen` into the ignored `.venv`, compiles
`code/`, `collective_ops/`, and `tests/`, then runs the 17-test pytest suite.
It is safe to run twice; the second run reuses the environment and lock. The
first sync downloads the ~210 MB CPU torch wheel.

## What runs offline

Everything in this module runs locally and without an external service:

- The queue mesh uses OS pipes through `multiprocessing.Queue`.
- The reference uses the gloo backend, a local `FileStore`, and loopback
  transport; there is no remote host, cloud sandbox, paid API, or GPU
  requirement.
- Tests are deterministic: fixed seeds, small integer fixtures, and no
  sleeps in the collective path.
- `verify.sh` needs network only to install the pinned wheels; tests make no
  external network calls.

## Design decisions

**N mailboxes, not N(N-1) channels.** The spec's ring is N queues. One inbox per
rank is enough because each collective step has one expected sender; direct
children in the tree and the ring successor are just destinations. The cost is
a per-source demux buffer, tested by injecting rank 2's message before rank 1's.

**Real gloo, not a Python imitation.** `_gloo_reference` spawns N processes,
initializes gloo with a fresh file store, runs the same collective, and returns
the per-rank result. Integer rows are compared with `torch.equal` (byte-equal);
float32 rows use `assert_close` because gloo and the ring reduce in different
orders. Tests never skip gloo.

**Spawn, deadlines, and cleanup.** Python 3.14 defaults multiprocessing to
`forkserver`; this code requests `spawn` explicitly and sets one torch thread
per worker. `Mesh.recv` has a per-message deadline (`recv_timeout_s`, 30 s
default) that raises `CommTimeout` with rank/source fields instead of hanging.
The parent also has an overall deadline and terminates workers on failure.
Successful collectives consume every queued message before joining, which
avoids the documented queue feeder-thread join deadlock.

**Ring versus tree.** Ring allreduce is bandwidth-optimal: per-rank traffic
approaches `2T` regardless of N, but it pays `2(N-1)` sequential steps. A tree
has `O(log N)` depth and wins when latency dominates, while its per-rank
traffic is not bandwidth-optimal. Choose by cluster shape and message size:
fat, homogeneous pipes and large tensors favor ring; small messages or
high-latency links favor tree. NCCL embodies this with a cost model that picks
ring for peak bandwidth and tree for logarithmic behavior, rather than a fixed
religion. This module implements the ring allreduce and a tree broadcast; it
does not ship a tree-allreduce variant.

## Measured results

Measured locally on a 12-vCPU x86_64 Linux host, Python 3.14.10, PyTorch
2.15.0+cpu, gloo CPU backend:

```text
Collective Ops From Scratch: queue mesh vs torch.distributed gloo
torch=2.15.0+cpu gloo=True python=3.14.10 cpus=12 platform=Linux-7.3.5-2-cachyos-x86_64-with-glibc2.44

primitive        N  max|mesh-gloo|  exact  steps  result
-----------------------------------------------------------------
allreduce        4    9.537e-07     no      6  PASS
allreduce        4    0.000e+00    yes      6  PASS
broadcast        4    0.000e+00    yes      2  PASS
allgather        4    0.000e+00    yes      3  PASS
reduce_scatter   4    0.000e+00    yes      3  PASS

Allreduce byte accounting (float32, T=2048 elements)
 N    T bytes  per-rank bytes    formula   bytes/T  steps  result
----------------------------------------------------------------------------
 2      8,192           8,192      8,192     1.000      2  PASS
 4      8,192          12,288     12,288     1.500      6  PASS
 8      8,192          14,336     14,336     1.750     14  PASS

broadcast N=4: wire bytes=98,304 (expected 98,304), rounds=2 (ceil(log2 N)=2)  PASS

ALL CHECKS PASSED
```

The float32 delta is reduction-order noise; the integer rows are exact. The
byte counts are integer payload counts from `MeshStats`, not estimates. The
`bytes/T` column approaches 2 as N grows, which is the `2T(N-1)/N` bound. This
run does not claim a latency crossover measurement; that would require a real
fabric and is deliberately not invented.

The test suite's 17 cases (14 collective cases plus 3 transport tests) pass
locally:

```text
17 passed
```

## Limitations

- **CPU process transport.** Pickling/serialization and Python process
  scheduling dominate; this is a correctness reference, not a substitute for
  NCCL or a bandwidth benchmark.
- **Allgather equality for gloo.** The mesh supports ragged shards (tested);
  gloo's `all_gather` path requires equal shard shapes.
- **No GPU/NCCL mode.** Tensors are CPU tensors and the queue mesh has no
  device-aware path. See below for how production would differ.
- **Not implemented:** tree-allreduce switching, TCP transport, JSONL byte
  logging, and wall-clock ring/tree crossover benchmarking. These are spec
  exercises, not claims of this build.
- **Static checks.** `verify.sh` runs `compileall` and pytest; it does not run
  a separate static type checker.

## Real GPU / NCCL path

Nothing is simulated behind a fake provider: the queue version is the real
implementation, and gloo is the real CPU reference. For a production NCCL
path, keep the four algorithm functions and replace the transport with
`torch.distributed` over CUDA tensors (backend `nccl`) or TCP sockets; the
verification harness would compare against the same gloo reference. That path
is not exercised here, so no GPU numbers are reported. The gloo setup in
`_gloo_worker` is the reference-side template: unique file store, spawn,
per-rank tensors, `destroy_process_group` in `finally`.

## References

1. [Collective Ops From Scratch — AI Engineering From Scratch capstone](https://aiengineeringfromscratch.com/learn/capstone-projects/collective-ops-from-scratch)
2. [Patarasuk and Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations* (JPDC 2009)](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) — ring reduce-scatter/allgather schedules and the bandwidth lower bound.
3. [Andrew Gibiansky, *Bringing HPC Techniques to Deep Learning*](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) — derivation of the two-pass ring and per-rank `2(N-1)K/N` traffic.
4. [PyTorch Distributed documentation (2.15)](https://docs.pytorch.org/docs/2.15/distributed.html) — gloo CPU support, `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter_tensor`, and file-store initialization.
5. [NVIDIA, *Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication*](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication) — cost-model algorithm selection, ring peak bandwidth versus tree's logarithmic behavior.

## License

MIT. See [LICENSE](LICENSE).
