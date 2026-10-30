# Track

A collection of 6 tested projects built from scratch, one module per capstone lesson.

Each module has its own `README.md`, `verify.sh`, tests, and a deterministic offline demo.

## Modules

| # | Module | What it demonstrates | Verify |
|---:|---|---|---|
| 76 | [Collective Ops From Scratch](76-collective-ops-from-scratch/) | The four collective operations that hold distributed training together are allreduce, broadcast, allgather, and reduce_scatter. Every other  | `cd 76-collective-ops-from-scratch && ./verify.sh` |
| 77 | [Data Parallel DDP From Scratch](77-data-parallel-ddp/) | DistributedDataParallel is a hook on top of allreduce. Wrap a model, broadcast the initial parameters from rank 0 so every rank starts ident | `cd 77-data-parallel-ddp && ./verify.sh` |
| 78 | [ZeRO Optimizer State Sharding](78-zero-parameter-sharding/) | Adam stores two moment estimates per parameter, both in float32. A 7B-parameter model carries 56 GB of optimiser state. ZeRO stage 1 shards  | `cd 78-zero-parameter-sharding && ./verify.sh` |
| 79 | [Pipeline Parallel and Bubble Analysis](79-pipeline-parallel/) | Tensor parallelism splits the matrix multiply across ranks. Pipeline parallelism splits the model across ranks, one stage per rank. Microbat | `cd 79-pipeline-parallel && ./verify.sh` |
| 80 | [Sharded Checkpoint and Atomic Resume](80-checkpoint-sharded-resume/) | A 70B-parameter training job is paused by a node failure every few hours. The checkpoint format decides whether you lose 30 minutes or 30 ho | `cd 80-checkpoint-sharded-resume && ./verify.sh` |
| 81 | [End-to-End Distributed Training](81-end-to-end-distributed-train/) | Lessons 76 through 80 each built one piece. This is the assembly: a tiny GPT trained across 4 simulated ranks with DDP for gradient sync, Ze | `cd 81-end-to-end-distributed-train && ./verify.sh` |

## Requirements

- Python 3.12+ and `uv` for Python modules
- Node 22+ / Bun 1.4 for TypeScript modules
- No API keys, GPU, or network needed for the test suites

## Verify all

```bash
make verify
```

## References

- Capstone source: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects
