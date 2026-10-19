#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
.venv/bin/python -m compileall -q code data_parallel_ddp tests
timeout 900 .venv/bin/python -m pytest -q
