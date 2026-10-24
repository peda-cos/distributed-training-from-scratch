#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
.venv/bin/python -m compileall -q code pipeline_parallel tests
.venv/bin/python code/main.py --no-gantt >/dev/null
timeout 900 .venv/bin/python -m pytest -q
