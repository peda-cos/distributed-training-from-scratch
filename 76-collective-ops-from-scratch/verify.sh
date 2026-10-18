#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
.venv/bin/python -m compileall -q code collective_ops tests
.venv/bin/python -m pytest -q
