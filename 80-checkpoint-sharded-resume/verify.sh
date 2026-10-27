#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
.venv/bin/python -m compileall -q code sharded_checkpoint tests
demo_dir="$(mktemp -d)"
trap 'rm -rf "$demo_dir"' EXIT
.venv/bin/python code/main.py --dir "$demo_dir/checkpoint" >/dev/null
timeout 900 .venv/bin/python -m pytest -q
