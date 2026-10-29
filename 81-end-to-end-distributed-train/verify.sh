#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
.venv/bin/python -m compileall -q code end_to_end_distributed tests

demo_root="$(mktemp -d)"
trap 'rm -rf "$demo_root"' EXIT
.venv/bin/python code/main.py --checkpoint-root "$demo_root" >/dev/null
timeout 900 .venv/bin/python -m pytest -q
