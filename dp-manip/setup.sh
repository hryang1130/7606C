#!/usr/bin/env bash
# Build the pinned Python environment on a cluster login node.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it in your home directory before running setup.sh" >&2
  exit 1
fi
unset UV_PROJECT_ENVIRONMENT
uv sync --frozen
.venv/bin/python scripts/verify_cuda.py --allow-cpu
