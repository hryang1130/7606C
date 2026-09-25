#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv in your user account: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
uv python install "$PYTHON_VERSION"
uv lock
uv sync --python "$PYTHON_VERSION"
uv run python - <<'PY'
import torch
import mani_skill
import diffusers

print(f"torch={torch.__version__}; CUDA runtime={torch.version.cuda}; CUDA available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU={torch.cuda.get_device_name(0)}; BF16={torch.cuda.is_bf16_supported()}")
else:
    print("No visible CUDA GPU. Only CPU smoke tests are available.")
print(f"ManiSkill={mani_skill.__version__}; Diffusers={diffusers.__version__}")
PY
echo "Environment ready. Run: uv run python run.py smoke --task <task_id>"
