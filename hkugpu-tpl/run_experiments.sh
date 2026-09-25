#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
TASK="${TASK:-}"
SEEDS="${SEEDS:-1 2 3}"
STEPS="${STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
DATASET="${DATASET:-}"
MODELS="${MODELS:-}"
EVAL_EPISODES="${EVAL_EPISODES:-}"

if [[ -z "$TASK" ]]; then
  echo "Set TASK to a task plugin name, for example: TASK=<task_id> bash run_experiments.sh" >&2
  exit 2
fi

if [[ -z "$MODELS" ]]; then
  MODELS="$(uv run python run.py models --task "$TASK" | tr '\n' ' ')"
fi
for model in $MODELS; do
  for seed in $SEEDS; do
    arguments=(run.py train --task "$TASK" --model "$model" --seed "$seed")
    if [[ -n "$STEPS" ]]; then
      arguments+=(--steps "$STEPS")
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
      arguments+=(--batch-size "$BATCH_SIZE")
    fi
    if [[ -n "$DATASET" ]]; then
      arguments+=(--dataset "$DATASET")
    fi
    uv run python "${arguments[@]}"
    checkpoint="runs/$TASK/$model/seed_$seed/best.pt"
    evaluation_arguments=(run.py evaluate --task "$TASK" --checkpoint "$checkpoint")
    if [[ -n "$EVAL_EPISODES" ]]; then
      evaluation_arguments+=(--episodes "$EVAL_EPISODES")
    fi
    uv run python "${evaluation_arguments[@]}"
  done
done
uv run python run.py report --task "$TASK"
