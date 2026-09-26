#!/usr/bin/env python3
"""Validate RGB dataset schemas and nested subsets before submitting jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.config import load  # noqa: E402
from dp_manip.data import compute_normalization, read_dataset_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, help="repeatable; default: all *_rgb.toml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, default=400)
    args = parser.parse_args()
    configs = args.config or sorted((ROOT / "configs").glob("*_rgb.toml"))
    failures = 0
    for config_path in configs:
        try:
            cfg = load(config_path, [f"data.num_demos={args.num_demos}"])
            train_path = args.data_root / cfg.data.train_path
            val_path = args.data_root / cfg.data.val_path
            train = read_dataset_info(train_path, cfg.data.num_demos)
            val = read_dataset_info(val_path, cfg.data.val_num_demos)
            if (train.env_id, train.control_mode) != (cfg.task.env_id, cfg.task.control_mode):
                raise ValueError("training metadata does not match task config")
            if (val.env_id, val.control_mode) != (cfg.task.env_id, cfg.task.control_mode):
                raise ValueError("validation metadata does not match task config")
            if (train.image_shape, train.proprio_dim, train.action_dim, train.cameras) != (
                val.image_shape,
                val.proprio_dim,
                val.action_dim,
                val.cameras,
            ):
                raise ValueError("training and validation schemas differ")
            stats = compute_normalization(train)
            print(
                f"PASS {cfg.task.env_id}: train {len(train.episodes)} demos/{train.num_transitions} steps; "
                f"val {len(val.episodes)}; RGB {train.image_shape} {list(train.cameras)}; "
                f"state {train.proprio_dim}; action {train.action_dim} ({train.control_mode}); "
                f"action range [{stats.action_low.min():.3f}, {stats.action_high.max():.3f}]"
            )
        except Exception as error:
            failures += 1
            print(f"FAIL {config_path}: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures} dataset(s) failed validation")


if __name__ == "__main__":
    main()
