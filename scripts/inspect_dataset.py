#!/usr/bin/env python3
"""Validate observation schemas and nested data-size subsets before submitting jobs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.config import load, load_experiment  # noqa: E402
from dp_manip.data import DatasetInfo, compute_normalization, read_dataset_info  # noqa: E402
from dp_manip.trainer import check_horizon  # noqa: E402

DATA_SIZE_EXPERIMENT = ROOT / "configs" / "experiments" / "data_size.toml"


def check_nested_subsets(path: Path, full: DatasetInfo, sizes: Sequence[int]) -> None:
    """Verify that every declared N selects the seed-ordered prefix of the pool."""
    for size in sorted(set(sizes)):
        if size > len(full.episodes):
            raise ValueError(
                f"{path}: data-size grid asks for {size} demos, pool has {len(full.episodes)}"
            )
        subset_seeds = read_dataset_info(path, size, obs_mode=full.obs_mode).seeds
        if subset_seeds != full.seeds[:size]:
            raise ValueError(f"{path}: N={size} is not the first {size} demos ordered by episode seed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, help="repeatable; default: all task configs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, help="limit the inspected training subset")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=DATA_SIZE_EXPERIMENT,
        help="data-size spec whose values must form nested training subsets",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="config override (repeatable), e.g. task.control_mode=pd_ee_delta_pose",
    )
    args = parser.parse_args(argv)
    configs = args.config or sorted((ROOT / "configs" / "tasks").glob("*.toml"))
    spec = load_experiment(args.experiment)
    if spec.variable != "data.num_demos":
        raise ValueError(f"{args.experiment}: expected data.num_demos, got {spec.variable!r}")
    grid = tuple(int(value) for value in spec.values)
    failures = 0
    for config_path in configs:
        try:
            overrides = list(args.overrides)
            if args.num_demos is not None:
                overrides.append(f"data.num_demos={args.num_demos}")
            cfg = load(config_path, overrides)
            train_path = args.data_root / cfg.data.train_path
            val_path = args.data_root / cfg.data.val_path
            train = read_dataset_info(train_path, args.num_demos, obs_mode=cfg.task.obs_mode)
            if args.num_demos is None:
                check_nested_subsets(train_path, train, grid)
            val = read_dataset_info(val_path, cfg.data.val_num_demos, obs_mode=cfg.task.obs_mode)
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
            check_horizon(cfg, train, val)
            lengths = [episode.length for info in (train, val) for episode in info.episodes]
            stats = compute_normalization(train)
            print(
                f"PASS {cfg.task.env_id}: train {len(train.episodes)} demos/{train.num_transitions} steps; "
                f"val {len(val.episodes)}; obs_mode {train.obs_mode}; "
                f"images {train.image_shape} {list(train.cameras)}; "
                f"lowdim {train.proprio_dim}; action {train.action_dim} ({train.control_mode}); "
                f"action range [{stats.action_low.min():.3f}, {stats.action_high.max():.3f}]; "
                f"demo length {min(lengths)}-{sum(lengths) / len(lengths):.0f}-{max(lengths)} "
                f"(min-mean-max), max_episode_steps {cfg.task.max_episode_steps}"
            )
        except Exception as error:
            failures += 1
            print(f"FAIL {config_path}: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures} dataset(s) failed validation")


if __name__ == "__main__":
    main()
