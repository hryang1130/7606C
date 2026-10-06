#!/usr/bin/env python3
"""Collect baseline-policy rollouts and build success/failure datasets (plan §3).

Per baseline checkpoint::

    collect_rollouts.py collect <final.pt> --split train     # until K of each class, cap 600
    collect_rollouts.py collect <final.pt> --split holdout   # 100 episodes, all kept
    collect_rollouts.py build <rollout dir>                  # K-prefixes + truncation

Rollouts go to ``<run root>/failure_aware/<task>/s<train seed>/`` by default,
where ``<run root>`` is the directory holding the checkpoint's run directory.
The task comes from the checkpoint's recorded config; nothing here is
task-specific. ``--max-episode-steps`` replaces the recorded horizon with the
study's locked one (plan §13); ``failure_study.py record-collection`` refuses
a collection made at another horizon.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.failure_protocol import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from dp_manip.failure_rollout import (  # noqa: E402
    RAW_HOLDOUT,
    RAW_TRAIN,
    ClassQuota,
    RolloutWriter,
    build_datasets,
    collect,
    rollout_dir_for,
)
from dp_manip.metadata import file_sha256, git_revision  # noqa: E402


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("collect", help="roll out one split of one checkpoint")
    run.add_argument("checkpoint", type=Path)
    run.add_argument("--split", choices=("train", "holdout"), required=True)
    run.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    run.add_argument("--run-root", type=Path, help="default: the checkpoint's run directory's parent")
    run.add_argument("--output-dir", type=Path, help="default: <run root>/failure_aware/<task>/s<seed>")
    run.add_argument("--device", default="cuda")
    run.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
    run.add_argument("--overwrite", action="store_true", help="replace an existing raw file")
    run.add_argument(
        "--max-episode-steps",
        type=positive_int,
        help="episode horizon; default: the checkpoint's recorded one (plan §13)",
    )

    build = commands.add_parser("build", help="build datasets from a checkpoint's raw rollouts")
    build.add_argument("rollout_dir", type=Path)
    build.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    build.add_argument("--overwrite", action="store_true", help="replace existing datasets")
    return parser.parse_args(argv)


def default_envs_factory(cfg, num_envs: int, render_backend: str | None):
    from dp_manip.envs import make_eval_envs

    return make_eval_envs(cfg, num_envs, render_backend)


def run_collect(args: argparse.Namespace, envs_factory=default_envs_factory) -> None:
    import torch

    from dp_manip.config import from_recorded
    from dp_manip.envs import environment_kwargs
    from dp_manip.policy import DiffusionPolicy

    protocol = load_protocol(args.protocol)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = from_recorded(checkpoint["config"])
    recorded_horizon = cfg.task.max_episode_steps
    if args.max_episode_steps is not None:
        cfg.task.max_episode_steps = args.max_episode_steps
    protocol.check_against(cfg)
    train_data = checkpoint["train_data"]
    policy = DiffusionPolicy.from_checkpoint(checkpoint, device)
    policy.eval()

    output_dir = args.output_dir or rollout_dir_for(
        checkpoint_path, cfg.task.name, cfg.train.seed, args.run_root
    )
    raw_path = output_dir / (RAW_TRAIN if args.split == "train" else RAW_HOLDOUT)
    seeds = protocol.train_seeds() if args.split == "train" else protocol.holdout_seeds()
    quota = ClassQuota(protocol.collection.dataset_size) if args.split == "train" else None
    num_envs = cfg.eval.num_envs  # checked by protocol.check_against to divide every count

    env_info = {"env_id": cfg.task.env_id, "env_kwargs": environment_kwargs(cfg, args.render_backend)}
    export_info = {"cameras": train_data["cameras"], "rgb_env_info": train_data.get("rgb_env_info")}
    print(f"collecting {args.split} rollouts of {checkpoint_path} into {raw_path}")
    writer = RolloutWriter(raw_path, env_info=env_info, export_info=export_info, overwrite=args.overwrite)
    started = dt.datetime.now(dt.timezone.utc)
    try:
        envs = envs_factory(cfg, num_envs, args.render_backend)
    except BaseException:
        writer.abort()
        raise
    try:
        result = collect(
            policy,
            envs,
            seeds,
            device,
            inference_seed=cfg.eval.inference_seed,
            writer=writer,
            quota=quota,
        )
    except BaseException:
        writer.abort()
        raise
    finally:
        envs.close()
    summary = result["summary"]
    used = [episode["seed"] for episode in result["episodes"]]
    writer.close(
        {
            "dataset_type": "rollout_raw",
            "split": args.split,
            "task": cfg.task.name,
            "env_id": cfg.task.env_id,
            "control_mode": cfg.task.control_mode,
            "max_episode_steps": cfg.task.max_episode_steps,
            "checkpoint_max_episode_steps": recorded_horizon,
            "act_horizon": cfg.policy.act_horizon,
            "source_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "step": int(checkpoint["step"]),
                "train_seed": cfg.train.seed,
                "num_demos": cfg.data.num_demos,
                "backbone": cfg.policy.backbone,
            },
            "protocol": {"path": str(protocol.path), "sha256": protocol.sha256},
            "git": git_revision(ROOT),
            "inference_seed": cfg.eval.inference_seed,
            "num_envs": num_envs,
            "device": str(device),
            "seeds_candidate": [seeds[0], seeds[-1] + 1],
            "seeds_used": used,
            "stopped_by_quota": quota is not None and quota.reached(),
            "started_utc": started.isoformat(),
            "wall_time_s": summary["wall_time_s"],
            "mean_inference_ms": summary["mean_inference_ms"],
        }
    )
    successes = sum(episode["success_once"] for episode in result["episodes"])
    print(
        f"{args.split}: {len(used)} rollouts, {successes} successes, "
        f"{len(used) - successes} failures, {summary['wall_time_s'] / 3600:.2f} h; {raw_path}"
    )
    if quota is not None and not quota.reached():
        print(
            f"rollout cap reached before {protocol.collection.dataset_size} of each class; "
            "build will lower K to the smaller class count"
        )


def run_build(args: argparse.Namespace) -> None:
    summary = build_datasets(args.rollout_dir, load_protocol(args.protocol), overwrite=args.overwrite)
    datasets = summary["datasets"]
    print(json.dumps({key: value["episodes"] for key, value in datasets.items()}))
    print(
        f"K={summary['dataset_size']} (requested {summary['dataset_size_requested']}), "
        f"L_fail={summary['l_fail']}, train success rate {summary['train_success_rate']:.3f}"
        + (" [LOW-SUCCESS MODE]" if summary["low_success"] else "")
    )


def main(argv: list[str] | None = None, envs_factory=default_envs_factory) -> None:
    args = parse_args(argv)
    if args.command == "collect":
        run_collect(args, envs_factory)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
