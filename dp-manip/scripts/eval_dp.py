#!/usr/bin/env python3
"""Evaluate an RGB or state checkpoint on fixed held-out or training seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.config import from_recorded  # noqa: E402
from dp_manip.envs import make_eval_envs  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--episodes", type=int, help="override the split's episode count")
    parser.add_argument("--num-envs", type=int, help="parallel physx_cpu worker processes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-episode-steps", type=int, help="evaluation-only horizon override; adds _h<N> to result filename")
    args = parser.parse_args(argv)
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    return args


def evaluation_config(recorded: dict, max_episode_steps: int | None):
    cfg = from_recorded(recorded)
    if max_episode_steps is not None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        cfg.task.max_episode_steps = max_episode_steps
    return cfg


def result_path(checkpoint: Path, split: str, max_episode_steps: int | None) -> Path:
    suffix = "" if max_episode_steps is None else f"_h{max_episode_steps}"
    return checkpoint.resolve().parents[1] / "eval" / f"{split}_{checkpoint.stem}{suffix}.json"


def main() -> None:
    import torch
    from dp_manip.evaluate import evaluate
    from dp_manip.policy import DiffusionPolicy

    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = evaluation_config(checkpoint["config"], args.max_episode_steps)
    train_data = checkpoint["train_data"]
    policy = DiffusionPolicy.from_checkpoint(checkpoint, device)
    policy.eval()

    if args.split == "test":
        seeds = cfg.test_seeds()
    elif args.split == "val":
        seeds = cfg.val_seeds()
    else:
        # Overfitting diagnostic on this checkpoint's training subset, in
        # ascending seed order: ``--episodes K`` selects the K seeds shared by
        # every nested subset of size >= K. The experiment spec chooses K.
        seeds = sorted(int(seed) for seed in train_data["seeds"])
    if args.episodes is not None:
        if args.episodes < 1 or args.episodes > len(seeds):
            raise ValueError(f"--episodes must be in [1, {len(seeds)}] for split {args.split}")
        seeds = seeds[: args.episodes]
    requested_envs = args.num_envs or cfg.eval.num_envs
    num_envs = math.gcd(len(seeds), requested_envs)
    if num_envs != requested_envs:
        print(f"adjusting num_envs from {requested_envs} to {num_envs} so all waves are full")

    envs = make_eval_envs(cfg, num_envs, args.render_backend)
    try:
        result = evaluate(
            policy,
            envs,
            seeds,
            device,
            inference_seed=cfg.eval.inference_seed,
        )
    finally:
        envs.close()
    result.update(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_step=int(checkpoint["step"]),
        split=args.split,
        env_id=cfg.task.env_id,
        obs_mode=cfg.task.obs_mode,
        sim_backend=cfg.task.sim_backend,
        num_envs=num_envs,
        inference_seed=cfg.eval.inference_seed,
        max_episode_steps=cfg.task.max_episode_steps,
        checkpoint_max_episode_steps=checkpoint["config"]["task"]["max_episode_steps"],
        evaluation_overrides={} if args.max_episode_steps is None else {"task.max_episode_steps": args.max_episode_steps},
    )
    output = args.output
    if output is None:
        output = result_path(args.checkpoint, args.split, args.max_episode_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(
        f"{args.split}: success_once={summary['success_once']:.3f} "
        f"success_at_end={summary['success_at_end']:.3f} "
        f"({summary['num_episodes']} episodes); {output}"
    )


if __name__ == "__main__":
    main()
