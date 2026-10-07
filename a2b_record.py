#!/usr/bin/env python3
"""Record the closed-loop gripper command trace of one checkpoint.

Diagnostic only: it replays exactly the same evaluation as ``eval_dp.py``
(same seeds, ``num_envs``, inference seed) but attaches a ``RolloutObserver``
that stores, for every step of every episode, the executed gripper command and
the pixel position of the red cube. That is enough to ask *when* a backbone
decides to close the gripper and whether it oscillates, which the denoising
loss cannot show.

Writes ``<output>`` as JSON next to (not inside) the run's ``eval/`` directory
so the project's evaluation-file discovery never sees it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent / "dp-manip"
sys.path.insert(0, str(ROOT))

from dp_manip.config import from_recorded  # noqa: E402
from dp_manip.envs import make_eval_envs  # noqa: E402
from dp_manip.evaluate import evaluate  # noqa: E402
from dp_manip.policy import DiffusionPolicy  # noqa: E402


def cube_px(image: np.ndarray) -> list[float] | None:
    """Normalised median pixel of the saturated red cube (None if not visible)."""
    red, green, blue = (image[..., channel].astype(np.int32) for channel in range(3))
    mask = (red - green > 120) & (red - blue > 120)
    if mask.sum() < 3:
        return None
    rows, cols = np.nonzero(mask)
    return [
        float(np.median(cols)) / image.shape[1],
        float(np.median(rows)) / image.shape[0],
        int(mask.sum()),
    ]


class Recorder:
    """``RolloutObserver`` that keeps the per-step gripper command and cube pixel."""

    def __init__(self, limit: int):
        self.limit = limit
        self.episodes: list[dict] = []
        self._seeds: list[int] = []
        self._current: dict[int, dict] = {}

    def on_reset(self, seeds, rgb, proprio):
        self._seeds = [int(seed) for seed in seeds]
        self._current = {
            seed: {"seed": seed, "gripper": [], "reward": [], "success": [], "cube": []}
            for seed in self._seeds
        }
        for index, seed in enumerate(self._seeds):
            self._current[seed]["cube"].append(cube_px(rgb[index]))

    def on_step(self, actions, rgb, proprio, reward, success):
        for index, seed in enumerate(self._seeds):
            episode = self._current[seed]
            episode["gripper"].append(float(actions[index, 3]))
            episode["reward"].append(float(reward[index]))
            episode["success"].append(bool(success[index]))
            episode["cube"].append(cube_px(rgb[index]))

    def on_wave_end(self, wave) -> bool:
        for record in wave:
            episode = self._current[record["seed"]]
            episode.update(
                success_once=record["success_once"],
                success_at_end=record["success_at_end"],
                episode_len=record["episode_len"],
                total_return=record["return"],
            )
            self.episodes.append(episode)
        return len(self.episodes) >= self.limit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = from_recorded(checkpoint["config"])
    policy = DiffusionPolicy.from_checkpoint(checkpoint, device)
    policy.eval()

    seeds = cfg.test_seeds()[: args.episodes]
    num_envs = math.gcd(len(seeds), cfg.eval.num_envs)
    envs = make_eval_envs(cfg, num_envs, None)
    recorder = Recorder(len(seeds))
    try:
        result = evaluate(
            policy,
            envs,
            seeds,
            device,
            inference_seed=cfg.eval.inference_seed,
            observer=recorder,
        )
    finally:
        envs.close()

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "split": "test",
        "num_envs": num_envs,
        "episodes_planned": len(seeds),
        "summary_of_run": result["summary"],
        "episodes": recorder.episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
