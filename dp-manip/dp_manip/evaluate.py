"""Closed-loop evaluation on an explicit list of held-out reset seeds.

The ManiSkill baseline resets without seeds and relies on auto-reset, so the
evaluated initial conditions are not a fixed, reportable set. Here every
episode is reset with a known seed and results are reported per seed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

METRICS = ("success_once", "success_at_end", "episode_len", "return")


def _episode_info(info: dict) -> dict:
    # gymnasium>=1.0 vector envs default to next-step autoreset, where the
    # truncating step's info already holds the final metrics; older versions
    # moved them to final_info.
    if "final_info" in info:
        final = info["final_info"]
        if isinstance(final, dict):
            return final["episode"]
        return {k: np.array([f["episode"][k] for f in final]) for k in final[0]["episode"]}
    return info["episode"]


@torch.no_grad()
def evaluate(policy, envs, seeds: Sequence[int], device: torch.device) -> dict:
    """Run one episode per seed. Returns per-episode records plus summary stats."""
    num_envs = envs.num_envs
    if len(seeds) % num_envs:
        raise ValueError("number of seeds must be a multiple of num_envs")
    was_training = policy.training
    policy.eval()
    episodes, infer_time, infer_calls = [], 0.0, 0
    start = time.time()
    for i in range(0, len(seeds), num_envs):
        chunk = list(seeds[i : i + num_envs])
        obs, _ = envs.reset(seed=chunk)
        done = False
        while not done:
            tick = time.time()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions = policy.get_action(obs_t).cpu().numpy()
            infer_time += time.time() - tick
            infer_calls += 1
            for j in range(actions.shape[1]):
                obs, _, _, truncated, info = envs.step(actions[:, j])
                if truncated.any():
                    if not truncated.all():
                        raise RuntimeError("sub-envs truncated on different steps")
                    ep = _episode_info(info)
                    for k, seed in enumerate(chunk):
                        episodes.append({"seed": int(seed), **{m: _scalar(ep[m][k]) for m in METRICS if m in ep}})
                    done = True
                    break
    if was_training:
        policy.train()
    summary = {m: float(np.mean([e[m] for e in episodes])) for m in METRICS if m in episodes[0]}
    summary.update(
        num_episodes=len(episodes),
        wall_time_s=time.time() - start,
        mean_inference_ms=1000 * infer_time / max(infer_calls, 1),
    )
    return {"summary": summary, "episodes": episodes}


def _scalar(x):
    x = np.asarray(x).item()
    return bool(x) if isinstance(x, (bool, np.bool_)) else float(x)
