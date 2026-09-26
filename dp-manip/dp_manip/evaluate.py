"""Closed-loop RGB policy evaluation on explicit held-out reset seeds."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch


def _numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _observations(observation: dict, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observation, dict) or "rgb" not in observation or "state" not in observation:
        raise ValueError(f"expected flattened RGB observation dict, got {type(observation).__name__}")
    rgb, state = _numpy(observation["rgb"]), _numpy(observation["state"])
    # Some ManiSkill wrappers preserve their internal num_envs=1 axis before
    # Gymnasium's outer process-vector axis.
    if rgb.ndim == 5 and rgb.shape[:2] == (num_envs, 1):
        rgb = rgb[:, 0]
    if state.ndim == 3 and state.shape[:2] == (num_envs, 1):
        state = state[:, 0]
    if rgb.ndim != 4 or state.ndim != 2 or len(rgb) != num_envs or len(state) != num_envs:
        raise ValueError(f"unexpected environment RGB/state shapes: {rgb.shape}, {state.shape}")
    return rgb, state


@torch.no_grad()
def evaluate(
    policy,
    envs,
    seeds: Sequence[int],
    device: torch.device,
    *,
    inference_seed: int,
) -> dict:
    """Run one fixed-length episode per seed and return per-seed metrics."""
    num_envs = envs.num_envs
    if len(seeds) % num_envs:
        raise ValueError("number of seeds must be divisible by the number of environments")
    was_training = policy.training
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(inference_seed)
    episodes: list[dict] = []
    inference_seconds = 0.0
    inference_calls = 0
    start_time = time.time()

    for offset in range(0, len(seeds), num_envs):
        chunk = list(seeds[offset : offset + num_envs])
        observation, _ = envs.reset(seed=chunk)
        rgb, state = _observations(observation, num_envs)
        rgb_history = np.repeat(rgb[:, None], policy.obs_horizon, axis=1)
        state_history = np.repeat(state[:, None], policy.obs_horizon, axis=1)
        success_once = np.zeros(num_envs, dtype=bool)
        success_at_end = np.zeros(num_envs, dtype=bool)
        returns = np.zeros(num_envs, dtype=np.float64)
        episode_length = np.zeros(num_envs, dtype=np.int64)
        finished = False

        while not finished:
            rgb_tensor = torch.as_tensor(
                np.transpose(rgb_history, (0, 1, 4, 2, 3)), device=device, dtype=torch.uint8
            )
            state_tensor = torch.as_tensor(state_history, device=device, dtype=torch.float32)
            tick = time.time()
            action_chunks = policy.get_action(rgb_tensor, state_tensor, generator=generator).cpu().numpy()
            inference_seconds += time.time() - tick
            inference_calls += 1

            for action_index in range(action_chunks.shape[1]):
                observation, reward, _, truncated, info = envs.step(action_chunks[:, action_index])
                rgb, state = _observations(observation, num_envs)
                rgb_history = np.concatenate((rgb_history[:, 1:], rgb[:, None]), axis=1)
                state_history = np.concatenate((state_history[:, 1:], state[:, None]), axis=1)
                returns += _numpy(reward).reshape(num_envs)
                episode_length += 1
                current_success = _numpy(info.get("success", np.zeros(num_envs))).reshape(num_envs).astype(bool)
                success_once |= current_success
                success_at_end = current_success
                truncated = _numpy(truncated).reshape(num_envs).astype(bool)
                if truncated.any():
                    if not truncated.all():
                        raise RuntimeError("parallel environments truncated on different steps")
                    finished = True
                    break

        for index, seed in enumerate(chunk):
            episodes.append(
                {
                    "seed": int(seed),
                    "success_once": bool(success_once[index]),
                    "success_at_end": bool(success_at_end[index]),
                    "episode_len": int(episode_length[index]),
                    "return": float(returns[index]),
                }
            )

    if was_training:
        policy.train()
    summary = {
        metric: float(np.mean([episode[metric] for episode in episodes]))
        for metric in ("success_once", "success_at_end", "episode_len", "return")
    }
    summary.update(
        num_episodes=len(episodes),
        wall_time_s=time.time() - start_time,
        mean_inference_ms=1_000 * inference_seconds / max(inference_calls, 1),
    )
    return {"summary": summary, "episodes": episodes}
