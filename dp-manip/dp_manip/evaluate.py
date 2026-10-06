"""Closed-loop RGB or state policy evaluation on explicit held-out reset seeds."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch


class RolloutObserver(Protocol):
    """Read-only hooks into :func:`evaluate`'s stepping loop.

    Observers see exactly the arrays the policy and the metrics see and must not
    consume policy or environment randomness, so an observed evaluation returns
    the same per-episode results as an unobserved one.
    """

    def on_reset(self, seeds: list[int], rgb: np.ndarray, proprio: np.ndarray) -> None: ...

    def on_step(
        self,
        actions: np.ndarray,
        rgb: np.ndarray,
        proprio: np.ndarray,
        reward: np.ndarray,
        success: np.ndarray,
    ) -> None: ...

    def on_wave_end(self, episodes: list[dict]) -> bool:
        """Receive the finished wave's episode records; return ``True`` to stop."""
        ...


def _numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _adapt_environment_observation(
    observation, num_envs: int, obs_mode: str = "rgb"
) -> tuple[np.ndarray | None, np.ndarray]:
    """Read RGB proprioception or a complete state vector, without falling back."""
    if obs_mode == "state":
        if isinstance(observation, dict):
            raise ValueError("expected a flat complete state vector from obs_mode=state")
        state = _numpy(observation)
        if state.ndim == 3 and state.shape[:2] == (num_envs, 1):
            state = state[:, 0]
        if state.ndim != 2 or len(state) != num_envs:
            raise ValueError(f"unexpected environment state shape: {state.shape}")
        return None, state
    if obs_mode != "rgb":
        raise ValueError("obs_mode must be rgb or state")
    if not isinstance(observation, dict) or "rgb" not in observation or "state" not in observation:
        raise ValueError(f"expected flattened RGB observation dict, got {type(observation).__name__}")
    rgb, proprio = _numpy(observation["rgb"]), _numpy(observation["state"])
    # Some ManiSkill wrappers preserve their internal num_envs=1 axis before
    # Gymnasium's outer process-vector axis.
    if rgb.ndim == 5 and rgb.shape[:2] == (num_envs, 1):
        rgb = rgb[:, 0]
    if proprio.ndim == 3 and proprio.shape[:2] == (num_envs, 1):
        proprio = proprio[:, 0]
    if rgb.ndim != 4 or proprio.ndim != 2 or len(rgb) != num_envs or len(proprio) != num_envs:
        raise ValueError(f"unexpected environment RGB/proprio shapes: {rgb.shape}, {proprio.shape}")
    return rgb, proprio


@torch.no_grad()
def evaluate(
    policy,
    envs,
    seeds: Sequence[int],
    device: torch.device,
    *,
    inference_seed: int,
    observer: RolloutObserver | None = None,
) -> dict:
    """Run one fixed-length episode per seed and return per-seed metrics.

    With an ``observer``, evaluation stops after any wave for which
    ``observer.on_wave_end`` returns ``True``. The waves that did run are
    identical to the same waves of an unobserved evaluation.
    """
    num_envs = envs.num_envs
    obs_mode = getattr(policy, "obs_mode", "rgb")
    if obs_mode == "state" and observer is not None:
        raise ValueError("RGB rollout observers are unsupported for state evaluation")
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
        rgb, proprio = _adapt_environment_observation(observation, num_envs, obs_mode)
        if observer is not None:
            observer.on_reset(chunk, rgb, proprio)
        rgb_history = np.repeat(rgb[:, None], policy.obs_horizon, axis=1) if rgb is not None else None
        proprio_history = np.repeat(proprio[:, None], policy.obs_horizon, axis=1)
        success_once = np.zeros(num_envs, dtype=bool)
        success_at_end = np.zeros(num_envs, dtype=bool)
        returns = np.zeros(num_envs, dtype=np.float64)
        episode_length = np.zeros(num_envs, dtype=np.int64)
        finished = False

        while not finished:
            rgb_tensor = (
                torch.as_tensor(
                    np.transpose(rgb_history, (0, 1, 4, 2, 3)), device=device, dtype=torch.uint8
                ) if rgb_history is not None else None
            )
            proprio_tensor = torch.as_tensor(proprio_history, device=device, dtype=torch.float32)
            tick = time.time()
            action_chunks = policy.get_action(
                rgb_tensor, proprio_tensor, generator=generator
            ).cpu().numpy()
            inference_seconds += time.time() - tick
            inference_calls += 1

            for action_index in range(action_chunks.shape[1]):
                observation, reward, _, truncated, info = envs.step(action_chunks[:, action_index])
                rgb, proprio = _adapt_environment_observation(observation, num_envs, obs_mode)
                if rgb_history is not None:
                    rgb_history = np.concatenate((rgb_history[:, 1:], rgb[:, None]), axis=1)
                proprio_history = np.concatenate(
                    (proprio_history[:, 1:], proprio[:, None]), axis=1
                )
                step_reward = _numpy(reward).reshape(num_envs)
                returns += step_reward
                episode_length += 1
                current_success = _numpy(info.get("success", np.zeros(num_envs))).reshape(num_envs).astype(bool)
                if observer is not None:
                    observer.on_step(
                        action_chunks[:, action_index], rgb, proprio, step_reward, current_success
                    )
                success_once |= current_success
                success_at_end = current_success
                truncated = _numpy(truncated).reshape(num_envs).astype(bool)
                if truncated.any():
                    if not truncated.all():
                        raise RuntimeError("parallel environments truncated on different steps")
                    finished = True
                    break

        wave: list[dict] = []
        for index, seed in enumerate(chunk):
            wave.append(
                {
                    "seed": int(seed),
                    "success_once": bool(success_once[index]),
                    "success_at_end": bool(success_at_end[index]),
                    "episode_len": int(episode_length[index]),
                    "return": float(returns[index]),
                }
            )
        episodes.extend(wave)
        if observer is not None and observer.on_wave_end(wave):
            break

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
