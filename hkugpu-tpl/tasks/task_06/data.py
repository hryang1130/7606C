from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def read_trajectories(path: str | Path, limit: int | None = None) -> list[dict[str, np.ndarray]]:
    trajectories = []
    with h5py.File(path, "r") as file:
        keys = sorted((key for key in file if key.startswith("traj_")), key=lambda key: int(key.split("_")[-1]))
        if not keys:
            raise ValueError(f"No traj_N groups found in {path}")
        if limit:
            keys = keys[:limit]
        for key in keys:
            group = file[key]
            if "obs" not in group or "actions" not in group:
                raise ValueError(f"{key} must contain state observations ('obs') and actions")
            obs = np.asarray(group["obs"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            if obs.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"Expected flat state observations/actions; got {obs.shape}, {actions.shape} in {key}")
            if len(obs) == len(actions) + 1:
                obs = obs[:-1]
            if len(obs) != len(actions):
                raise ValueError(f"Observation/action lengths disagree in {key}: {len(obs)} vs {len(actions)}")
            trajectories.append({"obs": obs, "actions": actions})
    return trajectories


class WindowDataset(Dataset):
    def __init__(self, trajectories: list[dict[str, np.ndarray]], obs_horizon: int, pred_horizon: int):
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.episode_count = len(trajectories)
        self.trajectories = [
            {key: torch.as_tensor(value, dtype=torch.float32) for key, value in trajectory.items()}
            for trajectory in trajectories
        ]
        self.windows = []
        for index, trajectory in enumerate(self.trajectories):
            length = len(trajectory["actions"])
            self.windows.extend((index, start) for start in range(-obs_horizon + 1, length))
        if not self.windows:
            raise ValueError("No trajectory windows available")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_index, start = self.windows[index]
        trajectory = self.trajectories[trajectory_index]
        observations, actions = trajectory["obs"], trajectory["actions"]
        length = len(actions)
        obs_indices = torch.arange(start, start + self.obs_horizon).clamp(0, length - 1)
        action_indices = torch.arange(start, start + self.pred_horizon)
        action_indices = action_indices.clamp(0, length - 1)
        action_window = actions[action_indices].clone()
        after_end = torch.arange(start, start + self.pred_horizon) >= length
        if after_end.any() and action_window.shape[-1] >= 2:
            action_window[after_end, :-1] = 0
            action_window[after_end, -1] = actions[-1, -1]
        return observations[obs_indices], action_window


def split_trajectories(trajectories: list[dict[str, np.ndarray]], seed: int, fraction: float = 0.1):
    if len(trajectories) < 2:
        raise ValueError("At least two expert trajectories are required for an episode-wise validation split")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajectories))
    count = min(len(trajectories) - 1, max(1, round(len(trajectories) * fraction)))
    validation = set(indices[:count].tolist())
    return ([t for i, t in enumerate(trajectories) if i not in validation],
            [t for i, t in enumerate(trajectories) if i in validation])


def load_datasets(path, seed, validation_fraction, limit, config):
    episodes = read_trajectories(path, limit)
    training, validation = split_trajectories(episodes, seed, validation_fraction)
    return (
        WindowDataset(training, config["obs_horizon"], config["prediction_horizon"]),
        WindowDataset(validation, config["obs_horizon"], config["prediction_horizon"]),
    )
