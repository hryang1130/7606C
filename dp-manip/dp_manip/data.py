"""ManiSkill state-trajectory loading, action normalization and DP training windows.

Episode boundaries come only from the JSON `episodes[].episode_id` and the
matching H5 `traj_<id>` group; per-step terminated/truncated flags are ignored
because they can turn true before the group ends (see AGENT.md §4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class Episode:
    episode_id: int
    seed: int
    obs: np.ndarray  # (T+1, obs_dim) float32
    actions: np.ndarray  # (T, act_dim) float32


@dataclass
class DemoSet:
    env_id: str
    control_mode: str
    obs_mode: str
    episodes: list[Episode]

    @property
    def seeds(self) -> list[int]:
        return [e.seed for e in self.episodes]

    @property
    def obs_dim(self) -> int:
        return self.episodes[0].obs.shape[1]

    @property
    def act_dim(self) -> int:
        return self.episodes[0].actions.shape[1]


def load_demos(h5_path: str | Path, num_demos: int | None = None) -> DemoSet:
    h5_path = Path(h5_path)
    meta = json.loads(h5_path.with_suffix(".json").read_text())
    env_kwargs = meta["env_info"]["env_kwargs"]
    entries = sorted(meta["episodes"], key=lambda e: e["episode_id"])
    if num_demos is not None:
        if not 0 < num_demos <= len(entries):
            raise ValueError(f"num_demos={num_demos} but file has {len(entries)} episodes")
        entries = entries[:num_demos]

    episodes = []
    with h5py.File(h5_path, "r") as f:
        for entry in entries:
            if not entry.get("success", False):
                raise ValueError(f"episode {entry['episode_id']} is not marked successful")
            group = f[f"traj_{entry['episode_id']}"]
            obs = np.asarray(group["obs"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            if obs.ndim != 2:
                raise ValueError(f"expected flat state obs, got shape {obs.shape}")
            if obs.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"traj_{entry['episode_id']}: obs {obs.shape} vs actions {actions.shape}")
            if not (np.isfinite(obs).all() and np.isfinite(actions).all()):
                raise ValueError(f"traj_{entry['episode_id']}: non-finite values")
            episodes.append(Episode(entry["episode_id"], entry["episode_seed"], obs, actions))

    return DemoSet(
        env_id=meta["env_info"]["env_id"],
        control_mode=env_kwargs["control_mode"],
        obs_mode=env_kwargs["obs_mode"],
        episodes=episodes,
    )


class ActionNormalizer:
    """Per-dimension min-max scaling of actions to [-1, 1].

    The DDPM scheduler clips samples to [-1, 1], so actions must live in that
    range. Absolute joint targets (pd_joint_pos) are in radians and do not.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, eps: float = 1e-4):
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        # Constant dims map to 0 instead of dividing by ~0.
        flat = (high - low) < eps
        center = (high + low) / 2
        self.low = np.where(flat, center - 1, low).astype(np.float32)
        self.high = np.where(flat, center + 1, high).astype(np.float32)

    @classmethod
    def fit(cls, demos: DemoSet) -> "ActionNormalizer":
        actions = np.concatenate([e.actions for e in demos.episodes])
        return cls(actions.min(axis=0), actions.max(axis=0))

    def normalize(self, x):
        low, high = self._like(x)
        return 2 * (x - low) / (high - low) - 1

    def unnormalize(self, x):
        low, high = self._like(x)
        return (x + 1) / 2 * (high - low) + low

    def _like(self, x):
        if isinstance(x, torch.Tensor):
            return (torch.as_tensor(self.low, device=x.device, dtype=x.dtype),
                    torch.as_tensor(self.high, device=x.device, dtype=x.dtype))
        return self.low, self.high

    def state_dict(self) -> dict[str, list[float]]:
        return {"low": self.low.tolist(), "high": self.high.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict[str, list[float]]) -> "ActionNormalizer":
        return cls(np.asarray(d["low"]), np.asarray(d["high"]), eps=0.0)


def is_delta_control(control_mode: str) -> bool:
    return "delta" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"


def build_windows(demos: DemoSet, obs_horizon: int, pred_horizon: int):
    """All (obs history, action chunk) training windows, as in ManiSkill's DP baseline.

    For a window starting at s the policy sees obs[s : s+obs_horizon] and predicts
    actions[s : s+pred_horizon]; the current time is t = s + obs_horizon - 1.
    Before the episode start, obs and actions repeat index 0. After the end, the
    robot is told to stay still: absolute modes repeat the last action, delta
    modes use a zero arm delta with the last gripper command.

    Unlike the baseline (whose range excludes it), the window with t = T-1 is
    included so the final real action is also a training target.
    """
    delta = is_delta_control(demos.control_mode)
    obs_windows, act_windows = [], []
    for ep in demos.episodes:
        T = ep.actions.shape[0]
        pad_before = obs_horizon - 1
        pad_after = pred_horizon - obs_horizon
        obs = np.concatenate([np.repeat(ep.obs[:1], pad_before, axis=0), ep.obs[:T]])
        still = ep.actions[-1:].copy()
        if delta:
            still[:, :-1] = 0
        acts = np.concatenate([
            np.repeat(ep.actions[:1], pad_before, axis=0),
            ep.actions,
            np.repeat(still, pad_after, axis=0),
        ])
        # Padded index i is original index i - pad_before, so the window whose
        # current time is t starts at padded index t.
        for t in range(T):
            obs_windows.append(obs[t : t + obs_horizon])
            act_windows.append(acts[t : t + pred_horizon])
    obs_arr = np.stack(obs_windows).astype(np.float32)
    act_arr = np.stack(act_windows).astype(np.float32)
    assert obs_arr.shape[1] == obs_horizon and act_arr.shape[1] == pred_horizon
    return obs_arr, act_arr


class WindowSampler:
    """Holds every training window on `device` and samples batches with replacement.

    The baseline used an epoch sampler with drop_last=True, which yields no
    batches at all when there are fewer windows than batch_size (10 PickCube
    demos give 726 windows < 1024).
    """

    def __init__(self, demos: DemoSet, normalizer: ActionNormalizer,
                 obs_horizon: int, pred_horizon: int, device: torch.device):
        obs, acts = build_windows(demos, obs_horizon, pred_horizon)
        self.obs = torch.from_numpy(obs).to(device)
        self.actions = torch.from_numpy(normalizer.normalize(acts)).to(device)
        self.device = device

    def __len__(self) -> int:
        return self.obs.shape[0]

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        idx = torch.randint(len(self), (batch_size,), generator=generator).to(self.device)
        return self.obs[idx], self.actions[idx]
