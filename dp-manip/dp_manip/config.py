"""Experiment configuration shared by training and evaluation.

Configs are TOML files under configs/. Everything that must agree between
training and evaluation (control mode, horizons, episode length, eval seeds)
lives here, and the resolved config is stored inside every checkpoint.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class TaskConfig:
    env_id: str
    control_mode: str
    obs_mode: str = "state"
    sim_backend: str = "physx_cpu"
    # Evaluation episode length. ManiSkill defaults are tuned for RL and can be
    # shorter than the demos (PickCube JSON says 50, demos reach 88 steps).
    max_episode_steps: int = 100


@dataclass
class DataConfig:
    demo_path: str
    # None = use every trajectory in the file; otherwise the first N by traj index.
    num_demos: int | None = None


@dataclass
class PolicyConfig:
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: list[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    num_diffusion_iters: int = 100


@dataclass
class TrainConfig:
    seed: int = 1
    total_iters: int = 30_000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    log_freq: int = 1000
    eval_freq: int = 5000
    # 0 = only keep best/final checkpoints.
    save_freq: int = 0


@dataclass
class EvalConfig:
    # Reset seeds for closed-loop evaluation. Validation seeds are used during
    # training to pick best.pt; test seeds are only used by scripts/eval_dp.py
    # for reported numbers, so checkpoint selection never sees them. Both
    # ranges must be disjoint from each other and from the demo seeds.
    val_seed_start: int = 5_000
    val_episodes: int = 50
    test_seed_start: int = 10_000
    test_episodes: int = 100
    num_envs: int = 10


@dataclass
class Config:
    task: TaskConfig
    data: DataConfig
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        p = self.policy
        if min(p.obs_horizon, p.act_horizon, p.pred_horizon) < 1:
            raise ValueError("horizons must be positive")
        if p.obs_horizon + p.act_horizon - 1 > p.pred_horizon:
            raise ValueError("need obs_horizon + act_horizon - 1 <= pred_horizon")
        e = self.eval
        if e.val_episodes % e.num_envs or e.test_episodes % e.num_envs:
            raise ValueError("eval episode counts must be multiples of eval.num_envs")
        if set(self.val_seeds()) & set(self.test_seeds()):
            raise ValueError("validation and test seed ranges overlap")

    def val_seeds(self) -> list[int]:
        return list(range(self.eval.val_seed_start, self.eval.val_seed_start + self.eval.val_episodes))

    def test_seeds(self) -> list[int]:
        return list(range(self.eval.test_seed_start, self.eval.test_seed_start + self.eval.test_episodes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "task": TaskConfig,
    "data": DataConfig,
    "policy": PolicyConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = Config(**{name: cls(**raw[name]) for name, cls in _SECTIONS.items() if name in raw})
    cfg.validate()
    return cfg


def load(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load a TOML config and apply `section.key=value` overrides (value parsed as TOML, else string)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for item in overrides:
        key, sep, value = item.partition("=")
        section, dot, name = key.partition(".")
        if not sep or not dot:
            raise ValueError(f"override must look like section.key=value: {item!r}")
        try:
            parsed = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            parsed = value  # bare string, e.g. data.demo_path=data/x.h5
        raw.setdefault(section, {})[name] = parsed
    return from_dict(raw)
