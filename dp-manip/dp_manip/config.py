"""Typed experiment configuration shared by cluster training and evaluation."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, TypeVar


@dataclass
class TaskConfig:
    name: str
    env_id: str
    control_mode: str
    max_episode_steps: int
    sim_backend: str = "physx_cpu"
    shader_pack: str = "minimal"


@dataclass
class DataConfig:
    root: str
    train_path: str
    val_path: str
    # The exporter sorts by seed, so these are nested subsets.
    num_demos: int = 100
    val_num_demos: int = 50


@dataclass
class VisionConfig:
    feature_dim: int = 128
    random_shift: int = 4
    share_camera_encoder: bool = True


@dataclass
class PolicyConfig:
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 256
    unet_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    n_groups: int = 8
    num_diffusion_iters: int = 100
    num_inference_iters: int = 100


@dataclass
class TrainConfig:
    seed: int = 1
    total_iters: int = 100_000
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    log_freq: int = 100
    resume_freq: int = 5_000
    validation_steps: list[int] = field(default_factory=lambda: [10_000, 30_000, 60_000])
    checkpoint_steps: list[int] = field(default_factory=lambda: [10_000, 30_000, 60_000])
    amp: bool = True


@dataclass
class EvalConfig:
    val_seed_start: int = 5_000
    val_episodes: int = 50
    test_seed_start: int = 10_000
    test_episodes: int = 100
    num_envs: int = 4
    inference_seed: int = 0


@dataclass
class Config:
    task: TaskConfig
    data: DataConfig
    vision: VisionConfig = field(default_factory=VisionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        if self.task.sim_backend != "physx_cpu":
            raise ValueError("evaluation must use physx_cpu to match the generated demonstrations")
        if self.data.num_demos not in {25, 50, 100, 200, 400}:
            raise ValueError("data.num_demos must be one of 25, 50, 100, 200, 400")
        if self.data.val_num_demos < 1:
            raise ValueError("data.val_num_demos must be positive")
        policy = self.policy
        if min(policy.obs_horizon, policy.act_horizon, policy.pred_horizon) < 1:
            raise ValueError("all horizons must be positive")
        if policy.obs_horizon + policy.act_horizon - 1 > policy.pred_horizon:
            raise ValueError("need obs_horizon + act_horizon - 1 <= pred_horizon")
        if policy.num_inference_iters > policy.num_diffusion_iters:
            raise ValueError("inference diffusion iterations cannot exceed training iterations")
        if self.vision.feature_dim < 1 or self.vision.random_shift < 0:
            raise ValueError("invalid vision encoder settings")
        train = self.train
        if min(train.total_iters, train.batch_size, train.log_freq, train.resume_freq) < 1:
            raise ValueError("training counts must be positive")
        for step in [*train.validation_steps, *train.checkpoint_steps]:
            if step < 1:
                raise ValueError(f"intermediate step {step} must be positive")
        evaluation = self.eval
        if evaluation.val_episodes < 1 or evaluation.test_episodes < 1 or evaluation.num_envs < 1:
            raise ValueError("evaluation counts must be positive")
        if set(self.val_seeds()) & set(self.test_seeds()):
            raise ValueError("validation and test rollout seed ranges overlap")

    def val_seeds(self) -> list[int]:
        return list(range(self.eval.val_seed_start, self.eval.val_seed_start + self.eval.val_episodes))

    def test_seeds(self) -> list[int]:
        return list(range(self.eval.test_seed_start, self.eval.test_seed_start + self.eval.test_episodes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_T = TypeVar("_T")
_SECTIONS = {
    "task": TaskConfig,
    "data": DataConfig,
    "vision": VisionConfig,
    "policy": PolicyConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _section(cls: type[_T], raw: dict[str, Any], name: str) -> _T:
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise ValueError(f"config section {name!r} must be a table")
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
    return cls(**values)


def from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = Config(**{name: _section(cls, raw, name) for name, cls in _SECTIONS.items()})
    cfg.validate()
    return cfg


def load(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load TOML and apply repeatable ``section.key=value`` overrides."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    for item in overrides:
        key, separator, value = item.partition("=")
        section, dot, name = key.partition(".")
        if not separator or not dot or section not in _SECTIONS:
            raise ValueError(f"override must look like section.key=value: {item!r}")
        try:
            parsed = tomllib.loads(f"value = {value}")["value"]
        except tomllib.TOMLDecodeError:
            parsed = value
        raw.setdefault(section, {})[name] = parsed
    return from_dict(raw)
