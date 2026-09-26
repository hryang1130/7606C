"""RGB ManiSkill evaluation environments matching ``maniskill-demogen``."""

from __future__ import annotations

import os
import sys
from typing import Any

import gymnasium as gym

from .config import Config


NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
LAVAPIPE_ICD = "/usr/share/vulkan/icd.d/lvp_icd.json"


def ensure_render_icd() -> None:
    """Use lavapipe only when Linux has no configured NVIDIA Vulkan ICD."""
    if sys.platform != "linux" or os.environ.get("VK_ICD_FILENAMES") or os.path.isfile(NVIDIA_ICD):
        return
    if os.path.isfile(LAVAPIPE_ICD):
        os.environ["VK_ICD_FILENAMES"] = LAVAPIPE_ICD


def environment_kwargs(cfg: Config, render_backend: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "obs_mode": "rgb",
        "control_mode": cfg.task.control_mode,
        "reward_mode": "sparse",
        "sim_backend": cfg.task.sim_backend,
        "max_episode_steps": cfg.task.max_episode_steps,
        "reconfiguration_freq": 1,
        "sensor_configs": {"shader_pack": cfg.task.shader_pack},
    }
    if render_backend is not None:
        kwargs["render_backend"] = render_backend
    return kwargs


def make_eval_envs(cfg: Config, num_envs: int, render_backend: str | None = None):
    """Create process-vectorized CPU-physics RGB environments.

    Physics stays on ``physx_cpu`` because the demonstrations were generated
    there and ManiSkill's CPU/GPU backends do not produce identical initial
    states for a fixed seed. The policy still runs on CUDA.
    """
    if cfg.task.sim_backend != "physx_cpu":
        raise ValueError("fair evaluation requires physx_cpu, matching the generated data")
    ensure_render_icd()
    import mani_skill.envs  # noqa: F401  registers environment IDs
    from mani_skill.utils.wrappers import CPUGymWrapper
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    def make():
        def thunk():
            env = gym.make(cfg.task.env_id, **environment_kwargs(cfg, render_backend))
            env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
            return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

        return thunk

    constructors = [make() for _ in range(num_envs)]
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(constructors)
    return gym.vector.AsyncVectorEnv(constructors, context="forkserver")
