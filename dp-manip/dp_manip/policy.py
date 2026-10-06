"""RGB- or state-conditioned Diffusion Policy using a shared training pipeline."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .backbones import build_noise_predictor
from .config import DiffusionConfig, PolicyConfig, VisionConfig, from_recorded
from .data import NormalizationStats
from .observation_encoder import ObservationEncoder, StateObservationEncoder


class DiffusionPolicy(nn.Module):
    """Encode the selected observation mode and denoise normalized actions."""

    def __init__(
        self,
        policy_cfg: PolicyConfig,
        vision_cfg: VisionConfig,
        diffusion_cfg: DiffusionConfig,
        *,
        image_shape: tuple[int, int, int] | None,
        proprio_dim: int,
        action_dim: int,
        stats: NormalizationStats,
        obs_mode: str = "rgb",
    ):
        super().__init__()
        self.obs_horizon = policy_cfg.obs_horizon
        self.act_horizon = policy_cfg.act_horizon
        self.pred_horizon = policy_cfg.pred_horizon
        self.action_dim = action_dim
        self.num_inference_iters = diffusion_cfg.num_inference_iters
        self.obs_mode = obs_mode
        if obs_mode == "state":
            if image_shape is not None:
                raise ValueError("state policy expects no image shape")
            self.observation_encoder = StateObservationEncoder(
                obs_horizon=policy_cfg.obs_horizon, state_dim=proprio_dim, stats=stats
            )
        elif obs_mode == "rgb":
            if image_shape is None:
                raise ValueError("RGB policy requires an image shape")
            self.observation_encoder = ObservationEncoder(
                vision_cfg,
                obs_horizon=policy_cfg.obs_horizon,
                image_shape=image_shape,
                proprio_dim=proprio_dim,
                stats=stats,
            )
        else:
            raise ValueError("obs_mode must be rgb or state")
        # The backbone consumes the shared ``(B, To, Dobs)`` sequence and decides
        # how to condition on it; the policy itself is backbone-agnostic.
        self.noise_predictor = build_noise_predictor(
            policy_cfg.backbone,
            policy_cfg,
            obs_dim=self.observation_encoder.output_dim,
            action_dim=action_dim,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_cfg.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.register_buffer("action_low", torch.as_tensor(stats.action_low))
        self.register_buffer("action_high", torch.as_tensor(stats.action_high))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        device: str | torch.device = "cpu",
    ) -> "DiffusionPolicy":
        """Build an inference policy from a training checkpoint payload.

        This is the single loader used by ``scripts/eval_dp.py`` and the
        lifecycle tests, so every consumer reads the config, normalization
        stats, dataset shapes and (possibly legacy) model keys exactly the way
        the trainer wrote them.
        """
        cfg = from_recorded(checkpoint["config"])
        train_data = checkpoint["train_data"]
        if train_data.get("obs_mode", "rgb") != cfg.task.obs_mode:
            raise ValueError("checkpoint observation mode differs from dataset metadata")
        image_shape = train_data["image_shape"]
        policy = cls(
            cfg.policy,
            cfg.vision,
            cfg.diffusion,
            image_shape=tuple(image_shape) if image_shape is not None else None,
            proprio_dim=int(train_data["proprio_dim"]),
            action_dim=int(train_data["action_dim"]),
            stats=NormalizationStats.from_dict(checkpoint["normalization"]),
            obs_mode=cfg.task.obs_mode,
        )
        load_policy_state_dict(policy, checkpoint["model"])
        return policy.to(device)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low

    def observation_features(self, rgb: torch.Tensor | None, proprio: torch.Tensor) -> torch.Tensor:
        """Return shared observation features shaped ``(B, To, Dobs)``."""
        return self.observation_encoder(rgb, proprio)

    def compute_loss(
        self,
        rgb: torch.Tensor | None,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        obs_features = self.observation_features(rgb, proprio)
        actions = self.normalize_action(actions.to(dtype=torch.float32))
        noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=generator)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            generator=generator,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.noise_predictor(noisy_actions, timesteps, obs_features)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def get_action(
        self,
        rgb: torch.Tensor | None,
        proprio: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``(B, act_horizon, action_dim)`` in the environment's units."""
        return self.sample_actions(self.observation_features(rgb, proprio), generator=generator)

    @torch.no_grad()
    def sample_actions(
        self,
        obs_features: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Denoise one action chunk from ``(B, To, Dobs)`` observation features.

        ``noise_fn(sample, timestep)`` replaces this policy's own noise
        prediction at every denoising step; everything else (initial noise,
        scheduler, RNG consumption, action slicing) is shared, so a wrapper that
        returns this policy's prediction reproduces :meth:`get_action` exactly.
        """
        device = obs_features.device
        sample = torch.randn(
            (obs_features.shape[0], self.pred_horizon, self.action_dim),
            device=device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_iters, device=device)
        # Unlike add_noise(), DDPMScheduler.step() does not move these tensors
        # to the sample device. A freshly loaded evaluation-only policy has not
        # called add_noise(), so move them explicitly before CUDA sampling.
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        self.noise_scheduler.one = self.noise_scheduler.one.to(device)
        for timestep in self.noise_scheduler.timesteps:
            if noise_fn is None:
                prediction = self.noise_predictor(sample, timestep, obs_features)
            else:
                prediction = noise_fn(sample, timestep)
            sample = self.noise_scheduler.step(
                prediction, timestep, sample, generator=generator
            ).prev_sample
        start = self.obs_horizon - 1
        return self.unnormalize_action(sample[:, start : start + self.act_horizon])


def num_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def adapt_legacy_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Map pre-refactor keys onto the current module layout.

    Inference checkpoints, resume checkpoints, and EMA shadows written before
    the observation encoder was extracted keep the camera weights under
    ``image_encoders.*`` and the proprio buffers at the top level; version-1
    checkpoints used the legacy ``state_*`` vocabulary. Checkpoints written
    before the backbone interface name the UNet directly as ``noise_pred_net``.
    """
    adapted = state_dict.copy()
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        adapted._metadata = metadata
    for legacy, canonical in (
        ("state_mean", "observation_encoder.proprio_mean"),
        ("state_std", "observation_encoder.proprio_std"),
        ("proprio_mean", "observation_encoder.proprio_mean"),
        ("proprio_std", "observation_encoder.proprio_std"),
    ):
        if legacy in adapted:
            if canonical in adapted:
                raise ValueError(f"checkpoint contains both {legacy!r} and {canonical!r}")
            adapted[canonical] = adapted.pop(legacy)
    for key in [key for key in adapted if key.startswith("image_encoders.")]:
        canonical = "observation_encoder." + key
        if canonical in adapted:
            raise ValueError(f"checkpoint contains both {key!r} and {canonical!r}")
        adapted[canonical] = adapted.pop(key)
    for key in [key for key in adapted if key.startswith("noise_pred_net.")]:
        canonical = "noise_predictor.unet." + key[len("noise_pred_net.") :]
        if canonical in adapted:
            raise ValueError(f"checkpoint contains both {key!r} and {canonical!r}")
        adapted[canonical] = adapted.pop(key)
    return adapted


def load_policy_state_dict(policy: DiffusionPolicy, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True):
    """Load a policy, adapting pre-refactor checkpoint key names."""
    return policy.load_state_dict(adapt_legacy_state_dict(state_dict), strict=strict)
