"""RGB-conditioned Diffusion Policy with a ResNet-18 observation encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .conditional_unet1d import ConditionalUnet1D
from .config import PolicyConfig, VisionConfig
from .data import NormalizationStats
from .vision import ResNet18Encoder, random_shift


class DiffusionPolicy(nn.Module):
    """Encode RGB + proprioception and denoise a normalized action sequence."""

    def __init__(
        self,
        policy_cfg: PolicyConfig,
        vision_cfg: VisionConfig,
        *,
        image_shape: tuple[int, int, int],
        proprio_dim: int,
        action_dim: int,
        stats: NormalizationStats,
    ):
        super().__init__()
        height, width, channels = image_shape
        if height < 32 or width < 32 or channels % 3:
            raise ValueError(f"invalid concatenated camera image shape: {image_shape}")
        self.obs_horizon = policy_cfg.obs_horizon
        self.act_horizon = policy_cfg.act_horizon
        self.pred_horizon = policy_cfg.pred_horizon
        self.action_dim = action_dim
        self.num_cameras = channels // 3
        self.num_inference_iters = policy_cfg.num_inference_iters
        self.random_shift_pad = vision_cfg.random_shift
        self.share_camera_encoder = vision_cfg.share_camera_encoder

        if self.share_camera_encoder:
            self.image_encoders = nn.ModuleList([ResNet18Encoder(vision_cfg.feature_dim)])
        else:
            self.image_encoders = nn.ModuleList(
                ResNet18Encoder(vision_cfg.feature_dim) for _ in range(self.num_cameras)
            )
        observation_dim = self.num_cameras * vision_cfg.feature_dim + proprio_dim
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=policy_cfg.obs_horizon * observation_dim,
            diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
            down_dims=policy_cfg.unet_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=policy_cfg.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.register_buffer("state_mean", torch.as_tensor(stats.state_mean))
        self.register_buffer("state_std", torch.as_tensor(stats.state_std))
        self.register_buffer("action_low", torch.as_tensor(stats.action_low))
        self.register_buffer("action_high", torch.as_tensor(stats.action_high))

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return ((state - self.state_mean) / self.state_std).clamp(-10.0, 10.0)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low

    def encode_observation(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return flattened conditioning from ``(B,To,3C,H,W)`` and ``(B,To,P)``."""
        if rgb.ndim != 5 or state.ndim != 3:
            raise ValueError(f"expected RGB/state histories, got {tuple(rgb.shape)} and {tuple(state.shape)}")
        batch, horizon, channels, height, width = rgb.shape
        if horizon != self.obs_horizon or channels != self.num_cameras * 3:
            raise ValueError(f"unexpected RGB history shape {tuple(rgb.shape)}")
        images = rgb.to(dtype=torch.float32).div_(127.5).sub_(1.0)
        images = images.reshape(batch * horizon, self.num_cameras, 3, height, width)
        images = images.reshape(batch * horizon * self.num_cameras, 3, height, width)
        if self.training and self.random_shift_pad:
            images = random_shift(images, self.random_shift_pad)
        if self.share_camera_encoder:
            features = self.image_encoders[0](images)
        else:
            by_camera = images.reshape(batch * horizon, self.num_cameras, 3, height, width)
            encoded = [encoder(by_camera[:, index]) for index, encoder in enumerate(self.image_encoders)]
            features = torch.stack(encoded, dim=1).reshape(batch * horizon * self.num_cameras, -1)
        features = features.reshape(batch, horizon, -1)
        state = self.normalize_state(state.to(dtype=torch.float32))
        return torch.cat((features, state), dim=-1).flatten(start_dim=1)

    def compute_loss(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.encode_observation(rgb, state)
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
        prediction = self.noise_pred_net(noisy_actions, timesteps, global_cond=condition)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def get_action(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``(B, act_horizon, action_dim)`` in the environment's units."""
        condition = self.encode_observation(rgb, state)
        sample = torch.randn(
            (rgb.shape[0], self.pred_horizon, self.action_dim),
            device=rgb.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_iters, device=rgb.device)
        # Unlike add_noise(), DDPMScheduler.step() does not move these tensors
        # to the sample device. A freshly loaded evaluation-only policy has not
        # called add_noise(), so move them explicitly before CUDA sampling.
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(rgb.device)
        self.noise_scheduler.one = self.noise_scheduler.one.to(rgb.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.noise_pred_net(sample, timestep, global_cond=condition)
            sample = self.noise_scheduler.step(
                prediction, timestep, sample, generator=generator
            ).prev_sample
        start = self.obs_horizon - 1
        return self.unnormalize_action(sample[:, start : start + self.act_horizon])


def num_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
