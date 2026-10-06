"""Shared RGB + proprioception observation encoder.

This module is the single observation boundary of the policy: every experiment
feeds ``(rgb, proprio)`` histories through the same encoder and receives a
shared feature sequence of shape ``(B, To, Dobs)``. It must not know how a
noise-prediction backbone consumes those features; flattening, tokenizing, or
pooling belongs to the backbone adapter (see the noise predictor interface) so
that UNet / Transformer / MLP experiments keep one visual encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VisionConfig
from .data import NormalizationStats
from .vision import ResNet18Encoder, random_shift


class ObservationEncoder(nn.Module):
    """Encode RGB and proprioception histories into ``(B, To, Dobs)``.

    ``Dobs = num_cameras * feature_dim + proprio_dim``. RGB is scaled to
    ``[-1, 1]``, optionally augmented with DrQ-style random shifts while
    training, encoded per camera (shared or per-camera weights), and
    concatenated with clamped, z-scored proprioception.
    """

    def __init__(
        self,
        vision_cfg: VisionConfig,
        *,
        obs_horizon: int,
        image_shape: tuple[int, int, int],
        proprio_dim: int,
        stats: NormalizationStats,
    ):
        super().__init__()
        height, width, channels = image_shape
        if height < 32 or width < 32 or channels % 3:
            raise ValueError(f"invalid concatenated camera image shape: {image_shape}")
        if obs_horizon < 1:
            raise ValueError("obs_horizon must be positive")
        if proprio_dim < 1:
            raise ValueError("proprio_dim must be positive")
        self.obs_horizon = obs_horizon
        self.num_cameras = channels // 3
        self.camera_feature_dim = vision_cfg.feature_dim
        self.proprio_dim = proprio_dim
        self.random_shift_pad = vision_cfg.random_shift
        self.share_camera_encoder = vision_cfg.share_camera_encoder
        def image_encoder() -> ResNet18Encoder:
            return ResNet18Encoder(
                vision_cfg.feature_dim, pool=vision_cfg.pool, num_keypoints=vision_cfg.num_keypoints
            )

        if self.share_camera_encoder:
            self.image_encoders = nn.ModuleList([image_encoder()])
        else:
            self.image_encoders = nn.ModuleList(image_encoder() for _ in range(self.num_cameras))
        self.register_buffer("proprio_mean", torch.as_tensor(stats.proprio_mean))
        self.register_buffer("proprio_std", torch.as_tensor(stats.proprio_std))

    @property
    def output_dim(self) -> int:
        """Feature width ``Dobs`` of the returned observation sequence."""
        return self.num_cameras * self.camera_feature_dim + self.proprio_dim

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return ((proprio - self.proprio_mean) / self.proprio_std).clamp(-10.0, 10.0)

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return per-timestep camera features shaped ``(B, To, cameras * Dcam)``."""
        batch, horizon, _, height, width = rgb.shape
        images = rgb.to(dtype=torch.float32).div_(127.5).sub_(1.0)
        images = images.reshape(batch * horizon * self.num_cameras, 3, height, width)
        if self.training and self.random_shift_pad:
            images = random_shift(images, self.random_shift_pad)
        if self.share_camera_encoder:
            features = self.image_encoders[0](images)
        else:
            by_camera = images.reshape(batch * horizon, self.num_cameras, 3, height, width)
            encoded = [
                encoder(by_camera[:, index]) for index, encoder in enumerate(self.image_encoders)
            ]
            features = torch.stack(encoded, dim=1)
        return features.reshape(batch, horizon, self.num_cameras * self.camera_feature_dim)

    def forward(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Return shared observation features shaped ``(B, To, Dobs)``."""
        if rgb.ndim != 5 or proprio.ndim != 3:
            raise ValueError(
                f"expected RGB/proprio histories, got {tuple(rgb.shape)} and {tuple(proprio.shape)}"
            )
        batch, horizon, channels, _, _ = rgb.shape
        if horizon != self.obs_horizon or channels != self.num_cameras * 3:
            raise ValueError(f"unexpected RGB history shape {tuple(rgb.shape)}")
        if proprio.shape[0] != batch or proprio.shape[1] != horizon:
            raise ValueError(f"proprio does not match the RGB history: {tuple(proprio.shape)}")
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio width {proprio.shape[-1]} does not match encoder width {self.proprio_dim}"
            )
        features = self.encode_rgb(rgb)
        proprio = self.normalize_proprio(proprio.to(dtype=torch.float32))
        return torch.cat((features, proprio), dim=-1)


class StateObservationEncoder(nn.Module):
    """Normalize complete state histories without constructing a visual encoder."""

    def __init__(self, *, obs_horizon: int, state_dim: int, stats: NormalizationStats):
        super().__init__()
        if obs_horizon < 1 or state_dim < 1:
            raise ValueError("state_dim and obs_horizon must be positive")
        if stats.proprio_mean.shape != (state_dim,) or stats.proprio_std.shape != (state_dim,):
            raise ValueError("normalization statistics do not match the complete state width")
        self.obs_horizon = obs_horizon
        self.output_dim = state_dim
        self.register_buffer("proprio_mean", torch.as_tensor(stats.proprio_mean))
        self.register_buffer("proprio_std", torch.as_tensor(stats.proprio_std))

    def forward(self, rgb: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        if rgb is not None:
            raise ValueError("state policy expects no RGB input")
        if state.ndim != 3 or state.shape[1:] != (self.obs_horizon, self.output_dim):
            raise ValueError(f"expected (B,{self.obs_horizon},{self.output_dim}) state history, got {tuple(state.shape)}")
        return ((state.float() - self.proprio_mean) / self.proprio_std).clamp(-10.0, 10.0)
