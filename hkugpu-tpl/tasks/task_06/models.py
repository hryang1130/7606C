from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timestep.device) * -scale)
        phase = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return F.pad(embedding, (0, self.dim - embedding.shape[-1]))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(F.silu(cond)).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class MLPDenoiser(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, horizon: int, width: int = 512):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.time = SinusoidalEmbedding(128)
        self.condition = nn.Sequential(nn.Linear(obs_dim + 128, width), nn.SiLU(), nn.Linear(width, width))
        self.net = nn.Sequential(
            nn.Linear(horizon * action_dim + width, width * 2), nn.SiLU(),
            nn.Linear(width * 2, width * 2), nn.SiLU(),
            nn.Linear(width * 2, horizon * action_dim),
        )

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = self.condition(torch.cat((obs.flatten(1), self.time(timestep)), dim=-1))
        values = self.net(torch.cat((noisy_actions.flatten(1), cond), dim=-1))
        return values.view_as(noisy_actions)


class UNetDenoiser(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, width: int = 128):
        super().__init__()
        cond_dim = width * 4
        self.time = SinusoidalEmbedding(width * 2)
        self.condition = nn.Sequential(nn.Linear(obs_dim + width * 2, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.input = nn.Conv1d(action_dim, width, 3, padding=1)
        self.down1 = ResidualBlock1D(width, width, cond_dim)
        self.downsample1 = nn.Conv1d(width, width * 2, 4, stride=2, padding=1)
        self.down2 = ResidualBlock1D(width * 2, width * 2, cond_dim)
        self.downsample2 = nn.Conv1d(width * 2, width * 3, 4, stride=2, padding=1)
        self.middle = ResidualBlock1D(width * 3, width * 3, cond_dim)
        self.up2 = nn.ConvTranspose1d(width * 3, width * 2, 4, stride=2, padding=1)
        self.up_block2 = ResidualBlock1D(width * 4, width * 2, cond_dim)
        self.up1 = nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1)
        self.up_block1 = ResidualBlock1D(width * 2, width, cond_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, action_dim, 3, padding=1))

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = self.condition(torch.cat((obs.flatten(1), self.time(timestep)), dim=-1))
        x0 = self.down1(self.input(noisy_actions.transpose(1, 2)), cond)
        x1 = self.down2(self.downsample1(x0), cond)
        x2 = self.middle(self.downsample2(x1), cond)
        x = self.up2(x2)
        x = self.up_block2(torch.cat((x, x1), dim=1), cond)
        x = self.up1(x)
        x = self.up_block1(torch.cat((x, x0), dim=1), cond)
        return self.output(x).transpose(1, 2)


class TransformerDenoiser(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, horizon: int, width: int = 256, layers: int = 6, heads: int = 8):
        super().__init__()
        self.action_in = nn.Linear(action_dim, width)
        self.position = nn.Parameter(torch.zeros(1, horizon, width))
        nn.init.normal_(self.position, std=0.02)
        self.time = SinusoidalEmbedding(width)
        self.condition = nn.Sequential(nn.Linear(obs_dim + width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = self.condition(torch.cat((obs.flatten(1), self.time(timestep)), dim=-1))
        tokens = self.transformer(self.action_in(noisy_actions) + self.position[:, :noisy_actions.shape[1]] + cond.unsqueeze(1))
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        return self.output(tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


def make_denoiser(name: str, obs_dim: int, action_dim: int, horizon: int) -> nn.Module:
    if name == "mlp":
        return MLPDenoiser(obs_dim, action_dim, horizon)
    if name == "unet":
        return UNetDenoiser(obs_dim, action_dim)
    if name == "transformer":
        return TransformerDenoiser(obs_dim, action_dim, horizon)
    raise ValueError(f"Unknown architecture {name!r}; choose mlp, unet, or transformer")
