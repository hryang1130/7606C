"""Small dependency-free ResNet-18 encoder for RGB observations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VISION_POOLS


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + residual)


class SpatialSoftmax(nn.Module):
    """robomimic-style spatial soft-argmax over a feature map.

    A 1x1 convolution maps the input channels to ``num_keypoints`` attention
    maps; a softmax over the ``H*W`` positions (learnable temperature, initial
    value 1) turns each map into a distribution whose expected ``(x, y)`` image
    coordinate in ``[-1, 1]`` is that keypoint. Returns ``(N, num_keypoints, 2)``.

    The softmax and the coordinate expectation run in fp32 even under AMP.
    The coordinate grid follows the input's spatial size, so no parameter
    depends on the image resolution.
    """

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.keypoints = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start the attention near-uniform, like robomimic's SpatialSoftmax.

        The keypoint conv must not inherit :meth:`ResNet18Encoder._initialize`:
        its ``kaiming_normal_(mode="fan_out")`` gives a ``512 -> K`` 1x1 conv a
        weight std of ``sqrt(2 / K)`` (~0.25 here), ten times PyTorch's default.
        The logits then start large enough that the softmax saturates to a
        one-hot pick, and a saturated soft-argmax has *exactly zero* gradient
        with respect to its logits, so the keypoints freeze wherever they
        happened to be and the camera branch collapses to a constant.

        That is not hypothetical: the first vision-pool run died this way. Its
        checkpoints show softmax entropy 0.000 of a possible 2.773 nats,
        keypoint coordinates with std 0.00000 across 24 windows from 50 demos,
        and a keypoint-conv gradient ~1/200 of the adjacent projection's
        (``report/exp_vision_pool_n100.md`` section 5). PyTorch's default
        ``kaiming_uniform_(a=sqrt(5))`` keeps the initial logits small.
        """
        nn.init.kaiming_uniform_(self.keypoints.weight, a=math.sqrt(5))
        nn.init.zeros_(self.keypoints.bias)
        nn.init.ones_(self.temperature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        logits = self.keypoints(feature)
        count, _, height, width = logits.shape
        with torch.amp.autocast(logits.device.type, enabled=False):
            logits = logits.float().reshape(count, self.num_keypoints, height * width)
            # Floor the temperature: shrinking it is another way to saturate the
            # softmax and kill the coordinate gradient.
            attention = F.softmax(logits / self.temperature.float().clamp(min=0.1), dim=-1)
            pos_y, pos_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, device=logits.device),
                torch.linspace(-1.0, 1.0, width, device=logits.device),
                indexing="ij",
            )
            positions = torch.stack((pos_x.reshape(-1), pos_y.reshape(-1)), dim=-1)
            return attention @ positions


class ResNet18Encoder(nn.Module):
    """ResNet-18 with GroupNorm and a compact output projection.

    ``pool`` selects how the final ``512``-channel map becomes a vector:

    * ``"avg"``: global average pooling, then ``Linear(512, feature_dim)``
      (the baseline; its parameter names are unchanged, so older checkpoints
      load as-is);
    * ``"spatial_softmax"``: :class:`SpatialSoftmax` with ``num_keypoints``
      keypoints, flattened to ``2 * num_keypoints`` coordinates, then
      ``Linear(2 * num_keypoints, feature_dim)``.

    Both produce ``feature_dim`` features, so nothing downstream changes.

    Limitation: like robomimic's ResNet18Conv, the spatial softmax reads
    ``layer4``, which is only 4x4 for the 128x128 camera images used here
    (total stride 32). The expected coordinates are still continuous, but the
    attention has only 16 positions to choose from; whether to read ``layer3``
    (8x8) instead is a separate decision.
    """

    def __init__(self, feature_dim: int, *, pool: str, num_keypoints: int):
        super().__init__()
        if pool not in VISION_POOLS:
            raise ValueError(f"unknown vision pool {pool!r}; expected one of {list(VISION_POOLS)}")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._layer(64, 64, stride=1)
        self.layer2 = self._layer(64, 128, stride=2)
        self.layer3 = self._layer(128, 256, stride=2)
        self.layer4 = self._layer(256, 512, stride=2)
        if pool == "avg":
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, feature_dim))
        else:
            self.head = nn.Sequential(
                SpatialSoftmax(512, num_keypoints),
                nn.Flatten(),
                nn.Linear(2 * num_keypoints, feature_dim),
            )
        self.apply(self._initialize)
        if pool != "avg":
            # ``_initialize`` above is the ResNet scheme and also matches the
            # keypoint conv; restore the head's own init (see
            # SpatialSoftmax.reset_parameters).
            self.head[0].reset_parameters()

    @staticmethod
    def _layer(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(BasicBlock(in_channels, out_channels, stride), BasicBlock(out_channels, out_channels))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def feature_map(self, image: torch.Tensor) -> torch.Tensor:
        """Return the ``layer4`` map the head pools, ``(N, 512, H/32, W/32)``."""
        value = self.stem(image)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        return self.layer4(value)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_map(image))


def random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply independent DrQ-style integer translations to NCHW images."""
    if pad == 0:
        return images
    count, _, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    padded_width = width + 2 * pad
    padded_height = height + 2 * pad
    # Base grid is the top-left HxW crop of the padded image. Integer shifts
    # move that crop through all (2*pad+1)^2 valid positions without scaling.
    base_x = (
        torch.arange(width, device=images.device, dtype=images.dtype) * (2.0 / padded_width)
        - 1.0
        + 1.0 / padded_width
    )
    base_y = (
        torch.arange(height, device=images.device, dtype=images.dtype) * (2.0 / padded_height)
        - 1.0
        + 1.0 / padded_height
    )
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(count, 1, 1, 1)
    shifts = torch.randint(0, 2 * pad + 1, (count, 2), device=images.device).to(images.dtype)
    grid[..., 0] += shifts[:, 0, None, None] * (2.0 / padded_width)
    grid[..., 1] += shifts[:, 1, None, None] * (2.0 / padded_height)
    return F.grid_sample(padded, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
