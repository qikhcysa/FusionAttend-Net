"""
Attention Mechanisms for Comparison Experiments.

Implements the seven attention modules used in the ablation study:
    * SE    – Squeeze-and-Excitation (Hu et al., 2018)
    * ECA   – Efficient Channel Attention (Wang et al., 2020)
    * ESE   – Effective Squeeze-and-Excitation (Lee et al., 2019)
    * CBAM  – Convolutional Block Attention Module (Woo et al., 2018)
    * CA    – Coordinate Attention (Hou et al., 2021)
    * ParNet – Parallel Network attention
    * PSA   – Pyramid Squeeze Attention (this work)

All modules follow a common interface: they take a ``(B, C, H, W)``
feature tensor and return a recalibrated tensor of the same shape.

Usage in comparison experiments::

    from models.attention import build_attention
    attn = build_attention("cbam", channels=256)
    out = attn(feature_map)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SE – Squeeze-and-Excitation ────────────────────────────────────────────

class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.gap(x).flatten(1)
        return x * self.fc(z).view(b, c, 1, 1)


# ── ECA – Efficient Channel Attention ─────────────────────────────────────

class ECA(nn.Module):
    """
    Efficient Channel Attention with adaptive 1-D convolution kernel size.
    Avoids dimensionality reduction; uses a single small conv over the GAP
    descriptor.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gap(x).squeeze(-1).transpose(-1, -2)   # (B, 1, C)
        z = self.sigmoid(self.conv(z)).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * z


# ── ESE – Effective Squeeze-and-Excitation ─────────────────────────────────

class ESE(nn.Module):
    """
    Effective SE: replaces the two-layer FC bottleneck with a single
    1×1 conv applied after GAP, avoiding the reduction ratio parameter.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(self.gap(x)))


# ── CBAM – Convolutional Block Attention Module ───────────────────────────

class _ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_z = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_z = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg_z + max_z).unsqueeze(-1).unsqueeze(-1)


class _SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return gate


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_gate = _ChannelGate(channels, reduction)
        self.spatial_gate = _SpatialGate(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x


# ── CA – Coordinate Attention ──────────────────────────────────────────────

class CA(nn.Module):
    """
    Coordinate Attention embeds spatial location information into channel
    attention by pooling separately along H and W dimensions.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))          # (B, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)                # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = y.split([h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))            # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))  # (B, C, 1, W)
        return x * a_h * a_w


# ── ParNet Attention ───────────────────────────────────────────────────────

class ParNet(nn.Module):
    """
    Parallel Network attention: combines a depth-wise conv branch with a
    squeeze-excitation branch and adds them before sigmoid gating.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)

        # Spatial branch (depth-wise conv)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Channel branch (SE-style)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.dw_conv(x) + self.se(x))


# ── Factory ────────────────────────────────────────────────────────────────

_ATTENTION_REGISTRY: dict[str, type] = {
    "se":      SE,
    "eca":     ECA,
    "ese":     ESE,
    "cbam":    CBAM,
    "ca":      CA,
    "parnet":  ParNet,
}


def build_attention(name: str, channels: int, **kwargs) -> nn.Module:
    """
    Instantiate an attention module by name.

    Args:
        name: One of ``"se"``, ``"eca"``, ``"ese"``, ``"cbam"``,
              ``"ca"``, ``"parnet"``, ``"psa"`` (case-insensitive).
        channels: Number of feature channels.
        **kwargs: Additional keyword arguments forwarded to the module
            constructor (e.g. ``reduction=8``).

    Returns:
        An :class:`nn.Module` that maps ``(B, C, H, W)`` → ``(B, C, H, W)``.

    Raises:
        ValueError: If *name* is not recognised.
    """
    name_lower = name.lower()
    if name_lower == "psa":
        from models.psa import PSA
        return PSA(channels=channels, **kwargs)
    if name_lower == "none" or name_lower == "identity":
        return nn.Identity()
    if name_lower not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention '{name}'. "
            f"Available: {sorted(_ATTENTION_REGISTRY) + ['psa', 'none']}"
        )
    return _ATTENTION_REGISTRY[name_lower](channels, **kwargs)


AVAILABLE_ATTENTIONS = sorted(_ATTENTION_REGISTRY.keys()) + ["psa"]
