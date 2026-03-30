"""
PSAN – Pyramid Squeeze Attention Network classification module.

Replaces the YOLOv5 detection head with a classification head designed for
fine-grained plant disease recognition.

Architecture (as described in the paper):
    P3, P4, P5  (from DFN)
        │
    Upsample P4/P5 → P3 spatial size → Concat  [C3+C4+C5 channels]
        │
    1×1 Conv  →  fused_channels
        │
    ┌───────────────────────────────────────┐
    │  Multi-scale Conv Branches            │
    │  (parallel 3×3 · 5×5 · 7×7 · 9×9)   │
    └───────────────────────────────────────┘
        │  concat → 4 × fused_channels
        │
    SEWeight channel recalibration
        │
    Softmax-based channel normalisation
        │
    PSA Pyramid Squeeze Attention
        │
    Global Average Pooling  →  (B, 4×fused_channels)
        │
    Dropout
        │
    Linear → num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .psa import PSA


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution to keep large-kernel branches efficient."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class SEWeight(nn.Module):
    """
    Squeeze-and-Excitation channel weighting with Softmax recalibration.

    Unlike standard SE (which uses Sigmoid), this variant applies Softmax
    across the channel dimension so attention weights sum to 1 – this
    'recalibrates' rather than just 'gates' the channels and has been shown
    to improve training stability for multi-scale feature aggregation.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.gap(x).flatten(1)              # (B, C)
        z = self.act(self.fc1(z))               # (B, reduced)
        w = F.softmax(self.fc2(z), dim=1)       # (B, C) – Softmax recalibration
        return x * w.view(b, c, 1, 1)


class PSAN(nn.Module):
    """
    PSA-Network classification head with 4 multi-scale conv branches,
    SEWeight channel recalibration, and Pyramid Squeeze Attention.

    Args:
        in_channels: List of three channel counts [c3, c4, c5] matching the
            DFN output.
        num_classes: Number of output classes.
        reduction: Channel reduction ratio for SEWeight and PSA (default 16).
        pyramid_levels: Pooling sizes for PSA (default ``[1, 2, 4, 8]``).
        dropout: Dropout probability before the final linear layer
            (default 0.3).
    """

    def __init__(
        self,
        in_channels: list,
        num_classes: int,
        reduction: int = 16,
        pyramid_levels: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if pyramid_levels is None:
            pyramid_levels = [1, 2, 4, 8]

        total_channels = sum(in_channels)

        # Initial channel fusion: concat(P3, P4↑, P5↑) → fused_channels
        fused_channels = max(total_channels // 2, 64)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.SiLU(inplace=True),
        )

        # 4 parallel multi-scale conv branches (3×3, 5×5, 7×7, 9×9)
        # All use depthwise-separable convolutions to keep parameter count low.
        self.branch3 = _DepthwiseSeparableConv(fused_channels, kernel_size=3)
        self.branch5 = _DepthwiseSeparableConv(fused_channels, kernel_size=5)
        self.branch7 = _DepthwiseSeparableConv(fused_channels, kernel_size=7)
        self.branch9 = _DepthwiseSeparableConv(fused_channels, kernel_size=9)

        # After concatenating the 4 branches: 4 × fused_channels
        ms_channels = fused_channels * 4

        # SEWeight channel recalibration with Softmax
        self.se_weight = SEWeight(ms_channels, reduction=reduction)

        # 1×1 conv to project back to fused_channels before PSA
        self.project = nn.Sequential(
            nn.Conv2d(ms_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.SiLU(inplace=True),
        )

        # Pyramid Squeeze Attention
        self.psa = PSA(fused_channels, reduction=reduction, pyramid_levels=pyramid_levels)

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(fused_channels, num_classes)

    def forward(self, features) -> torch.Tensor:
        """
        Args:
            features: Tuple/list of three tensors (P3, P4, P5) from the DFN.

        Returns:
            Class logits tensor of shape ``(B, num_classes)``.
        """
        p3, p4, p5 = features
        target_size = p3.shape[2:]

        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)

        # Fusion
        fused = torch.cat([p3, p4_up, p5_up], dim=1)   # (B, C3+C4+C5, H, W)
        fused = self.fusion_conv(fused)                  # (B, fused_ch, H, W)

        # Multi-scale branches
        b3 = self.branch3(fused)
        b5 = self.branch5(fused)
        b7 = self.branch7(fused)
        b9 = self.branch9(fused)
        ms = torch.cat([b3, b5, b7, b9], dim=1)         # (B, 4*fused_ch, H, W)

        # SEWeight + Softmax recalibration
        ms = self.se_weight(ms)                          # (B, 4*fused_ch, H, W)

        # Project back & apply PSA
        ms = self.project(ms)                            # (B, fused_ch, H, W)
        ms = self.psa(ms)                                # (B, fused_ch, H, W)

        # Global pooling → classifier
        out = self.gap(ms).flatten(1)                    # (B, fused_ch)
        out = self.dropout(out)
        return self.classifier(out)                      # (B, num_classes)
