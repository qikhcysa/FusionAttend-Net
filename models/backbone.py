"""
YOLOv5-based Backbone for DFN feature extraction.

Implements the convolutional stem, C3 blocks, and SPPF (Spatial Pyramid
Pooling Fast) that form the feature extraction backbone.
"""

import torch
import torch.nn as nn


def autopad(kernel_size, padding=None):
    """Compute 'same' padding for a convolution kernel."""
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding


class Conv(nn.Module):
    """Standard convolution: Conv + BN + SiLU."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            autopad(kernel_size, padding), groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard YOLOv5 bottleneck block."""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(hidden, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Cross Stage Partial bottleneck with 3 convolutions (C3)."""

    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(in_channels, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, out_channels, 1, 1)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut, groups, expansion=1.0) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (SPPF) – replaces SPP with sequential pooling."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = Conv(in_channels, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class YOLOv5Backbone(nn.Module):
    """
    YOLOv5-style backbone for multi-scale feature extraction.

    Produces three feature maps at 1/8, 1/16 and 1/32 of the input
    resolution which are consumed by :class:`YOLOv5Neck`.

    Args:
        in_channels: Number of input image channels (default 3).
        width_multiple: Channel width multiplier (default 0.5, i.e. YOLOv5-s).
        depth_multiple: Block depth multiplier (default 0.33, i.e. YOLOv5-s).
    """

    def __init__(self, in_channels: int = 3, width_multiple: float = 0.5, depth_multiple: float = 0.33):
        super().__init__()

        def ch(c):
            return max(round(c * width_multiple), 1)

        def nl(n):
            return max(round(n * depth_multiple), 1)

        # Stem: 640 -> 320
        self.stem = Conv(in_channels, ch(64), 6, 2, 2)

        # Stage 1: 320 -> 160
        self.stage1 = nn.Sequential(
            Conv(ch(64), ch(128), 3, 2),
            C3(ch(128), ch(128), nl(3)),
        )

        # Stage 2: 160 -> 80  (P3)
        self.stage2 = nn.Sequential(
            Conv(ch(128), ch(256), 3, 2),
            C3(ch(256), ch(256), nl(6)),
        )

        # Stage 3: 80 -> 40  (P4)
        self.stage3 = nn.Sequential(
            Conv(ch(256), ch(512), 3, 2),
            C3(ch(512), ch(512), nl(9)),
        )

        # Stage 4: 40 -> 20  (P5)
        self.stage4 = nn.Sequential(
            Conv(ch(512), ch(1024), 3, 2),
            C3(ch(1024), ch(1024), nl(3)),
            SPPF(ch(1024), ch(1024), 5),
        )

        # Record output channels for downstream modules
        self.out_channels = [ch(256), ch(512), ch(1024)]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32
        return p3, p4, p5
