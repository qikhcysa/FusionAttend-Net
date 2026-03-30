"""
YOLOv5-based Neck (FPN + PANet) for the DFN feature extraction module.

The neck fuses the three feature maps produced by :class:`YOLOv5Backbone`
at 1/8, 1/16 and 1/32 using a top-down FPN path and a bottom-up PANet
path, yielding three enriched feature maps that capture both fine-grained
local texture and coarse global semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Conv, C3


class YOLOv5Neck(nn.Module):
    """
    Feature Pyramid Network (FPN) + Path Aggregation Network (PANet) neck.

    Args:
        in_channels: List of three channel counts from the backbone for
            P3, P4 and P5 feature maps.
        width_multiple: Channel width multiplier, kept consistent with the
            backbone so channel counts align.
    """

    def __init__(self, in_channels: list, width_multiple: float = 0.5):
        super().__init__()

        def ch(c):
            return max(round(c * width_multiple), 1)

        c3, c4, c5 = in_channels  # channels for P3, P4, P5

        # ── Top-down (FPN) path ───────────────────────────────────────────
        # P5 -> P4
        self.reduce_p5 = Conv(c5, c4, 1, 1)
        self.c3_p4_fpn = C3(c4 + c4, c4, 1, shortcut=False)

        # P4 -> P3
        self.reduce_p4 = Conv(c4, c3, 1, 1)
        self.c3_p3_fpn = C3(c3 + c3, c3, 1, shortcut=False)

        # ── Bottom-up (PANet) path ────────────────────────────────────────
        # P3 -> P4
        self.downsample_p3 = Conv(c3, c3, 3, 2)
        self.c3_p4_pan = C3(c3 + c4, c4, 1, shortcut=False)

        # P4 -> P5
        self.downsample_p4 = Conv(c4, c4, 3, 2)
        self.c3_p5_pan = C3(c4 + c5, c5, 1, shortcut=False)

        self.out_channels = [c3, c4, c5]

    def forward(self, features):
        p3, p4, p5 = features  # from backbone

        # ── Top-down path ──────────────────────────────────────────────────
        p5_reduced = self.reduce_p5(p5)
        p4_fpn = self.c3_p4_fpn(
            torch.cat([F.interpolate(p5_reduced, size=p4.shape[2:], mode="nearest"), p4], dim=1)
        )

        p4_reduced = self.reduce_p4(p4_fpn)
        p3_fpn = self.c3_p3_fpn(
            torch.cat([F.interpolate(p4_reduced, size=p3.shape[2:], mode="nearest"), p3], dim=1)
        )

        # ── Bottom-up path ─────────────────────────────────────────────────
        p4_pan = self.c3_p4_pan(
            torch.cat([self.downsample_p3(p3_fpn), p4_fpn], dim=1)
        )

        p5_pan = self.c3_p5_pan(
            torch.cat([self.downsample_p4(p4_pan), p5], dim=1)
        )

        return p3_fpn, p4_pan, p5_pan
