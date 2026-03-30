"""
FusionAttend-Net – Multi-scale Feature Fusion and Attention Network
for plant disease classification.

The model is composed of:
  1. DFN  – Deep Feature Network (YOLOv5 Backbone + Neck)
  2. PSAN – PSA-enhanced classification module

Usage::

    from models import FusionAttendNet

    model = FusionAttendNet(num_classes=38)
    logits = model(images)          # (B, num_classes)
    features = model.extract(images)  # (B, fused_channels)  for t-SNE etc.
"""

import torch
import torch.nn as nn

from .dfn import DFN
from .psan import PSAN


class FusionAttendNet(nn.Module):
    """
    FusionAttend-Net: complete plant-disease classification model.

    Args:
        num_classes: Number of disease / healthy categories.
        in_channels: Input image channels (default 3).
        width_multiple: YOLOv5 channel width multiplier (default 0.5).
        depth_multiple: YOLOv5 block depth multiplier (default 0.33).
        psa_reduction: PSA channel reduction ratio (default 16).
        psa_pyramid_levels: PSA pooling sizes (default ``[1, 2, 4, 8]``).
        dropout: Dropout probability in the classification head (default 0.3).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        psa_reduction: int = 16,
        psa_pyramid_levels: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if psa_pyramid_levels is None:
            psa_pyramid_levels = [1, 2, 4, 8]

        self.dfn = DFN(in_channels, width_multiple, depth_multiple)
        self.psan = PSAN(
            in_channels=self.dfn.out_channels,
            num_classes=num_classes,
            reduction=psa_reduction,
            pyramid_levels=psa_pyramid_levels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        features = self.dfn(x)
        return self.psan(features)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the penultimate feature vector (before the linear classifier)
        for use with t-SNE visualization or other downstream analyses.

        Returns:
            Feature tensor of shape ``(B, fused_channels)``.
        """
        features = self.dfn(x)
        # Re-use PSAN internals up to the dropout step
        p3, p4, p5 = features
        import torch.nn.functional as F
        target_size = p3.shape[2:]
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        fused = torch.cat([p3, p4_up, p5_up], dim=1)
        fused = self.psan.fusion_conv(fused)
        fused = self.psan.psa(fused)
        fused = self.psan.gap(fused).flatten(1)
        return fused

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
