"""
PSAN – PSA-enhanced Classification Module.

Aggregates the three multi-scale feature maps from the DFN via global
average pooling, fuses them, applies the PSA attention mechanism, then
feeds the result to a linear classifier.

Architecture:
    P3, P4, P5  ──GAP──►  concat (C3+C4+C5)
                             │
                          PSA (channel attention)
                             │
                          Dropout
                             │
                          Linear → num_classes
"""

import torch
import torch.nn as nn

from .psa import PSA


class PSAN(nn.Module):
    """
    PSA-Network classification head.

    Args:
        in_channels: List of three channel counts [c3, c4, c5] matching the
            DFN output.
        num_classes: Number of output classes.
        reduction: PSA channel-reduction ratio (default 16).
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

        # Fuse the concatenated multi-scale channels to a fixed size
        fused_channels = max(total_channels // 2, 64)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.SiLU(inplace=True),
        )

        self.psa = PSA(fused_channels, reduction=reduction, pyramid_levels=pyramid_levels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(fused_channels, num_classes)

    def forward(self, features):
        """
        Args:
            features: Tuple/list of three tensors (P3, P4, P5) from the DFN.

        Returns:
            Class logits tensor of shape ``(B, num_classes)``.
        """
        p3, p4, p5 = features
        target_size = p3.shape[2:]  # up-sample P4 / P5 to P3 spatial size

        import torch.nn.functional as F
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([p3, p4_up, p5_up], dim=1)  # (B, C3+C4+C5, H, W)
        fused = self.fusion_conv(fused)                # (B, fused_ch, H, W)
        fused = self.psa(fused)                        # attention recalibration
        fused = self.gap(fused).flatten(1)             # (B, fused_ch)
        fused = self.dropout(fused)
        return self.classifier(fused)                  # (B, num_classes)
