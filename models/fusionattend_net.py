"""
FusionAttend-Net – Multi-scale Feature Fusion and Attention Network
for plant disease classification.

The model is composed of:
  1. DFN  – Deep Feature Network (YOLOv5 Backbone + Neck)
  2. PSAN – PSA-enhanced classification module with multi-scale conv
            branches and SEWeight channel recalibration

Usage::

    from models import FusionAttendNet

    model = FusionAttendNet(num_classes=38)
    logits = model(images)            # (B, num_classes)
    features = model.extract(images)  # (B, fused_channels)  for t-SNE

Comparison experiments::

    # Use a different attention module
    from models.attention import build_attention
    model = FusionAttendNet(num_classes=12, attention_name="se")

    # Count GFLOPs
    gflops = model.count_gflops(input_size=(1, 3, 256, 256))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        psa_reduction: PSA / SEWeight channel reduction ratio (default 16).
        psa_pyramid_levels: PSA pooling sizes (default ``[1, 2, 4, 8]``).
        dropout: Dropout probability in the classification head (default 0.3).
        attention_name: Name of the attention module to use in PSAN
            (default ``"psa"``; see :mod:`models.attention` for alternatives).
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
        attention_name: str = "psa",
    ):
        super().__init__()
        if psa_pyramid_levels is None:
            psa_pyramid_levels = [1, 2, 4, 8]

        self.dfn = DFN(in_channels, width_multiple, depth_multiple)

        # Allow swapping the PSA module with any registered attention
        if attention_name.lower() != "psa":
            # Patch PSA inside PSAN with the requested attention module
            from models.attention import build_attention as _build_attn
            self.psan = PSAN(
                in_channels=self.dfn.out_channels,
                num_classes=num_classes,
                reduction=psa_reduction,
                pyramid_levels=psa_pyramid_levels,
                dropout=dropout,
            )
            # Replace the PSA submodule with the requested attention
            fused_ch = self.psan.psa.squeeze.in_features  # channel count
            self.psan.psa = _build_attn(attention_name, channels=fused_ch)
        else:
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
        p3, p4, p5 = features
        target_size = p3.shape[2:]

        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        fused = torch.cat([p3, p4_up, p5_up], dim=1)
        fused = self.psan.fusion_conv(fused)

        b3 = self.psan.branch3(fused)
        b5 = self.psan.branch5(fused)
        b7 = self.psan.branch7(fused)
        b9 = self.psan.branch9(fused)
        ms = torch.cat([b3, b5, b7, b9], dim=1)
        ms = self.psan.se_weight(ms)
        ms = self.psan.project(ms)
        ms = self.psan.psa(ms)
        return self.psan.gap(ms).flatten(1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_gflops(self, input_size: tuple = (1, 3, 256, 256)) -> float:
        """
        Estimate the model's computational cost in GFLOPs for a given input
        size using the ``thop`` library.

        Args:
            input_size: Input tensor shape ``(B, C, H, W)`` (default
                ``(1, 3, 256, 256)``).

        Returns:
            GFLOPs as a float (multiply-accumulate operations × 2 / 1e9).
            Returns ``-1.0`` if ``thop`` is not installed.
        """
        try:
            from thop import profile
        except ImportError:
            return -1.0

        device = next(self.parameters()).device
        dummy = torch.randn(*input_size, device=device)
        self.eval()
        with torch.no_grad():
            macs, _ = profile(self, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9
