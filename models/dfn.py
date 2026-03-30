"""
DFN – Deep Feature Network (Backbone + Neck).

Combines :class:`YOLOv5Backbone` and :class:`YOLOv5Neck` into a single
feature extraction module that outputs a rich, multi-scale representation
ready for the PSAN classification head.
"""

import torch
import torch.nn as nn

from .backbone import YOLOv5Backbone
from .neck import YOLOv5Neck


class DFN(nn.Module):
    """
    Deep Feature Network built from a YOLOv5-style backbone and neck.

    The module returns the three fused feature maps (P3, P4, P5) from
    the PANet neck.  A downstream classification head should aggregate
    these maps (e.g. via global average pooling and concatenation) before
    the final linear classifier.

    Args:
        in_channels: Number of input image channels (default 3).
        width_multiple: Channel-width multiplier (default 0.5 → YOLOv5-s).
        depth_multiple: Block-depth multiplier (default 0.33 → YOLOv5-s).
    """

    def __init__(
        self,
        in_channels: int = 3,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
    ):
        super().__init__()
        self.backbone = YOLOv5Backbone(in_channels, width_multiple, depth_multiple)
        self.neck = YOLOv5Neck(self.backbone.out_channels, width_multiple)
        self.out_channels = self.neck.out_channels  # [c3, c4, c5]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tuple of three feature tensors at 1/8, 1/16 and 1/32 scale.
        """
        features = self.backbone(x)
        return self.neck(features)
