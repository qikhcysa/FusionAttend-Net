"""
Image preprocessing pipeline for FusionAttend-Net.

Implements the core preprocessing steps applied to all images before
augmentation:
    1. BGR → RGB (OpenCV loads BGR by default)
    2. Resize to 256×256
    3. Gaussian filter denoising (ksize=3)
    4. Non-local means denoising (h=3, hColor=3)
    5. Brightness / contrast adjustment (alpha=1.0, beta=0)

These steps are exposed both as a callable OpenCV-based function and as a
torchvision-compatible PIL transform so they can be inserted at the front
of any transform pipeline.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── Per-dataset normalization statistics ───────────────────────────────────
# Computed on the training split of each dataset.
DATASET_STATS = {
    "katra_twelve": {
        "mean": [0.2357, 0.2659, 0.2438],
        "std":  [0.1203, 0.1426, 0.1120],
    },
    "bari_sunflower": {
        "mean": [0.4517, 0.4882, 0.2605],
        "std":  [0.2268, 0.2133, 0.1806],
    },
    "fgvc8": {
        "mean": [0.4057, 0.5142, 0.3238],
        "std":  [0.2018, 0.1879, 0.1887],
    },
    # PlantVillage uses ImageNet stats as a sensible default.
    "plantvillage": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
}

# Default / fallback stats (ImageNet)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD  = [0.229, 0.224, 0.225]


def get_normalization_stats(dataset_name: str) -> tuple[list, list]:
    """
    Return (mean, std) lists for the requested dataset.

    Falls back to ImageNet statistics for unknown dataset names.
    """
    stats = DATASET_STATS.get(dataset_name.lower(), None)
    if stats is None:
        return DEFAULT_MEAN, DEFAULT_STD
    return stats["mean"], stats["std"]


def preprocess_image_cv2(
    img_bgr: np.ndarray,
    target_size: int = 256,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a single BGR image.

    Steps:
        1. BGR → RGB
        2. Resize to *target_size* × *target_size*
        3. Gaussian denoising (ksize=3)
        4. Non-local means denoising (h=3, hColor=3)
        5. Brightness / contrast: ``out = alpha * in + beta``

    Args:
        img_bgr: Input BGR image as a NumPy uint8 array (H×W×3).
        target_size: Target square resolution (default 256).
        alpha: Contrast factor (default 1.0 – no change).
        beta: Brightness offset (default 0.0 – no change).

    Returns:
        Preprocessed RGB image as a uint8 NumPy array of shape
        ``(target_size, target_size, 3)``.
    """
    # 1. BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Resize
    img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # 3. Gaussian filter denoising
    img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=0)

    # 4. Non-local means denoising
    img_rgb = cv2.fastNlMeansDenoisingColored(img_rgb, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    # 5. Brightness / contrast
    if alpha != 1.0 or beta != 0.0:
        img_rgb = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)

    return img_rgb


def preprocess_pil(
    pil_image: Image.Image,
    target_size: int = 256,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Image.Image:
    """
    Apply the preprocessing pipeline to a PIL image.

    This is a convenience wrapper around :func:`preprocess_image_cv2`
    that accepts and returns PIL images, making it directly usable as a
    torchvision transform.

    Args:
        pil_image: Input PIL image (any mode; converted to RGB internally).
        target_size: Target square resolution (default 256).
        alpha: Contrast factor (default 1.0).
        beta: Brightness offset (default 0.0).

    Returns:
        Preprocessed PIL image in RGB mode.
    """
    # PIL images are already RGB; convert to NumPy and then to BGR for cv2
    img_np = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_out = preprocess_image_cv2(img_bgr, target_size=target_size, alpha=alpha, beta=beta)
    return Image.fromarray(img_out)


class Preprocess:
    """
    Torchvision-compatible PIL transform that applies the preprocessing
    pipeline (resize → Gaussian denoise → NLM denoise → contrast/brightness).

    Use as the *first* element of a :class:`torchvision.transforms.Compose`
    pipeline::

        transforms.Compose([
            Preprocess(target_size=256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    Args:
        target_size: Output square side length in pixels (default 256).
        alpha: Contrast factor (default 1.0 – no change).
        beta: Brightness offset (default 0.0 – no change).
    """

    def __init__(self, target_size: int = 256, alpha: float = 1.0, beta: float = 0.0):
        self.target_size = target_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img: Image.Image) -> Image.Image:
        return preprocess_pil(img, target_size=self.target_size, alpha=self.alpha, beta=self.beta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_size={self.target_size}, alpha={self.alpha}, beta={self.beta})"
        )
