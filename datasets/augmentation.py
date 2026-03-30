"""
Data-augmentation pipelines for FusionAttend-Net.

Implements three augmentation strategies:

1. **Strong** (``build_strong_train_transforms``) – used for FGVC8 minority
   classes with fewer than 500 samples; each image is augmented 6 times.
2. **Weak** (``build_weak_train_transforms``) – used for Katra-Twelve and
   BARI-Sunflower; each image is augmented 2 times.
3. **Validation** (``build_val_transforms``) – only resize and normalise.

Both train pipelines include **weather simulation augmentations** (sun flare,
rain, shadow, fog – each applied with probability 0.6) via the
``albumentations`` library, reflecting real agricultural shooting conditions.

Per-dataset normalisation statistics (computed on each training split) are
supported via the ``dataset_name`` parameter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

import torchvision.transforms as T

from datasets.preprocessing import get_normalization_stats

# ── Weather augmentation ──────────────────────────────────────────────────
# albumentations is optional; falls back gracefully if not installed.
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False


def _weather_transforms(p: float = 0.6):
    """
    Return an albumentations ``Compose`` of four weather effects, each
    applied with probability *p*.  Returns ``None`` if albumentations is
    not installed.
    """
    if not _ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=p),
        A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=p),
        A.RandomShadow(num_shadows_limit=(1, 2), p=p),
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=p),
    ])


class WeatherAugment:
    """
    PIL-compatible transform that applies albumentations weather effects.
    If albumentations is unavailable the image is returned unchanged.
    """

    def __init__(self, p: float = 0.6):
        self._pipeline = _weather_transforms(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self._pipeline is None:
            return img
        img_np = np.array(img.convert("RGB"))
        result = self._pipeline(image=img_np)["image"]
        return Image.fromarray(result)

    def __repr__(self) -> str:
        return f"WeatherAugment(albumentations_available={_ALBUMENTATIONS_AVAILABLE})"


# ── Per-dataset normalization defaults ────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _get_norm(dataset_name: Optional[str]):
    """Return (mean, std) for the given dataset name."""
    if dataset_name is None:
        return _IMAGENET_MEAN, _IMAGENET_STD
    return get_normalization_stats(dataset_name)


# ── Augmentation pipelines ────────────────────────────────────────────────

def build_strong_train_transforms(
    image_size: int = 256,
    dataset_name: Optional[str] = None,
    weather_p: float = 0.6,
) -> T.Compose:
    """
    Strong training augmentation – for FGVC8 minority classes (< 500 samples).

    Each source image is passed through this pipeline **6 times** by the
    dataset loader to expand the minority class.

    Includes:
        * Weather simulation (sun flare, rain, shadow, fog – each at *weather_p*)
        * Random resized crop (scale 0.08–1.0, ratio 3/4–4/3)
        * Random horizontal flip (p=0.5), vertical flip (p=0.1)
        * Colour jitter (brightness/contrast/saturation/hue = 0.2 each)
        * Normalisation to per-dataset statistics

    Args:
        image_size: Target square side length in pixels (default 256).
        dataset_name: Dataset key for per-dataset normalisation stats.
        weather_p: Per-weather-effect probability (default 0.6).
    """
    mean, std = _get_norm(dataset_name)
    return T.Compose([
        WeatherAugment(p=weather_p),
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_weak_train_transforms(
    image_size: int = 256,
    dataset_name: Optional[str] = None,
    weather_p: float = 0.6,
) -> T.Compose:
    """
    Weak training augmentation – for Katra-Twelve and BARI-Sunflower.

    Each source image is passed through this pipeline **2 times** to
    moderately expand the dataset while avoiding overfitting on images that
    already have sufficient class coverage.

    Includes:
        * Weather simulation (sun flare, rain, shadow, fog – each at *weather_p*)
        * Random resized crop (scale 0.08–1.0)
        * Random horizontal flip (p=0.5), vertical flip (p=0.1)
        * Light colour jitter (brightness/contrast/saturation/hue = 0.2)
        * Normalisation to per-dataset statistics

    Args:
        image_size: Target square side length in pixels (default 256).
        dataset_name: Dataset key for per-dataset normalisation stats.
        weather_p: Per-weather-effect probability (default 0.6).
    """
    mean, std = _get_norm(dataset_name)
    return T.Compose([
        WeatherAugment(p=weather_p),
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# Convenience alias: train pipeline selector
def build_train_transforms(
    image_size: int = 256,
    dataset_name: Optional[str] = None,
    strong: bool = False,
    weather_p: float = 0.6,
) -> T.Compose:
    """
    Select the appropriate training transform pipeline.

    Args:
        image_size: Target square side length (default 256).
        dataset_name: Dataset key for per-dataset normalisation.
        strong: If ``True`` return the strong pipeline (6× augmentation);
            otherwise return the weak pipeline (2×).
        weather_p: Per-weather-effect probability (default 0.6).
    """
    if strong:
        return build_strong_train_transforms(image_size, dataset_name, weather_p)
    return build_weak_train_transforms(image_size, dataset_name, weather_p)


def build_val_transforms(
    image_size: int = 256,
    dataset_name: Optional[str] = None,
) -> T.Compose:
    """
    Validation / test augmentation pipeline.

    Only resizes to *image_size* × *image_size* and normalises.

    Args:
        image_size: Target square side length in pixels (default 256).
        dataset_name: Dataset key for per-dataset normalisation stats.
    """
    mean, std = _get_norm(dataset_name)
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
