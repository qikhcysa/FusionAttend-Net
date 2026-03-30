"""
Data-augmentation pipelines for FusionAttend-Net.

Training augmentations are aggressive (colour jitter, random crops,
flips, rotations, etc.) to help the model generalise across the
diverse backgrounds and lighting conditions found in real agricultural
scenes.  Validation / test augmentations only resize and normalise.

All transforms follow the torchvision.transforms API.
"""

import torchvision.transforms as T

# ImageNet statistics used to normalise every image
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Build the training augmentation pipeline.

    Includes:
        * Random resized crop (scale 0.4–1.0, ratio 3/4–4/3)
        * Random horizontal and vertical flips
        * Random rotation (±45°)
        * Colour jitter (brightness, contrast, saturation, hue)
        * Random greyscale (p=0.05) – simulates disease-scanning variability
        * Random Gaussian blur
        * Normalisation to ImageNet statistics

    Args:
        image_size: Target square side length in pixels (default 224).

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Build the validation / test augmentation pipeline.

    Only resizes (256 px), centre-crops (224 px) and normalises.

    Args:
        image_size: Target square side length in pixels (default 224).

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    resize_size = int(image_size * 256 / 224)
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
