"""
Plant Disease Dataset loader.

Supports the three benchmark datasets used in FusionAttend-Net:
  * Katra-Twelve
  * BARI-Sunflower
  * FGVC8  (FGVC Plant Pathology challenge)
  * PlantVillage (open-source validation set)

Each dataset is expected to be organised as an ImageFolder-style
directory tree::

    <root>/
        <class_name_1>/
            img_001.jpg
            img_002.jpg
            ...
        <class_name_2>/
            ...

Class imbalance is handled by an optional **WeightedRandomSampler**
that over-samples minority classes during training.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder


class PlantDiseaseDataset(ImageFolder):
    """
    ImageFolder-based dataset wrapper for plant disease classification.

    Args:
        root: Root directory of the dataset (ImageFolder layout).
        transform: Optional transform applied to each PIL image.
        target_transform: Optional transform applied to each label.
    """

    SUPPORTED_DATASETS = ("katra_twelve", "bari_sunflower", "fgvc8", "plantvillage")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

    # ------------------------------------------------------------------
    # Helper: compute per-sample weights for WeightedRandomSampler
    # ------------------------------------------------------------------
    def compute_sample_weights(self) -> torch.Tensor:
        """
        Return a 1-D float tensor of length ``len(self)`` with weights
        inversely proportional to each sample's class frequency.  Pass
        this to :class:`torch.utils.data.WeightedRandomSampler` to
        balance training across imbalanced classes.
        """
        label_counts = Counter(self.targets)
        # Weight for class c = 1 / count(c)
        class_weights = {c: 1.0 / n for c, n in label_counts.items()}
        sample_weights = torch.tensor(
            [class_weights[label] for label in self.targets], dtype=torch.float
        )
        return sample_weights

    def class_distribution(self) -> dict:
        """Return a dict mapping class name → number of samples."""
        counts = Counter(self.targets)
        return {self.classes[c]: n for c, n in sorted(counts.items())}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: PlantDiseaseDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    is_train: bool = True,
    balance_classes: bool = True,
) -> DataLoader:
    """
    Create a :class:`DataLoader` for a :class:`PlantDiseaseDataset`.

    When *is_train* and *balance_classes* are both ``True`` a
    :class:`~torch.utils.data.WeightedRandomSampler` is used so that
    each mini-batch is approximately class-balanced.

    Args:
        dataset: The dataset to load from.
        batch_size: Mini-batch size (default 32).
        num_workers: Number of worker processes for data loading
            (default 4).
        is_train: Whether this loader is for training.  Controls
            shuffling / balanced sampling behaviour.
        balance_classes: Use weighted sampling to balance class
            frequencies (only effective when *is_train* is ``True``).

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    sampler = None
    shuffle = is_train

    if is_train and balance_classes:
        weights = dataset.compute_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )


def build_kfold_datasets(
    root: str,
    train_indices: list,
    val_indices: list,
    train_transform: Callable,
    val_transform: Callable,
) -> Tuple[PlantDiseaseDataset, PlantDiseaseDataset]:
    """
    Create train / validation dataset splits from pre-computed index lists.

    Args:
        root: Dataset root directory.
        train_indices: Sample indices for training.
        val_indices: Sample indices for validation.
        train_transform: Transform applied to training images.
        val_transform: Transform applied to validation images.

    Returns:
        A ``(train_dataset, val_dataset)`` tuple backed by
        :class:`torch.utils.data.Subset`.
    """
    from torch.utils.data import Subset

    base = PlantDiseaseDataset(root, transform=None)
    train_ds = Subset(PlantDiseaseDataset(root, transform=train_transform), train_indices)
    val_ds = Subset(PlantDiseaseDataset(root, transform=val_transform), val_indices)
    return train_ds, val_ds
