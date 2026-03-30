"""
Dataset Preparation Script
============================
Prepares the three benchmark datasets (Katra-Twelve, BARI-Sunflower, FGVC8)
for training by:

  1. Applying 8:1:1 stratified random splitting (train / val / test).
  2. Merging FGVC8's original 12 multi-label classes into 6 single-label
     categories (Complex, Frog_Eye_Leaf_Spot, Healthy, Powdery_Mildew,
     Rust, Scab).
  3. Applying the full preprocessing pipeline (resize 256×256, Gaussian
     denoise, NLM denoise) and saving processed images to an output
     directory tree ready for ImageFolder loading.

Usage::

    # Prepare Katra-Twelve
    python datasets/prepare_dataset.py \\
        --dataset katra_twelve \\
        --src_dir /raw/Katra_Twelve \\
        --dst_dir datasets/Katra_Twelve

    # Prepare FGVC8 (with class merging)
    python datasets/prepare_dataset.py \\
        --dataset fgvc8 \\
        --src_dir /raw/FGVC8 \\
        --dst_dir datasets/FGVC8

Expected source layout (ImageFolder)::

    <src_dir>/
        <class_name>/
            img_001.jpg
            ...

Output layout::

    <dst_dir>/
        train/  <class_name>/  *.jpg
        val/    <class_name>/  *.jpg
        test/   <class_name>/  *.jpg
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from datasets.preprocessing import preprocess_image_cv2


# ── FGVC8 class mapping ────────────────────────────────────────────────────
# Maps every original FGVC8 folder name to one of the 6 merged categories.
# Any class containing more than one disease label is mapped to "Complex".
FGVC8_CLASS_MAP: dict[str, str] = {
    # Single-disease classes
    "frog_eye_leaf_spot":           "Frog_Eye_Leaf_Spot",
    "healthy":                      "Healthy",
    "powdery_mildew":               "Powdery_Mildew",
    "rust":                         "Rust",
    "scab":                         "Scab",
    # Compound-disease classes → all merged into Complex
    "complex":                      "Complex",
    "frog_eye_leaf_spot complex":   "Complex",
    "powdery_mildew complex":       "Complex",
    "rust frog_eye_leaf_spot":      "Complex",
    "rust complex":                 "Complex",
    "scab frog_eye_leaf_spot":      "Complex",
    "scab frog_eye_leaf_spot complex": "Complex",
}

FGVC8_6_CLASSES = ["Complex", "Frog_Eye_Leaf_Spot", "Healthy", "Powdery_Mildew", "Rust", "Scab"]


def discover_samples(src_dir: str) -> dict[str, list[str]]:
    """
    Walk *src_dir* (ImageFolder layout) and return a mapping
    ``{class_name: [file_path, ...]}``.
    """
    class_samples: dict[str, list[str]] = defaultdict(list)
    src = Path(src_dir)
    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                class_samples[class_dir.name].append(str(img_path))
    return dict(class_samples)


def split_class_samples(
    samples: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Stratified 8:1:1 split for a single class list.

    Returns:
        ``(train_paths, val_paths, test_paths)``
    """
    if len(samples) < 3:
        # Too few samples: put everything in train
        return samples, [], []

    test_size = val_ratio / (val_ratio + train_ratio)  # relative to total
    val_size_after_test = val_ratio / train_ratio       # relative to remaining

    train_val, test = train_test_split(samples, test_size=val_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio / (1.0 - val_ratio), random_state=seed)
    return train, val, test


def save_image(src_path: str, dst_path: str, apply_preprocessing: bool, target_size: int = 256) -> None:
    """Read, optionally preprocess, and save a single image."""
    img = cv2.imread(src_path)
    if img is None:
        print(f"  WARNING: could not read {src_path}, skipping.")
        return
    if apply_preprocessing:
        img_out = preprocess_image_cv2(img, target_size=target_size)
        # img_out is RGB → convert back to BGR for cv2.imwrite
        img_save = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    else:
        img_save = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, img_save)


def prepare_dataset(
    dataset: str,
    src_dir: str,
    dst_dir: str,
    apply_preprocessing: bool = True,
    target_size: int = 256,
    seed: int = 42,
) -> None:
    """
    Prepare and split a dataset.

    Args:
        dataset: Dataset identifier (``"katra_twelve"``, ``"bari_sunflower"``,
            ``"fgvc8"``, or ``"plantvillage"``).
        src_dir: Path to raw ImageFolder-layout source directory.
        dst_dir: Destination root (will be created; must not exist or be empty).
        apply_preprocessing: Whether to run the full preprocessing pipeline
            (Gaussian + NLM denoise etc.) – set to ``False`` for a quick dry
            run that only resizes.
        target_size: Resize target (default 256).
        seed: Random seed for reproducible splits (default 42).
    """
    print(f"\nPreparing dataset: {dataset}")
    print(f"  Source : {src_dir}")
    print(f"  Dest   : {dst_dir}")

    class_samples = discover_samples(src_dir)
    if not class_samples:
        raise ValueError(f"No images found under {src_dir}")

    print(f"  Found {sum(len(v) for v in class_samples.values())} images "
          f"across {len(class_samples)} classes.")

    # For FGVC8: remap class names to merged categories
    if dataset.lower() == "fgvc8":
        merged: dict[str, list[str]] = defaultdict(list)
        unmapped = []
        for orig_class, paths in class_samples.items():
            mapped = FGVC8_CLASS_MAP.get(orig_class.lower().strip())
            if mapped is None:
                unmapped.append(orig_class)
                # Best-effort: check if name contains a known compound hint
                mapped = "Complex"
            merged[mapped].extend(paths)
        if unmapped:
            print(f"  WARNING: unmapped FGVC8 classes (→ Complex): {unmapped}")
        class_samples = dict(merged)
        print(f"  After merging: {len(class_samples)} classes: {sorted(class_samples)}")

    splits = {"train": [], "val": [], "test": []}
    for class_name, paths in sorted(class_samples.items()):
        train_p, val_p, test_p = split_class_samples(paths, seed=seed)
        splits["train"].append((class_name, train_p))
        splits["val"].append((class_name, val_p))
        splits["test"].append((class_name, test_p))
        print(f"  {class_name:35s}  train={len(train_p):4d}  val={len(val_p):4d}  test={len(test_p):4d}")

    # Copy / process images into destination directory
    dst = Path(dst_dir)
    total = sum(len(p) for _, lst in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]
                for _, p in lst)
    done = 0
    for split_name, class_list in splits.items():
        for class_name, paths in class_list:
            for src_path in paths:
                fname = Path(src_path).name
                dst_path = str(dst / split_name / class_name / fname)
                save_image(src_path, dst_path, apply_preprocessing, target_size)
                done += 1
                if done % 500 == 0:
                    print(f"  Processed {done}/{total} images …")

    print(f"  Done. {done} images written to {dst_dir}")


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and split a plant disease dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        choices=["katra_twelve", "bari_sunflower", "fgvc8", "plantvillage"],
                        help="Dataset identifier")
    parser.add_argument("--src_dir", required=True,
                        help="Raw ImageFolder-layout source directory")
    parser.add_argument("--dst_dir", required=True,
                        help="Output directory root (will contain train/val/test subdirs)")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip denoising preprocessing (only resize)")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        dataset=args.dataset,
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        apply_preprocessing=not args.no_preprocess,
        target_size=args.target_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
