"""
SHAP-based Explanation Heatmaps for FusionAttend-Net.

Generates SHAP (SHapley Additive exPlanations) pixel-importance maps for
individual predictions.  Red regions contribute positively (disease signals);
blue regions contribute negatively (background / noise).

Usage::

    python utils/shap_analysis.py \\
        --config   configs/fgvc8.yaml \\
        --checkpoint outputs/fgvc8/fold1_best.pth \\
        --data_root  datasets/FGVC8/test \\
        --save_dir   outputs/fgvc8/shap \\
        --num_samples 20

Requires ``shap>=0.41`` and ``matplotlib``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ── Core analysis function ─────────────────────────────────────────────────

def run_shap_analysis(
    model: nn.Module,
    images: torch.Tensor,
    labels: List[int],
    class_names: Optional[List[str]],
    save_dir: str,
    background_size: int = 50,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Compute GradientExplainer SHAP values for a batch of images and save
    red-blue heat-map overlays.

    Args:
        model: Trained :class:`FusionAttendNet` (eval mode, on *device*).
        images: Float tensor of shape ``(N, 3, H, W)`` (already normalised).
        labels: Ground-truth integer label for each image.
        class_names: Optional list of class name strings.
        save_dir: Directory to write PNG overlays.
        background_size: Number of random background samples for SHAP
            baseline estimation (default 50).
        device: Device on which the model resides.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP analysis. "
            "Install it with: pip install shap"
        ) from exc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Use a small random subset of the batch as the SHAP background
    n = images.shape[0]
    bg_idx = np.random.choice(n, size=min(background_size, n), replace=False)
    background = images[bg_idx].to(device)

    # Wrap model so SHAP sees a plain function
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(t)
        return logits.cpu().numpy()

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(images.to(device))
    # shap_values: list of (N, 3, H, W) – one per class

    images_np = images.cpu().numpy()

    for i in range(n):
        true_label = labels[i]
        # Use SHAP values for the predicted class
        with torch.no_grad():
            pred_label = int(model(images[i : i + 1].to(device)).argmax(dim=1).item())

        sv = shap_values[pred_label][i]           # (3, H, W)
        sv_mean = sv.mean(axis=0)                  # aggregate over channels (H, W)

        # Recover a displayable image (un-normalise approximately)
        img_display = images_np[i].transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img_display)
        axes[0].set_title(
            f"GT: {class_names[true_label] if class_names else true_label}\n"
            f"Pred: {class_names[pred_label] if class_names else pred_label}",
            fontsize=9,
        )
        axes[0].axis("off")

        vmax = float(np.abs(sv_mean).max()) + 1e-8
        im = axes[1].imshow(sv_mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.85)
        axes[1].imshow(img_display, alpha=0.35)
        axes[1].set_title("SHAP (red=positive, blue=negative)", fontsize=9)
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        correct = "correct" if pred_label == true_label else "wrong"
        fig.suptitle(f"Sample {i} [{correct}]", fontsize=11)
        fig.tight_layout()

        fname = os.path.join(save_dir, f"shap_{i:04d}_{correct}.png")
        fig.savefig(fname, dpi=120)
        plt.close(fig)

    print(f"SHAP overlays saved to {save_dir}  ({n} images)")


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP heatmaps for FusionAttend-Net")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--save_dir",    default="outputs/shap")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of images to analyse (default 20)")
    parser.add_argument("--device",      type=str, default="")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from datasets import PlantDiseaseDataset, build_val_transforms
    from models import FusionAttendNet

    ds_cfg = cfg["dataset"]
    if args.data_root:
        ds_cfg["root"] = args.data_root

    val_tf = build_val_transforms(ds_cfg["image_size"])
    dataset = PlantDiseaseDataset(ds_cfg["root"], transform=val_tf)

    # Randomly pick num_samples images
    indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    images_list, labels_list = [], []
    for idx in indices:
        img, lbl = dataset[int(idx)]
        images_list.append(img)
        labels_list.append(lbl)
    images_tensor = torch.stack(images_list)

    model_cfg = cfg["model"]
    model = FusionAttendNet(
        num_classes=model_cfg["num_classes"],
        in_channels=model_cfg["in_channels"],
        width_multiple=model_cfg["width_multiple"],
        depth_multiple=model_cfg["depth_multiple"],
        psa_reduction=model_cfg["psa_reduction"],
        psa_pyramid_levels=model_cfg["psa_pyramid_levels"],
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    run_shap_analysis(
        model=model,
        images=images_tensor,
        labels=labels_list,
        class_names=dataset.classes,
        save_dir=args.save_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
