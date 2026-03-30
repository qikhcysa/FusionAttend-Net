"""
FusionAttend-Net Evaluation Script
====================================
Evaluates a trained FusionAttend-Net checkpoint on a held-out test set and
optionally produces t-SNE and confusion-matrix visualizations.

Quick start::

    python evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint outputs/katra_twelve/fold1_best.pth \\
        --data_root  data/katra_twelve/test \\
        --save_dir   outputs/katra_twelve/eval
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import PlantDiseaseDataset, build_dataloader, build_val_transforms
from models import FusionAttendNet
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix, plot_tsne


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_features_and_predictions(model, loader, device):
    model.eval()
    all_features, all_labels, all_preds = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = model.extract(images).cpu().numpy()
        logits = model(images).cpu()
        preds = logits.argmax(dim=1).numpy()

        all_features.append(features)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())

    return np.concatenate(all_features, axis=0), all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate FusionAttend-Net")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pth checkpoint produced by train.py")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override dataset root from config (e.g. path to test split)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_tsne", action="store_true",
                        help="Skip t-SNE visualization (faster)")
    parser.add_argument("--device", type=str, default="",
                        help="cuda device index or 'cpu'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    if args.data_root:
        ds_cfg["root"] = args.data_root
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers:
        ds_cfg["num_workers"] = args.num_workers

    save_dir = Path(args.save_dir or cfg["output"]["save_dir"]) / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Dataset
    val_tf = build_val_transforms(ds_cfg["image_size"])
    dataset = PlantDiseaseDataset(ds_cfg["root"], transform=val_tf)
    loader = build_dataloader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=ds_cfg["num_workers"],
        is_train=False,
        balance_classes=False,
    )
    print(f"Evaluating on {len(dataset)} samples  |  {len(dataset.classes)} classes")

    # Model
    model = FusionAttendNet(
        num_classes=model_cfg["num_classes"],
        in_channels=model_cfg["in_channels"],
        width_multiple=model_cfg["width_multiple"],
        depth_multiple=model_cfg["depth_multiple"],
        psa_reduction=model_cfg["psa_reduction"],
        psa_pyramid_levels=model_cfg["psa_pyramid_levels"],
        dropout=0.0,  # disable dropout at inference
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Extract features and compute predictions
    features, all_labels, all_preds = extract_features_and_predictions(model, loader, device)

    # Metrics
    metrics = compute_metrics(all_labels, all_preds, class_names=dataset.classes)
    print("\nEvaluation Results")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1  : {metrics['macro_f1'] * 100:.2f}%")
    print("\n" + metrics["report"])

    # Save metrics
    results = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "checkpoint": args.checkpoint,
        "dataset_root": ds_cfg["root"],
        "num_samples": len(dataset),
        "num_classes": len(dataset.classes),
    }
    with open(save_dir / "metrics.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Metrics saved to {save_dir / 'metrics.json'}")

    # Confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=dataset.classes,
        save_path=str(save_dir / "confusion_matrix.png"),
        title=f"Confusion Matrix – {ds_cfg['name']}",
    )

    # t-SNE
    if not args.no_tsne:
        plot_tsne(
            features,
            np.array(all_labels),
            class_names=dataset.classes,
            save_path=str(save_dir / "tsne.png"),
            title=f"t-SNE Feature Visualization – {ds_cfg['name']}",
        )


if __name__ == "__main__":
    main()
