"""
FusionAttend-Net Training Script
=================================
Trains FusionAttend-Net with K-fold cross-validation on a plant disease
dataset arranged in the ImageFolder layout.

Quick start::

    python train.py --config configs/default.yaml

Override any config key from the command line::

    python train.py --config configs/fgvc8.yaml --epochs 150 --lr 5e-4

Checkpoints and metrics are written to the ``output.save_dir`` directory
specified in the config (or overridden via ``--save_dir``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from datasets import (
    PlantDiseaseDataset,
    build_dataloader,
    build_train_transforms,
    build_val_transforms,
)
from models import FusionAttendNet
from utils.metrics import AverageMeter, compute_metrics, topk_accuracy


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config loading ─────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Apply CLI overrides (flat key → nested path not supported; pass dicts)
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg


# ── Optimiser / Scheduler ─────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict):
    name = cfg["optimizer"]["name"].lower()
    lr = cfg["optimizer"]["lr"]
    wd = cfg["optimizer"]["weight_decay"]
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=cfg["optimizer"].get("momentum", 0.9), weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, total_steps: int):
    name = cfg["scheduler"]["name"].lower()
    if name == "cosine":
        warmup = cfg["scheduler"].get("warmup_epochs", 0)
        return CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"] - warmup, eta_min=1e-6)
    elif name == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        return None


# ── One epoch helpers ──────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> tuple:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        top1 = topk_accuracy(logits.detach(), labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1 / 100.0, images.size(0))

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}]  "
                f"loss={loss_meter.avg:.4f}  acc={acc_meter.avg * 100:.2f}%"
            )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), images.size(0))

        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return loss_meter.avg, metrics


# ── K-fold training loop ───────────────────────────────────────────────────

def run_kfold(cfg: dict, device: torch.device) -> None:
    set_seed(cfg["training"]["seed"])

    ds_cfg = cfg["dataset"]
    train_tf = build_train_transforms(ds_cfg["image_size"])
    val_tf = build_val_transforms(ds_cfg["image_size"])

    # Load full dataset (without transform – indices only needed here)
    full_ds = PlantDiseaseDataset(ds_cfg["root"])
    targets = np.array(full_ds.targets)
    class_names = full_ds.classes
    print(f"Dataset: {ds_cfg['name']}  |  samples={len(full_ds)}  |  classes={len(class_names)}")
    print("Class distribution:", full_ds.class_distribution())

    k = cfg["training"]["k_folds"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["training"]["seed"])

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'=' * 60}")
        print(f"  Fold {fold + 1}/{k}")
        print(f"{'=' * 60}")

        # Per-fold datasets
        from torch.utils.data import Subset
        train_ds = Subset(PlantDiseaseDataset(ds_cfg["root"], transform=train_tf), train_idx.tolist())
        val_ds = Subset(PlantDiseaseDataset(ds_cfg["root"], transform=val_tf), val_idx.tolist())

        # Balanced sampler weights for this fold
        fold_targets = targets[train_idx]
        from collections import Counter
        counts = Counter(fold_targets.tolist())
        sample_weights = torch.tensor(
            [1.0 / counts[t] for t in fold_targets], dtype=torch.float
        )
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True) \
            if ds_cfg.get("balance_classes", True) else None

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=ds_cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=ds_cfg["num_workers"] > 0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=ds_cfg["num_workers"],
            pin_memory=True,
            persistent_workers=ds_cfg["num_workers"] > 0,
        )

        # Model
        model_cfg = cfg["model"]
        model = FusionAttendNet(
            num_classes=model_cfg["num_classes"],
            in_channels=model_cfg["in_channels"],
            width_multiple=model_cfg["width_multiple"],
            depth_multiple=model_cfg["depth_multiple"],
            psa_reduction=model_cfg["psa_reduction"],
            psa_pyramid_levels=model_cfg["psa_pyramid_levels"],
            dropout=model_cfg["dropout"],
        ).to(device)
        print(f"  Parameters: {model.count_parameters():,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, total_steps=len(train_loader) * cfg["training"]["epochs"])

        best_val_acc = 0.0
        best_ckpt_path = save_dir / f"fold{fold + 1}_best.pth"
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        warmup_epochs = cfg["scheduler"].get("warmup_epochs", 0) if scheduler else 0

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            # Linear warm-up for the initial epochs
            if epoch <= warmup_epochs:
                lr_scale = epoch / max(warmup_epochs, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["optimizer"]["lr"] * lr_scale

            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, cfg["output"]["log_interval"],
            )
            vl_loss, vl_metrics = validate(model, val_loader, criterion, device)

            if scheduler and epoch > warmup_epochs:
                scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_accs.append(tr_acc)
            val_accs.append(vl_metrics["accuracy"])

            elapsed = time.time() - t0
            print(
                f"  [Epoch {epoch:3d}/{cfg['training']['epochs']}]  "
                f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                f"val_acc={vl_metrics['accuracy'] * 100:.2f}%  "
                f"val_f1={vl_metrics['macro_f1'] * 100:.2f}%  "
                f"({elapsed:.1f}s)"
            )

            if vl_metrics["accuracy"] > best_val_acc:
                best_val_acc = vl_metrics["accuracy"]
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(), "metrics": vl_metrics},
                    best_ckpt_path,
                )

        # Final evaluation on best checkpoint
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        _, final_metrics = validate(model, val_loader, criterion, device)
        fold_results.append(final_metrics)

        print(f"\n  Fold {fold + 1} results:")
        print(f"  Best val acc = {final_metrics['accuracy'] * 100:.2f}%")
        print(f"  Macro F1     = {final_metrics['macro_f1'] * 100:.2f}%")
        print(final_metrics["report"])

        # Save training curves
        from utils.visualization import plot_training_curves
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path=str(save_dir / f"fold{fold + 1}_curves.png"),
            title=f"Fold {fold + 1} Training Curves",
        )

    # ── Cross-validation summary ──────────────────────────────────────────
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    print("\n" + "=" * 60)
    print("  K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, r in enumerate(fold_results):
        print(f"  Fold {i + 1}: acc={r['accuracy'] * 100:.2f}%  f1={r['macro_f1'] * 100:.2f}%")
    print(f"\n  Mean accuracy : {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
    print(f"  Mean macro-F1 : {np.mean(f1s) * 100:.2f}% ± {np.std(f1s) * 100:.2f}%")
    print(f"  Acc fluctuation (max-min): {(max(accs) - min(accs)) * 100:.2f}%")

    summary = {
        "fold_accuracies": [float(a) for a in accs],
        "fold_f1s": [float(f) for f in f1s],
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
    }
    with open(save_dir / "kfold_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to {save_dir / 'kfold_summary.json'}")


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train FusionAttend-Net")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="",
                        help="cuda device, e.g. '0' or 'cpu' (auto-detected if blank)")
    # Allow overriding nested config keys as 'section.key=value'
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["optimizer.lr"] = args.lr
    if args.save_dir is not None:
        overrides["output.save_dir"] = args.save_dir
    if args.data_root is not None:
        overrides["dataset.root"] = args.data_root

    cfg = load_config(args.config, overrides)

    if args.device:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    run_kfold(cfg, device)


if __name__ == "__main__":
    main()
