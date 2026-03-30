"""
FusionAttend-Net Training Script
=================================
Trains FusionAttend-Net with K-fold cross-validation on a plant disease
dataset arranged in the ImageFolder layout.

Key training details (matching the paper):
  * batch_size = 64
  * epochs = 30
  * lr = 0.01
  * Two-phase optimizer: Adam for fast convergence → SGD for fine-tuning
  * momentum = 0.9, weight_decay = 5e-4
  * Label smoothing = 0.1
  * Early stopping: patience = 6 (based on validation accuracy)
  * FGVC8 minority classes (< 500 samples) → strong augmentation × 6
  * Katra-Twelve / BARI-Sunflower → weak augmentation × 2

Quick start::

    python train.py --config configs/default.yaml

Override any config key from the command line::

    python train.py --config configs/fgvc8.yaml --epochs 30 --lr 0.01

Checkpoints and metrics are written to the ``output.save_dir`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import WeightedRandomSampler

from datasets import (
    PlantDiseaseDataset,
    build_dataloader,
    build_train_transforms,
    build_val_transforms,
)
from datasets.augmentation import build_strong_train_transforms, build_weak_train_transforms
from models import FusionAttendNet
from utils.metrics import AverageMeter, compute_metrics, topk_accuracy, compute_model_stats


# ── Constants ─────────────────────────────────────────────────────────────
# FGVC8 minority threshold: classes with fewer samples use strong augmentation
FGVC8_MINORITY_THRESHOLD = 500
# Augmentation repeat counts
STRONG_AUG_REPEAT = 6
WEAK_AUG_REPEAT = 2


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
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg


# ── Two-phase optimizer ────────────────────────────────────────────────────

def build_two_phase_optimizers(model: nn.Module, cfg: dict):
    """
    Build Adam (phase 1: fast convergence) and SGD (phase 2: fine-tuning).

    Phase boundary is controlled by ``training.adam_epochs`` (default:
    half of total epochs).  Returns ``(adam_opt, sgd_opt)``.
    """
    lr = cfg["optimizer"]["lr"]
    wd = cfg["optimizer"]["weight_decay"]
    momentum = cfg["optimizer"].get("momentum", 0.9)

    adam_opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    sgd_opt  = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    return adam_opt, sgd_opt


# ── LR scheduler ──────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict):
    name = cfg.get("scheduler", {}).get("name", "cosine").lower()
    epochs = cfg["training"]["epochs"]
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    return None


# ── Augmented dataset builder ──────────────────────────────────────────────

class RepeatedAugDataset(torch.utils.data.Dataset):
    """
    Wraps a base dataset and repeats each sample *repeat* times, each time
    applying the transform independently.  Used to physically expand the
    dataset for minority classes without loading extra images.
    """

    def __init__(self, base_dataset, repeat: int = 1):
        self.base = base_dataset
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.base) * self.repeat

    def __getitem__(self, idx: int):
        return self.base[idx % len(self.base)]


def build_augmented_train_dataset(
    root: str,
    train_idx: np.ndarray,
    dataset_name: str,
    image_size: int,
    class_counts: Counter,
) -> torch.utils.data.Dataset:
    """
    Build a training dataset that applies strong augmentation × 6 to FGVC8
    minority classes (< 500 samples) and weak augmentation × 2 to all others.

    For simplicity, a single unified transform is applied per fold:
      * FGVC8: strong augmentation applied to the whole training split.
      * Others: weak augmentation.

    The repeat factor simulates the paper's per-image augmentation counts.
    """
    is_fgvc8 = dataset_name.lower() == "fgvc8"

    if is_fgvc8:
        # Determine if any minority class exists
        has_minority = any(c < FGVC8_MINORITY_THRESHOLD for c in class_counts.values())
        transform = build_strong_train_transforms(image_size, dataset_name)
        repeat = STRONG_AUG_REPEAT if has_minority else WEAK_AUG_REPEAT
    else:
        transform = build_weak_train_transforms(image_size, dataset_name)
        repeat = WEAK_AUG_REPEAT

    base_ds = torch.utils.data.Subset(
        PlantDiseaseDataset(root, transform=transform),
        train_idx.tolist(),
    )
    return RepeatedAugDataset(base_ds, repeat=repeat)


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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
    class_names=None,
) -> tuple:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), images.size(0))

        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    return loss_meter.avg, metrics


# ── K-fold training loop ───────────────────────────────────────────────────

def run_kfold(cfg: dict, device: torch.device) -> None:
    set_seed(cfg["training"]["seed"])

    ds_cfg     = cfg["dataset"]
    train_cfg  = cfg["training"]
    optim_cfg  = cfg["optimizer"]
    out_cfg    = cfg["output"]

    dataset_name = ds_cfg["name"]
    image_size   = ds_cfg["image_size"]
    val_tf = build_val_transforms(image_size, dataset_name)

    # Load full dataset to get targets for stratified splitting
    full_ds     = PlantDiseaseDataset(ds_cfg["root"])
    targets     = np.array(full_ds.targets)
    class_names = full_ds.classes
    print(f"Dataset: {dataset_name}  |  samples={len(full_ds)}  |  classes={len(class_names)}")
    print("Class distribution:", full_ds.class_distribution())

    k   = train_cfg["k_folds"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=train_cfg["seed"])

    save_dir = Path(out_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    total_epochs  = train_cfg["epochs"]
    adam_epochs   = train_cfg.get("adam_epochs", total_epochs // 2)
    patience      = train_cfg.get("early_stop_patience", 6)
    batch_size    = train_cfg["batch_size"]
    num_workers   = ds_cfg["num_workers"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'=' * 60}")
        print(f"  Fold {fold + 1}/{k}")
        print(f"{'=' * 60}")

        # Class counts for this fold's training split
        fold_targets = targets[train_idx]
        class_counts = Counter(fold_targets.tolist())

        # Training dataset (with repeated augmentation)
        train_ds = build_augmented_train_dataset(
            root=ds_cfg["root"],
            train_idx=train_idx,
            dataset_name=dataset_name,
            image_size=image_size,
            class_counts=class_counts,
        )

        # Balanced sampler (inverse frequency weighting over expanded dataset)
        if ds_cfg.get("balance_classes", True):
            # Map expanded indices back to original targets for weighting
            repeat = len(train_ds) // len(train_idx)
            expanded_targets = np.tile(fold_targets, repeat)[:len(train_ds)]
            exp_counts = Counter(expanded_targets.tolist())
            sample_weights = torch.tensor(
                [1.0 / exp_counts[int(t)] for t in expanded_targets], dtype=torch.float
            )
            sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        val_ds = torch.utils.data.Subset(
            PlantDiseaseDataset(ds_cfg["root"], transform=val_tf),
            val_idx.tolist(),
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
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
            attention_name=model_cfg.get("attention", "psa"),
        ).to(device)

        stats = compute_model_stats(model)
        print(f"  Parameters: {stats['params']:,}")
        if stats["gflops"] > 0:
            print(f"  GFLOPs    : {stats['gflops']:.2f}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        adam_opt, sgd_opt = build_two_phase_optimizers(model, cfg)
        optimizer = adam_opt
        scheduler = build_scheduler(optimizer, cfg)

        best_val_acc = 0.0
        best_ckpt_path = save_dir / f"fold{fold + 1}_best.pth"
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        epochs_no_improve = 0

        for epoch in range(1, total_epochs + 1):
            # Switch from Adam to SGD after adam_epochs
            if epoch == adam_epochs + 1:
                print(f"  [Epoch {epoch}] Switching optimizer: Adam → SGD")
                optimizer  = sgd_opt
                scheduler  = build_scheduler(optimizer, cfg)

            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, out_cfg["log_interval"],
            )
            vl_loss, vl_metrics = validate(
                model, val_loader, criterion, device, class_names=class_names,
            )
            if scheduler:
                scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_accs.append(tr_acc)
            val_accs.append(vl_metrics["accuracy"])

            elapsed = time.time() - t0
            print(
                f"  [Epoch {epoch:3d}/{total_epochs}]  "
                f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                f"val_acc={vl_metrics['accuracy'] * 100:.2f}%  "
                f"val_f1={vl_metrics['macro_f1'] * 100:.2f}%  "
                f"val_prec={vl_metrics['macro_precision'] * 100:.2f}%  "
                f"val_rec={vl_metrics['macro_recall'] * 100:.2f}%  "
                f"({elapsed:.1f}s)"
            )

            if vl_metrics["accuracy"] > best_val_acc:
                best_val_acc = vl_metrics["accuracy"]
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "metrics": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                    for k, v in vl_metrics.items()},
                    },
                    best_ckpt_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping triggered at epoch {epoch} (patience={patience})")
                    break

        # Final evaluation on best checkpoint
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        _, final_metrics = validate(model, val_loader, criterion, device, class_names=class_names)
        fold_results.append(final_metrics)

        print(f"\n  Fold {fold + 1} results:")
        print(f"  Best val acc  = {final_metrics['accuracy'] * 100:.2f}%")
        print(f"  Macro F1      = {final_metrics['macro_f1'] * 100:.2f}%")
        print(f"  Macro Prec.   = {final_metrics['macro_precision'] * 100:.2f}%")
        print(f"  Macro Recall  = {final_metrics['macro_recall'] * 100:.2f}%")
        print(final_metrics["report"])

        from utils.visualization import plot_training_curves
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path=str(save_dir / f"fold{fold + 1}_curves.png"),
            title=f"Fold {fold + 1} Training Curves",
        )

    # ── Cross-validation summary ──────────────────────────────────────────
    accs  = [r["accuracy"]         for r in fold_results]
    f1s   = [r["macro_f1"]         for r in fold_results]
    precs = [r["macro_precision"]  for r in fold_results]
    recs  = [r["macro_recall"]     for r in fold_results]

    print("\n" + "=" * 60)
    print("  K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, r in enumerate(fold_results):
        print(
            f"  Fold {i + 1}: acc={r['accuracy'] * 100:.2f}%  "
            f"f1={r['macro_f1'] * 100:.2f}%  "
            f"prec={r['macro_precision'] * 100:.2f}%  "
            f"rec={r['macro_recall'] * 100:.2f}%"
        )
    print(f"\n  Mean accuracy  : {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
    print(f"  Mean macro-F1  : {np.mean(f1s) * 100:.2f}% ± {np.std(f1s) * 100:.2f}%")
    print(f"  Mean precision : {np.mean(precs) * 100:.2f}% ± {np.std(precs) * 100:.2f}%")
    print(f"  Mean recall    : {np.mean(recs) * 100:.2f}% ± {np.std(recs) * 100:.2f}%")
    print(f"  Acc fluctuation (max-min): {(max(accs) - min(accs)) * 100:.2f}%")

    summary = {
        "fold_accuracies":   [float(a) for a in accs],
        "fold_f1s":          [float(f) for f in f1s],
        "fold_precisions":   [float(p) for p in precs],
        "fold_recalls":      [float(r) for r in recs],
        "mean_accuracy":     float(np.mean(accs)),
        "std_accuracy":      float(np.std(accs)),
        "mean_f1":           float(np.mean(f1s)),
        "std_f1":            float(np.std(f1s)),
        "mean_precision":    float(np.mean(precs)),
        "mean_recall":       float(np.mean(recs)),
    }
    with open(save_dir / "kfold_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSummary saved to {save_dir / 'kfold_summary.json'}")


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train FusionAttend-Net")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--device",     type=str, default="",
                        help="cuda device, e.g. '0' or 'cpu' (auto-detected if blank)")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--save_dir",   type=str,   default=None)
    parser.add_argument("--data_root",  type=str,   default=None)
    parser.add_argument("--attention",  type=str,   default=None,
                        help="Attention module name for ablation study (e.g. se, eca, cbam)")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.epochs     is not None: overrides["training.epochs"]     = args.epochs
    if args.batch_size is not None: overrides["training.batch_size"] = args.batch_size
    if args.lr         is not None: overrides["optimizer.lr"]        = args.lr
    if args.save_dir   is not None: overrides["output.save_dir"]     = args.save_dir
    if args.data_root  is not None: overrides["dataset.root"]        = args.data_root
    if args.attention  is not None: overrides["model.attention"]     = args.attention

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
