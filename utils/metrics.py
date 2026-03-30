"""
Evaluation metrics for FusionAttend-Net.

Provides:
    * :func:`compute_metrics` – accuracy, precision, recall, per-class and
      macro-averaged F1-score.
    * :func:`compute_model_stats` – parameter count and GFLOPs.
    * :class:`AverageMeter` – running mean tracker for loss / accuracy.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score


class AverageMeter:
    """Computes and stores the running average of a scalar quantity."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: avg={self.avg:.4f}"


def compute_metrics(
    all_labels: list,
    all_preds: list,
    class_names: Optional[list] = None,
) -> Dict:
    """
    Compute classification metrics from flat lists of ground-truth labels
    and model predictions.

    Args:
        all_labels: Ground-truth integer class labels.
        all_preds: Predicted integer class labels.
        class_names: Optional list of human-readable class names.

    Returns:
        Dictionary with keys:
            * ``"accuracy"``         – top-1 accuracy (float, 0–1).
            * ``"macro_f1"``         – macro-averaged F1-score.
            * ``"macro_precision"``  – macro-averaged precision.
            * ``"macro_recall"``     – macro-averaged recall.
            * ``"report"``           – full per-class ``classification_report`` string.
            * ``"confusion_matrix"`` – NumPy confusion matrix (C×C).
    """
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    accuracy = float((labels_arr == preds_arr).mean())
    report = classification_report(
        labels_arr,
        preds_arr,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(labels_arr, preds_arr)

    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0))
    macro_precision = float(precision_score(labels_arr, preds_arr, average="macro", zero_division=0))
    macro_recall = float(recall_score(labels_arr, preds_arr, average="macro", zero_division=0))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "report": report,
        "confusion_matrix": cm,
    }


def compute_model_stats(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 256, 256),
) -> Dict:
    """
    Compute the parameter count and GFLOPs for a model.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape ``(B, C, H, W)`` (default
            ``(1, 3, 256, 256)``).

    Returns:
        Dictionary with keys:
            * ``"params"``   – total trainable parameter count (int).
            * ``"gflops"``   – GFLOPs estimate (float; ``-1.0`` if ``thop``
              is not installed).
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gflops = -1.0
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(*input_size, device=device)
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = macs * 2 / 1e9
    except ImportError:
        pass

    return {"params": params, "gflops": gflops}


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    Compute top-k accuracies in a single forward pass.

    Args:
        output: Model logits of shape ``(B, C)``.
        target: Ground-truth labels of shape ``(B,)``.
        topk: Tuple of k values to compute (e.g. ``(1, 5)``).

    Returns:
        List of top-k accuracy floats (0–100 %).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(float(correct_k.mul_(100.0 / batch_size)))
        return results
