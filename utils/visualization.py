"""
Visualization utilities for FusionAttend-Net.

Provides:
    * :func:`plot_tsne`              – t-SNE 2-D scatter plot of feature embeddings.
    * :func:`plot_confusion_matrix`  – annotated heat-map of a confusion matrix.
    * :func:`plot_training_curves`   – loss and accuracy curves over epochs.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend (safe in training scripts)


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "t-SNE Feature Visualization",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> None:
    """
    Reduce high-dimensional feature vectors to 2-D with t-SNE and plot a
    colour-coded scatter diagram to reveal class separability.

    Args:
        features: Float array of shape ``(N, D)`` – penultimate layer
            activations extracted via :meth:`FusionAttendNet.extract`.
        labels: Integer array of shape ``(N,)`` – class indices.
        class_names: Optional list of human-readable class names.
        save_path: If provided the figure is saved to this path; otherwise
            the figure is shown interactively.
        title: Plot title.
        perplexity: t-SNE perplexity (default 30).
        n_iter: Number of t-SNE optimisation iterations (default 1000).
        random_state: NumPy / sklearn random seed for reproducibility.
    """
    from sklearn.manifold import TSNE

    print(f"Running t-SNE on {features.shape[0]} samples …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    embedded = tsne.fit_transform(features)

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        label_name = class_names[lbl] if class_names is not None else str(lbl)
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=[cmap(idx)],
            label=label_name,
            s=10,
            alpha=0.7,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=6, markerscale=2, ncol=2)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Draw an annotated confusion-matrix heat-map.

    Args:
        cm: Square integer array of shape ``(C, C)`` from
            :func:`utils.metrics.compute_metrics`.
        class_names: Optional list of class labels for axes ticks.
        save_path: Optional file path to save the figure.
        title: Plot title.
        normalize: If ``True``, normalize each row to show recall.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums == 0, 0.0, cm.astype(float) / row_sums)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    num_classes = cm.shape[0]
    fig_size = max(8, num_classes * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    if class_names:
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

    thresh = cm_plot.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = f"{cm_plot[i, j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                    fontsize=max(4, 8 - num_classes // 10))

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> None:
    """
    Plot loss and accuracy curves over training epochs.

    Args:
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        train_accs: Per-epoch training accuracy values (0–1).
        val_accs: Per-epoch validation accuracy values (0–1).
        save_path: Optional file path to save the figure.
        title: Suptitle for the figure.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    # Loss
    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train acc")
    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
