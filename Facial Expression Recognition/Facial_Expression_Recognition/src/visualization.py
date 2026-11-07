"""Matplotlib / Seaborn figures (saved to disk, no GUI backend required)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.config import DISPLAY_NAMES, EMOTIONS


def _save(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_class_distribution(table: pd.DataFrame, path: Path) -> Path:
    """Bar chart of image counts per emotion (``table`` comes from class_distribution)."""
    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    totals = table["total"] if "total" in table.columns else table.sum(axis=1)
    names = [str(i).capitalize() for i in table.index]
    bars = ax.bar(names, totals.values, color=sns.color_palette("viridis", len(names)))
    for bar, value in zip(bars, totals.values):
        ax.annotate(f"{int(value):,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("FER-2013 class distribution")
    ax.set_ylabel("Number of images")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_sample_images(images: np.ndarray, labels: np.ndarray, path: Path,
                       per_class: int = 6) -> Path:
    """Grid showing the first few images of each emotion."""
    fig = Figure(figsize=(per_class * 1.4, len(EMOTIONS) * 1.5))
    axes = fig.subplots(len(EMOTIONS), per_class)
    for row, name in enumerate(DISPLAY_NAMES):
        indices = np.flatnonzero(labels == row)[:per_class]
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(indices):
                ax.imshow(images[indices[col]], cmap="gray", vmin=0, vmax=255)
            if col == 0:
                ax.text(-0.1, 0.5, name, transform=ax.transAxes, ha="right", va="center")
    fig.suptitle("Sample images per class", y=0.995)
    return _save(fig, path)


def plot_training_history(history: Dict[str, List[float]], path: Path) -> Path:
    """Training/validation accuracy and loss curves."""
    fig = Figure(figsize=(12, 4.5))
    acc_ax, loss_ax = fig.subplots(1, 2)
    epochs = range(1, len(history.get("loss", [])) + 1)
    for ax, metric, title in ((acc_ax, "accuracy", "Accuracy"), (loss_ax, "loss", "Loss")):
        if metric in history:
            ax.plot(epochs, history[metric], label=f"Training {metric}")
        if f"val_{metric}" in history:
            ax.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}")
        ax.set_title(f"{title} per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(matrix: np.ndarray, path: Path,
                          labels: Sequence[str] = tuple(DISPLAY_NAMES)) -> Path:
    """Side-by-side confusion matrices: raw counts and row-normalised."""
    matrix = np.asarray(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros(matrix.shape, dtype=float),
                           where=row_sums != 0)
    fig = Figure(figsize=(15, 6))
    left, right = fig.subplots(1, 2)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=left, cbar=False)
    left.set_title("Confusion matrix (counts)")
    sns.heatmap(normalised, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels,
                yticklabels=labels, ax=right, cbar=False, vmin=0, vmax=1)
    right.set_title("Confusion matrix (row-normalised = recall)")
    for ax in (left, right):
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    return _save(fig, path)
