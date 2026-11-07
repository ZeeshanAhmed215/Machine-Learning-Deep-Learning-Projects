"""Evaluate the trained model on the held-out test set.

Usage:  python -m src.evaluate

Writes to results/: metrics.json, classification_report.txt, confusion_matrix.png
(and regenerates training_history.png if a training history file exists).
All numbers come from running the model on the test images; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config
from src.data_loader import DatasetError, load_raw_dataset, make_splits
from src.preprocessing import encode_labels, images_to_model_input
from src.visualization import plot_confusion_matrix, plot_training_history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Accuracy, macro/weighted precision-recall-F1, per-class report, confusion matrix."""
    labels = list(range(config.NUM_CLASSES))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=config.DISPLAY_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]),
                     "f1": float(weighted[2])},
        "per_class": {name: report[name] for name in config.DISPLAY_NAMES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_model(model_path: Path = config.MODEL_PATH,
                   dataset_dir: Path = config.DATASET_DIR,
                   val_size: float = 0.15, seed: int = config.RANDOM_SEED) -> Dict:
    from tensorflow import keras

    raw = load_raw_dataset(dataset_dir)
    splits = make_splits(raw, val_size=val_size, seed=seed)
    model = keras.models.load_model(str(model_path))

    X_test = images_to_model_input(splits.X_test)
    y_test = splits.y_test
    test_loss, _ = model.evaluate(X_test, encode_labels(y_test), batch_size=256, verbose=0)
    y_pred = model.predict(X_test, batch_size=256, verbose=0).argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred)
    metrics["test_loss"] = float(test_loss)
    metrics["num_test_images"] = int(len(y_test))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    text_report = classification_report(y_test, y_pred, labels=list(range(config.NUM_CLASSES)),
                                        target_names=config.DISPLAY_NAMES, digits=4, zero_division=0)
    (config.RESULTS_DIR / "classification_report.txt").write_text(
        f"Test images: {len(y_test)}\nTest accuracy: {metrics['accuracy']:.4f}\n"
        f"Test loss: {metrics['test_loss']:.4f}\n\n{text_report}", encoding="utf-8")
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                          config.RESULTS_DIR / "confusion_matrix.png")
    if config.HISTORY_PATH.is_file():
        history = json.loads(config.HISTORY_PATH.read_text(encoding="utf-8"))
        plot_training_history(history, config.RESULTS_DIR / "training_history.png")

    print(f"Test images: {len(y_test)}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(text_report)
    print(f"Results written to {config.RESULTS_DIR}")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args(argv)
    if not args.model_path.is_file():
        print(f"[ERROR] Model not found: {args.model_path}\nTrain it first: python -m src.train",
              file=sys.stderr)
        return 2
    try:
        evaluate_model(args.model_path, args.dataset_dir, args.val_size, args.seed)
    except DatasetError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    except ImportError:
        print("[ERROR] TensorFlow is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
