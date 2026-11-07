"""Train the expression CNN on FER-2013.

Usage (from the project root):
    python -m src.train
    python -m src.train --epochs 50 --batch-size 64

The best model (lowest validation loss) is saved to models/emotion_model.keras.
After training, the model is evaluated on the held-out test set and all results
are written to results/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from src import config
from src.data_loader import DatasetError, assert_no_leakage, load_raw_dataset, make_splits
from src.preprocessing import encode_labels, images_to_model_input
from src.visualization import plot_training_history


def compute_class_weights(labels: np.ndarray, power: float = 0.5) -> dict:
    """Softened inverse-frequency class weights (helps rare classes such as Disgust)."""
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(float)
    weights = counts.sum() / (config.NUM_CLASSES * np.maximum(counts, 1.0))
    return {i: float(w ** power) for i, w in enumerate(weights)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the facial expression CNN.")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Do not evaluate after training.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        raw = load_raw_dataset(args.dataset_dir)
    except DatasetError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    splits = make_splits(raw, val_size=args.val_size, seed=args.seed)
    assert_no_leakage(splits)
    print("Dataset split:", json.dumps(splits.report, indent=2))

    try:
        from tensorflow import keras
    except ImportError:
        print("[ERROR] TensorFlow is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 3
    from src.model import build_model

    keras.utils.set_random_seed(args.seed)
    X_train, X_val = images_to_model_input(splits.X_train), images_to_model_input(splits.X_val)
    Y_train, Y_val = encode_labels(splits.y_train), encode_labels(splits.y_val)

    model = build_model(learning_rate=args.learning_rate)
    model.summary()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(args.model_path), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                          min_lr=1e-6, verbose=1),
    ]
    class_weight = None if args.no_class_weights else compute_class_weights(splits.y_train)

    history = model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=class_weight, callbacks=callbacks, verbose=2,
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    config.HISTORY_PATH.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
    plot_training_history(history_dict, config.RESULTS_DIR / "training_history.png")
    (config.RESULTS_DIR / "training_config.json").write_text(json.dumps({
        "epochs_requested": args.epochs, "epochs_run": len(history_dict.get("loss", [])),
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "class_weights": class_weight, "seed": args.seed, "split": splits.report,
    }, indent=2), encoding="utf-8")
    print(f"\nBest model saved to: {args.model_path}")

    if not args.skip_eval:
        from src.evaluate import evaluate_model
        evaluate_model(args.model_path, args.dataset_dir, val_size=args.val_size, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
