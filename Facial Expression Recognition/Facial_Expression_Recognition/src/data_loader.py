"""FER-2013 dataset inspection, loading and leakage-safe splitting.

Two common distributions of FER-2013 are supported automatically:

1. Image folders (Kaggle "msambare/fer2013"):
       dataset/train/<emotion>/*.jpg   and   dataset/test/<emotion>/*.jpg
2. The original CSV file (``fer2013.csv`` with columns emotion, pixels, Usage).

Leakage-safe strategy
---------------------
FER-2013 is known to contain exact duplicate images. If a duplicate lands in
both the training and the test set, the test score is inflated. This module:

* hashes every image and keeps only ONE copy of each identical image, giving
  priority to test > validation > training (so a training image that also
  occurs in the test set is dropped from training);
* uses the official test partition when the dataset provides one, otherwise a
  stratified split;
* carves the validation set out of the training pool with a stratified split;
* provides ``assert_no_leakage`` to verify that the three sets are disjoint.

Run ``python -m src.data_loader`` to print an inspection report and save the
class-distribution plot and sample images to ``results/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.config import (CLASS_ALIASES, CSV_EMOTION_CODES, CSV_USAGE_TO_PARTITION,
                        EMOTIONS, IMAGE_EXTENSIONS, IMG_SIZE, PARTITION_FOLDERS,
                        RANDOM_SEED)


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or has an unsupported structure."""


@dataclass
class RawDataset:
    """All images found in the dataset before any splitting."""

    images: np.ndarray          # (N, 48, 48) uint8
    labels: np.ndarray          # (N,) int, index into config.EMOTIONS
    partitions: np.ndarray      # (N,) str: "train", "val", "test" or ""
    layout: str                 # "folders" or "csv"
    report: Dict = field(default_factory=dict)


@dataclass
class Splits:
    """Train / validation / test arrays plus a description of what was done."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    report: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Locating and reading the data
# --------------------------------------------------------------------------- #
def _iter_dirs(base: Path, max_depth: int = 3) -> Iterator[Path]:
    """Yield ``base`` and its sub-directories down to ``max_depth`` levels."""
    yield base
    if max_depth <= 0:
        return
    try:
        children = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return
    for child in children:
        yield from _iter_dirs(child, max_depth - 1)


def _find_folder_root(base: Path) -> Optional[Path]:
    for root in _iter_dirs(base):
        names = {p.name.lower() for p in root.iterdir() if p.is_dir()}
        if any(PARTITION_FOLDERS.get(n) == "train" for n in names):
            return root
    return None


def _find_csv(base: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for root in _iter_dirs(base, max_depth=2):
        candidates.extend(sorted(root.glob("*.csv")))
    candidates.sort(key=lambda p: 0 if "fer" in p.name.lower() else 1)
    for csv_path in candidates:
        try:
            header = pd.read_csv(csv_path, nrows=0).columns
        except Exception:  # unreadable CSV: ignore and keep looking
            continue
        if {"emotion", "pixels"}.issubset(set(header)):
            return csv_path
    return None


def _decode_gray(path: Path) -> Optional[np.ndarray]:
    """Read an image as grayscale. Uses imdecode so non-ASCII Windows paths work."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _class_from_folder(name: str) -> Optional[str]:
    key = name.strip().lower()
    key = CLASS_ALIASES.get(key, key)
    return key if key in EMOTIONS else None


def _load_folders(root: Path) -> RawDataset:
    images, labels, parts = [], [], []
    corrupt: List[str] = []
    unknown_dirs: List[str] = []
    resized = 0
    for part_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        partition = PARTITION_FOLDERS.get(part_dir.name.lower())
        if partition is None:
            continue
        for class_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            emotion = _class_from_folder(class_dir.name)
            if emotion is None:
                unknown_dirs.append(str(class_dir.relative_to(root)))
                continue
            label = EMOTIONS.index(emotion)
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                img = _decode_gray(path)
                if img is None or img.ndim != 2 or img.size == 0:
                    corrupt.append(str(path.relative_to(root)))
                    continue
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    resized += 1
                images.append(img)
                labels.append(label)
                parts.append(partition)
    if not images:
        raise DatasetError(f"No readable images with known emotion folders found in {root}.")
    report = {"corrupt_files": corrupt, "unknown_class_folders": unknown_dirs,
              "resized_images": resized, "invalid_rows": 0, "source": str(root)}
    return RawDataset(np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64),
                      np.array(parts), "folders", report)


def _load_csv(csv_path: Path) -> RawDataset:
    frame = pd.read_csv(csv_path)
    images, labels, parts = [], [], []
    invalid = 0
    expected = IMG_SIZE * IMG_SIZE
    usages = frame["Usage"] if "Usage" in frame.columns else pd.Series([""] * len(frame))
    for emotion_code, pixel_text, usage in zip(frame["emotion"], frame["pixels"], usages):
        try:
            emotion = CSV_EMOTION_CODES[int(emotion_code)]
            values = np.array(str(pixel_text).split(), dtype=np.int64)
        except (KeyError, ValueError, TypeError):
            invalid += 1
            continue
        if values.size != expected or values.min() < 0 or values.max() > 255:
            invalid += 1
            continue
        images.append(values.astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE))
        labels.append(EMOTIONS.index(emotion))
        parts.append(CSV_USAGE_TO_PARTITION.get(str(usage), ""))
    if not images:
        raise DatasetError(f"No valid rows found in {csv_path}.")
    report = {"corrupt_files": [], "unknown_class_folders": [], "resized_images": 0,
              "invalid_rows": invalid, "source": str(csv_path)}
    return RawDataset(np.stack(images), np.array(labels, dtype=np.int64),
                      np.array(parts), "csv", report)


def load_raw_dataset(dataset_dir: Path = config.DATASET_DIR) -> RawDataset:
    """Detect the dataset layout inside ``dataset_dir`` and load every image."""
    dataset_dir = Path(dataset_dir)
    help_text = (
        f"Place the FER-2013 dataset inside '{dataset_dir}'. Supported layouts:\n"
        "  1) dataset/train/<emotion>/*.jpg and dataset/test/<emotion>/*.jpg\n"
        "  2) dataset/fer2013.csv (columns: emotion, pixels, Usage)\n"
        "Emotion folders: angry, disgust, fear, happy, neutral, sad, surprise."
    )
    if not dataset_dir.is_dir():
        raise DatasetError(f"Dataset directory not found: {dataset_dir}\n{help_text}")
    folder_root = _find_folder_root(dataset_dir)
    if folder_root is not None:
        return _load_folders(folder_root)
    csv_path = _find_csv(dataset_dir)
    if csv_path is not None:
        return _load_csv(csv_path)
    raise DatasetError(f"No FER-2013 data recognised in {dataset_dir}.\n{help_text}")


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def image_hash(image: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(image).tobytes()).hexdigest()


def class_distribution(raw: RawDataset) -> pd.DataFrame:
    """Image counts per emotion and partition (a pandas DataFrame)."""
    frame = pd.DataFrame({
        "emotion": [EMOTIONS[i] for i in raw.labels],
        "partition": [p if p else "unassigned" for p in raw.partitions],
    })
    table = pd.crosstab(frame["emotion"], frame["partition"])
    table = table.reindex(list(EMOTIONS), fill_value=0)
    table["total"] = table.sum(axis=1)
    return table


def inspect_dataset(raw: RawDataset) -> Dict:
    """Summarise the dataset: layout, counts, corrupt files and duplicates."""
    hashes = [image_hash(img) for img in raw.images]
    duplicates = len(hashes) - len(set(hashes))
    table = class_distribution(raw)
    return {
        "layout": raw.layout,
        "source": raw.report.get("source"),
        "image_shape": [IMG_SIZE, IMG_SIZE],
        "total_images": int(len(raw.images)),
        "corrupt_or_unreadable_files": len(raw.report.get("corrupt_files", [])),
        "invalid_csv_rows": int(raw.report.get("invalid_rows", 0)),
        "resized_images": int(raw.report.get("resized_images", 0)),
        "unknown_class_folders": raw.report.get("unknown_class_folders", []),
        "exact_duplicate_images": int(duplicates),
        "class_counts": {k: {c: int(v) for c, v in row.items()}
                         for k, row in table.to_dict(orient="index").items()},
    }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(raw: RawDataset, val_size: float = 0.15, test_size: float = 0.15,
                seed: int = RANDOM_SEED) -> Splits:
    """Create deduplicated, stratified train / validation / test sets.

    ``test_size`` is only used when the dataset has no official test partition.
    ``val_size`` is the fraction of the remaining training pool used for
    validation when the dataset has no official validation partition.
    """
    priority = {"test": 0, "val": 1, "train": 2}
    order = sorted(range(len(raw.images)), key=lambda i: priority.get(raw.partitions[i], 3))
    seen, kept = set(), []
    for idx in order:  # sorted() is stable, so ties keep their original order
        digest = image_hash(raw.images[idx])
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(idx)
    kept = np.array(sorted(kept))
    removed = len(raw.images) - len(kept)

    X, y, parts = raw.images[kept], raw.labels[kept], raw.partitions[kept]
    train_pool = np.isin(parts, ["train", ""])
    has_test = bool((parts == "test").any())
    has_val = bool((parts == "val").any())

    X_pool, y_pool = X[train_pool], y[train_pool]
    if has_test:
        X_test, y_test = X[parts == "test"], y[parts == "test"]
    else:
        X_pool, X_test, y_pool, y_test = train_test_split(
            X_pool, y_pool, test_size=test_size, stratify=y_pool, random_state=seed)

    if has_val:
        X_val, y_val = X[parts == "val"], y[parts == "val"]
        X_train, y_train = X_pool, y_pool
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool, test_size=val_size, stratify=y_pool, random_state=seed)

    splits = Splits(X_train, y_train, X_val, y_val, X_test, y_test, report={
        "duplicates_removed": int(removed),
        "official_test_partition_used": has_test,
        "official_val_partition_used": has_val,
        "sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "seed": seed,
    })
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: Splits) -> None:
    """Raise AssertionError if any identical image appears in two different sets."""
    sets = {
        "train": {image_hash(i) for i in splits.X_train},
        "val": {image_hash(i) for i in splits.X_val},
        "test": {image_hash(i) for i in splits.X_test},
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"Data leakage: {len(overlap)} identical images in both {a} and {b}."


# --------------------------------------------------------------------------- #
# Command line: inspection report
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    from src import visualization

    parser = argparse.ArgumentParser(description="Inspect the FER-2013 dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    args = parser.parse_args(argv)
    try:
        raw = load_raw_dataset(args.dataset_dir)
    except DatasetError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    summary = inspect_dataset(raw)
    table = class_distribution(raw)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATASET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    visualization.plot_class_distribution(table, config.RESULTS_DIR / "class_distribution.png")
    visualization.plot_sample_images(raw.images, raw.labels, config.RESULTS_DIR / "sample_images.png")

    print(f"Layout: {summary['layout']}   Source: {summary['source']}")
    print(f"Total readable images: {summary['total_images']}")
    print(f"Corrupt/unreadable files: {summary['corrupt_or_unreadable_files']}   "
          f"Invalid CSV rows: {summary['invalid_csv_rows']}")
    print(f"Exact duplicate images: {summary['exact_duplicate_images']}")
    print("\nImages per class and partition:")
    print(table.to_string())
    print(f"\nSaved plots and dataset_summary.json to {config.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
