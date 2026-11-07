"""Data-loader tests use tiny SYNTHETIC images, only to test code paths (never for training)."""
import cv2
import numpy as np
import pandas as pd
import pytest

from src.config import EMOTIONS
from src.data_loader import (DatasetError, assert_no_leakage, class_distribution,
                             image_hash, inspect_dataset, load_raw_dataset, make_splits)

RNG = np.random.default_rng(0)


def _write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())


@pytest.fixture()
def folder_dataset(tmp_path):
    for emotion in EMOTIONS:
        for i in range(12):
            _write_png(tmp_path / "train" / emotion / f"{i}.png",
                       RNG.integers(0, 256, (48, 48), dtype=np.uint8))
        for i in range(4):
            _write_png(tmp_path / "test" / emotion / f"{i}.png",
                       RNG.integers(0, 256, (48, 48), dtype=np.uint8))
    # a corrupt file
    (tmp_path / "train" / "happy" / "broken.png").write_bytes(b"garbage")
    # the same image in train and test -> must not leak
    shared = RNG.integers(0, 256, (48, 48), dtype=np.uint8)
    _write_png(tmp_path / "train" / "sad" / "dup.png", shared)
    _write_png(tmp_path / "test" / "sad" / "dup.png", shared)
    return tmp_path


def test_folder_layout_loads_and_reports_corrupt(folder_dataset):
    raw = load_raw_dataset(folder_dataset)
    assert raw.layout == "folders"
    assert raw.images.shape[1:] == (48, 48)
    assert len(raw.report["corrupt_files"]) == 1
    summary = inspect_dataset(raw)
    assert summary["exact_duplicate_images"] == 1
    assert set(class_distribution(raw).index) == set(EMOTIONS)


def test_splits_are_leakage_free_and_stratified(folder_dataset):
    splits = make_splits(load_raw_dataset(folder_dataset), val_size=0.25, seed=1)
    assert_no_leakage(splits)
    assert splits.report["duplicates_removed"] == 1
    assert splits.report["official_test_partition_used"] is True
    assert set(np.unique(splits.y_val)) == set(range(7))
    train_hashes = {image_hash(i) for i in splits.X_train}
    assert not train_hashes & {image_hash(i) for i in splits.X_test}


def test_csv_layout(tmp_path):
    rows = []
    for usage in ("Training", "PublicTest", "PrivateTest"):
        for code in range(7):
            for _ in range(3):
                rows.append({"emotion": code, "Usage": usage,
                             "pixels": " ".join(map(str, RNG.integers(0, 256, 48 * 48)))})
    rows.append({"emotion": 1, "Usage": "Training", "pixels": "1 2 3"})  # invalid row
    pd.DataFrame(rows).to_csv(tmp_path / "fer2013.csv", index=False)
    raw = load_raw_dataset(tmp_path)
    assert raw.layout == "csv" and raw.report["invalid_rows"] == 1
    splits = make_splits(raw)
    assert splits.report["sizes"] == {"train": 21, "val": 21, "test": 21}


def test_missing_dataset_gives_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="Place the FER-2013 dataset"):
        load_raw_dataset(tmp_path / "does_not_exist")
    with pytest.raises(DatasetError):
        load_raw_dataset(tmp_path)  # exists but empty
