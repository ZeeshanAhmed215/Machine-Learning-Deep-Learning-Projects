"""Central configuration: relative paths, class names and constants.

Every path is derived from the project root, so the project can live anywhere.
The dataset location can be overridden with the FER_DATASET_DIR environment
variable if the data is stored elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = Path(os.environ.get("FER_DATASET_DIR", PROJECT_ROOT / "dataset"))
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"

MODEL_PATH = MODELS_DIR / "emotion_model.keras"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_summary.json"

IMG_SIZE = 48
NUM_CLASSES = 7
RANDOM_SEED = 42

# Alphabetical order = the order of the class folders in the common Kaggle
# version of FER-2013. The model's output index i corresponds to EMOTIONS[i].
EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
DISPLAY_NAMES = [name.capitalize() for name in EMOTIONS]

EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

# Label codes used by the original fer2013.csv file.
CSV_EMOTION_CODES = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral",
}
CSV_USAGE_TO_PARTITION = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}

# Alternative folder names that some FER-2013 copies use.
CLASS_ALIASES = {
    "anger": "angry", "disgusted": "disgust", "fearful": "fear",
    "happiness": "happy", "sadness": "sad", "surprised": "surprise",
}
PARTITION_FOLDERS = {"train": "train", "training": "train", "val": "val",
                     "validation": "val", "test": "test", "testing": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Annotation colours (BGR, as used by OpenCV).
EMOTION_COLORS_BGR = {
    "angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (180, 80, 160),
    "happy": (60, 200, 255), "neutral": (190, 190, 190), "sad": (230, 140, 60),
    "surprise": (0, 150, 255),
}
