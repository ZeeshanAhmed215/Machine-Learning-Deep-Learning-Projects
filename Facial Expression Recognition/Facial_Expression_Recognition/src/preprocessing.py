"""Image preprocessing shared by training, evaluation and inference.

Keeping a single implementation guarantees that the model sees exactly the same
kind of input at training time and at prediction time. This module deliberately
does not import TensorFlow.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from src.config import IMG_SIZE, NUM_CLASSES


class InvalidImageError(ValueError):
    """Raised when bytes or arrays cannot be interpreted as a usable image."""


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0.0, 1.0] as float32."""
    return np.asarray(images, dtype=np.float32) / 255.0


def images_to_model_input(images: np.ndarray) -> np.ndarray:
    """Convert (N, 48, 48) uint8 images to a (N, 48, 48, 1) float32 batch."""
    batch = normalize_pixels(images)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    if batch.ndim != 4 or batch.shape[1:] != (IMG_SIZE, IMG_SIZE, 1):
        raise ValueError(
            f"Expected images shaped (N, {IMG_SIZE}, {IMG_SIZE}[, 1]); got {np.shape(images)}."
        )
    return batch


def encode_labels(labels: Iterable[int], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """One-hot encode integer labels (needed for categorical crossentropy)."""
    labels = np.asarray(list(labels), dtype=int)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}].")
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a 2-D grayscale version of a BGR, BGRA or gray image."""
    if image is None or image.size == 0:
        raise InvalidImageError("The image is empty.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise InvalidImageError(f"Unsupported image shape: {image.shape}.")


def prepare_face(face: np.ndarray) -> np.ndarray:
    """Turn one cropped face into a (48, 48, 1) float32 array in [0, 1]."""
    gray = to_grayscale(face)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    interpolation = cv2.INTER_AREA if min(gray.shape) > IMG_SIZE else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=interpolation)
    return normalize_pixels(resized)[..., np.newaxis]


def prepare_batch(faces: Iterable[np.ndarray]) -> np.ndarray:
    """Preprocess several face crops into one (N, 48, 48, 1) batch."""
    prepared = [prepare_face(face) for face in faces]
    if not prepared:
        return np.zeros((0, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    return np.stack(prepared, axis=0)


def decode_image(data: bytes, max_side: Optional[int] = None) -> np.ndarray:
    """Decode uploaded image bytes into a BGR array, optionally limiting its size."""
    if not data:
        raise InvalidImageError("The file is empty.")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise InvalidImageError("The file is not a valid or supported image.")
    if max_side and max(decoded.shape[:2]) > max_side:
        scale = max_side / max(decoded.shape[:2])
        decoded = cv2.resize(decoded, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return decoded
