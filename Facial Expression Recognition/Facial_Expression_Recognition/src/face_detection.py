"""Face detection with OpenCV's Haar cascade classifier."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.preprocessing import to_grayscale

Box = Tuple[int, int, int, int]  # x, y, width, height

CASCADE_FILENAME = r"D:\Projects\Facial Expression Recognition\Facial_Expression_Recognition\haarcascade_frontalface_default.xml"
# New safe code



class FaceDetectorError(RuntimeError):
    """Raised when the face detector cannot be initialised."""


def cascade_path() -> Path:
    """Location of the Haar cascade file shipped with the opencv-python package."""
    return Path(cv2.data.haarcascades) / CASCADE_FILENAME


class FaceDetector:
    """Detect frontal faces and return their bounding boxes."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (48, 48)) -> None:
        path = cascade_path()
        if not path.is_file():
            raise FaceDetectorError(
                f"Haar cascade file not found at {path}. "
                "Reinstall OpenCV with: pip install --force-reinstall opencv-python"
            )
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise FaceDetectorError(f"Could not load the Haar cascade from {path}.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> List[Box]:
        """Return face boxes (x, y, w, h), sorted left to right."""
        gray = cv2.equalizeHist(to_grayscale(image).astype(np.uint8))
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size,
        )
        boxes = [tuple(int(v) for v in box) for box in faces]
        return sorted(boxes, key=lambda b: b[0])

    @staticmethod
    def crop(image: np.ndarray, boxes: List[Box]) -> List[np.ndarray]:
        """Crop each box out of the image (boxes are clipped to the image bounds)."""
        height, width = image.shape[:2]
        crops = []
        for x, y, w, h in boxes:
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, width), min(y + h, height)
            crops.append(image[y0:y1, x0:x1])
        return crops
