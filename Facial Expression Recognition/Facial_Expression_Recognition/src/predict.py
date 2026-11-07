"""Expression prediction for images and live webcam video.

Command line:
    python -m src.predict --image path/to/photo.jpg [--output annotated.jpg]
    python -m src.predict --webcam            (press Q or ESC to stop)

Predictions are *visual expression classifications* of the face pixels. They do
not reveal a person's true emotional state.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src import config
from src.face_detection import Box, FaceDetector, FaceDetectorError
from src.preprocessing import InvalidImageError, decode_image, prepare_batch, to_grayscale


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained model is missing, unreadable or incompatible."""


@dataclass
class FacePrediction:
    box: Box
    label: str                       # e.g. "happy"
    confidence: float                # 0..1
    probabilities: Dict[str, float]  # all seven classes


class ExpressionPredictor:
    """Detect faces and classify their expression with the trained CNN."""

    def __init__(self, model_path: Path = config.MODEL_PATH,
                 detector: Optional[FaceDetector] = None) -> None:
        self.model_path = Path(model_path)
        self._detector = detector
        self._model = None

    # -- model handling ---------------------------------------------------- #
    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    def model_file_exists(self) -> bool:
        return self.model_path.is_file()

    def load(self) -> None:
        """Load the model once. Raises ModelNotAvailableError with a clear message."""
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise ModelNotAvailableError(
                f"No trained model found at '{self.model_path}'. "
                "Train one first with: python -m src.train")
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ModelNotAvailableError(
                "TensorFlow is not installed. Run: pip install -r requirements.txt") from exc
        try:
            model = keras.models.load_model(str(self.model_path), compile=False)
        except Exception as exc:  # corrupt file, version mismatch, ...
            raise ModelNotAvailableError(
                f"Could not load the model '{self.model_path}': {exc}") from exc
        if tuple(model.input_shape[1:]) != (config.IMG_SIZE, config.IMG_SIZE, 1) \
                or model.output_shape[-1] != config.NUM_CLASSES:
            raise ModelNotAvailableError(
                f"Model has an unexpected shape (input {model.input_shape}, "
                f"output {model.output_shape}); expected 48x48x1 -> 7 classes.")
        self._model = model

    # -- prediction -------------------------------------------------------- #
    def predict_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return an (N, 7) array of class probabilities for N face crops."""
        self.load()
        if len(crops) == 0:
            return np.zeros((0, config.NUM_CLASSES), dtype=np.float32)
        batch = prepare_batch(crops)
        return np.asarray(self._model(batch, training=False))

    def predict_boxes(self, image: np.ndarray, boxes: List[Box]) -> List[FacePrediction]:
        """Classify the faces at the given boxes of a BGR image."""
        gray = to_grayscale(image)
        crops = FaceDetector.crop(gray, boxes)
        probs = self.predict_crops(crops)
        results = []
        for box, row in zip(boxes, probs):
            top = int(np.argmax(row))
            results.append(FacePrediction(
                box=box, label=config.EMOTIONS[top], confidence=float(row[top]),
                probabilities={name: float(p) for name, p in zip(config.EMOTIONS, row)}))
        return results

    def analyze(self, image: np.ndarray) -> List[FacePrediction]:
        """Detect every face in a BGR image and classify each one."""
        boxes = self.detector.detect(image)
        return self.predict_boxes(image, boxes) if boxes else []


# -- drawing --------------------------------------------------------------- #
def _put_label(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = y - 8 if y - th - baseline - 8 > 0 else y + th + 12
    cv2.rectangle(image, (x, y_text - th - 4), (x + tw + 6, y_text + baseline), color, -1)
    cv2.putText(image, text, (x + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (20, 20, 20), 2, cv2.LINE_AA)


def draw_predictions(image: np.ndarray, predictions: Sequence[FacePrediction],
                     no_face_message: bool = True) -> np.ndarray:
    """Return a copy of ``image`` with boxes, labels and confidence drawn on it."""
    output = image.copy()
    for pred in predictions:
        x, y, w, h = pred.box
        color = config.EMOTION_COLORS_BGR[pred.label]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        _put_label(output, f"{pred.label.capitalize()} {pred.confidence * 100:.1f}%", x, y, color)
    if not predictions and no_face_message:
        cv2.putText(output, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def draw_boxes(image: np.ndarray, boxes: Sequence[Box], text: str = "Face") -> np.ndarray:
    """Draw plain boxes (used when no trained model is available)."""
    output = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), (200, 160, 40), 2)
        _put_label(output, text, x, y, (200, 160, 40))
    return output


# -- webcam ---------------------------------------------------------------- #
def open_camera(index: int = 0) -> Optional[cv2.VideoCapture]:
    """Open a webcam; returns None (instead of raising) if it is unavailable."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    try:
        capture = cv2.VideoCapture(index, backend)
    except Exception:
        return None
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def run_webcam(camera_index: int = 0, predictor: Optional[ExpressionPredictor] = None) -> int:
    """Show live predictions in an OpenCV window. Press Q or ESC to stop."""
    predictor = predictor or ExpressionPredictor()
    try:
        predictor.load()
        predictor.detector  # initialise the detector now to fail early
    except (ModelNotAvailableError, FaceDetectorError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    capture = open_camera(camera_index)
    if capture is None:
        print(f"[ERROR] Could not open webcam {camera_index}. Check that it is connected, "
              "not used by another application, and that camera access is allowed.",
              file=sys.stderr)
        return 1
    window = "Facial Expression Recognition - press Q or ESC to quit"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[ERROR] Lost connection to the webcam.", file=sys.stderr)
                return 1
            frame = cv2.flip(frame, 1)  # mirror view
            annotated = draw_predictions(frame, predictor.analyze(frame))
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless OpenCV builds have no GUI
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predict facial expressions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a JPG/PNG image.")
    group.add_argument("--webcam", action="store_true", help="Use the webcam.")
    parser.add_argument("--output", type=Path, help="Save the annotated image here.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args(argv)

    predictor = ExpressionPredictor(args.model_path)
    if args.webcam:
        return run_webcam(args.camera_index, predictor)
    try:
        image = decode_image(args.image.read_bytes(), max_side=1280)
        predictions = predictor.analyze(image)
    except (OSError, InvalidImageError, ModelNotAvailableError, FaceDetectorError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    if not predictions:
        print("No face detected.")
    for i, pred in enumerate(predictions, start=1):
        print(f"Face {i}: {pred.label.capitalize()} ({pred.confidence * 100:.1f}%) box={pred.box}")
    if args.output:
        cv2.imwrite(str(args.output), draw_predictions(image, predictions))
        print(f"Annotated image saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
