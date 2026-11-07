"""Prediction-pipeline tests.

The TensorFlow tests build an UNTRAINED network purely to check shapes and the
plumbing (probabilities sum to 1, etc.). They say nothing about accuracy.
"""
import numpy as np
import pytest

from src.config import EMOTIONS
from src.face_detection import FaceDetector, FaceDetectorError, cascade_path
from src.predict import (ExpressionPredictor, FacePrediction, ModelNotAvailableError,
                         draw_predictions)


def test_missing_model_raises_clear_error(tmp_path):
    predictor = ExpressionPredictor(tmp_path / "nope.keras")
    assert not predictor.model_file_exists()
    with pytest.raises(ModelNotAvailableError, match="python -m src.train"):
        predictor.load()


def test_corrupt_model_file_raises_model_error(tmp_path):
    pytest.importorskip("tensorflow")
    bad = tmp_path / "bad.keras"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelNotAvailableError):
        ExpressionPredictor(bad).load()


def test_draw_predictions_handles_no_face_and_faces():
    frame = np.zeros((200, 200, 3), np.uint8)
    assert draw_predictions(frame, []).shape == frame.shape
    pred = FacePrediction((50, 50, 80, 80), "happy", 0.9, {e: 1 / 7 for e in EMOTIONS})
    drawn = draw_predictions(frame, [pred])
    assert drawn.shape == frame.shape and drawn.any()


@pytest.mark.skipif(not cascade_path().is_file(), reason="Haar cascade not available")
def test_detector_returns_no_faces_for_blank_image():
    assert FaceDetector().detect(np.full((240, 320, 3), 127, np.uint8)) == []


def test_detector_error_if_cascade_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("src.face_detection.cascade_path", lambda: tmp_path / "missing.xml")
    with pytest.raises(FaceDetectorError):
        FaceDetector()


def test_pipeline_with_untrained_network(tmp_path):
    pytest.importorskip("tensorflow")
    from src.model import build_model

    model = build_model()
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 7)
    path = tmp_path / "untrained.keras"
    model.save(path)

    predictor = ExpressionPredictor(path)
    crops = [np.random.randint(0, 256, (90, 80, 3), dtype=np.uint8),
             np.random.randint(0, 256, (64, 64), dtype=np.uint8)]
    probs = predictor.predict_crops(crops)
    assert probs.shape == (2, 7)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    boxes = [(10, 10, 80, 90), (100, 20, 64, 64)]
    image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    results = predictor.predict_boxes(image, boxes)
    assert len(results) == 2 and all(r.label in EMOTIONS for r in results)
