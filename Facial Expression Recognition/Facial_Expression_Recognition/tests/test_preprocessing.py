import cv2
import numpy as np
import pytest

from src.preprocessing import (InvalidImageError, decode_image, encode_labels,
                               images_to_model_input, normalize_pixels, prepare_batch,
                               prepare_face, to_grayscale)


def test_normalize_range_and_dtype():
    out = normalize_pixels(np.array([0, 128, 255], dtype=np.uint8))
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0


def test_images_to_model_input_shape():
    batch = images_to_model_input(np.zeros((5, 48, 48), dtype=np.uint8))
    assert batch.shape == (5, 48, 48, 1)


def test_images_to_model_input_rejects_wrong_size():
    with pytest.raises(ValueError):
        images_to_model_input(np.zeros((2, 64, 64), dtype=np.uint8))


def test_encode_labels_one_hot():
    encoded = encode_labels([0, 3, 6])
    assert encoded.shape == (3, 7)
    assert encoded.sum(axis=1).tolist() == [1, 1, 1]
    assert encoded[1, 3] == 1


def test_encode_labels_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode_labels([7])


@pytest.mark.parametrize("shape", [(120, 90), (120, 90, 3), (30, 30, 4), (48, 48, 1)])
def test_prepare_face_any_input_gives_model_shape(shape):
    face = np.random.randint(0, 256, shape, dtype=np.uint8)
    out = prepare_face(face)
    assert out.shape == (48, 48, 1)
    assert out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0


def test_prepare_batch_empty_and_multiple():
    assert prepare_batch([]).shape == (0, 48, 48, 1)
    faces = [np.zeros((100, 100, 3), np.uint8), np.zeros((60, 80), np.uint8)]
    assert prepare_batch(faces).shape == (2, 48, 48, 1)


def test_to_grayscale_rejects_empty():
    with pytest.raises(InvalidImageError):
        to_grayscale(np.zeros((0, 0), dtype=np.uint8))


def test_decode_image_roundtrip_and_resize():
    img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", img)
    assert ok
    decoded = decode_image(buffer.tobytes())
    assert decoded.shape == (200, 300, 3)
    assert max(decode_image(buffer.tobytes(), max_side=100).shape[:2]) == 100


@pytest.mark.parametrize("payload", [b"", b"this is not an image"])
def test_decode_image_invalid(payload):
    with pytest.raises(InvalidImageError):
        decode_image(payload)
