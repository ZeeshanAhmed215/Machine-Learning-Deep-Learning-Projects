"""Streamlit dashboard for AI-based facial expression recognition.

Run with:  streamlit run app.py
All processing happens locally on this computer; no image is uploaded anywhere.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st


from src import config
from src.face_detection import FaceDetectorError
from src.predict import (ExpressionPredictor, ModelNotAvailableError, draw_boxes,
                         draw_predictions, open_camera)
from src.preprocessing import InvalidImageError, decode_image

st.set_page_config(page_title="Facial Expression Recognition", page_icon="🙂", layout="wide")

PAGES = ["🏠 Home", "🖼️ Image Recognition", "🎥 Webcam Recognition",
         "📊 Model Performance", "ℹ️ About Project"]

CSS = """
<style>
.hero {padding: 2rem 2.2rem; border-radius: 18px; color: #fff;
       background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
       margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .4rem 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 0; font-size: 1.05rem; opacity: .95;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1rem;
       text-align: center; background: rgba(128,128,128,.07); height: 100%;}
.card .emoji {font-size: 2.2rem;}
.card .name {font-weight: 700; font-size: 1.1rem; margin-top: .2rem;}
.note {border-left: 4px solid #7c3aed; padding: .6rem 1rem; border-radius: 6px;
       background: rgba(124,58,237,.08); font-size: .92rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

RESPONSIBLE_NOTE = (
    "**Responsible use:** results are *visual expression classifications* of face pixels. "
    "They do **not** reveal a person's true feelings. The model can be wrong and may be "
    "biased by its training data (FER-2013 has noisy labels and uneven demographics). "
    "Images are processed locally and never uploaded or stored."
)


@st.cache_resource(show_spinner=False)
def get_predictor() -> ExpressionPredictor:
    return ExpressionPredictor()


def model_status(predictor: ExpressionPredictor) -> Optional[str]:
    """Return None if the model is ready, otherwise a human-readable problem."""
    try:
        predictor.load()
        return None
    except ModelNotAvailableError as error:
        return str(error)


def get_detector_or_stop(predictor: ExpressionPredictor):
    try:
        return predictor.detector
    except FaceDetectorError as error:
        st.error(f"Face detector unavailable: {error}")
        st.stop()


def read_json(path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.markdown(
        '<div class="hero"><h1>🙂 AI-Based Facial Expression Recognition</h1>'
        "<p>A convolutional neural network (CNN) trained on FER-2013 that classifies "
        "faces into seven visual expression categories, from photos or a live webcam.</p></div>",
        unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Project objectives")
        st.markdown(
            "- Detect faces in images and live video with OpenCV.\n"
            "- Classify each face into one of seven expressions with a Keras CNN.\n"
            "- Report a confidence score and the full probability distribution.\n"
            "- Evaluate honestly on a leakage-safe, held-out test set.\n"
            "- Keep all processing local to protect privacy.")
    with right:
        st.subheader("Quick start")
        st.markdown("1. Open **Image Recognition** and upload a photo.\n"
                    "2. Or open **Webcam Recognition** for live predictions.\n"
                    "3. See **Model Performance** for measured test results.")

    st.subheader("The seven expression categories")
    columns = st.columns(7)
    for column, name in zip(columns, config.EMOTIONS):
        column.markdown(f'<div class="card"><div class="emoji">{config.EMOJI[name]}</div>'
                        f'<div class="name">{name.capitalize()}</div></div>',
                        unsafe_allow_html=True)

    summary = read_json(config.DATASET_SUMMARY_PATH)
    st.subheader("Dataset overview")
    if summary:
        counts = pd.DataFrame(summary["class_counts"]).T
        st.bar_chart(counts["total"].rename(index=str.capitalize), height=280)
        st.caption(f"{summary['total_images']:,} readable images; "
                   f"{summary['exact_duplicate_images']:,} exact duplicates detected "
                   "(removed from the splits to prevent leakage).")
    else:
        st.info("Class-distribution chart will appear here after you run "
                "`python -m src.data_loader` on your FER-2013 dataset.")
    st.markdown(f'<div class="note">{RESPONSIBLE_NOTE}</div>', unsafe_allow_html=True)


def page_image() -> None:
    st.header("🖼️ Image-based expression recognition")
    predictor = get_predictor()
    problem = model_status(predictor)
    if problem:
        st.warning(f"{problem}\n\nFaces can still be detected, but expressions cannot be predicted.")

    uploaded = st.file_uploader("Upload a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image containing one or more frontal faces.")
        return
    try:
        image = decode_image(uploaded.getvalue(), max_side=1280)
    except InvalidImageError as error:
        st.error(f"Could not read this image: {error}")
        return

    detector = get_detector_or_stop(predictor)
    boxes = detector.detect(image)
    if not boxes:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image",
                 use_container_width=True)
        st.warning("No face detected. Try a clearer, well-lit, front-facing photo.")
        return

    predictions = [] if problem else predictor.predict_boxes(image, boxes)
    annotated = draw_boxes(image, boxes) if problem else draw_predictions(image, predictions)

    col_a, col_b = st.columns(2)
    col_a.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image",
                use_container_width=True)
    col_b.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"{len(boxes)} face(s) detected", use_container_width=True)
    if problem:
        return

    st.subheader("Results")
    for number, pred in enumerate(predictions, start=1):
        x, y, w, h = pred.box
        crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1.2, 3])
            c1.image(crop, caption=f"Face {number}", width=120)
            c2.metric("Predicted expression",
                      f"{config.EMOJI[pred.label]} {pred.label.capitalize()}")
            c2.metric("Confidence", f"{pred.confidence * 100:.1f}%")
            frame = pd.DataFrame({"Probability": pred.probabilities.values()},
                                 index=[n.capitalize() for n in pred.probabilities])
            c3.bar_chart(frame, height=230)
    st.markdown(f'<div class="note">{RESPONSIBLE_NOTE}</div>', unsafe_allow_html=True)


def page_webcam() -> None:
    st.header("🎥 Real-time webcam recognition")
    predictor = get_predictor()
    problem = model_status(predictor)
    if problem:
        st.error(problem)
        return
    detector = get_detector_or_stop(predictor)

    st.markdown('<div class="note">Video frames are analysed in memory only. '
                "Nothing is recorded, saved or sent anywhere.</div>", unsafe_allow_html=True)
    consent = st.checkbox("I agree to let this app use my webcam for live analysis.")
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    if not consent:
        st.info("Tick the checkbox above to enable the camera switch.")
        return

    running = st.toggle("Camera on (switch off to stop)", key="webcam_on")
    frame_slot = st.empty()
    info_slot = st.empty()
    if not running:
        frame_slot.info("The camera is off.")
        return

    capture = open_camera(int(camera_index))
    if capture is None:
        st.error("Could not open the webcam. Check that it is connected, not in use by "
                 "another application, and that Windows camera privacy settings allow access.")
        return
    try:
        while st.session_state.get("webcam_on", False):
            ok, frame = capture.read()
            if not ok:
                st.error("Lost connection to the webcam.")
                break
            frame = cv2.flip(frame, 1)
            boxes = detector.detect(frame)
            predictions = predictor.predict_boxes(frame, boxes) if boxes else []
            annotated = draw_predictions(frame, predictions)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB",
                             use_container_width=True)
            if predictions:
                info_slot.markdown(" &nbsp;|&nbsp; ".join(
                    f"**Face {i}:** {config.EMOJI[p.label]} {p.label.capitalize()} "
                    f"({p.confidence * 100:.0f}%)" for i, p in enumerate(predictions, 1)))
            else:
                info_slot.warning("No face detected.")
            time.sleep(0.01)
    finally:
        capture.release()  # always free the camera, also when Streamlit reruns the script


def page_performance() -> None:
    st.header("📊 Model performance")
    metrics = read_json(config.METRICS_PATH)
    if not metrics:
        st.warning("No evaluation results found yet. Train and evaluate the model first:\n\n"
                   "`python -m src.train`  (evaluation runs automatically afterwards), or "
                   "`python -m src.evaluate`.")
        return
    st.caption(f"Measured on {metrics.get('num_test_images', '?'):,} held-out test images.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Macro precision", f"{metrics['macro']['precision']:.3f}")
    c3.metric("Macro recall", f"{metrics['macro']['recall']:.3f}")
    c4.metric("Macro F1-score", f"{metrics['macro']['f1']:.3f}")

    st.subheader("Class-wise performance")
    rows = {name: {"Precision": v["precision"], "Recall": v["recall"],
                   "F1-score": v["f1-score"], "Support": int(v["support"])}
            for name, v in metrics["per_class"].items()}
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                                     "F1-score": "{:.3f}"}), use_container_width=True)
    st.bar_chart(table[["Precision", "Recall", "F1-score"]], height=300)

    for title, filename in (("Confusion matrix", "confusion_matrix.png"),
                            ("Training and validation curves", "training_history.png")):
        path = config.RESULTS_DIR / filename
        st.subheader(title)
        if path.is_file():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"{filename} not found in the results folder.")
    report = config.RESULTS_DIR / "classification_report.txt"
    if report.is_file():
        with st.expander("Full classification report"):
            st.code(report.read_text(encoding="utf-8"))


def page_about() -> None:
    st.header("ℹ️ About this project")
    st.subheader("Dataset: FER-2013")
    st.markdown("FER-2013 contains grayscale 48×48 face images labelled with seven expressions: "
                "angry, disgust, fear, happy, neutral, sad and surprise. Labels are crowd-sourced "
                "and noisy, classes are imbalanced (Disgust is rare), and some images are "
                "duplicated.")
    st.subheader("Preprocessing")
    st.markdown("- Corrupt or invalid images are detected and skipped.\n"
                "- Pixels are normalised from 0–255 to 0–1; images are shaped 48×48×1.\n"
                "- Labels are one-hot encoded.\n"
                "- Exact duplicate images are removed so none appear in more than one split; "
                "validation data is a stratified split of the training pool; the test set is "
                "never used for training or model selection.\n"
                "- At inference, detected faces are cropped, converted to grayscale, resized to "
                "48×48 and normalised in the same way.")
    st.subheader("CNN architecture")
    st.code("Input 48×48×1\n"
            "Data augmentation (flip, rotation, zoom, shift; training only)\n"
            "4 × [Conv3×3 → BatchNorm → ReLU] ×2 → MaxPool → Dropout   (32, 64, 128, 256 filters)\n"
            "GlobalAveragePooling → Dense(256) → BatchNorm → ReLU → Dropout(0.5)\n"
            "Dense(7, softmax)", language="text")
    st.subheader("Training")
    st.markdown("Adam optimiser, categorical cross-entropy, batch size 64, up to 50 epochs with "
                "early stopping, learning-rate reduction on plateau and checkpointing of the best "
                "model (lowest validation loss). Softened class weights compensate for rare classes.")
    st.subheader("Limitations")
    st.markdown("- Expression labels are not emotions: facial appearance does not reliably reveal "
                "internal feelings, and expressions vary across cultures and individuals.\n"
                "- FER-2013 is small, low-resolution and noisy; accuracy is limited and "
                "Fear/Sad/Neutral are often confused.\n"
                "- Possible dataset bias across age, gender, skin tone and image conditions.\n"
                "- The Haar-cascade detector works best on well-lit, frontal faces and can miss "
                "angled, small or occluded faces.\n"
                "- Do not use this system for decisions about people (hiring, security, "
                "education, etc.).")
    st.markdown(f'<div class="note">{RESPONSIBLE_NOTE}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def main() -> None:
    logo = config.ASSETS_DIR / "project_logo.png"
    if logo.is_file():
        st.sidebar.image(str(logo), use_container_width=True)
    st.sidebar.title("Expression AI")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    status = model_status(get_predictor())
    if status is None:
        st.sidebar.success("Model loaded")
    else:
        st.sidebar.warning("Model not trained yet")
    st.sidebar.caption("All processing is local. Nothing is uploaded or saved.")

    {"🏠 Home": page_home, "🖼️ Image Recognition": page_image,
     "🎥 Webcam Recognition": page_webcam, "📊 Model Performance": page_performance,
     "ℹ️ About Project": page_about}[page]()


main()
