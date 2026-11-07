# AI-Based Facial Expression Recognition (CNN · TensorFlow · OpenCV · Streamlit)

Classifies faces into **seven visual expression categories** — Happy, Sad, Angry, Surprise,
Fear, Disgust, Neutral — from uploaded images or a live webcam, using a CNN trained on **FER-2013**.

> **Status of this package:** the complete code, tests and documentation are included, but
> **the model is NOT trained yet** and the **FER-2013 dataset is NOT included**.
> `models/emotion_model.keras` and the metrics in `results/` are produced when *you* run the
> training step below. No accuracy figures are quoted here because none have been measured.

## 1. Installation (Windows 10/11)

Install Python 3.10, 3.11 or 3.12 (64-bit) from python.org (tick "Add Python to PATH"), then in
Command Prompt or PowerShell inside the project folder:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

(PowerShell may need: `Set-ExecutionPolicy -Scope Process RemoteSigned` before `activate`.)
On macOS/Linux use `source .venv/bin/activate`.

## 2. Add the FER-2013 dataset

Put the data in `dataset/` as described in [`dataset/README.md`](dataset/README.md)
(either `dataset/train/<emotion>/…` + `dataset/test/<emotion>/…`, or `dataset/fer2013.csv`).
Inspect it (structure, class counts, corrupt files, duplicates, sample images, class plot):

    python -m src.data_loader

## 3. Train the model

    python -m src.train

Defaults: batch 64, up to 50 epochs, Adam (1e-3), early stopping, learning-rate scheduling,
model checkpointing. Options: `--epochs 30 --batch-size 32 --no-class-weights --val-size 0.15`.
Training on a CPU can take from tens of minutes to a few hours. (TensorFlow ≥ 2.11 has no native
Windows GPU support; use WSL2 if you need a GPU.)

This produces `models/emotion_model.keras` and, after evaluating on the test set, everything in
`results/`: `metrics.json`, `classification_report.txt`, `confusion_matrix.png`,
`training_history.png`, `training_history.json`, `training_config.json`.
Re-run evaluation alone with `python -m src.evaluate`.

## 4. Run the application

    streamlit run app.py

Pages: **Home**, **Image Recognition** (upload JPG/PNG, multi-face, per-class probabilities),
**Webcam Recognition** (live boxes, labels, confidence; stop with the toggle),
**Model Performance** (reads your real results), **About Project**.

Command-line alternatives:

    python -m src.predict --image photo.jpg --output annotated.jpg
    python -m src.predict --webcam            (press Q or ESC to stop)

Without a trained model the app still starts; it shows a clear message, and the image page
only draws face boxes.

## 5. Tests

    pytest

Data-loader tests use tiny synthetic images solely to exercise code paths. TensorFlow-based tests
(shape/pipeline checks with an *untrained* network) run automatically when TensorFlow is installed.

## Project structure

    app.py                  Streamlit dashboard
    requirements.txt  pytest.ini  .gitignore  README.md
    dataset/README.md       where to put FER-2013
    models/                 emotion_model.keras appears here after training
    results/                metrics and figures appear here after training
    assets/project_logo.png
    src/
      config.py             relative paths, class names, constants
      data_loader.py        dataset detection, validation, dedup, leakage-safe split, inspection CLI
      preprocessing.py      normalisation, 48x48x1 conversion, label encoding (shared train/inference)
      visualization.py      class distribution, samples, history, confusion matrix
      model.py              CNN definition
      train.py              training CLI (callbacks, class weights, saving)
      evaluate.py           test-set metrics, report, confusion matrix
      face_detection.py     OpenCV Haar-cascade face detector
      predict.py            predictor, drawing helpers, webcam loop, CLI
    notebooks/              data_exploration.ipynb, model_training.ipynb (no stored outputs)
    tests/                  test_preprocessing.py, test_data_loader.py, test_prediction.py

## How it works

* **Leakage-safe splits:** FER-2013 contains exact duplicate images. Every image is hashed and only
  one copy is kept (priority test > validation > train), so no identical image appears in two
  sets. The official test partition is used when present; validation is a stratified split of the
  training pool. The test set is never used for training or model selection.
* **CNN:** input 48×48×1 → augmentation (flip/rotation/zoom/shift, training only) → 4 blocks of
  2×[Conv3×3-BatchNorm-ReLU] + MaxPool + Dropout (32/64/128/256 filters) → GlobalAveragePooling →
  Dense(256)-BatchNorm-ReLU-Dropout(0.5) → Dense(7, softmax). L2 regularisation throughout.
* **Inference:** Haar-cascade detection → crop → grayscale → resize 48×48 → /255 → CNN → softmax.
  Class order is alphabetical (angry, disgust, fear, happy, neutral, sad, surprise).

## Privacy and responsible use

Everything runs locally; images are never uploaded, and webcam frames are analysed in memory
and never saved (the webcam page asks for explicit consent). Outputs are *visual expression
classifications*, **not** a measurement of anyone's true emotional state. FER-2013 has noisy labels,
class imbalance and demographic bias, so errors are expected (Fear/Sad/Neutral are commonly
confused; Disgust is rare). Do not use this for decisions about people.

## Known limitations

* Model untrained and dataset absent in this package (see top).
* Haar cascades miss angled, small, dark or occluded faces; thresholds are in `src/face_detection.py`.
* 48×48 grayscale input limits achievable accuracy.
* Webcam access in the Streamlit page uses OpenCV on the machine running Streamlit (local use only;
  it will not access a visitor's camera if the app is deployed on a server).
* Dependency versions were chosen for compatibility but could not be install-tested in the build
  environment; if `pip install` reports a conflict, relax the TensorFlow/NumPy pins.
