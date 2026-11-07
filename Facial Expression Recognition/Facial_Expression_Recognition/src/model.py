"""CNN architecture for 48x48 grayscale facial expression classification."""
from __future__ import annotations

from src.config import IMG_SIZE, NUM_CLASSES


def build_model(learning_rate: float = 1e-3, dropout_rate: float = 0.25,
                l2_weight: float = 1e-4, augment: bool = True):
    """Build and compile the CNN.

    Input pixels must already be normalised to [0, 1] (see preprocessing.py).
    Augmentation layers (random flip / rotation / zoom / shift) are part of the
    model but only active during training; they are bypassed at inference time.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(l2_weight)

    def conv_block(x, filters: int, name: str):
        for i in (1, 2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_regularizer=reg, name=f"{name}_conv{i}")(x)
            x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
        return layers.Dropout(dropout_rate, name=f"{name}_drop")(x)

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="face_48x48")
    x = inputs
    if augment:
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(0.08, name="aug_rotate")(x)
        x = layers.RandomZoom(0.1, name="aug_zoom")(x)
        x = layers.RandomTranslation(0.08, 0.08, name="aug_shift")(x)

    for idx, filters in enumerate((32, 64, 128, 256), start=1):
        x = conv_block(x, filters, f"block{idx}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="expression")(x)

    model = keras.Model(inputs, outputs, name="expression_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
