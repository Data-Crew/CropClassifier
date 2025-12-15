"""
resnet.py
---------
ResNet-style architectures for crop classification from temporal satellite data.

This module provides residual network models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using residual connections and skip connections.

Available models:
* **baseline_resnet1d** – Compact plain ResNet-1D with residual blocks
* **baseline_resunet1d** – U-Net + residual hybrid (ResUNet-1D) with encoder-decoder

All models include:
* **EarlyStopping** on validation loss
* **ReduceLROnPlateau** learning rate scheduling
* **TensorBoard** logging with configurable update frequency
* **ExtendedMetricsCallback** – confusion matrix, balanced accuracy, macro-F1 & per-class recall
* **ModelCheckpoint** for best model saving

Intended for crop classification from multi-temporal satellite imagery where each
pixel contains a time series of spectral bands and vegetation indices.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam

# Extended metrics -----------------------------------------------------------
from models.callbacks import ExtendedMetricsCallback  

# ---------------------------------------------------------------------------
# Helper ▸ residual blocks & builders
# ---------------------------------------------------------------------------

def _residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    l2: float | None = 1e-2,
) -> tf.Tensor:
    """Standard pre‑activation residual block: Conv → BN → ReLU ×2 + skip."""
    shortcut = x
    reg = regularizers.l2(l2) if l2 else None

    # f(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    # adjust shortcut if channel dim changes
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([shortcut, x])
    return layers.ReLU()(x)


# ─────────────────────────────────────────────────────────────────────────────
#   ResNet‑1D
# ─────────────────────────────────────────────────────────────────────────────

def _build_resnet1d(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Plain 1‑D ResNet (≈ ~150‑K params)."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = _residual_block(x, 32)
    x = layers.MaxPooling1D(2)(x)

    x = _residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)

    x = _residual_block(x, 128)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="ResNet1D")


# ─────────────────────────────────────────────────────────────────────────────
#   ResUNet‑1D (hybrid)
# ─────────────────────────────────────────────────────────────────────────────

class _CropToMatch(layers.Layer):
    """Crop *skip* along temporal axis to match *x* (used in decoder)."""

    def call(self, inputs: list[tf.Tensor]):  # noqa: D401, N802 – Keras signature
        x, skip = inputs
        tgt = tf.shape(x)[1]
        return skip[:, :tgt, :]


def _conv_block(x: tf.Tensor, filters: int, dropout: float, l2: float | None = 1e-2) -> tf.Tensor:
    """Conv‑BN‑ReLU‑Dropout ×2 with residual connection."""
    shortcut = x
    reg = regularizers.l2(l2) if l2 else None

    for _ in range(2):
        x = layers.Conv1D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def _build_resunet1d(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """UNet encoder–decoder with residual blocks, squeeze‑excite & MH‑Attention."""
    inputs = layers.Input(shape=input_shape)

    # Encoder ----------------------------------------------------------------
    def enc_block(x: tf.Tensor, filters: int, drop: float):
        skip = _conv_block(x, filters, drop)
        x = layers.MaxPooling1D(2)(skip)
        return skip, x

    skip1, x = enc_block(inputs, 32, 0.3)
    skip2, x = enc_block(x, 64, 0.4)
    skip3, x = enc_block(x, 128, 0.5)

    # Bottleneck -------------------------------------------------------------
    x = _conv_block(x, 256, 0.5)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(256 // 16, activation="relu")(se)
    se = layers.Dense(256, activation="sigmoid")(se)
    se = layers.Reshape((1, 256))(se)
    x = layers.Multiply()([x, se])

    # Decoder ----------------------------------------------------------------
    def dec_block(x: tf.Tensor, skip: tf.Tensor, filters: int, drop: float):
        x = layers.Conv1DTranspose(filters, 3, strides=2, padding="same")(x)
        skip = _CropToMatch()([x, skip])
        x = layers.Concatenate()([x, skip])
        return _conv_block(x, filters, drop)

    x = dec_block(x, skip3, 128, 0.5)
    x = dec_block(x, skip2, 64, 0.4)
    x = dec_block(x, skip1, 32, 0.3)

    # Head -------------------------------------------------------------------
    x = layers.MultiHeadAttention(num_heads=4, key_dim=x.shape[-1] // 4, dropout=0.3)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="ResUNet1D")

# ---------------------------------------------------------------------------
# Helper ▸ shared training wrapper
# ---------------------------------------------------------------------------

def _train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    xaxis_callback: Union[int, str],
    max_epochs: int,
    es_patience: int,
    model_save_dir: str,
    model_name: str,
    log_dir: str,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Compile → add callbacks → fit → save → return ``(model, history)``."""

    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{model_name}.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        TensorBoard(log_dir=log_dir, update_freq=("epoch" if xaxis_callback == "epoch" else xaxis_callback)),
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, history


def baseline_resnet1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_resnet1d",
    log_dir: str = "logs/fit",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a compact ResNet‑1D model.

    Parameters match the other baseline helpers.
    """
    input_shape = (max_images_per_series, num_features)
    model = _build_resnet1d(input_shape, len(label_legend))
    return _train_model(model, train_ds, val_ds, label_legend, xaxis_callback,
                        max_epochs, es_patience, model_save_dir, model_name, log_dir)


def baseline_resunet1d(
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    label_legend: list[str],
    max_images_per_series: int,
    num_features: int,
    *,
    use_time_channel: bool = False,         
    xaxis_callback: int|str = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_resunet1d",
    log_dir: str = "logs/fit",
):

    # ---------------------------------------------------------- 0. Data pipeline
    if use_time_channel:
        def add_time_channel(x, y):
            t = tf.linspace(0.0, 1.0, tf.shape(x)[1])
            t = tf.reshape(t, (1, -1, 1))
            t = tf.tile(t, [tf.shape(x)[0], 1, 1])
            return tf.concat([x, t], axis=-1), y

        def random_jitter(x, y):
            return x + tf.random.normal(tf.shape(x), 0., 0.01), y

        def random_scaling(x, y):
            return x * tf.random.uniform([], 0.95, 1.05), y

        def random_time_shift(x, y):
            shift = tf.random.uniform([], -2, 2, tf.int32)
            return tf.roll(x, shift=shift, axis=1), y

        train_ds = (train_ds
                    .map(add_time_channel)
                    .map(random_jitter)
                    .map(random_scaling)
                    .map(random_time_shift))
        val_ds   = val_ds.map(add_time_channel)
        num_features += 1        # ¡ahora son 13!

    # ---------------------------------------------------------- 1. Modelo
    input_shape = (max_images_per_series, num_features)
    model = _build_resunet1d(input_shape, len(label_legend))

    # ---------------------------------------------------------- 2. Entrenamiento (idéntico)
    return _train_model(
        model, train_ds, val_ds, label_legend,
        xaxis_callback, max_epochs, es_patience,
        model_save_dir, model_name, log_dir
    )
