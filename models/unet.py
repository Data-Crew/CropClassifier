"""
unet.py
---------
U-Net architectures for crop classification from temporal satellite data.

This module provides U-Net models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using encoder-decoder architectures with skip connections.

Available models:
* **baseline_unet1d** – Full-capacity U-Net-1D with encoder-decoder structure
* **baseline_unet1d_light** – Lighter variant (~¼ parameters) with regularization

All models include:
* **EarlyStopping** on validation loss
* **ReduceLROnPlateau** learning rate scheduling (factor 0.5, patience 3)
* **TensorBoard** logging with configurable update frequency
* **ExtendedMetricsCallback** – confusion matrix, balanced accuracy & macro-F1
* **ModelCheckpoint** for best model saving

Intended for crop classification from multi-temporal satellite imagery where each
pixel contains a time series of spectral bands and vegetation indices.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union, List

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models.callbacks import ExtendedMetricsCallback


# -----------------------------------------------------------------------------
# Helper ▸ building blocks
# -----------------------------------------------------------------------------

def _conv_block(x: tf.Tensor, filters: int, l2: float | None = None) -> tf.Tensor:
    """2×(Conv ➜ BN ➜ ReLU) stack used in both encoder and decoder."""
    reg = regularizers.l2(l2) if l2 else None
    for _ in range(2):
        x = layers.Conv1D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
    return x

class _CropToMatch(layers.Layer):
    """Cropping layer that trims *skip* to *x* along the temporal axis."""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:  # noqa: D401, N802 – Keras signature
        x, skip = inputs
        target_len = tf.shape(x)[1]
        return skip[:, :target_len, :]


def _build_unet1d(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Full‑capacity UNet‑1D."""
    inputs = layers.Input(shape=input_shape)

    # Encoder -----------------------------------------------------------------
    def encoder_block(x: tf.Tensor, filters: int) -> tuple[tf.Tensor, tf.Tensor]:
        skip = _conv_block(x, filters)
        x = layers.MaxPooling1D(pool_size=2)(skip)
        return skip, x

    skip1, x = encoder_block(inputs, 32)   # T/2
    skip2, x = encoder_block(x, 64)        # T/4
    skip3, x = encoder_block(x, 128)       # T/8

    # Bridge ------------------------------------------------------------------
    x = _conv_block(x, 256)

    # Decoder -----------------------------------------------------------------
    def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
        x = layers.UpSampling1D(size=2)(x)
        skip = _CropToMatch()([x, skip])
        x = layers.Concatenate()([x, skip])
        return _conv_block(x, filters)

    x = decoder_block(x, skip3, 128)       # T/4
    x = decoder_block(x, skip2, 64)        # T/2
    x = decoder_block(x, skip1, 32)        # T

    # Head --------------------------------------------------------------------
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="UNet1D")


def _build_unet1d_light(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Smaller UNet‑1D variant (~¼ params, added L2 + Dropout for regularisation)."""

    inputs = layers.Input(shape=input_shape)

    def encoder_block(x: tf.Tensor, filters: int, dropout: float) -> tuple[tf.Tensor, tf.Tensor]:
        skip = _conv_block(x, filters, l2=1e-2)
        x = layers.MaxPooling1D(pool_size=2)(skip)
        x = layers.Dropout(dropout)(x)
        return skip, x

    def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
        x = layers.UpSampling1D(size=2)(x)
        skip = _CropToMatch()([x, skip])
        x = layers.Concatenate()([x, skip])
        return _conv_block(x, filters, l2=1e-2)

    # Encoder -----------------------------------------------------------------
    skip1, x = encoder_block(inputs, 16, 0.3)
    skip2, x = encoder_block(x, 32, 0.3)
    skip3, x = encoder_block(x, 64, 0.4)

    # Bridge ------------------------------------------------------------------
    x = _conv_block(x, 128, l2=1e-2)

    # Decoder -----------------------------------------------------------------
    x = decoder_block(x, skip3, 64)
    x = decoder_block(x, skip2, 32)
    x = decoder_block(x, skip1, 16)

    # Head --------------------------------------------------------------------
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="UNet1D_Light")

# -----------------------------------------------------------------------------
# Helper ▸ training wrapper
# -----------------------------------------------------------------------------

def _train_unet(
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
    """Compile, add common callbacks, train, save, and return *(model, history)*."""

    # ── compile
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )

    # ── I/O prep
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_name}.keras")

    # ── callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        TensorBoard(log_dir=log_dir, update_freq=("epoch" if xaxis_callback == "epoch" else xaxis_callback)),
        # extended metrics
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics")),
    ]

    # ── train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── save & return
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, history


def baseline_unet1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_unet1d",
    log_dir: str = "logs/fit",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a full‑capacity UNet‑1D model.

    Parameters
    ----------
    train_ds, val_ds : tf.data.Dataset
        Pre‑batched datasets yielding *(X, y)* pairs.
    label_legend : list[str]
        Ordered list of class names (used for metrics & final Dense(*n_classes*) layer).
    max_images_per_series : int
        Sequence / temporal length *T*.
    num_features : int
        Feature dimensionality *F* at each timestep.
    xaxis_callback : int | str, default=1
        `1` → log TensorBoard by *batch*; `'epoch'` → by epoch.
    max_epochs : int, default=60
    es_patience : int, default=10
        Early‑stopping patience (epochs).
    model_save_dir : str, default="results"
    model_name : str, default="baseline_unet1d"
    log_dir : str, default="logs/fit"

    Returns
    -------
    model : tf.keras.Model
    history : tf.keras.callbacks.History
    """
    input_shape = (max_images_per_series, num_features)
    model = _build_unet1d(input_shape, len(label_legend))
    return _train_unet(model, train_ds, val_ds, label_legend, xaxis_callback,
                       max_epochs, es_patience, model_save_dir, model_name, log_dir)


def baseline_unet1d_light(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_unet1d_light",
    log_dir: str = "logs/fit",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a lighter UNet‑1D variant suitable for quicker experimentation."""
    input_shape = (max_images_per_series, num_features)
    model = _build_unet1d_light(input_shape, len(label_legend))
    return _train_unet(model, train_ds, val_ds, label_legend, xaxis_callback,
                       max_epochs, es_patience, model_save_dir, model_name, log_dir)

