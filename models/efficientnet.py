"""
efficientnet.py
----------------
EfficientNet-style architectures for crop classification from temporal satellite data.

This module provides EfficientNet-inspired models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using mobile-inverted bottleneck blocks.

Available models:
* **baseline_efficientnet1d** – Compact EfficientNet-1D model with mobile-inverted bottlenecks

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
from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------------------------
# Extended metrics callback (your existing impl in models.callbacks)
# -----------------------------------------------------------------------------
from models.callbacks import ExtendedMetricsCallback  # noqa: E402 – after TF import

# -----------------------------------------------------------------------------
# Helper ▸ building blocks
# -----------------------------------------------------------------------------

def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    dropout_rate: float | None = 0.3,
) -> tf.Tensor:
    """Conv‑BN‑ReLU (+ optional Dropout)."""
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def _mb_conv_block(
    x: tf.Tensor,
    filters: int,
    expansion: int = 4,
    kernel_size: int = 3,
    stride: int = 1,
    dropout_rate: float | None = 0.2,
) -> tf.Tensor:
    """Mobile‑Inverted Bottleneck (MBConv) style residual block."""
    in_channels = x.shape[-1]

    # Expansion
    expanded = _conv_block(x, in_channels * expansion, kernel_size=1, stride=1, dropout_rate=None)

    # Depthwise conv
    expanded = layers.DepthwiseConv1D(kernel_size, strides=stride, padding="same", use_bias=False)(expanded)
    expanded = layers.BatchNormalization()(expanded)
    expanded = layers.ReLU()(expanded)

    # Projection
    projected = layers.Conv1D(filters, kernel_size=1, padding="same", use_bias=False)(expanded)
    projected = layers.BatchNormalization()(projected)

    # Skip‑connection if possible
    if (in_channels == filters) and (stride == 1):
        x = layers.Add()([x, projected])
    else:
        x = projected

    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x

# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------

def _build_efficientnet1d(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Construct a lightweight EfficientNet‑inspired 1‑D network."""
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = _conv_block(inputs, 32, kernel_size=3, stride=1, dropout_rate=0.1)

    # Stages (very small scaling – suited for 1‑D signals)
    x = _mb_conv_block(x, 32, expansion=1, kernel_size=3, stride=1, dropout_rate=0.1)
    x = _mb_conv_block(x, 64, expansion=4, kernel_size=3, stride=2, dropout_rate=0.2)
    x = _mb_conv_block(x, 128, expansion=4, kernel_size=3, stride=2, dropout_rate=0.3)
    x = _mb_conv_block(x, 128, expansion=4, kernel_size=3, stride=1, dropout_rate=0.3)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="EfficientNet1D")

# -----------------------------------------------------------------------------
# Training wrapper (shared across baselines)
# -----------------------------------------------------------------------------

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
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )

    # I/O prep ----------------------------------------------------------------
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


def baseline_efficientnet1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_efficientnet1d",
    log_dir: str = "logs/fit",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a compact EfficientNet‑1D baseline.

    Parameters
    ----------
    train_ds, val_ds : tf.data.Dataset
        Pre‑batched datasets yielding *(X, y)* pairs.
    label_legend : list[str]
        Ordered list of class names.
    max_images_per_series : int
        Temporal length *T*.
    num_features : int
        Feature dimensionality *F* at each timestep.
    xaxis_callback : int | str, default=1
        `1` → log TensorBoard by *batch*; `'epoch'` → by epoch.
    max_epochs : int, default=60
    es_patience : int, default=10
        Early‑stopping patience (epochs).
    model_save_dir : str, default="results"
    model_name : str, default="baseline_efficientnet1d"
    log_dir : str, default="logs/fit"

    Returns
    -------
    model : tf.keras.Model
    history : tf.keras.callbacks.History
    """
    input_shape = (max_images_per_series, num_features)
    model = _build_efficientnet1d(input_shape, len(label_legend))
    return _train_model(
        model,
        train_ds,
        val_ds,
        label_legend,
        xaxis_callback,
        max_epochs,
        es_patience,
        model_save_dir,
        model_name,
        log_dir,
    )
