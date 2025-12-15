"""
vgg.py
-------
VGG-style architectures for crop classification from temporal satellite data.

This module provides VGG-inspired models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using deep convolutional stacks.

Available models:
* **baseline_vgg1d** – Deeper VGG-like network with 3 convolutional blocks
* **baseline_vgg1d_compact** – Lighter/regularized variant with 2 blocks + GAP

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
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

# External callback with Confusion-Matrix + macro metrics
from models.callbacks import ExtendedMetricsCallback

# -----------------------------------------------------------------------------
#  Model builders
# -----------------------------------------------------------------------------

def _build_vgg1d_model(input_shape: tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Classic *deeper* VGG-style stack (3 conv blocks).

    Parameters
    ----------
    input_shape : tuple
        ``(timesteps, features)``
    num_classes : int
        Number of target classes
    """
    model = Sequential([
        layers.Input(shape=input_shape),

        # --- Block 1
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),

        # --- Block 2
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),

        # --- Block 3
        layers.Conv1D(256, 3, padding="same", activation="relu"),
        layers.Conv1D(256, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )
    return model


def _build_vgg1d_compact_model(input_shape: tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Smaller regularised VGG-style model (2 conv blocks + GAP)."""
    l2 = regularizers.l2(1e-2)

    model = Sequential([
        layers.Input(shape=input_shape),

        # --- Block 1
        layers.Conv1D(32, 3, padding="same", activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv1D(32, 3, padding="same", activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        # --- Block 2
        layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),

        # --- Classification
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )
    return model

# -----------------------------------------------------------------------------
#  Training helper (shared by both baselines)
# -----------------------------------------------------------------------------

def _train_vgg(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    *,
    xaxis_callback: int | str,
    max_epochs: int,
    es_patience: int,
    model_save_dir: str,
    model_name: str,
    log_dir: str,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Internal helper running ``model.fit`` with unified callbacks."""

    # --- IO setup
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_name}.keras")

    # --- Callbacks
    callbacks = [
        TensorBoard(log_dir=log_dir, update_freq=xaxis_callback),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        # extra metrics
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, history


def baseline_vgg1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: int | str = "epoch",
    max_epochs: int = 40,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_vgg1d",
    log_dir: str = "logs/fit",
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a *deeper* VGG-style 1-D CNN.

    Parameters
    ----------
    train_ds, val_ds : tf.data.Dataset
        Prepared datasets yielding ``(x, y)`` batches.
    label_legend : list[str]
        Ordered class names, used for ExtendedMetrics.
    max_images_per_series : int
        Sequence length (time steps).
    num_features : int
        Features per time step.
    xaxis_callback : int | str
        ``1`` → log per batch, ``"epoch"`` → per epoch.
    max_epochs : int
        Max training epochs.
    es_patience : int
        Early stopping patience on *validation loss*.
    model_save_dir : str
        Directory where ``<model_name>.keras`` will be written.
    model_name : str
        Base filename for the saved model.
    log_dir : str
        TensorBoard + ExtendedMetrics root directory.

    Returns
    -------
    model : tf.keras.Model
    history : tf.keras.callbacks.History
    """
    input_shape = (max_images_per_series, num_features)
    model = _build_vgg1d_model(input_shape, len(label_legend))

    return _train_vgg(
        model,
        train_ds,
        val_ds,
        label_legend,
        xaxis_callback=xaxis_callback,
        max_epochs=max_epochs,
        es_patience=es_patience,
        model_save_dir=model_save_dir,
        model_name=model_name,
        log_dir=log_dir,
    )


def baseline_vgg1d_compact(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: int | str = "epoch",
    max_epochs: int = 40,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_vgg1d_compact",
    log_dir: str = "logs/fit",
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train the compact VGG-style 1-D CNN (fewer params, heavy regularisation)."""

    input_shape = (max_images_per_series, num_features)
    model = _build_vgg1d_compact_model(input_shape, len(label_legend))

    return _train_vgg(
        model,
        train_ds,
        val_ds,
        label_legend,
        xaxis_callback=xaxis_callback,
        max_epochs=max_epochs,
        es_patience=es_patience,
        model_save_dir=model_save_dir,
        model_name=model_name,
        log_dir=log_dir,
    )
