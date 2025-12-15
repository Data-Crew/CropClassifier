"""
tcn.py
-------
Temporal Convolutional Network architectures for crop classification from temporal satellite data.

This module provides TCN models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using dilated causal convolutions and residual connections.

Available models:
* **baseline_tcn** – Deep TCN with dilation stack and global pooling

All models include:
* **EarlyStopping** on validation loss
* **ReduceLROnPlateau** learning rate scheduling
* **TensorBoard** logging with configurable update frequency
* **ExtendedMetricsCallback** – confusion matrix, balanced accuracy & macro-F1
* **ModelCheckpoint** for best model saving

Intended for crop classification from multi-temporal satellite imagery where each
pixel contains a time series of spectral bands and vegetation indices.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.callbacks import ExtendedMetricsCallback

# -----------------------------------------------------------------------------
# Helper ▸ Residual block for TCN
# -----------------------------------------------------------------------------

def _residual_block(x: tf.Tensor, filters: int, kernel_size: int,
                    dilation_rate: int, dropout_rate: float = 0.3) -> tf.Tensor:
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding='causal',
                      dilation_rate=dilation_rate)(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)

    x = layers.Conv1D(filters, kernel_size, padding='causal',
                      dilation_rate=dilation_rate)(x)
    x = layers.LayerNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# -----------------------------------------------------------------------------
# Architecture ▸ TCN
# -----------------------------------------------------------------------------

def _build_tcn(input_shape: tuple[int, int], num_classes: int) -> Model:
    """Constructs a TCN model with residual dilated blocks."""
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Stack of residual dilated blocks
    for dilation in [1, 2, 4, 8, 16]:
        x = _residual_block(x, filters=64, kernel_size=3,
                            dilation_rate=dilation, dropout_rate=0.3)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="TCN_Model")

# -----------------------------------------------------------------------------
# Training wrapper ▸ baseline_tcn
# -----------------------------------------------------------------------------

def baseline_tcn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_tcn",
    log_dir: str = "logs/fit",
) -> tuple[Model, tf.keras.callbacks.History]:
    """Train a TCN model with standard training loop and extended metrics.

    Parameters
    ----------
    train_ds, val_ds : tf.data.Dataset
        Pre-batched datasets yielding (X, y) pairs.
    label_legend : list[str]
        Ordered list of class names (used for metrics & Dense(n_classes)).
    max_images_per_series : int
        Sequence length (T).
    num_features : int
        Feature dimensionality (F).
    xaxis_callback : int | str, default=1
        If 1 → log TensorBoard by batch, if 'epoch' → by epoch.
    max_epochs : int, default=60
    es_patience : int, default=10
        Early stopping patience.
    model_save_dir : str, default="results"
    model_name : str, default="baseline_tcn"
    log_dir : str, default="logs/fit"

    Returns
    -------
    model : tf.keras.Model
    history : tf.keras.callbacks.History
    """
    input_shape = (max_images_per_series, num_features)
    num_classes = len(label_legend)

    model = _build_tcn(input_shape, num_classes)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=10,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["categorical_accuracy"]
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(model_save_dir, f"{model_name}.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        #ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1), # Cosin decay
        TensorBoard(log_dir=log_dir, update_freq=("epoch" if xaxis_callback == "epoch" else xaxis_callback)),
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics"))
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, history
