"""
transformer.py
--------------
Self-attention-based architectures for crop classification from temporal satellite data.

This module provides Transformer-based models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using self-attention mechanisms.

Available models:
* **baseline_transformer1d** – Pure Transformer encoder-only model for temporal classification
* **baseline_cnn_transformer1d** – CNN front-end + Transformer encoder (faster convergence)

All models include:
* **EarlyStopping** on validation loss
* **TensorBoard** logging with configurable update frequency
* **ExtendedMetricsCallback** – confusion matrix, balanced accuracy & macro-F1
* **ModelCheckpoint** for best model saving
* **CosineDecayRestarts** learning rate schedule (no ReduceLROnPlateau)

Intended for crop classification from multi-temporal satellite imagery where each
pixel contains a time series of spectral bands and calculated spectral indices
(NDVI, EVI, NDWI, NDBI).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Model

# Extended metrics -----------------------------------------------------------
from models.callbacks import ExtendedMetricsCallback  

# ---------------------------------------------------------------------------
# Positional encoding & Transformer blocks
# ---------------------------------------------------------------------------

class PositionalEmbedding(layers.Layer):
    """Learnable ±sinusoidal positional embedding added to the inputs."""

    def __init__(self, sequence_length: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = self.add_weight(
            shape=(sequence_length, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # noqa: D401 (Keras call)
        return inputs + self.pos_embedding


def _transformer_encoder_block(
    x: tf.Tensor,
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout_rate: float = 0.3,
) -> tf.Tensor:
    """Standard encoder block: MHA ➜ Add+Norm ➜ FFN ➜ Add+Norm."""

    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim // num_heads,
        dropout=dropout_rate,
    )(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(x.shape[-1]),
    ])
    ffn_out = ffn(x)
    x = layers.Add()([x, ffn_out])
    x = layers.LayerNormalization()(x)
    return x

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_transformer1d(
    input_shape: Tuple[int, int],
    num_classes: int,
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.3,
) -> Model:
    """Pure Transformer encoder (≈200‑K params for default dims)."""

    inputs = keras.Input(shape=input_shape)
    x = PositionalEmbedding(*input_shape)(inputs)
    for _ in range(3):
        x = _transformer_encoder_block(x, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="Transformer1D")


def _build_cnn_transformer1d(
    input_shape: Tuple[int, int],
    num_classes: int,
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.3,
) -> Model:
    """CNN front‑end + Transformer encoder (≈160‑K params)."""

    inputs = keras.Input(shape=input_shape)

    # Shallow CNN encoder ---------------------------------------------------
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)

    # Positional embedding + Transformer blocks ----------------------------
    x = PositionalEmbedding(input_shape[0], x.shape[-1])(x)
    for _ in range(2):
        x = _transformer_encoder_block(x, num_heads, ff_dim, dropout)

    # Head ------------------------------------------------------------------
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="CNNTransformer1D")

# ---------------------------------------------------------------------------
# Shared training wrapper
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
    """Compile ➜ callbacks ➜ fit ➜ save ➜ return *(model, history)*."""

    # ── compile ------------------------------------------------------------
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=10,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )

    # ── I/O ----------------------------------------------------------------
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{model_name}.keras")

    # ── callbacks ----------------------------------------------------------
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        TensorBoard(log_dir=log_dir, update_freq=("epoch" if xaxis_callback == "epoch" else xaxis_callback)),
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics")),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    # ── train --------------------------------------------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Model saved to {model_path}")
    return model, history


def baseline_transformer1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_transformer1d",
    log_dir: str = "logs/fit",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a pure Transformer encoder‑only model.

    Parameters
    ----------
    train_ds / val_ds : *tf.data.Dataset*
        Batched *(X, y)* datasets.
    label_legend : list[str]
        Class names.
    max_images_per_series : int
        Sequence length *T*.
    num_features : int
        Feature dimension *F*.
    xaxis_callback : int | str, default=1
        `1` → TensorBoard by batch, `'epoch'` → by epoch.
    max_epochs : int, default=60
    es_patience : int, default=10
    model_save_dir / model_name / log_dir : str
        I/O paths.
    """
    input_shape = (max_images_per_series, num_features)
    model = _build_transformer1d(input_shape, len(label_legend))
    return _train_model(model, train_ds, val_ds, label_legend, xaxis_callback,
                        max_epochs, es_patience, model_save_dir, model_name, log_dir)


def baseline_cnn_transformer1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: List[str],
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: Union[int, str] = 1,
    max_epochs: int = 60,
    es_patience: int = 10,
    model_save_dir: str = "results",
    model_name: str = "baseline_cnn_transformer1d",
    log_dir: str = "logs/fit",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a lightweight CNN+Transformer hybrid."""
    input_shape = (max_images_per_series, num_features)
    model = _build_cnn_transformer1d(input_shape, len(label_legend))
    return _train_model(model, train_ds, val_ds, label_legend, xaxis_callback,
                        max_epochs, es_patience, model_save_dir, model_name, log_dir)

