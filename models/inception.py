"""
inception.py
------------
Inception-family architectures for crop classification from temporal satellite data.

This module provides Inception-based models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using multi-scale convolutional operations.

Available models:
* **baseline_inception1d** – Vanilla Inception-1D model for temporal classification
* **baseline_inception1d_se_augmented** – Inception + Squeeze-Excite + data augmentation
* **baseline_inception1d_se_mixup_focal_attention_residual** – Advanced variant with SE, MixUp, 
  Focal Loss, multi-head attention and residual connections (optional oversampling)

All models include:
* **EarlyStopping** with appropriate criterion per variant
* **TensorBoard** logging with configurable update frequency
* **ExtendedMetricsCallback** – confusion matrix, balanced accuracy, macro-F1 and per-class recall
* **ModelCheckpoint** for best model saving
* **ReduceLROnPlateau** learning rate scheduling

Intended for crop classification from multi-temporal satellite imagery where each
pixel contains a time series of spectral bands and vegetation indices.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing   import List, Tuple, Sequence, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model, Input
import numpy as np
from collections import Counter

# ---------------------------------------------------------------------
# Reusable utilities
# ---------------------------------------------------------------------
from models.callbacks import ExtendedMetricsCallback, AnnealedAlphaCallback

# ---------- Inception and helper modules -----------------------------
def inception_module(x: tf.Tensor, filters: int) -> tf.Tensor:
    p1 = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)

    p2 = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    p2 = layers.Conv1D(filters, 3, padding="same", activation="relu")(p2)

    p3 = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    p3 = layers.Conv1D(filters, 5, padding="same", activation="relu")(p3)

    p4 = layers.MaxPooling1D(3, strides=1, padding="same")(x)
    p4 = layers.Conv1D(filters, 1, padding="same", activation="relu")(p4)

    return layers.Concatenate()([p1, p2, p3, p4])



def squeeze_excite_block(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(x.shape[-1] // ratio, activation="relu")(se)
    se = layers.Dense(x.shape[-1],        activation="sigmoid")(se)
    se = layers.Reshape((1, x.shape[-1]))(se)
    return layers.Multiply()([x, se])

def mixup(ds: tf.data.Dataset, alpha: float = 0.4) -> tf.data.Dataset:
    def _mix(x, y):
        g1 = tf.random.gamma([], alpha)
        g2 = tf.random.gamma([], alpha)
        lam = g1 / (g1 + g2)

        idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        x2, y2 = tf.gather(x, idx), tf.gather(y, idx)
        return lam * x + (1 - lam) * x2, lam * y + (1 - lam) * y2
    return ds.map(_mix, num_parallel_calls=tf.data.AUTOTUNE)

def focal_loss(gamma: float = 2.0, alpha: float | Sequence[float] = 0.25,
               label_smoothing: float = 0.0):
    alpha = tf.constant(alpha, dtype=tf.float32)
    ce = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing,
                                              reduction="none")
    def _loss(y_true, y_pred):
        ce_val = ce(y_true, y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha_w = tf.reduce_sum(alpha * y_true, axis=-1)
        return tf.reduce_mean(alpha_w * tf.pow(1. - pt, gamma) * ce_val)
    return _loss

def categorical_focal_loss_var(alpha_var, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1. - tf.keras.backend.epsilon()
        )
        ce = -y_true * tf.math.log(y_pred)
        alpha_w = tf.reduce_sum(alpha_var * y_true, axis=-1, keepdims=True)
        focal_w = tf.pow(1. - y_pred, gamma)
        return tf.reduce_sum(alpha_w * focal_w * ce, axis=-1)
    return loss

def categorical_focal_loss_per_class(alpha_vector, gamma=2.0):
    alpha_vector = tf.constant(alpha_vector, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        alpha_weight = tf.reduce_sum(alpha_vector * y_true, axis=1, keepdims=True)
        focal_weight = tf.math.pow(1 - y_pred, gamma)
        loss = alpha_weight * focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    
    return loss 

def count_labels(dataset):
    counts = Counter()
    for _, y in dataset:
        labels = np.argmax(y.numpy(), axis=1)  # Assuming one-hot encoded labels
        counts.update(labels)
    return counts



# ---------- Configurable backbone constructor ------------------------
def _build_backbone(input_shape: Tuple[int, int],
                    num_classes: int,
                    *,
                    use_se: bool = False,
                    use_residual: bool = False,
                    use_attention: bool = False,
                    use_bilstm: bool = False) -> Model:

    inp = Input(shape=input_shape)
    x   = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)

    def block(x, filters):
        out = inception_module(x, filters)
        if use_se:
            out = squeeze_excite_block(out)
        if use_residual:
            if out.shape[-1] != x.shape[-1]:
                x = layers.Conv1D(out.shape[-1], 1, padding="same")(x)
            out = layers.Add()([out, x])
        return layers.BatchNormalization()(out)

    for f in (32, 64):
        x = block(x, f)
        x = layers.SpatialDropout1D(0.1)(x)
        x = layers.MaxPooling1D(2)(x)

    x = block(x, 128)

    if use_attention:
        x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)

    if use_bilstm:
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    #x = layers.Dropout(0.5)(x) basline
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)



# ---------- Simple data augmentation -------------------------------
def _augment(ds: tf.data.Dataset, sigma=0.01, scale=(0.9, 1.1)):
    def jitter(x, y):
        n = tf.random.normal(tf.shape(x), stddev=sigma)
        return x + n, y
    def scale_time(x, y):
        f = tf.random.uniform([], scale[0], scale[1])
        return x * f, y
    return ds.map(jitter).map(scale_time)

# ---------------------------------------------------------------------
# 1. baseline_inception1d  (vanilla model)
# ---------------------------------------------------------------------
def baseline_inception1d(
    train_ds, val_ds, label_legend: List[str],
    max_images_per_series: int, num_features: int,
    xaxis_callback, max_epochs: int, es_patience: int,
    model_save_dir: str, model_name: str, log_dir: str
):
    model = _build_backbone((max_images_per_series, num_features),
                            len(label_legend))
    return _fit(model, train_ds, val_ds, label_legend, xaxis_callback,
                max_epochs, es_patience, model_save_dir, model_name, log_dir)

# ---------------------------------------------------------------------
# 2. baseline_inception1d_se_augmented
# ---------------------------------------------------------------------
def baseline_inception1d_se_augmented(
    train_ds, val_ds, label_legend: List[str],
    max_images_per_series: int, num_features: int,
    xaxis_callback, max_epochs: int, es_patience: int,
    model_save_dir: str, model_name: str, log_dir: str,
):
    train_ds = _augment(train_ds)

    model = _build_backbone((max_images_per_series, num_features),
                            len(label_legend),
                            use_se=True)
    return _fit(model, train_ds, val_ds, label_legend, xaxis_callback,
                max_epochs, es_patience, model_save_dir, model_name, log_dir)

# ---------------------------------------------------------------------
# 3. baseline_inception1d_se_mixup_focal_attention_residual
# ---------------------------------------------------------------------
def baseline_inception1d_se_mixup_focal_attention_residual(
    train_ds, val_ds, label_legend: List[str],
    max_images_per_series: int, num_features: int,
    xaxis_callback, max_epochs: int, es_patience: int,
    model_save_dir: str, model_name: str, log_dir: str,
    apply_mixup: bool = True
):
    if apply_mixup:
        train_ds_final = mixup(train_ds)
    else:
        train_ds_final = train_ds

    model = _build_backbone((max_images_per_series, num_features),
                            len(label_legend),
                            use_se=True, use_residual=True, 
                            use_attention=True, use_bilstm=False)

    alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
    alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)

    #loss_fn = categorical_focal_loss_per_class(alpha_vector, gamma=2.0)
    loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)

    return _fit(model, train_ds_final, val_ds, label_legend, xaxis_callback,
                max_epochs, es_patience, model_save_dir, model_name, log_dir,
                loss_fn=loss_fn, early_stop_metric="val_categorical_accuracy",
                use_adam=True, use_annealed_alpha=True, alpha_vector=alpha_vector,
                alpha_var=alpha_var, alpha_class="Cultivated")





# TODO: explore inception variants
# - SE + Attention + BiLSTM
# - SE + Attention + BiLSTM + label smotthing
# - SE + Attention + BiLSTM + class_weight
# - SE + Attention + BiLSTM + class_weight + label smotthing
# - SE + Attention + BiLSTM + class_weight + label smotthing + class_weight
# - SE + Attention + BiLSTM + class_weight + label smotthing + class_weight + class_weight

# ---------------------------------------------------------------------
# Shared training routine
# ---------------------------------------------------------------------
from typing import Optional, List

def _fit(model: Model,
         train_ds, val_ds,
         label_legend: List[str],
         xaxis_callback, max_epochs: int, es_patience: int,
         model_save_dir: str, model_name: str, log_dir: str,
         *,
         loss_fn=None,
         early_stop_metric: str = "val_categorical_accuracy",
         class_weights=None,
         use_adam: bool = False,
         use_annealed_alpha: bool = False,
         alpha_vector: Optional[List[float]] = None,
         alpha_var: Optional[tf.Variable] = None,
         alpha_class: str = "Cultivated"):

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(model_save_dir, f"{model_name}.keras")

    if use_adam:
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
        
    model.compile(optimizer=optimizer,
                  loss=(loss_fn or keras.losses.CategoricalCrossentropy()),
                  metrics=["categorical_accuracy"])

    cbs = [
        keras.callbacks.TensorBoard(log_dir=log_dir,
                                    update_freq=xaxis_callback),
        keras.callbacks.EarlyStopping(monitor=early_stop_metric,
                                      mode="max" if "val_categorical_accuracy" in early_stop_metric else "min", #val_loss
                                      patience=es_patience,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                        save_best_only=True),
        ExtendedMetricsCallback(val_ds, label_legend,
                                output_dir=os.path.join(log_dir, "metrics")),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                          factor=0.5, patience=7,
                                          min_lr=1e-6, verbose=1)
    ]
    

    if use_annealed_alpha:
        assert alpha_vector is not None, "alpha_vector must be provided when using AnnealedAlphaCallback"

        cbs.append(AnnealedAlphaCallback(
            alpha_vector=alpha_vector,
            alpha_var=alpha_var,
            class_index=label_legend.index(alpha_class),
            max_alpha=1.3,
            total_epochs=max_epochs
        ))

    hist = model.fit(train_ds,
                     validation_data=val_ds,
                     epochs=max_epochs,
                     callbacks=cbs,
                     class_weight=class_weights,
                     verbose=1)

    print(f"[inception.py] model saved to → {ckpt_path}")
    return model, hist


