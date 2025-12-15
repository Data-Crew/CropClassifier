"""
cnn.py
-------
Convolutional Neural Network architectures for crop classification from temporal satellite data.

This module provides CNN models for sequence classification of crop types
from Sentinel-2 time series data. Each pixel's temporal sequence is classified into
crop categories using 1D convolutional operations.

Available models:
* **baseline_simplecnn** – Simple 1D CNN with basic convolutional layers
* **baseline_bigcnn** – Larger CNN with more filters and additional layers

All models include:
* **EarlyStopping** on validation loss
* **TensorBoard** logging with configurable update frequency
* **ExtendedMetricsCallback** – confusion matrix, balanced accuracy & macro-F1
* **ModelCheckpoint** for best model saving

Intended for crop classification from multi-temporal satellite imagery where each
pixel contains a time series of spectral bands and vegetation indices.
"""

import tensorflow as tf

def baseline_simplecnn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: list,
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: int | str = 1,
    max_epochs: int = 40,
    es_patience: int = 40,
    model_save_dir: str = 'results',
    model_name: str = 'baseline_simplecnn',
    log_dir: str = 'logs/fit'
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains a baseline 1D CNN model for temporal series classification.

    Parameters:
    - train_ds (tf.data.Dataset): Training dataset.
    - val_ds (tf.data.Dataset): Validation dataset.
    - label_legend (list): List of label classes.
    - max_images_per_series (int): Number of time steps.
    - num_features (int): Number of features per time step.
    - xaxis_callback (int|str): 1 to analyze accuracy/loss by batch or 'epoch' to do it by epochs.
    - max_epochs (int, optional): Maximum number of training epochs. Default is 40.
    - es_patience (int, optional): Early stopping patience. Default is 40.
    - model_save_dir (str, optional): Directory to save the model. Default is 'results'.
    - model_name (str, optional): Name for the saved model. Default is 'baseline_simplecnn'.
    - log_dir (str, optional): Directory for TensorBoard logs. Default is 'logs/fit'.

    Returns:
    - model (tf.keras.Model): Trained Keras model.
    - history (tf.keras.callbacks.History): Training history.
    """
    import os
    import tensorflow as tf
    from models.callbacks import ExtendedMetricsCallback

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(max_images_per_series, num_features)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=50, activation='relu'),
        tf.keras.layers.Dense(units=len(label_legend)),
        tf.keras.layers.Softmax(),
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'{model_name}.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience, mode='min'),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=xaxis_callback),
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics"))
    ]

    history = model.fit(
        train_ds,
        epochs=max_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, history

def baseline_bigcnn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    label_legend: list,
    max_images_per_series: int,
    num_features: int,
    xaxis_callback: int | str = 1,
    max_epochs: int = 40,
    es_patience: int = 40,
    model_save_dir: str = 'results',
    model_name: str = 'baseline_bigcnn',
    log_dir: str = 'logs/fit'
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains a bigger 1D CNN model for temporal series classification.

    This version uses more convolutional filters, an extra Conv1D layer, 
    reduced dropout, and a larger dense layer for increased capacity.

    Parameters:
    - train_ds (tf.data.Dataset): Training dataset.
    - val_ds (tf.data.Dataset): Validation dataset.
    - label_legend (list): List of label classes.
    - max_images_per_series (int): Number of time steps.
    - num_features (int): Number of features per time step.
    - xaxis_callback (int|str): 1 to analyze accuracy/loss by batch or 'epoch' to do it by epochs.
    - max_epochs (int, optional): Maximum number of training epochs. Default is 40.
    - es_patience (int, optional): Early stopping patience. Default is 40.
    - model_save_dir (str, optional): Directory to save the model. Default is 'results'.
    - model_name (str, optional): Name for the saved model. Default is 'baseline_bigcnn'.
    - log_dir (str, optional): Directory for TensorBoard logs. Default is 'logs/fit'.

    Returns:
    - model (tf.keras.Model): Trained Keras model.
    - history (tf.keras.callbacks.History): Training history.
    """
    import os
    import tensorflow as tf
    from models.callbacks import ExtendedMetricsCallback

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(max_images_per_series, num_features)),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=len(label_legend)),
        tf.keras.layers.Softmax(),
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'{model_name}.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience, mode='min'),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=xaxis_callback),
        ExtendedMetricsCallback(val_ds, label_legend, output_dir=os.path.join(log_dir, "metrics"))
    ]

    history = model.fit(
        train_ds,
        epochs=max_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, history

