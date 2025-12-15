#!/usr/bin/env python3
"""
test.py
-------
Test script for crop classification models.

This script loads a trained model and evaluates it on test data, generating
comprehensive metrics that can be compared with training results.

Usage:
    python test.py -model_name simplecnn
    python test.py -model_name inception1d_se_mixup_focal_attention_residual
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# === GPU STABILITY MODE (optional) ===
# Set STABLE_GPU_MODE=0 to disable stability settings and use full XLA performance
# Default is STABLE_GPU_MODE=1 (stable mode enabled) to prevent CUDA crashes
STABLE_GPU_MODE = os.environ.get('STABLE_GPU_MODE', '1') == '1'

if STABLE_GPU_MODE:
    # Disable XLA JIT to prevent 'CUDA_ERROR_LAUNCH_FAILED' on consumer GPUs
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
    os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = '0'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='  # Disable XLA CUDA
    os.environ['TF_DISABLE_JIT'] = '1'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# Always enable memory growth (good practice)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf

# === FORCE DISABLE XLA AFTER TF IMPORT ===
if STABLE_GPU_MODE:
    # This MUST be called after importing TensorFlow
    tf.config.optimizer.set_jit(False)
    # Force eager execution for maximum stability (slower but no XLA)
    tf.config.run_functions_eagerly(True)
    print("🛡️  Eager execution enabled - XLA completely disabled")

import gc
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

# Configure GPU (stable_mode based on STABLE_GPU_MODE env var)
from config.gpu.gpu_utils import configure_tensorflow_gpu
configure_tensorflow_gpu(stable_mode=STABLE_GPU_MODE)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dataloader import make_from_pandas, compute_normalization_stats, test_parser
import subprocess


def get_gpu_temperature():
    """Get GPU temperature using nvidia-smi. Returns temperature in Celsius or None if unavailable."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def wait_for_gpu_cooldown(max_temp=75, check_interval=10):
    """
    Wait for GPU to cool down if temperature is too high.
    This helps prevent CUDA_ERROR_LAUNCH_FAILED due to thermal throttling.
    """
    temp = get_gpu_temperature()
    if temp is None:
        return
    
    if temp > max_temp:
        print(f"🌡️  GPU temperature: {temp}°C (above {max_temp}°C threshold)")
        print(f"⏳ Waiting for GPU to cool down...")
        while temp and temp > max_temp - 5:  # Wait until 5 degrees below threshold
            time.sleep(check_interval)
            temp = get_gpu_temperature()
            if temp:
                print(f"🌡️  GPU temperature: {temp}°C")
        print(f"✅ GPU cooled down to {temp}°C, continuing...")
    else:
        print(f"🌡️  GPU temperature: {temp}°C (OK)")


def load_model(model_name, days_in_series):
    """
    Load a trained model based on the model name.
    
    Args:
        model_name (str): Name of the model to load
        days_in_series (int): Number of days in the time series
    
    Returns:
        tf.keras.Model: Loaded model
    """
    # Models are stored directly as {model_name}_{days_in_series}days.keras
    model_file = Path(f"results/models/{model_name}_{days_in_series}days.keras")
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    # Disable JIT compilation when loading model
    # This prevents XLA from recompiling the model
    with tf.device('/CPU:0'):
        # Load model weights only, then transfer to GPU
        # This avoids XLA compilation issues
        pass
    
    # Special handling for models with custom loss functions
    if "inception1d_se_mixup_focal_attention_residual" in model_name:
        from models.inception import categorical_focal_loss_var
        alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
        alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)
        loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)
        
        model = tf.keras.models.load_model(str(model_file), custom_objects={'loss': loss_fn}, compile=False)
        
    elif "inception1d_se_augmented" in model_name:
        from models.inception import categorical_focal_loss_var
        alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
        alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)
        loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)
        
        model = tf.keras.models.load_model(str(model_file), custom_objects={'loss': loss_fn}, compile=False)
        
    elif "inception1d" in model_name:
        from models.inception import categorical_focal_loss_var
        alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
        alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)
        loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)
        
        model = tf.keras.models.load_model(str(model_file), custom_objects={'loss': loss_fn}, compile=False)
        
    elif "bigcnn_focal" in model_name:
        from models.cnn import categorical_focal_loss
        model = tf.keras.models.load_model(str(model_file), custom_objects={'categorical_focal_loss': categorical_focal_loss}, compile=False)
        
    else:
        # Standard models (simplecnn, bigcnn, vgg1d, etc.)
        model = tf.keras.models.load_model(str(model_file), compile=False)
    
    # Disable any remaining JIT compilation on the model
    # model.jit_compile = False  # Only works in TF 2.9+
    
    print(f"✅ Loaded model: {model_file} (compile=False to avoid XLA)")
    return model


def main():
    parser = argparse.ArgumentParser(description="Test crop classification models")
    parser.add_argument("-model_name", required=True, help="Name of the model to test")
    parser.add_argument("-days_in_series", type=int, default=120, help="Days in time series")
    parser.add_argument("-batch_size", type=int, default=1028, help="Batch size for testing")
    parser.add_argument("-num_features", type=int, default=16, help="Number of features")
    parser.add_argument("-days_per_bucket", type=int, default=5, help="Days per bucket")
    parser.add_argument("-frames_to_check", type=int, default=2, help="Frames to check")
    parser.add_argument("-bucketing_strategy", default="random", help="Bucketing strategy")
    parser.add_argument("-targeted_crops", required=True, help="Comma-separated list of targeted crops")
    parser.add_argument("-other_crops", required=True, help="Comma-separated list of other cultivated crops")
    parser.add_argument("-label_legend", required=True, help="Comma-separated list of label legend")
    parser.add_argument("-train_path", required=True, help="Path pattern for training files")
    parser.add_argument("-test_path", required=True, help="Path pattern for test files")
    
    args = parser.parse_args()
    
    # Constants
    date_range = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 200]
    
    # Parse crop lists from arguments
    targeted_cultivated_crops_list = [crop.strip() for crop in args.targeted_crops.split(',')]
    other_cultivated_crops_list = [crop.strip() for crop in args.other_crops.split(',')]
    label_legend = [label.strip() for label in args.label_legend.split(',')]
    
    # Derived constants
    max_images_per_series = (args.days_in_series // args.days_per_bucket) + 1
    
    print(f"🔢 Configuration:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Days in series: {args.days_in_series}")
    print(f"   - Max images per series: {max_images_per_series}")
    print(f"   - Batch shape: [{args.batch_size} x {max_images_per_series}]")
    print(f"   - Targeted crops: {targeted_cultivated_crops_list}")
    print(f"   - Other crops: {other_cultivated_crops_list}")
    print(f"   - Label legend: {label_legend}")
    
    # Load train and test files
    train_files = sorted(glob.glob(args.train_path))
    test_files = sorted(glob.glob(args.test_path))
    
    print(f"📁 Train files: {len(train_files)}")
    print(f"📁 Test files: {len(test_files)}")
    
    # Warn if test dataset is very large (potential memory issues)
    if len(test_files) > 500:
        print(f"⚠️  WARNING: Large test dataset detected ({len(test_files)} files)")
        print(f"   This may cause GPU memory fragmentation. Consider:")
        print(f"   - Using a smaller test dataset")
        print(f"   - Reducing batch_size (current: {args.batch_size})")
        print(f"   - Processing in smaller chunks")
    
    if len(test_files) == 0:
        print(f"❌ ERROR: No test files found at: {args.test_path}")
        print(f"   Please check the path and try again.")
        sys.exit(1)
    
    # Compute normalization parameters
    print("📈 Computing normalization parameters...")
    means, stds = compute_normalization_stats(
        train_files=train_files,
        label_legend=label_legend,
        targeted_cultivated_crops_list=targeted_cultivated_crops_list,
        other_cultivated_crops_list=other_cultivated_crops_list,
        days_in_series=args.days_in_series,
        days_per_bucket=args.days_per_bucket,
        max_images_per_series=max_images_per_series,
        frames_to_check=args.frames_to_check,
        bucketing_strategy=args.bucketing_strategy,
        batch_size=args.batch_size,
        num_features=args.num_features
    )
    
    # Load the test data
    print("📊 Loading test data...")
    test_files_ds = make_from_pandas(test_files)
    
    # === CHECKPOINTING SETUP ===
    # Save progress after each start_day so we can resume if it crashes
    checkpoint_dir = Path("results/test/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{args.model_name}_{args.days_in_series}days_checkpoint.npz"
    
    # Check if we have a checkpoint to resume from
    completed_start_days = []
    preds = []
    trues = []
    lons = []
    lats = []
    raw_cdl_labels = []
    start_days = []
    
    if checkpoint_file.exists():
        print(f"📂 Found checkpoint file: {checkpoint_file}")
        try:
            checkpoint = np.load(checkpoint_file, allow_pickle=True)
            completed_start_days = list(checkpoint['completed_start_days'])
            preds = list(checkpoint['preds'])
            trues = list(checkpoint['trues'])
            lons = list(checkpoint['lons'])
            lats = list(checkpoint['lats'])
            raw_cdl_labels = list(checkpoint['raw_cdl_labels'])
            start_days = list(checkpoint['start_days'])
            print(f"✅ Resuming from checkpoint! Already completed start_days: {completed_start_days}")
            print(f"   Loaded {len(preds)} prediction batches from checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Starting from scratch...")
            completed_start_days = []
            preds = []
            trues = []
            lons = []
            lats = []
            raw_cdl_labels = []
            start_days = []
    
    # Predict on the test data
    print("🔮 Running predictions on test data...")
    
    for start_day_idx, start_day in enumerate(date_range):
        # Skip if already completed (resuming from checkpoint)
        if start_day in completed_start_days:
            print(f"\n⏭️  Skipping start_day={start_day} (already completed in checkpoint)")
            continue
        
        # === GPU THERMAL MANAGEMENT ===
        # Check GPU temperature before each iteration to prevent thermal crashes
        wait_for_gpu_cooldown(max_temp=78, check_interval=10)
        
        # Reload model for each start_day to prevent memory accumulation
        print(f"\n🔄 Processing start_day={start_day} ({start_day_idx+1}/{len(date_range)})")
        print(f"🤖 Loading model: {args.model_name}")
        start_time = time.time()
        trained_model = load_model(args.model_name, args.days_in_series)
        test_ds = test_files_ds.map(
            lambda x: test_parser(
                x, norm=True, 
                means=means,
                stds=stds,
                label_legend=label_legend,
                targeted_cultivated_crops_list=targeted_cultivated_crops_list,
                other_cultivated_crops_list=other_cultivated_crops_list,
                days_in_series=args.days_in_series,
                days_per_bucket=args.days_per_bucket,
                max_images_per_series=max_images_per_series,
                frames_to_check=args.frames_to_check,
                bucketing_strategy=args.bucketing_strategy,
                start_day=start_day,
                num_features=args.num_features
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(args.batch_size)
        
        # Process in smaller chunks to prevent GPU memory fragmentation
        # The real problem: after processing thousands of large batches (1028 samples),
        # CUDA memory becomes fragmented even though we free memory.
        # Solution: process each dataset batch in smaller chunks
        # Adjust chunk size based on dataset size to prevent fragmentation
        if len(test_files) > 500:
            # For very large datasets, use smaller chunks to prevent fragmentation
            processing_chunk_size = 64
            print(f"   📊 Large dataset detected: using smaller chunks (64 samples) to prevent memory fragmentation")
        elif len(test_files) > 200:
            # For medium-large datasets, use medium chunks
            processing_chunk_size = 128
        else:
            # For smaller datasets, can use larger chunks
            processing_chunk_size = 256
        
        batch_count = 0
        for X, y, lon, lat, raw_CDL in test_ds:
            try:
                # Split large dataset batches into smaller processing chunks
                batch_size_actual = X.shape[0]
                prediction_batch_size = 32  # Small batch size for model.predict()
                
                # Process the entire dataset batch in smaller chunks
                all_preds = []
                all_trues = []
                all_lons = []
                all_lats = []
                all_raw_cdls = []
                
                for chunk_start in range(0, batch_size_actual, processing_chunk_size):
                    chunk_end = min(chunk_start + processing_chunk_size, batch_size_actual)
                    chunk_X = X[chunk_start:chunk_end]
                    chunk_y = y[chunk_start:chunk_end]
                    chunk_lon = lon[chunk_start:chunk_end] if hasattr(lon, '__getitem__') else lon
                    chunk_lat = lat[chunk_start:chunk_end] if hasattr(lat, '__getitem__') else lat
                    chunk_raw_CDL = raw_CDL[chunk_start:chunk_end] if hasattr(raw_CDL, '__getitem__') else raw_CDL
                    
                    # Process this chunk in even smaller batches for prediction
                    if chunk_X.shape[0] > prediction_batch_size:
                        chunk_preds = []
                        for pred_start in range(0, chunk_X.shape[0], prediction_batch_size):
                            pred_end = min(pred_start + prediction_batch_size, chunk_X.shape[0])
                            pred_chunk_X = chunk_X[pred_start:pred_end]
                            pred_probs = trained_model.predict(pred_chunk_X, verbose=0, batch_size=prediction_batch_size)
                            chunk_preds.append(tf.argmax(pred_probs, axis=1).numpy())
                            del pred_probs, pred_chunk_X
                            gc.collect()
                        chunk_pred = np.concatenate(chunk_preds, axis=0)
                        del chunk_preds
                    else:
                        pred_probs = trained_model.predict(chunk_X, verbose=0, batch_size=prediction_batch_size)
                        chunk_pred = tf.argmax(pred_probs, axis=1).numpy()
                        del pred_probs
                    
                    chunk_true = tf.argmax(chunk_y, axis=1).numpy()
                    
                    all_preds.append(chunk_pred)
                    all_trues.append(chunk_true)
                    all_lons.append(chunk_lon.numpy() if hasattr(chunk_lon, 'numpy') else chunk_lon)
                    all_lats.append(chunk_lat.numpy() if hasattr(chunk_lat, 'numpy') else chunk_lat)
                    all_raw_cdls.append(chunk_raw_CDL.numpy() if hasattr(chunk_raw_CDL, 'numpy') else chunk_raw_CDL)
                    
                    # Clean up chunk tensors immediately
                    del chunk_X, chunk_y, chunk_pred, chunk_true
                    gc.collect()
                
                # Concatenate all chunks from this dataset batch
                pred = np.concatenate(all_preds, axis=0)
                true = np.concatenate(all_trues, axis=0)
                
                preds.append(pred)
                trues.append(true)
                lons.extend(all_lons)
                lats.extend(all_lats)
                raw_cdl_labels.extend(all_raw_cdls)
                start_days.append([start_day] * len(pred))
                
                del all_preds, all_trues, all_lons, all_lats, all_raw_cdls
                
                batch_count += 1
                
                # Memory cleanup after each batch (but NOT clear_session - that destroys the model!)
                gc.collect()
                
                # Print progress every 50 batches (or more frequently for large datasets)
                progress_interval = 25 if len(test_files) > 500 else 50
                if batch_count % progress_interval == 0:
                    elapsed_time = time.time() - start_time
                    elapsed_min = int(elapsed_time // 60)
                    elapsed_sec = int(elapsed_time % 60)
                    print(f"⏳ Batch {batch_count} | ⏱️  {elapsed_min}m {elapsed_sec}s elapsed | 🧹 GPU memory cleared (start_day={start_day})")
                    
            except tf.errors.ResourceExhaustedError as e:
                print(f"❌ GPU out of memory at batch {batch_count}, start_day {start_day}")
                print(f"   Error: {e}")
                print(f"   Try reducing batch_size or prediction_batch_size")
                raise
            except Exception as e:
                print(f"❌ Error during prediction at batch {batch_count}, start_day {start_day}")
                print(f"   Error: {e}")
                raise
        
        # Clear memory after processing all batches for this start_day
        total_time = time.time() - start_time
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        print(f"✅ Completed {batch_count} batches for start_day={start_day} in {total_min}m {total_sec}s")
        
        # === AGGRESSIVE GPU MEMORY CLEANUP ===
        # Step 1: Delete model reference
        del trained_model
        
        # Step 2: Force synchronization - wait for GPU to finish all operations
        # This prevents race conditions where we try to free memory while GPU is still working
        try:
            # Synchronize all GPU streams before cleanup
            tf.test.experimental.sync_devices()
        except Exception:
            # Fallback: do a dummy operation to force sync
            try:
                _ = tf.constant([0.0])
            except Exception:
                pass
        
        # Step 3: Clear Keras session (frees model graph)
        tf.keras.backend.clear_session()
        
        # Step 4: Python garbage collection
        gc.collect()
        
        # Step 5: Reset CUDA memory stats if available
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.reset_memory_stats(gpu.name)
        except Exception:
            pass  # Ignore if not available
        
        # Step 6: Longer pause to let CUDA driver stabilize and GPU cool down
        # This helps prevent "unspecified launch failure" after many iterations
        time.sleep(5)
        print(f"🧹 GPU memory cleared, cooling down for 5s...")
        
        # === SAVE CHECKPOINT ===
        # Save progress so we can resume if it crashes
        completed_start_days.append(start_day)
        try:
            np.savez(
                checkpoint_file,
                completed_start_days=np.array(completed_start_days),
                preds=np.array(preds, dtype=object),
                trues=np.array(trues, dtype=object),
                lons=np.array(lons, dtype=object),
                lats=np.array(lats, dtype=object),
                raw_cdl_labels=np.array(raw_cdl_labels, dtype=object),
                start_days=np.array(start_days, dtype=object)
            )
            print(f"💾 Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")
    
    # Concatenate results (already numpy arrays)
    print("\n📊 Concatenating results...")
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    lons = np.concatenate(lons, axis=0)
    lats = np.concatenate(lats, axis=0)
    raw_cdl_labels = np.concatenate(raw_cdl_labels, axis=0)
    start_days = list(np.concatenate(start_days, axis=0))
    
    # Create results DataFrame
    results = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'Raw CDL Label': raw_cdl_labels,
        'start_day': start_days,
        'true_label': trues,
        'predictions': preds,
        'predicted_label': [label_legend[p] for p in preds]
    })
    
    # Calculate comprehensive metrics
    print("📊 Calculating metrics...")
    test_accuracy = accuracy_score(results['true_label'], results['predictions'])
    test_micro_f1 = f1_score(results['true_label'], results['predictions'], average='micro')
    test_macro_f1 = f1_score(results['true_label'], results['predictions'], average='macro')
    test_weighted_f1 = f1_score(results['true_label'], results['predictions'], average='weighted')
    
    # Generate classification report
    class_report = classification_report(
        results['true_label'], 
        results['predictions'], 
        target_names=label_legend, 
        digits=4,
        zero_division=0
    )
    
    # Print metrics
    print("\n" + "="*60)
    print(f"🧪 TEST RESULTS FOR {args.model_name.upper()}")
    print("="*60)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print(f"Test micro F1 score: {test_micro_f1:.4f}")
    print(f"Test macro F1 score: {test_macro_f1:.4f}")
    print(f"Test weighted F1 score: {test_weighted_f1:.4f}")
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT:")
    print("="*60)
    print(class_report)
    
    # Save results and metrics
    print("💾 Saving results...")
    
    # Create test results directory
    test_dir = Path("results/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results DataFrame
    results_path = test_dir / f"{args.model_name}_{args.days_in_series}days_results.parquet"
    results.to_parquet(results_path)
    print(f"✅ Results saved to: {results_path}")
    
    # Save metrics to text file
    metrics_path = test_dir / f"{args.model_name}_{args.days_in_series}days_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"TEST RESULTS FOR {args.model_name.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Test set accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test micro F1 score: {test_micro_f1:.4f}\n")
        f.write(f"Test macro F1 score: {test_macro_f1:.4f}\n")
        f.write(f"Test weighted F1 score: {test_weighted_f1:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write("="*60 + "\n")
        f.write(class_report)
    
    print(f"✅ Metrics saved to: {metrics_path}")
    
    # Clean up checkpoint file after successful completion
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print(f"🗑️  Checkpoint file removed (test completed successfully)")
        except Exception:
            pass


if __name__ == "__main__":
    main() 