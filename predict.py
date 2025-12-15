#!/usr/bin/env python3
"""
predict.py
---------
Prediction script for crop classification models.

This script loads a trained model and makes predictions on new data,
generating crop classification maps and confidence scores.

Usage:
    python predict.py -model_name simplecnn -input_path data/new_data.parquet
    python predict.py -model_name inception1d_se_mixup_focal_attention_residual -input_path data/new_data.parquet

Input Data Format:
    The script expects Parquet files with Sentinel-2 time series data.
    
    Expected structure:
    - File path pattern: data/region/year/*.parquet
    - Each Parquet file contains time series for a specific location
    - Columns depend on NUM_FEATURES parameter:
      * NUM_FEATURES=12: ['longitude', 'latitude', 'date', 'B02', 'B03', 'B04', 'B08', 'B11', 'B12']
      * NUM_FEATURES=16: ['longitude', 'latitude', 'date', 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'NDVI', 'NDRE', 'EVI', 'SAVI']
    - Time series: Multiple observations per location across different dates
    
    Example file structure:
    data/
    ├── s2_final.parquet/
    │   ├── bbox=390747,1195097,437820,1284288/
    │   │   ├── year=2021/
    │   │   │   ├── part-00000.parquet
    │   │   │   └── part-00001.parquet
    │   │   └── year=2022/
    │   │       └── part-00000.parquet
    │   └── bbox=414748,1149262,439833,1193858/
    │       └── year=2021/
    │           └── part-00000.parquet
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dataloader import make_from_pandas, compute_normalization_stats, test_parser


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
    
    # Special handling for models with custom loss functions
    if "inception1d_se_mixup_focal_attention_residual" in model_name:
        from models.inception import categorical_focal_loss_var
        alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
        alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)
        loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)
        
        model = tf.keras.models.load_model(str(model_file), custom_objects={'loss': loss_fn})
        
    elif "inception1d_se_augmented" in model_name:
        from models.inception import categorical_focal_loss_var
        alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
        alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)
        loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)
        
        model = tf.keras.models.load_model(str(model_file), custom_objects={'loss': loss_fn})
        
    elif "inception1d" in model_name:
        from models.inception import categorical_focal_loss_var
        alpha_vector = [0.25, 0.80, 0.25, 0.25, 0.25, 0.25, 0.25]
        alpha_var = tf.Variable(alpha_vector, dtype=tf.float32, trainable=False)
        loss_fn = categorical_focal_loss_var(alpha_var, gamma=2.0)
        
        model = tf.keras.models.load_model(str(model_file), custom_objects={'loss': loss_fn})
        
    elif "bigcnn_focal" in model_name:
        from models.cnn import categorical_focal_loss
        model = tf.keras.models.load_model(str(model_file), custom_objects={'categorical_focal_loss': categorical_focal_loss})
        
    else:
        # Standard models (simplecnn, bigcnn, vgg1d, etc.)
        model = tf.keras.models.load_model(str(model_file))
    
    print(f"✅ Loaded model: {model_file}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Make predictions with crop classification models")
    parser.add_argument("-model_name", required=True, help="Name of the model to use for prediction")
    parser.add_argument("-input_path", required=True, help="Path to input data (parquet file or directory)")
    parser.add_argument("-output_path", default="results/predictions", help="Output directory for predictions")
    parser.add_argument("-days_in_series", type=int, default=120, help="Days in time series")
    parser.add_argument("-batch_size", type=int, default=1028, help="Batch size for prediction")
    parser.add_argument("-num_features", type=int, default=16, help="Number of features")
    parser.add_argument("-days_per_bucket", type=int, default=5, help="Days per bucket")
    parser.add_argument("-frames_to_check", type=int, default=2, help="Frames to check")
    parser.add_argument("-bucketing_strategy", default="random", help="Bucketing strategy")
    parser.add_argument("-start_day", type=int, default=0, help="Starting day for prediction (single day) - deprecated, always uses full date range")
    parser.add_argument("-save_probabilities", action="store_true", help="Save prediction probabilities")
    parser.add_argument("-save_confidence", action="store_true", help="Save confidence scores")
    parser.add_argument("-targeted_crops", required=True, help="Comma-separated list of targeted crops")
    parser.add_argument("-other_crops", required=True, help="Comma-separated list of other cultivated crops")
    parser.add_argument("-label_legend", required=True, help="Comma-separated list of label legend")
    parser.add_argument("-train_path", required=True, help="Path pattern for training files (for normalization)")
    
    args = parser.parse_args()
    
    # Constants - Always use the same date range as test.py for consistency
    date_range = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 200]
    
    # Parse crop lists from arguments
    targeted_cultivated_crops_list = [crop.strip() for crop in args.targeted_crops.split(',')]
    other_cultivated_crops_list = [crop.strip() for crop in args.other_crops.split(',')]
    label_legend = [label.strip() for label in args.label_legend.split(',')]
    
    # Derived constants
    max_images_per_series = (args.days_in_series // args.days_per_bucket) + 1
    
    print(f"🔢 Configuration:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Input path: {args.input_path}")
    print(f"   - Output path: {args.output_path}")
    print(f"   - Days in series: {args.days_in_series}")
    print(f"   - Max images per series: {max_images_per_series}")
    print(f"   - Batch shape: [{args.batch_size} x {max_images_per_series}]")
    print(f"   - Date range: {date_range}")
    print(f"   - Num features: {args.num_features} ({'bands only' if args.num_features == 12 else 'bands + indices'})")
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train files for normalization (using training data to compute stats)
    train_files = sorted(glob.glob(args.train_path))
    
    print(f"📁 Train files for normalization: {len(train_files)}")
    
    # Compute normalization parameters from training data
    print("📈 Computing normalization parameters from training data...")
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
    
    # Load the trained model
    print(f"🤖 Loading model: {args.model_name}")
    trained_model = load_model(args.model_name, args.days_in_series)
    
    # Load the input data
    print(f"📊 Loading input data from: {args.input_path}")
    
    # Handle different input types
    if os.path.isfile(args.input_path):
        # Single file
        input_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # Directory - find all parquet files
        input_files = sorted(glob.glob(f"{args.input_path}/**/*.parquet", recursive=True))
    else:
        # Pattern matching
        input_files = sorted(glob.glob(args.input_path))
    
    if not input_files:
        raise FileNotFoundError(f"No input files found at: {args.input_path}")
    
    print(f"📁 Input files: {len(input_files)}")
    
    # Load the input data
    input_files_ds = make_from_pandas(input_files)
    
    # Make predictions
    print("🔮 Running predictions...")
    preds = []
    probs = []
    lons = []
    lats = []
    raw_cdl_labels = []
    start_days = []
    
    total_start_days = len(date_range)
    for idx, start_day in enumerate(date_range):
        print(f"📅 Processing start_day {start_day} ({idx+1}/{total_start_days})...")
        input_ds = input_files_ds.map(
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
        
        batch_count = 0
        samples_this_day = 0
        for X, y, lon, lat, raw_CDL in input_ds:
            # Get raw predictions (probabilities)
            raw_pred = trained_model.predict(X, verbose=0)
            
            # Get class predictions
            pred = tf.argmax(raw_pred, axis=1)
            
            # Get confidence scores (max probability)
            confidence = tf.reduce_max(raw_pred, axis=1)
            
            preds.append(pred)
            probs.append(raw_pred)
            lons.append(lon)
            lats.append(lat)
            raw_cdl_labels.append(raw_CDL)
            start_days.append([start_day] * tf.shape(pred)[0].numpy())
            
            batch_count += 1
            samples_this_day += tf.shape(pred)[0].numpy()
            if batch_count % 10 == 0:
                print(f"   📊 Processed {batch_count} batches ({samples_this_day} samples) for start_day={start_day}")
        
        print(f"   ✅ Completed start_day={start_day}: {samples_this_day} samples in {batch_count} batches")
    
    # Concatenate results
    preds = tf.concat(preds, axis=0).numpy()
    probs = tf.concat(probs, axis=0).numpy()
    lons = tf.concat(lons, axis=0).numpy()
    lats = tf.concat(lats, axis=0).numpy()
    raw_cdl_labels = tf.concat(raw_cdl_labels, axis=0).numpy()
    start_days = list(np.concatenate(start_days, axis=0))
    
    # Calculate confidence scores
    confidence_scores = np.max(probs, axis=1)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'Raw CDL Label': raw_cdl_labels,
        'start_day': start_days,
        'predicted_class': preds,
        'predicted_label': [label_legend[p] for p in preds],
        'confidence_score': confidence_scores
    })
    
    # Add probability columns if requested
    if args.save_probabilities:
        for i, label in enumerate(label_legend):
            results[f'prob_{label}'] = probs[:, i]
    
    # Print prediction summary
    print("\n" + "="*60)
    print(f"🔮 PREDICTION RESULTS FOR {args.model_name.upper()}")
    print("="*60)
    print(f"Total predictions: {len(results)}")
    print(f"Average confidence: {confidence_scores.mean():.4f}")
    print(f"Min confidence: {confidence_scores.min():.4f}")
    print(f"Max confidence: {confidence_scores.max():.4f}")
    
    # Class distribution
    print("\n📊 Predicted Class Distribution:")
    class_counts = results['predicted_label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(results)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save main results
    results_path = output_dir / f"predictions_{args.model_name}_{args.days_in_series}days.parquet"
    results.to_parquet(results_path)
    print(f"✅ Predictions saved to: {results_path}")
    
    # Save confidence analysis
    confidence_path = output_dir / f"confidence_{args.model_name}_{args.days_in_series}days.txt"
    with open(confidence_path, 'w') as f:
        f.write(f"PREDICTION CONFIDENCE ANALYSIS FOR {args.model_name.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Total predictions: {len(results)}\n")
        f.write(f"Average confidence: {confidence_scores.mean():.4f}\n")
        f.write(f"Min confidence: {confidence_scores.min():.4f}\n")
        f.write(f"Max confidence: {confidence_scores.max():.4f}\n")
        f.write(f"Std confidence: {confidence_scores.std():.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("CONFIDENCE BY CLASS:\n")
        f.write("="*60 + "\n")
        for label in label_legend:
            mask = results['predicted_label'] == label
            if mask.sum() > 0:
                class_conf = confidence_scores[mask]
                f.write(f"{label}:\n")
                f.write(f"  Count: {mask.sum()}\n")
                f.write(f"  Avg confidence: {class_conf.mean():.4f}\n")
                f.write(f"  Min confidence: {class_conf.min():.4f}\n")
                f.write(f"  Max confidence: {class_conf.max():.4f}\n\n")
    
    print(f"✅ Confidence analysis saved to: {confidence_path}")
    
    # Save class distribution
    dist_path = output_dir / f"distribution_{args.model_name}_{args.days_in_series}days.txt"
    with open(dist_path, 'w') as f:
        f.write(f"PREDICTED CLASS DISTRIBUTION FOR {args.model_name.upper()}\n")
        f.write("="*60 + "\n")
        for label, count in class_counts.items():
            percentage = (count / len(results)) * 100
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")
    
    print(f"✅ Class distribution saved to: {dist_path}")
    
    # Save high-confidence predictions separately
    high_conf_threshold = 0.8
    high_conf_mask = confidence_scores >= high_conf_threshold
    high_conf_results = results[high_conf_mask]
    
    if len(high_conf_results) > 0:
        high_conf_path = output_dir / f"high_confidence_{args.model_name}_{args.days_in_series}days.parquet"
        high_conf_results.to_parquet(high_conf_path)
        print(f"✅ High-confidence predictions (≥{high_conf_threshold}) saved to: {high_conf_path}")
        print(f"   High-confidence count: {len(high_conf_results)} ({len(high_conf_results)/len(results)*100:.1f}%)")
    
    print(f"\n🎉 Prediction completed successfully!")
    print(f"📁 All results saved in: {output_dir}")


if __name__ == "__main__":
    main() 