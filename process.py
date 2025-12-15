#!/usr/bin/env python3
"""
process.py

Script to prepare training and validation datasets from Parquet files.

Usage:
    python process.py [--batch-size BATCH_SIZE] [--days-in-series DAYS] [--days-per-bucket DAYS] [--frames-to-check FRAMES] \
                      [--bucketing-strategy STRATEGY] [--train-path PATH] [--val-path PATH] \
                      [--targeted-crops CROP1 CROP2 ...] [--other-crops CROP1 CROP2 ...] [--label-legend LABEL1 LABEL2 ...] \
                      [--num-features FEATURES]

Parameters:
    --batch-size: Number of samples per batch (default: from config)
    --days-in-series: Number of days in time series (default: from config)
    --days-per-bucket: Days per bucket for temporal aggregation (default: from config)
    --frames-to-check: Number of frames to check for vegetation (default: from config)
    --bucketing-strategy: Time window selection strategy (default: from config)
    --train-path: Training data path pattern (default: from config)
    --val-path: Validation data path pattern (default: from config)
    --targeted-crops: List of targeted crops (default: from config)
    --other-crops: List of other cultivated crops (default: from config)
    --label-legend: List of label classes (default: from config)
    --num-features: Number of features to use (12=only bands, 16=bands+indices) (default: from config)
"""

import sys
import os
import argparse
import glob
import tensorflow as tf
import numpy as np

# Add project root to path
project_root = os.path.abspath(".")
sys.path.append(project_root)

from dataloader import make_from_pandas, filter_double_croppings, parse, make_dataset, compute_normalization_stats, NUM_FEATURES

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def main():
    parser = argparse.ArgumentParser(description='Prepare training and validation datasets')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training (default: from config)')
    parser.add_argument('--days-in-series', type=int, default=None,
                       help='Number of days in time series (default: from config)')
    parser.add_argument('--days-per-bucket', type=int, default=None,
                       help='Days per bucket for temporal aggregation (default: from config)')
    parser.add_argument('--frames-to-check', type=int, default=None,
                       help='Number of frames to check for vegetation (default: from config)')
    parser.add_argument('--bucketing-strategy', type=str, default=None,
                       help='Bucketing strategy for time series (default: from config)')
    parser.add_argument('--train-path', type=str, default=None,
                       help='Training data path pattern (default: from config)')
    parser.add_argument('--val-path', type=str, default=None,
                       help='Validation data path pattern (default: from config)')
    parser.add_argument('--targeted-crops', nargs='+', default=[],
                       help='Targeted crops list (default: from config)')
    parser.add_argument('--other-crops', nargs='+', default=[],
                       help='Other crops list (default: from config)')
    parser.add_argument('--label-legend', nargs='+', default=[],
                       help='Label legend (default: from config)')
    parser.add_argument('--num-features', type=int, default=NUM_FEATURES,
                       help='Number of features to use (12=only bands, 16=bands+indices)')
    
    args = parser.parse_args()
    
    # Get parameters from command line arguments (set by bash script)
    BATCH_SIZE = args.batch_size
    DAYS_IN_SERIES = args.days_in_series
    DAYS_PER_BUCKET = args.days_per_bucket
    FRAMES_TO_CHECK = args.frames_to_check
    BUCKETING_STRATEGY = args.bucketing_strategy
    TRAIN_PATH = args.train_path
    VAL_PATH = args.val_path
    
    # Get crop lists from command line arguments
    targeted_cultivated_crops_list = args.targeted_crops
    other_cultivated_crops_list = args.other_crops
    label_legend = args.label_legend

    # Parse lists if passed as a single string with '|' or ','
    if len(label_legend) == 1:
        if '|' in label_legend[0]:
            label_legend = label_legend[0].split('|')
        elif ',' in label_legend[0]:
            label_legend = label_legend[0].split(',')
    if len(targeted_cultivated_crops_list) == 1:
        if '|' in targeted_cultivated_crops_list[0]:
            targeted_cultivated_crops_list = targeted_cultivated_crops_list[0].split('|')
        elif ',' in targeted_cultivated_crops_list[0]:
            targeted_cultivated_crops_list = targeted_cultivated_crops_list[0].split(',')
    if len(other_cultivated_crops_list) == 1:
        if '|' in other_cultivated_crops_list[0]:
            other_cultivated_crops_list = other_cultivated_crops_list[0].split('|')
        elif ',' in other_cultivated_crops_list[0]:
            other_cultivated_crops_list = other_cultivated_crops_list[0].split(',')
    
    MAX_IMAGES_PER_SERIES = (DAYS_IN_SERIES // DAYS_PER_BUCKET) + 1
    
    print(f"🔢 MAX_IMAGES_PER_SERIES: {MAX_IMAGES_PER_SERIES}")
    print(f"📦 Batch shape: [{BATCH_SIZE} x {MAX_IMAGES_PER_SERIES}]")
    print(f"🔍 Frames to check: {FRAMES_TO_CHECK}")
    print(f"⏰ Bucketing strategy: {BUCKETING_STRATEGY}")
    print(f"🌾 Targeted crops: {targeted_cultivated_crops_list}")
    print(f"🌱 Other crops: {other_cultivated_crops_list}")
    print(f"🏷️ Labels: {label_legend}")
    
    # Load data
    print("📊 Loading data from Parquet files...")
    train_files = glob.glob(TRAIN_PATH)
    val_files = glob.glob(VAL_PATH)
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    
    # Estimate normalization parameters
    print("📈 Computing normalization parameters...")
    num_features = args.num_features
    means, stds = compute_normalization_stats(
        train_files=train_files,
        label_legend=label_legend,
            targeted_cultivated_crops_list=targeted_cultivated_crops_list,
            other_cultivated_crops_list=other_cultivated_crops_list,
            days_in_series=DAYS_IN_SERIES,
            days_per_bucket=DAYS_PER_BUCKET,
            max_images_per_series=MAX_IMAGES_PER_SERIES,
            frames_to_check=FRAMES_TO_CHECK,
        bucketing_strategy=BUCKETING_STRATEGY,
        batch_size=BATCH_SIZE,
        num_features=num_features
    )
    
    print(f"📊 Computed normalization parameters for {means.shape[0]} features")
    
    # Create datasets
    print("🔄 Creating training and validation datasets...")
    train_ds, val_ds = make_dataset(
        train_files,
        val_files,
        method="pandas",
        batch_size=BATCH_SIZE,
        means=means[0:num_features],  # e.g. 16 features: 12 bands + 4 indices
        stds=stds[0:num_features],
        label_legend=label_legend,
        targeted_cultivated_crops_list=targeted_cultivated_crops_list,
        other_cultivated_crops_list=other_cultivated_crops_list,
        days_in_series=DAYS_IN_SERIES,
        days_per_bucket=DAYS_PER_BUCKET,
        max_images_per_series=MAX_IMAGES_PER_SERIES,
        frames_to_check=FRAMES_TO_CHECK,
        bucketing_strategy=BUCKETING_STRATEGY,
        num_features=num_features
    )
    
    # Save datasets
    print("💾 Saving datasets to disk...")
    train_save_path = f"./data/train_ds_{num_features}feat"
    val_save_path = f"./data/val_ds_{num_features}feat"
    train_ds.save(train_save_path)
    val_ds.save(val_save_path)
    
    print("✅ Datasets saved successfully!")
    print(f"   - Training dataset: {train_save_path}")
    print(f"   - Validation dataset: {val_save_path}")
    
    # Print dataset info
    print(f"📋 Dataset element spec: {train_ds.element_spec}")

if __name__ == "__main__":
    main() 