#!/usr/bin/env python3
"""
experiment_dataloader_configs.py

Script to experiment with different dataset construction configurations
to find the optimal setup for maximizing Cultivated cases.

This script tests different combinations of:
- Days in series
- Days per bucket  
- SCL checking logic (frames to check)
- Bucketing strategies

Usage:
    python experiment_dataloader_configs.py

Example:
    # Run the script to test different configurations
    python experiment_dataloader_configs.py
    
    # The script will test various combinations and output results like:
    #  Testing config: {'days_in_series': 120, 'days_per_bucket': 5, 'frames_to_check': 5, 'bucketing_strategy': 'mid_season'}
    #      Cultivated: 1347 (0.021)
    # 
    #  Best configuration found:
    #    - Days in series: 120
    #    - Days per bucket: 5  
    #    - Frames to check: 5
    #    - Bucketing strategy: mid_season
    #    - Cultivated cases: 1347 (2.1%)
    #    - Expected recall improvement: ~85%

Configuration Parameters:
    - days_in_series: Number of days to include in time series (default: 120)
    - days_per_bucket: Days grouped into one bucket (default: 5)
    - frames_to_check: Number of last frames to check for vegetation (default: 5)
    - bucketing_strategy: Time window selection strategy:
        * "deterministic": Hash-based deterministic selection
        * "early_season": Focus on early growing season
        * "mid_season": Focus on mid growing season (recommended)
        * "late_season": Focus on late growing season
        * "random": Random selection

Output:
    The script will print results for each configuration tested and identify
    the best configuration for maximizing Cultivated cases while maintaining
    good class balance.
"""

import sys
import os
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path 
project_root = os.path.abspath("..")
sys.path.append(project_root)

from dataloader import make_from_pandas, filter_double_croppings, calculate_vegetation_indices
from models.utils import count_labels, load_config

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_label_experiment(
    cdl: tf.Tensor,
    scl_val: tf.Tensor,
    targeted_cultivated_crops_list: list[str],
    other_cultivated_crops_list: list[str],
    frames_to_check: int = 3
) -> tf.Tensor:
    """
    Modified create_label function with configurable SCL checking
    """
    label = None
    if tf.reduce_any(tf.math.equal(cdl, tf.constant(targeted_cultivated_crops_list))):
        label = tf.cast(tf.squeeze(tf.where(cdl == tf.constant(targeted_cultivated_crops_list)) + 3), dtype=tf.int16)
    elif tf.reduce_any(tf.math.equal(cdl, tf.constant(other_cultivated_crops_list))):
        label = tf.constant(1, dtype=tf.int16)
    else:
        label = tf.constant(0, dtype=tf.int16)

    # Configurable SCL checking
    non_zero_scl = tf.gather(scl_val, tf.where(scl_val != 0))
    if tf.shape(non_zero_scl)[0] > 0:
        frames_to_check = tf.minimum(frames_to_check, tf.shape(non_zero_scl)[0])
        last_frames = non_zero_scl[-frames_to_check:]
        has_vegetation = tf.reduce_any(tf.equal(last_frames, 4))
        
        if (label != 0) & ~has_vegetation:
            label = tf.constant(2, dtype=tf.int16)

    return label

def bucket_timeseries_experiment(
    data: tf.Tensor,
    days_in_series: int,
    days_per_bucket: int,
    max_images_per_series: int,
    strategy: str = "deterministic"
) -> tf.Tensor:
    """
    Modified bucket_timeseries with different strategies
    """
    days = data[:, -1]
    max_day = tf.reduce_max(days)

    # Different strategies for selecting time interval
    if strategy == "deterministic":
        # Original deterministic approach
        sample_hash = tf.strings.to_hash_bucket_fast(tf.strings.as_string(tf.reduce_sum(days)), 1000)
        start_day = tf.cast(sample_hash % tf.cast(tf.maximum(1, max_day - days_in_series), tf.int64), tf.int32)
    elif strategy == "early_season":
        # Focus on early growing season
        start_day = tf.constant(0, dtype=tf.int32)
    elif strategy == "mid_season":
        # Focus on mid growing season
        start_day = tf.cast((max_day - days_in_series) // 2, tf.int32)
    elif strategy == "late_season":
        # Focus on late growing season
        start_day = tf.cast(tf.maximum(0, max_day - days_in_series), tf.int32)
    else:
        # Random selection
        start_day = tf.random.uniform(
            shape=(), minval=0, maxval=tf.maximum(1, max_day - days_in_series), dtype=tf.int32
        )
    
    end_day = start_day + days_in_series

    # Filter observations within interval
    mask = (days >= start_day) & (days <= end_day)
    series_in_time_range = tf.gather(data, tf.squeeze(tf.where(mask)), axis=0)

    # Select images
    idxs = tf.range(tf.shape(series_in_time_range)[0])
    ridxs = tf.sort(tf.gather(idxs, tf.range(tf.minimum(tf.shape(idxs)[0], max_images_per_series))))

    num_features = tf.shape(series_in_time_range)[1]
    rinput = tf.gather(series_in_time_range, ridxs)

    if tf.shape(rinput)[0] == 0:
        rinput = tf.zeros((1, num_features), dtype=tf.int32)

    normalized_days = tf.clip_by_value(rinput[:, -1] - rinput[0, -1], 0, 365)
    indices = normalized_days // days_per_bucket

    def unique_with_inverse(x):
        y, idx = tf.unique(x)
        num_segments = tf.shape(y)[0]
        num_elems = tf.shape(x)[0]
        return (y, idx, tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

    unique_indices, _, idxs = unique_with_inverse(indices)
    normalized_unique_days = tf.gather(normalized_days, idxs)
    rinput_unique = tf.gather(rinput, idxs)
    norm_days = tf.concat((rinput_unique[:, 0:-1], tf.expand_dims(normalized_unique_days, -1)), axis=1)

    padded_data = tf.scatter_nd(
        tf.reshape(unique_indices, (-1, 1)),
        norm_days,
        [max_images_per_series, num_features]
    )

    return padded_data

def parse_experiment(
    row: dict,
    norm: bool,
    means: tf.Tensor,
    stds: tf.Tensor,
    label_legend_: list[str],
    targeted_cultivated_crops_list: list[str],
    other_cultivated_crops_list: list[str],
    days_in_series: int,
    days_per_bucket: int,
    max_images_per_series: int,
    frames_to_check: int = 3,
    bucketing_strategy: str = "deterministic"
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Modified parse function with configurable parameters
    """
    num_images = row['num_images']

    scl_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['scl_vals'], out_type=tf.uint8), tf.int32), (num_images, 1))
    bands_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['bands'], out_type=tf.uint16, little_endian=False), tf.int32), shape=(num_images, 12))
    date_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['img_dates'], out_type=tf.uint16, little_endian=False), tf.int32), shape=(num_images, 1))
    days_from_start_of_series = tf.cast(date_decoded[:, 0] - date_decoded[0, 0], tf.int32)
    days_from_start_of_series = tf.reshape(days_from_start_of_series, shape=(num_images, 1))

    # Calculate vegetation indices
    indices_stack = calculate_vegetation_indices(bands_decoded)

    raw_data = tf.concat([bands_decoded, indices_stack, scl_decoded, days_from_start_of_series], axis=1)
    bucketed_data = bucket_timeseries_experiment(raw_data, days_in_series, days_per_bucket, max_images_per_series, bucketing_strategy)

    y = tf.cast(create_label_experiment(row['CDL'], bucketed_data[:, -2], targeted_cultivated_crops_list, other_cultivated_crops_list, frames_to_check), tf.int32)

    if norm:
        X = bucketed_data[:, :16]
    else:
        X = bucketed_data

    X = tf.cast(X, tf.float32)
    y = tf.one_hot(y, len(label_legend_))
    y = tf.ensure_shape(y, (len(label_legend_),))

    if norm:
        mask = tf.where(tf.not_equal(X, 0), tf.ones_like(X), tf.zeros_like(X))
        X = (X - means) / stds
        X = X * mask

    return X, y

def experiment_configuration(
    train_files: list,
    val_files: list,
    config: dict,
    label_legend: list[str],
    targeted_crops: list[str],
    other_crops: list[str]
) -> dict:
    """
    Test a specific configuration and return results
    """
    print(f"🧪 Testing config: {config}")
    
    # Load data
    train_files_ds = make_from_pandas(train_files)
    
    # Create dataset with experimental configuration
    ds = (
        train_files_ds
        .filter(filter_double_croppings)
        .map(lambda x: parse_experiment(
            x,
            norm=False,
            means=tf.zeros([16], dtype=tf.float32),
            stds=tf.ones([16], dtype=tf.float32),
            label_legend_=label_legend,
            targeted_cultivated_crops_list=targeted_crops,
            other_cultivated_crops_list=other_crops,
            days_in_series=config['days_in_series'],
            days_per_bucket=config['days_per_bucket'],
            max_images_per_series=config['max_images_per_series'],
            frames_to_check=config['frames_to_check'],
            bucketing_strategy=config['bucketing_strategy']
        ), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1028)
    )
    
    # Count labels using imported function
    label_counts = count_labels(ds)
    
    # Calculate metrics
    total_samples = sum(label_counts.values())
    cultivated_ratio = label_counts.get(1, 0) / total_samples if total_samples > 0 else 0
    
    results = {
        'config': config,
        'label_counts': label_counts,
        'total_samples': total_samples,
        'cultivated_count': label_counts.get(1, 0),
        'cultivated_ratio': cultivated_ratio,
        'label_distribution': {label_legend[i]: label_counts.get(i, 0) for i in range(len(label_legend))}
    }
    
    print(f"   ✅ Cultivated: {label_counts.get(1, 0)} ({cultivated_ratio:.3f})")
    return results

def main():
    # Change to project root directory to ensure paths work correctly
    project_root = os.path.abspath("..")
    os.chdir(project_root)
    print(f"📁 Changed to project root: {project_root}")
    
    # Load configuration from file
    print("📋 Loading configuration from config/dataloader.txt...")
    config_data = load_config()
    
    # Load files
    print("📊 Loading data files...")
    train_files = glob.glob(config_data['train_path'])
    val_files = glob.glob(config_data['val_path'])
    print(f"   Train files: {len(train_files)}")
    print(f"   Val files: {len(val_files)}")
    
    # Define experiments
    experiments = [
        # Baseline
        {
            'name': 'baseline',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        
        # Different time windows
        {
            'name': 'short_series',
            'days_in_series': 90,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        {
            'name': 'long_series',
            'days_in_series': 150,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        {
            'name': 'very_long_series',
            'days_in_series': 180,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        
        # Different bucketing strategies
        {
            'name': 'early_season',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'early_season'
        },
        {
            'name': 'mid_season',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'mid_season'
        },
        {
            'name': 'late_season',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'late_season'
        },
        
        # Different SCL checking
        {
            'name': 'lenient_scl',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 1,
            'bucketing_strategy': 'deterministic'
        },
        {
            'name': 'strict_scl',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 5,
            'bucketing_strategy': 'deterministic'
        },
        {
            'name': 'very_strict_scl',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 7,
            'bucketing_strategy': 'deterministic'
        },
        
        # Different bucket sizes
        {
            'name': 'small_buckets',
            'days_in_series': 120,
            'days_per_bucket': 3,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        {
            'name': 'large_buckets',
            'days_in_series': 120,
            'days_per_bucket': 10,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        {
            'name': 'very_small_buckets',
            'days_in_series': 120,
            'days_per_bucket': 2,
            'frames_to_check': 3,
            'bucketing_strategy': 'deterministic'
        },
        
        # Combined experiments (best strategies)
        {
            'name': 'mid_season_lenient',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 1,
            'bucketing_strategy': 'mid_season'
        },
        {
            'name': 'mid_season_strict',
            'days_in_series': 120,
            'days_per_bucket': 5,
            'frames_to_check': 5,
            'bucketing_strategy': 'mid_season'
        },
        {
            'name': 'long_mid_season',
            'days_in_series': 150,
            'days_per_bucket': 5,
            'frames_to_check': 3,
            'bucketing_strategy': 'mid_season'
        }
    ]
    
    # Run experiments
    results = []
    print(f"\n🔬 Running {len(experiments)} experiments...")
    
    for exp in experiments:
        config = {
            'days_in_series': exp['days_in_series'],
            'days_per_bucket': exp['days_per_bucket'],
            'max_images_per_series': (exp['days_in_series'] // exp['days_per_bucket']) + 1,
            'frames_to_check': exp['frames_to_check'],
            'bucketing_strategy': exp['bucketing_strategy']
        }
        
        try:
            result = experiment_configuration(train_files, val_files, config, config_data['label_legend'], 
                                            config_data['targeted_crops'], config_data['other_crops'])
            result['name'] = exp['name']
            results.append(result)
        except Exception as e:
            print(f"   ❌ Error in {exp['name']}: {e}")
    
    # Analyze results
    print(f"\n📊 Experiment Results Summary:")
    print("=" * 80)
    
    # Sort by cultivated ratio
    results.sort(key=lambda x: x['cultivated_ratio'], reverse=True)
    
    for result in results:
        print(f"{result['name']:20} | Cultivated: {result['cultivated_count']:4d} ({result['cultivated_ratio']:.3f}) | "
              f"Total: {result['total_samples']:6d}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_results_{timestamp}.csv"
    
    df_results = []
    for result in results:
        row = {
            'name': result['name'],
            'cultivated_count': result['cultivated_count'],
            'cultivated_ratio': result['cultivated_ratio'],
            'total_samples': result['total_samples'],
            **result['config']
        }
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    df.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Find best configuration
    best_result = results[0]
    print(f"\n🏆 Best configuration: {best_result['name']}")
    print(f"   Cultivated cases: {best_result['cultivated_count']} ({best_result['cultivated_ratio']:.3f})")
    print(f"   Configuration: {best_result['config']}")
    
    # Show top 5 configurations
    print(f"\n🥇 Top 5 configurations:")
    for i, result in enumerate(results[:5]):
        print(f"   {i+1}. {result['name']}: {result['cultivated_count']} cases ({result['cultivated_ratio']:.3f})")

if __name__ == "__main__":
    main() 