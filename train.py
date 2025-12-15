#!/usr/bin/env python3
"""
train.py

Script to train crop classification models using different architectures.
This script loads the prepared datasets and trains a specified model architecture.

Usage:
    python train.py --model MODEL_NAME [--epochs EPOCHS] [--days-in-series DAYS] [--days-per-bucket DAYS] \
                    [--es-patience PATIENCE] [--num-features FEATURES] [--label-legend LABEL1 LABEL2 ...]

Available models:
    - simplecnn, bigcnn
    - vgg1d, vgg1d_compact
    - unet1d, unet1d_light
    - resnet1d, resunet1d
    - tcn
    - transformer1d, cnn_transformer1d
    - efficientnet1d
    - inception1d, inception1d_se_augmented, inception1d_se_mixup_focal_attention_residual

Parameters:
    --model: Model architecture to train (required)
    --epochs: Number of training epochs (default: from config)
    --days-in-series: Number of days in time series (default: from config)
    --days-per-bucket: Days per bucket (default: from config)
    --es-patience: Early stopping patience (default: from config)
    --num-features: Number of features (default: from config)
    --label-legend: Label legend (default: from config)
"""

import sys
import os
import argparse
import tensorflow as tf
import datetime

# Add project root to path
project_root = os.path.abspath(".")
sys.path.append(project_root)

from models.utils import count_labels, evaluate_class_balance, get_model_function

def main():
    parser = argparse.ArgumentParser(description='Train crop classification model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['simplecnn', 'bigcnn', 'vgg1d', 'vgg1d_compact', 'unet1d', 'unet1d_light',
                               'resnet1d', 'resunet1d', 'tcn', 'transformer1d', 'cnn_transformer1d',
                               'efficientnet1d', 'inception1d', 'inception1d_se_augmented', 
                               'inception1d_se_mixup_focal_attention_residual'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: from config)')
    parser.add_argument('--days-in-series', type=int, default=None,
                       help='Number of days in time series (default: from config)')
    parser.add_argument('--days-per-bucket', type=int, default=None,
                       help='Days per bucket (default: from config)')
    parser.add_argument('--es-patience', type=int, default=None,
                       help='Early stopping patience (default: from config)')
    parser.add_argument('--num-features', type=int, default=None,
                       help='Number of features (default: from config)')
    parser.add_argument('--label-legend', nargs='+', default=[],
                       help='Label legend (default: from config)')
    
    args = parser.parse_args()
    
    # Get parameters from command line arguments (set by bash script)
    MODEL_NAME = args.model
    MAX_EPOCHS = args.epochs
    ES_PATIENCE = args.es_patience
    DAYS_IN_SERIES = args.days_in_series
    DAYS_PER_BUCKET = args.days_per_bucket
    NUM_FEATURES = args.num_features
    MAX_IMAGES_PER_SERIES = (DAYS_IN_SERIES // DAYS_PER_BUCKET) + 1
    
    # Get label legend from command line arguments
    label_legend = args.label_legend
    
    # Parse lists if passed as a single string with '|' or ','
    if len(label_legend) == 1:
        if '|' in label_legend[0]:
            label_legend = label_legend[0].split('|')
        elif ',' in label_legend[0]:
            label_legend = label_legend[0].split(',')
    
    print(f"🚀 Starting training for model: {MODEL_NAME}")
    print(f"📊 Parameters:")
    print(f"   - Max epochs: {MAX_EPOCHS}")
    print(f"   - Early stopping patience: {ES_PATIENCE}")
    print(f"   - Days in series: {DAYS_IN_SERIES}")
    print(f"   - Days per bucket: {DAYS_PER_BUCKET}")
    print(f"   - Max images per series: {MAX_IMAGES_PER_SERIES}")
    print(f"   - Number of features: {NUM_FEATURES} ")
    print(f"   - Number of classes: {len(label_legend)}")
    
    # Load datasets
    print("📂 Loading datasets...")
    try:
        train_ds_path = f"./data/train_ds_{NUM_FEATURES}feat"
        val_ds_path = f"./data/val_ds_{NUM_FEATURES}feat"
        train_ds = tf.data.Dataset.load(train_ds_path)
        val_ds = tf.data.Dataset.load(val_ds_path)
        print("✅ Datasets loaded successfully")
        print(f"   - Training dataset: {train_ds_path}")
        print(f"   - Validation dataset: {val_ds_path}")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        print("💡 Make sure to run 'python process.py' first to prepare the datasets")
        sys.exit(1)
    
    # Assess data quality
    print("📊 Assessing data quality...")
    train_label_counts = count_labels(train_ds)
    val_label_counts = count_labels(val_ds)
    
    print("Train Label Distribution:")
    for idx, label in enumerate(label_legend):
        print(f"  {label}: {train_label_counts.get(idx, 0)}")
    
    print("\nValidation Label Distribution:")
    for idx, label in enumerate(label_legend):
        print(f"  {label}: {val_label_counts.get(idx, 0)}")
    
    # Evaluate class balance
    print("\n🔍 Evaluating class balance...")
    evaluate_class_balance(val_label_counts, label_legend, threshold=0.1, plot=False)
    
    # Get model function
    model_function = get_model_function(MODEL_NAME)
    
    # Create output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_dir = f'./results/models'
    log_dir = f"./results/logs/fit_{MODEL_NAME}/{timestamp}"
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📁 Output directories:")
    print(f"   - Model: {model_save_dir}")
    print(f"   - Logs: {log_dir}")
    
    # Train model
    print(f"🎯 Starting training for {MODEL_NAME}...")
    try:
        # Prepare arguments for the model function
        model_args = {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'label_legend': label_legend,
            'max_images_per_series': MAX_IMAGES_PER_SERIES,
            'num_features': NUM_FEATURES,
            'xaxis_callback': 'epoch',
            'max_epochs': MAX_EPOCHS,
            'es_patience': ES_PATIENCE,
            'model_save_dir': model_save_dir,
            'model_name': f'{MODEL_NAME}_{DAYS_IN_SERIES}days',
            'log_dir': log_dir
        }
        
        # Add special parameters for specific models
        if MODEL_NAME == 'resunet1d':
            model_args['use_time_channel'] = True
        
        elif MODEL_NAME == 'inception1d_se_mixup_focal_attention_residual':
            model_args['apply_mixup'] = True
        


        
        # Call the model function
        model, history = model_function(**model_args)
        
        print("✅ Training completed successfully")
        
        # Print training summary
        print("\n📊 Training Summary:")
        print(f"   - Model: {MODEL_NAME}")
        print(f"   - Model saved to: {model_save_dir}")
        print(f"   - Logs saved to: {log_dir}")
        print(f"   - Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 