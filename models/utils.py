import os
# Set matplotlib to use non-interactive backend to prevent crashes
import matplotlib
# Force Agg backend unless we're explicitly in a working Jupyter notebook
try:
    from IPython import get_ipython
    ipython_instance = get_ipython()
    if ipython_instance is not None and hasattr(ipython_instance, 'kernel'):
        pass  # In real notebook, use default backend
    else:
        try:
            matplotlib.use('Agg', force=True)
        except TypeError:
            # Older matplotlib versions don't support force parameter
            matplotlib.use('Agg')
except (ImportError, NameError, AttributeError):
    try:
        matplotlib.use('Agg', force=True)
    except TypeError:
        # Older matplotlib versions don't support force parameter
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tensorflow as tf

def count_labels(dataset):
    counts = Counter()
    for _, y in dataset:
        labels = np.argmax(y.numpy(), axis=1)  # Assuming one-hot encoded labels
        counts.update(labels)
    return counts

def evaluate_class_balance(counts, label_legend, threshold=0.1, plot=True):
    total = sum(counts.values())
    imbalance_detected = False

    print(f"{'Class':<20}{'Count':<10}{'Percentage':<10}")
    print("-" * 40)
    for idx, label in enumerate(label_legend):
        count = counts.get(idx, 0)
        percentage = count / total if total else 0
        flag = "⚠️" if percentage < threshold else ""
        if percentage < threshold:
            imbalance_detected = True
        print(f"{label:<20}{count:<10}{percentage*100:>6.2f}% {flag}")

    if imbalance_detected:
        print("\n⚠️  Class imbalance detected based on the threshold.\n")
    else:
        print("\n✅  No significant imbalance detected based on the threshold.\n")

    if plot:
        values = [counts.get(idx, 0) for idx in range(len(label_legend))]
        plt.figure(figsize=(8, 4))
        plt.bar(label_legend, values, color='skyblue')
        plt.axhline(total * threshold, color='red', linestyle='--', label=f"Threshold ({threshold*100:.0f}%)")
        plt.xticks(rotation=45)
        plt.title("Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Samples")
        plt.legend()
        plt.tight_layout()
        # Only show in notebooks, otherwise save or skip
        try:
            from IPython import get_ipython
            ipython_instance = get_ipython()
            if ipython_instance is not None and hasattr(ipython_instance, 'kernel'):
                plt.show()
            else:
                # Not in notebook - save instead
                os.makedirs('results/debug', exist_ok=True)
                plt.savefig('results/debug/class_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()
        except (ImportError, NameError, AttributeError):
            # Not in notebook - save instead
            os.makedirs('results/debug', exist_ok=True)
            plt.savefig('results/debug/class_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()

def get_model_function(model_name):
    """
    Get the appropriate model training function based on model name.
    
    Parameters:
    -----------
    model_name : str
        Name of the model architecture
        
    Returns:
    --------
    function
        Model training function
    """
    # Import model modules
    from models.cnn import baseline_simplecnn, baseline_bigcnn
    from models.vgg import baseline_vgg1d, baseline_vgg1d_compact
    from models.unet import baseline_unet1d, baseline_unet1d_light
    from models.resnet import baseline_resnet1d, baseline_resunet1d
    from models.tcn import baseline_tcn
    from models.transformer import baseline_transformer1d, baseline_cnn_transformer1d
    from models.efficientnet import baseline_efficientnet1d
    from models.inception import baseline_inception1d, baseline_inception1d_se_augmented, baseline_inception1d_se_mixup_focal_attention_residual
    
    # Model mapping
    model_functions = {
        # CNN models
        'simplecnn': baseline_simplecnn,
        'bigcnn': baseline_bigcnn,
        
        # VGG models
        'vgg1d': baseline_vgg1d,
        'vgg1d_compact': baseline_vgg1d_compact,
        
        # U-Net models
        'unet1d': baseline_unet1d,
        'unet1d_light': baseline_unet1d_light,
        
        # ResNet models
        'resnet1d': baseline_resnet1d,
        'resunet1d': baseline_resunet1d,
        
        # TCN model
        'tcn': baseline_tcn,
        
        # Transformer models
        'transformer1d': baseline_transformer1d,
        'cnn_transformer1d': baseline_cnn_transformer1d,
        
        # EfficientNet model
        'efficientnet1d': baseline_efficientnet1d,
        
        # Inception models
        'inception1d': baseline_inception1d,
        'inception1d_se_augmented': baseline_inception1d_se_augmented,
        'inception1d_se_mixup_focal_attention_residual': baseline_inception1d_se_mixup_focal_attention_residual
    }
    
    if model_name not in model_functions:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_functions.keys())}")
    
    return model_functions[model_name]

def load_config():
    """Load configuration from dataloader.txt"""
    config = {}
    current_section = None
    
    # Try different possible paths for the config file
    possible_paths = [
        'config/dataloader.txt',  # From project root
        'dataloader.txt',         # From config directory
        '../config/dataloader.txt'  # From subdirectories
    ]
    
    config_file = None
    for path in possible_paths:
        if os.path.exists(path):
            config_file = path
            break
    
    if config_file is None:
        raise FileNotFoundError(f"Could not find dataloader.txt in any of these locations: {possible_paths}")
    
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                config[current_section] = []
            elif line and current_section:
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[current_section].append((key.strip(), value.strip()))
                else:
                    config[current_section].append(line)
    
    # Extract paths
    train_path = None
    val_path = None
    for key, value in config.get('paths', []):
        if key == 'train_path':
            train_path = value
        elif key == 'val_path':
            val_path = value
    
    # Extract lists
    targeted_crops = config.get('targeted_crops', [])
    other_crops = config.get('other_crops', [])
    label_legend = config.get('label_legend', [])
    
    return {
        'train_path': train_path,
        'val_path': val_path,
        'targeted_crops': targeted_crops,
        'other_crops': other_crops,
        'label_legend': label_legend
    }
