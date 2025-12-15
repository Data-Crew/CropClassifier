#!/usr/bin/env python3
"""
analyze_cdl_classes.py

Analyzes parquet files to discover which CDL classes are present in a region.
This helps build the appropriate dataloader.txt for training region-specific models.

Usage:
    python utils/analyze_cdl_classes.py --data-path "data/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet"
    python utils/analyze_cdl_classes.py --data-path "data/valtrain/CDL_multiple_scene_ts.parquet/*/*2020*/*.parquet" --min-count 100
"""

import argparse
import pandas as pd
import glob
from collections import Counter
from pathlib import Path
import sys

# List of classes that are generally NOT crops (for filtering)
NON_CROP_CLASSES = {
    'Background', 'Open Water', 'Developed/Open Space', 'Developed/Low Intensity',
    'Developed/Med Intensity', 'Developed/High Intensity', 'Barren',
    'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Shrubland',
    'Grass/Pasture', 'Woody Wetlands', 'Herbaceous Wetlands', 'Aquaculture',
    'Fallow/Idle Cropland', 'Clouds/No Data', 'Nonag/Undefined'
}

# Classes that are crops but generally less important (for "other_crops")
COMMON_OTHER_CROPS = {
    'Other Hay/Non Alfalfa', 'Alfalfa', 'Other Crops', 'Clover/Wildflowers',
    'Sod/Grass Seed', 'Herbs', 'Misc Vegs & Fruits'
}

def normalize_cdl_label(label):
    """Normalizes CDL label to string."""
    if isinstance(label, bytes):
        return label.decode('utf-8')
    elif pd.isna(label):
        return None
    else:
        return str(label)

def analyze_cdl_classes(data_path_pattern, min_count=0, top_n=50):
    """
    Analyzes parquet files to find all CDL classes present.
    
    Parameters:
    -----------
    data_path_pattern : str
        Glob pattern for parquet files (e.g., "data/test/*.parquet")
    min_count : int
        Minimum number of occurrences to show a class
    top_n : int
        Maximum number of classes to show in summary
    
    Returns:
    --------
    dict with statistics and suggestions
    """
    print(f"🔍 Analyzing CDL classes in: {data_path_pattern}")
    print("=" * 80)
    
    # Find parquet files
    parquet_files = glob.glob(data_path_pattern)
    
    if not parquet_files:
        print(f"❌ Error: No parquet files found with pattern: {data_path_pattern}")
        return None
    
    print(f"📁 Found {len(parquet_files)} parquet files")
    print(f"📖 Reading data...")
    
    # Read files and extract CDL classes
    all_cdl_classes = []
    total_rows = 0
    
    for i, file_path in enumerate(parquet_files):
        if (i + 1) % 10 == 0:
            print(f"   Processing file {i+1}/{len(parquet_files)}...")
        
        try:
            # Read only CDL column for efficiency
            df = pd.read_parquet(file_path, columns=['CDL'])
            total_rows += len(df)
            
            # Normalize and extract classes
            cdl_values = df['CDL'].apply(normalize_cdl_label).dropna()
            all_cdl_classes.extend(cdl_values.tolist())
            
        except Exception as e:
            print(f"⚠️  Warning: Error reading {file_path}: {e}")
            continue
    
    if not all_cdl_classes:
        print("❌ Error: No CDL classes found in files")
        return None
    
    print(f"✅ Processed {total_rows:,} total rows")
    print()
    
    # Count frequency of each class
    cdl_counter = Counter(all_cdl_classes)
    total_samples = len(all_cdl_classes)
    
    print(f"📊 CDL Class Statistics")
    print("=" * 80)
    print(f"Total samples: {total_samples:,}")
    print(f"Unique classes found: {len(cdl_counter)}")
    print()
    
    # Separate into categories
    crop_classes = {}
    non_crop_classes = {}
    
    for class_name, count in cdl_counter.items():
        if class_name in NON_CROP_CLASSES or any(nc in class_name for nc in NON_CROP_CLASSES):
            non_crop_classes[class_name] = count
        else:
            crop_classes[class_name] = count
    
    # Show crop classes sorted by frequency
    print("🌾 CROP CLASSES (sorted by frequency):")
    print("-" * 80)
    
    sorted_crops = sorted(crop_classes.items(), key=lambda x: x[1], reverse=True)
    
    for i, (class_name, count) in enumerate(sorted_crops[:top_n], 1):
        percentage = (count / total_samples) * 100
        if count >= min_count:
            marker = "⭐" if class_name in COMMON_OTHER_CROPS else "  "
            print(f"{i:3d}. {marker} {class_name:40s} {count:8,} ({percentage:5.2f}%)")
    
    if len(sorted_crops) > top_n:
        print(f"\n   ... and {len(sorted_crops) - top_n} more classes")
    
    print()
    
    # Show non-crop classes (if any)
    if non_crop_classes:
        print("🏞️  NON-CROP CLASSES (for reference):")
        print("-" * 80)
        sorted_non_crops = sorted(non_crop_classes.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_non_crops[:20]:
            percentage = (count / total_samples) * 100
            print(f"     {class_name:40s} {count:8,} ({percentage:5.2f}%)")
        print()
    
    # Generate suggestions for dataloader.txt
    print("💡 SUGGESTIONS FOR dataloader.txt:")
    print("=" * 80)
    
    # Most frequent classes that should probably be "targeted_crops"
    top_crops = [name for name, count in sorted_crops[:10] if count >= min_count]
    
    # Filter common classes that generally go in "other_crops"
    suggested_targeted = [c for c in top_crops if c not in COMMON_OTHER_CROPS]
    suggested_other = [c for c in top_crops if c in COMMON_OTHER_CROPS]
    
    # Add other frequent classes to "other_crops"
    for name, count in sorted_crops:
        if name not in suggested_targeted and name not in suggested_other and count >= min_count:
            if len(suggested_other) < 20:  # Limit to 20 classes
                suggested_other.append(name)
    
    print("\n[targeted_crops]")
    print("# Most important classes (main crops in the region)")
    for crop in suggested_targeted[:6]:  # Maximum 6 main crops
        print(crop)
    
    print("\n[other_crops]")
    print("# Other cultivated crops (less frequent or secondary)")
    for crop in suggested_other[:25]:  # Maximum 25 other crops
        count = crop_classes.get(crop, 0)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{crop}  # {count:,} samples ({percentage:.2f}%)")
    
    print("\n[label_legend]")
    print("Uncultivated")
    print("Cultivated")
    print("No Crop Growing")
    for crop in suggested_targeted[:6]:
        print(crop)
    
    print()
    print("=" * 80)
    print("✅ Analysis complete!")
    print()
    print("📝 Notes:")
    print("   - Classes in 'targeted_crops' will be classified individually")
    print("   - Classes in 'other_crops' will be grouped into a single 'Cultivated' category")
    print("   - Adjust lists according to your specific needs")
    print("   - Unlisted classes will be treated as 'Uncultivated'")
    
    return {
        'total_samples': total_samples,
        'unique_classes': len(cdl_counter),
        'crop_classes': crop_classes,
        'non_crop_classes': non_crop_classes,
        'suggested_targeted': suggested_targeted[:6],
        'suggested_other': suggested_other[:25],
        'all_crops_sorted': sorted_crops
    }

def save_suggested_config(results, output_path, data_path_pattern):
    """
    Saves a suggested dataloader.txt file based on the analysis.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_cdl_classes
    output_path : str
        Path where to save the configuration file
    data_path_pattern : str
        Path pattern used for analysis (for documentation)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract year from path pattern if possible
    import re
    year_match = re.search(r'(\d{4})', data_path_pattern)
    suggested_year = year_match.group(1) if year_match else "YEAR"
    
    with open(output_file, 'w') as f:
        f.write("# Suggested configuration generated by analyze_cdl_classes.py\n")
        f.write(f"# Based on analysis of: {data_path_pattern}\n")
        f.write(f"# Total samples analyzed: {results['total_samples']:,}\n")
        f.write(f"# Unique classes found: {results['unique_classes']}\n")
        f.write("#\n")
        f.write("# ⚠️  IMPORTANT: Review and adjust these lists according to your needs\n")
        f.write("#     - Classes in 'targeted_crops' will be classified individually\n")
        f.write("#     - Classes in 'other_crops' will be grouped as 'Cultivated'\n")
        f.write("#     - Make sure to also update paths in [paths]\n")
        f.write("\n")
        
        f.write("[paths]\n")
        f.write(f"train_path=data/valtrain/CDL_multiple_scene_ts.parquet/*/*{suggested_year}*/*.parquet\n")
        f.write(f"val_path=data/valtrain/CDL_multiple_scene_ts.parquet/*/*{suggested_year}*/*.parquet\n")
        f.write(f"test_path=data/test/CDL_unique_scene_ts.parquet/*/*{suggested_year}*/*.parquet\n")
        f.write("\n")
        
        f.write("[hyperparams]\n")
        f.write("model_name=simplecnn\n")
        f.write("batch_size=1028\n")
        f.write("days_in_series=120\n")
        f.write("days_per_bucket=5\n")
        f.write("frames_to_check=2\n")
        f.write("num_features=16\n")
        f.write("max_epochs=60\n")
        f.write("es_patience=15\n")
        f.write("bucketing_strategy=random\n")
        f.write(f"test_year={suggested_year}\n")
        f.write("\n")
        
        f.write("[targeted_crops]\n")
        f.write("# Main crops in the region (classified individually)\n")
        for crop in results['suggested_targeted']:
            count = results['crop_classes'].get(crop, 0)
            percentage = (count / results['total_samples']) * 100 if results['total_samples'] > 0 else 0
            f.write(f"{crop}  # {count:,} samples ({percentage:.2f}%)\n")
        f.write("\n")
        
        f.write("[other_crops]\n")
        f.write("# Other cultivated crops (grouped as 'Cultivated')\n")
        for crop in results['suggested_other']:
            count = results['crop_classes'].get(crop, 0)
            percentage = (count / results['total_samples']) * 100 if results['total_samples'] > 0 else 0
            f.write(f"{crop}  # {count:,} samples ({percentage:.2f}%)\n")
        f.write("\n")
        
        f.write("[label_legend]\n")
        f.write("Uncultivated\n")
        f.write("Cultivated\n")
        f.write("No Crop Growing\n")
        for crop in results['suggested_targeted']:
            f.write(f"{crop}\n")
    
    print(f"\n💾 Suggested configuration saved to: {output_path}")
    print(f"   Review and adjust the file before using it for training.")

def main():
    parser = argparse.ArgumentParser(
        description='Analyzes CDL classes in parquet files to build dataloader.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze test data
  python utils/analyze_cdl_classes.py --data-path "data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet"
  
  # Analyze training data with minimum filter
  python utils/analyze_cdl_classes.py --data-path "data/demodata/valtrain/CDL_multiple_scene_ts.parquet/*/*2020*/*.parquet" --min-count 100
  
  # Analyze only top 30 crops
  python utils/analyze_cdl_classes.py --data-path "data/demodata/test/*.parquet" --top-n 30
  
  # Generate suggested config file
  python utils/analyze_cdl_classes.py --data-path "data/demodata/test/*.parquet" --output-config "config/dataloader_california.txt"
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Glob pattern for parquet files (e.g., "data/test/*.parquet")'
    )
    
    parser.add_argument(
        '--min-count',
        type=int,
        default=0,
        help='Minimum number of occurrences to show a class (default: 0)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Maximum number of classes to show in summary (default: 50)'
    )
    
    parser.add_argument(
        '--output-config',
        type=str,
        default=None,
        help='Optional path to save a suggested dataloader.txt file (e.g., "config/dataloader_california.txt")'
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_cdl_classes(args.data_path, args.min_count, args.top_n)
        if results is None:
            sys.exit(1)
        
        # If requested, save suggested configuration
        if args.output_config:
            save_suggested_config(results, args.output_config, args.data_path)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
