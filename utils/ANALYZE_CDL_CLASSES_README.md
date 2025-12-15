# CDL Class Analysis Tool

## Problem

When training models with data from one region (e.g., Mississippi Delta) and then testing with data from another region (e.g., California), CDL classes may not match. This results in warnings like:

```
⚠️  Warning: 30 unmapped CDL classes found: ['Tomatoes' 'Barren' 'Dry Beans' ...]
```

## Solution

The `analyze_cdl_classes.py` script analyzes your parquet files to discover which CDL classes are present in a specific region, helping you build the appropriate `dataloader.txt`.

## Basic Usage

### 1. Analyze data from a region

```bash
# Analyze test data (e.g., California)
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet"

# Analyze training data
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/valtrain/CDL_multiple_scene_ts.parquet/*/*2020*/*.parquet"
```

### 2. Filter infrequent classes

If you want to see only classes with at least N occurrences:

```bash
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet" \
    --min-count 100
```

### 3. Generate suggested configuration file

To automatically generate a suggested `dataloader.txt`:

```bash
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet" \
    --output-config "config/dataloader_california.txt"
```

Then review and adjust the generated file according to your needs.

## Interpreting Results

The script shows:

1. **Crop classes sorted by frequency**: Most common classes appear first
2. **Non-crop classes**: For reference (e.g., "Open Water", "Developed/Open Space")
3. **Suggestions for dataloader.txt**:
   - `[targeted_crops]`: Main crops that will be classified individually
   - `[other_crops]`: Other crops that will be grouped as "Cultivated"
   - `[label_legend]`: Order of classes for the model

## Example Workflow

### Step 1: Download California data

```bash
# Configure bbox_config.txt with California bboxes
bash build_training_data.sh multiple all
```

### Step 2: Analyze present CDL classes

```bash
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/valtrain/CDL_multiple_scene_ts.parquet/*/*2020*/*.parquet" \
    --output-config "config/dataloader_california.txt"
```

### Step 3: Review and adjust configuration

Edit `config/dataloader_california.txt`:
- Adjust paths in `[paths]` if necessary
- Review `[targeted_crops]` - these are the main crops
- Review `[other_crops]` - these are grouped as "Cultivated"
- Ensure `[label_legend]` has the correct order

### Step 4: Train with new configuration

```bash
# Copy configuration as dataloader.txt temporarily
cp config/dataloader_california.txt config/dataloader.txt

# Train
bash cropclassifier.sh -action train
```

## Important Notes

- **targeted_crops**: Maximum ~6 main crops. These will be classified individually by the model.
- **other_crops**: You can have many (20-30). All are grouped into a single "Cultivated" category.
- **Unlisted classes**: Will be treated as "Uncultivated" (class 0).
- **Order in label_legend**: Must match the order of classes in the model:
  1. Uncultivated (0)
  2. Cultivated (1)
  3. No Crop Growing (2)
  4. targeted_crops[0] (3)
  5. targeted_crops[1] (4)
  6. ...

## Example Output

```
🔍 Analyzing CDL classes in: data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet
================================================================================
📁 Found 15 parquet files
📖 Reading data...
✅ Processed 45,230 total rows

📊 CDL Class Statistics
================================================================================
Total samples: 45,230
Unique classes found: 42

🌾 CROP CLASSES (sorted by frequency):
--------------------------------------------------------------------------------
  1.    Almonds                                   12,450 (27.52%)
  2.    Tomatoes                                   8,230 (18.19%)
  3.    Grapes                                     5,120 (11.32%)
  4.    Cotton                                     3,890 (8.60%)
  ...

💡 SUGGESTIONS FOR dataloader.txt:
================================================================================

[targeted_crops]
# Most important classes (main crops in the region)
Almonds
Tomatoes
Grapes
Cotton
...

[other_crops]
# Other cultivated crops (less frequent or secondary)
Walnuts  # 1,230 samples (2.72%)
Peaches  # 890 samples (1.97%)
...
```
