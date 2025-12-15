# Crop Rotation Cycle Predictor

A deep learning tool for predicting crop rotation cycles using Sentinel-2 satellite imagery and USDA Cropland Data Layer (CDL) labels. This system analyzes temporal patterns in multispectral satellite data to classify crop types and predict rotation sequences across agricultural regions in the United States.

## Visual Predictions

The following animations show predicted crop rotation cycles for 2024 across four different agricultural zones:

<div align="center">

| Zone 1 | Zone 2 |
|:------:|:------:|
| ![Zone 1 Predictions](doc/crop_rotation_cycles/animation_predictions_only_prediction_zn1.gif) | ![Zone 2 Predictions](doc/crop_rotation_cycles/animation_predictions_only_prediction_zn2.gif) |
| **Zone 3** | **Zone 5** |
| ![Zone 3 Predictions](doc/crop_rotation_cycles/animation_predictions_only_prediction_zn3.gif) | ![Zone 5 Predictions](doc/crop_rotation_cycles/animation_predictions_only_prediction_zn5.gif) |

</div>

---

## Overview

This tool combines:
- **Sentinel-2 satellite imagery**: Multispectral temporal data capturing crop growth patterns
- **USDA CDL labels**: Ground truth crop classifications for training
- **Deep learning models**: Time series classifiers that learn temporal signatures of different crops
- **Rotation cycle prediction**: Spatial-temporal analysis to predict crop sequences across agricultural zones

---

# Initial Setup

This project provides two options for setting up your development environment:
1. **Local Virtual Environment** (recommended for direct development)
2. **Docker Environment** (recommended for reproducibility and isolation)

Choose the option that best fits your workflow. For detailed instructions, troubleshooting, and GPU configuration, see [Environment Setup Guide](doc/environment_setup.md).

---

## Option 1: Local Virtual Environment

**Prerequisites:** Python 3.10, NVIDIA drivers, CUDA 12.2

```bash
# 1. Create and activate environment
virtualenv venv --python=python3.10
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Verify GPU
python config/gpu/check_gpu.py

# 4. Ready! Run your first command
bash cropclassifier.sh -action test -model simplecnn
```

---

## Option 2: Docker Environment

**Prerequisites:** Docker, Docker Compose, nvidia-container-toolkit

```bash
# 1. Verify GPU support
./docker-run.sh check

# 2. Build image (only first time, ~15 min)
./docker-run.sh build

# 3. Start container
./docker-run.sh start

# 4. Verify GPU
./docker-run.sh exec 'python config/gpu/check_gpu.py'

# 5. Ready! Run your first command
./docker-run.sh exec 'bash cropclassifier.sh -action test -model simplecnn'
```

### Running Bash Scripts

```bash
# Option A: Open interactive shell
./docker-run.sh shell
# Then run commands inside:
bash build_training_data.sh multiple all
bash cropclassifier.sh -action train -model inception1d

# Option B: Execute commands directly
./docker-run.sh exec 'bash build_training_data.sh multiple all'
./docker-run.sh exec 'bash cropclassifier.sh -action train -model inception1d'
```

### Starting JupyterLab

```bash
./docker-run.sh jupyter
# Access at http://localhost:8888
```

### Docker Helper Commands

```bash
./docker-run.sh check      # Verify GPU support
./docker-run.sh build      # Build Docker image
./docker-run.sh start      # Start main container
./docker-run.sh shell      # Open shell in container
./docker-run.sh exec 'cmd' # Execute command
./docker-run.sh jupyter    # Start JupyterLab
./docker-run.sh logs       # Show container logs
./docker-run.sh stop       # Stop all containers
```

### Stopping and Cleaning Up

```bash
# Stop all running containers
docker compose stop

# Stop and remove containers
docker compose down

# Remove everything (containers, images, volumes)
docker compose down --volumes --rmi all
```

---

## Downloading Training Data

The `data/` directory structure is included in this repository, but the actual data files are not tracked by Git due to their large size. You need to download the preprocessed training data separately.

### Data Structure

The project expects the following directory structure:
```
data/
├── test/          # Test dataset (CDL and Sentinel-2 time series)
└── valtrain/      # Training and validation datasets (CDL and Sentinel-2 time series)
```

### Download Preprocessed Data

Preprocessed training data (CDL samples and Sentinel-2 time series) is available for download:

**📥 [Download Data from Google Drive](https://drive.google.com/drive/folders/19oRelb1hS0zHS49-m0DCU4KITtuak26y?usp=drive_link)**

After downloading, extract the data into the corresponding directories:
- Place test data in `data/test/`
- Place training/validation data in `data/valtrain/`

### Alternative: Build Your Own Training Data

If you prefer to build the training data from scratch, follow the instructions in the [Crop Classifier Configuration](#crop-classifier-configuration) section below. This involves:
1. Downloading CDL raster files from the USDA CDL data portal
2. Querying Sentinel-2 STAC API for satellite imagery
3. Processing and transforming the data into time series format

---

# Crop Classifier Configuration

Before training models, you need to configure which CDL (Cropland Data Layer) areas and years will be used for training/validation and testing. This configuration is done by selecting bounding boxes (bboxes) from CDL data files.

## Overview

The classifier uses a configuration file (`config/bbox_config.txt`) that specifies:
- **Scene identifiers**: Names for each area of interest
- **Bounding boxes**: Geographic coordinates defining the area boundaries
- **Years**: Which years of data to process for each area

These configurations are split into:
- **Training/Validation sets**: Used to train and validate models
- **Test sets**: Used for final model evaluation

## Using the Configuration Notebook

The notebook `notebooks/0.get_bbox_from_CDL.ipynb` provides a graphical interface to extract bounding boxes from CDL `.zip` files downloaded from the USDA CDL data portal.

### Workflow

1. **Download CDL files**: Download `.zip` files from the USDA CDL data portal for your areas of interest
2. **Process with notebook**: Use the notebook to extract bounding box coordinates
3. **Copy configuration**: Copy the generated configuration lines into `config/bbox_config.txt`

### Configuration Modes

#### Single Scene Mode

For processing one area with one year:

```python
# Process a single CDL zip file
aoi_path = "../data/test/CDL_1675300360.zip"
# ... processing code ...
# Output: example_unique_scene|484932,1401912,489035,1405125|2024
```

**Format**: `scene_name|bbox_coordinates|year`

#### Multiple Scenes Mode

For processing multiple areas with multiple years:

```python
# Process multiple CDL zip files
zip_files = glob.glob("../data/valtrain/*.zip")
years = [2020, 2021, 2022, 2023]
# ... processing code ...
# Output: Multiple configuration lines
```

**Format**: `scene_name|bbox_coordinates|year1,year2,year3,...`

### Configuration File Format

The `config/bbox_config.txt` file uses a simple pipe-delimited format:

```
example_multiple_scene01|426362,1405686,520508,1432630|2019,2020,2021
example_multiple_scene02|390747,1195097,437820,1284288|2019,2020,2021
example_unique_scene|484932,1401912,489035,1405125|2024
```

Each line represents:
- **Scene identifier**: A unique name for the area
- **Bounding box**: `minx,miny,maxx,maxy` coordinates in EPSG:5070 projection
- **Years**: Comma-separated list of years to process

### Execution Workflow

Once your `bbox_config.txt` is configured, follow these steps in order:

#### Step 1: Download Training Data

Use `build_training_data.sh` to download CDL and Sentinel-2 data. The execution mode depends on your configuration:

**Unique Mode** (single bbox/year):
```bash
# Run all steps: CDL download → Sentinel-2 download → Transform to time series
bash build_training_data.sh unique all

# Or run steps individually:
bash build_training_data.sh unique cdl        # Step 1: Download CDL data
bash build_training_data.sh unique sentinel  # Step 2: Download Sentinel-2 scenes
bash build_training_data.sh unique transform # Step 3: Transform to time series
```

**Multiple Mode** (multiple bboxes/years):
```bash
# Run all steps for all configured scenes
bash build_training_data.sh multiple all

# Or run steps individually:
bash build_training_data.sh multiple cdl
bash build_training_data.sh multiple sentinel
bash build_training_data.sh multiple transform
```

**Output**: This generates Parquet files containing time series data:
- Unique mode: `data/test/CDL_unique_scene_ts.parquet/`
- Multiple mode: `data/valtrain/CDL_multiple_scene_ts.parquet/`

#### Step 2: Configure Training Parameters

After downloading data, configure `config/dataloader.txt` with all parameters needed for training. This file contains four main sections:

##### 2.1 Data Paths (`[paths]`)

Configure paths to your downloaded Parquet files. The `[paths]` section specifies where to find training, validation, and test datasets.

**Understanding the data structure**:

- **Unique mode** (`build_training_data.sh unique`): Downloads **one bbox for one year**. Typically used for **test data**.
  - Output: `data/test/CDL_unique_scene_ts.parquet/`

- **Multiple mode** (`build_training_data.sh multiple`): Downloads **multiple bboxes for one or more years**. Used for **training and validation data**.
  - Output: `data/valtrain/CDL_multiple_scene_ts.parquet/`
  - When you configure multiple years per bbox (e.g., `2020,2022`), all data goes into the same Parquet file
  - You then **separate by year** in `dataloader.txt` to create train/val splits

**Configuration example**:

If you downloaded:
- **Test data** using `unique mode` (e.g., year 2019)
- **Train/Val data** using `multiple mode` with multiple bboxes and years (e.g., 2020, 2021)

Configure `dataloader.txt` like this:
```ini
[paths]
train_path=data/valtrain/s2_multiple_scene.parquet/*/*2021*/*.parquet
val_path=data/valtrain/s2_multiple_scene.parquet/*/*2020*/*.parquet
test_path=data/test/s2_unique_scene.parquet/*/*2019*/*.parquet
```

**Key points**:
- Both `train_path` and `val_path` point to the **same Parquet file** (`CDL_multiple_scene_ts.parquet`)
- They are separated by **year** using glob patterns (`*2021*` vs `*2020*`)
- `test_path` points to the **separate test dataset** from unique mode
- Paths use glob patterns: `*` matches any directory level, `*2021*` filters by year, `*.parquet` matches Parquet files

##### 2.2 Hyperparameters (`[hyperparams]`)

Configure model architecture and training parameters:

```ini
[hyperparams]
model_name=simplecnn                    # Model architecture (simplecnn, inception1d, resnet1d, etc.)
batch_size=1028                         # Batch size for training/testing
days_in_series=120                       # Time window length (days)
days_per_bucket=5                        # Temporal bucketing interval (days)
frames_to_check=2                        # Number of frames to sample per series
num_features=16                          # Number of input features (bands + indices)
max_epochs=60                            # Maximum training epochs
es_patience=15                           # Early stopping patience (epochs)
bucketing_strategy=random                # Sampling strategy: random, early_season, late_season, deterministic
test_year=2019                          # Year used for test set evaluation
```

**Key parameters**:
- **`model_name`**: Choose from available architectures (see Step 3 for full list)
- **`days_in_series`**: Length of temporal window to analyze (typically 120 days)
- **`days_per_bucket`**: Groups observations into time buckets (e.g., 5 days per bucket)
- **`frames_to_check`**: Number of temporal samples per series (affects dataset size and class balance)
- **`bucketing_strategy`**: How to sample temporal data (`random` recommended for balanced datasets)
- **`es_patience`**: Early stopping patience to prevent overfitting

##### 2.3 Crop Classification (`[targeted_crops]`, `[other_crops]`, `[label_legend]`)

Define which crops to classify and how to group them:

```ini
[targeted_crops]
Soybeans
Rice
Corn
Cotton

[other_crops]
Other Hay/Non Alfalfa
Pop or Orn Corn
Peanuts
Sorghum
# ... (list of other cultivated crops)

[label_legend]
Uncultivated
Cultivated
No Crop Growing
Soybeans
Rice
Corn
Cotton
```

**Understanding crop categories**:

- **`[targeted_crops]`**: Crops that receive **individual labels** (e.g., Soybeans=3, Rice=4, Corn=5, Cotton=6)
  - These are the main crops you want to distinguish and classify separately
  - Each crop gets its own class in the final model output

- **`[other_crops]`**: Other cultivated crops that are **grouped together** as "Cultivated" (label=1)
  - These crops are recognized as cultivated but not individually classified
  - Useful for crops that are less common or less important for your specific use case

- **`[label_legend]`**: Defines the **final label mapping** for model output
  - Order matters: `[Uncultivated, Cultivated, No Crop Growing, Soybeans, Rice, Corn, Cotton]`
  - Labels 0-2 are special classes (Uncultivated, Cultivated, No Crop Growing)
  - Labels 3+ correspond to targeted crops in order

**Label encoding logic**:
- Label 0: Uncultivated land
- Label 1: Other cultivated crops (from `[other_crops]`)
- Label 2: Ambiguous (cultivated but no vegetation detected in final frames)
- Label 3+: Targeted crops (position in `[targeted_crops]` + 3)

**🔧 Discovering CDL Classes for Your Region**:

When working with data from different regions, CDL classes may vary significantly. For example, California data might contain classes like "Almonds", "Tomatoes", and "Grapes" that aren't present in Mississippi Delta data.

Use the `analyze_cdl_classes.py` utility to automatically discover which CDL classes are present in your data:

```bash
# Analyze your data to discover present CDL classes
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet"

# Generate a suggested dataloader.txt configuration
python utils/analyze_cdl_classes.py \
    --data-path "data/demodata/test/CDL_unique_scene_ts.parquet/*/*2024*/*.parquet" \
    --output-config "config/dataloader_california.txt"
```

The script will:
- List all CDL classes found in your data, sorted by frequency
- Suggest which classes should go in `[targeted_crops]` vs `[other_crops]`
- Optionally generate a complete `dataloader.txt` file ready for review

See `utils/ANALYZE_CDL_CLASSES_README.md` for detailed usage instructions and examples.

Once `dataloader.txt` is fully configured, you're ready to train models.


#### Step 3: Train and Evaluate Models

Use `cropclassifier.sh` to process data, train models, and run evaluations. The script reads configuration from `config/dataloader.txt` but allows command-line overrides.

**Basic Usage**:
```bash
# Process training/validation datasets
bash cropclassifier.sh -action process

# Train a model
bash cropclassifier.sh -action train -model simplecnn

# Test a trained model
bash cropclassifier.sh -action test -model simplecnn

# Run complete pipeline: process → train → test
bash cropclassifier.sh -action "process train test" -model simplecnn
```

**Common Flags**:
- `-action`: Actions to execute (`process`, `train`, `test`, `predict`, or combinations)
- `-model`: Model architecture (e.g., `simplecnn`, `inception1d`, `resnet1d`)
- `-batch-size`: Batch size for training/testing
- `-epochs`: Number of training epochs
- `-days-in-series`: Time window length (default: 120)
- `-days-per-bucket`: Temporal bucketing interval (default: 3)
- `-frames-to-check`: Number of frames to sample per series (default: 2)
- `-bucketing-strategy`: Sampling strategy (`random`, `early_season`, `late_season`, `deterministic`)

**Prediction on New Data**:
```bash
bash cropclassifier.sh -action predict \
    -model simplecnn \
    -input-path "data/new_data.parquet/*/*2024*/*.parquet" \
    -output-path "results/my_predictions" \
    -save-probabilities \
    -pred-year 2024
```

**Available Models**: `simplecnn`, `bigcnn`, `vgg1d`, `resnet1d`, `inception1d`, `inception1d_se_mixup_focal_attention_residual`, and others (see script header for full list).

#### Summary

1. **Configure bounding boxes** → Edit `bbox_config.txt` with your areas of interest
2. **Download data** → Run `build_training_data.sh` in `unique` (test) or `multiple` (train/val) mode
3. **Configure training parameters** → Edit `config/dataloader.txt` with paths, hyperparameters, and crop classifications
4. **Train and evaluate** → Use `cropclassifier.sh` to process data, train models, and evaluate

The Parquet files generated by `build_training_data.sh` serve as input for the crop classifier training pipeline, configured via `dataloader.txt`.

---

See the [Crop Classification Workflow](#crop-classification-workflow) section for details on the complete pipeline.

---

# Crop Classification Workflow

This document outlines the structure and logic of the full Sentinel-2 crop classification pipeline. 
Each section represents a major step in the process.

---

## 1. CDL Data Preparation

- **Objective**: Download and process Cropland Data Layer (CDL) raster labels.

This step creates a table of "ground truth" crop labels by retrieving and clipping portions of the USDA Cropland Data Layer (CDL) raster dataset. The raster images are converted into a tabular format suitable for training, with each row representing a unique lat/lon coordinate and year.

CDL data is retrieved from the USDA NASS GeoSpatial Data Gateway via API or local fallback, then downsampled to reduce spatial density and written to a Parquet file.

---

#### 🔧 Function Highlight: `CDL_clip_retrieve()`

**Purpose:**
Downloads a raster image (GeoTIFF) of CDL data for a specified bounding box and year, either from the USDA API or local storage.

- Accepts a bounding box (`bbox`), year, and an optional fallback `local_path`.
- Returns GeoTIFF data as raw bytes.
- Used as the entry point for CDL sampling.

```python
cdl_bytes = CDL_clip_retrieve(bbox="484932, 1401912, 489035, 1405125", year=2019)
```

---

#### 🔧 Function Highlight: `sample_data_into_dataframe_write_parquet()`

**Purpose:**
Converts the raster content from CDL into a tabular PySpark DataFrame and stores the sampled data as a partitioned Parquet file.

- Calls internal helper `sample_raster_data()` to downsample pixel data.
- Converts sampled raster values into CDL labels with lat/lon positions.
- Appends data to a `data/*.parquet` file, partitioned by `bbox` and `year`.

```python
sample_data_into_dataframe_write_parquet(cdl_bytes, bbox="484932, 1401912, 489035, 1405125", year=2019)
```

**Expected Output Format:**
```text
+------------------+------------------+-----------------+----------------------------+------+
| CDL              | lon              | lat             | bbox                       | year |
+------------------+------------------+-----------------+----------------------------+------+
| Soybeans         | -90.5993         | 35.5765         | 484932, 1401912, ...       | 2019 |
| Woody Wetlands   | -90.5990         | 35.5765         | 484932, 1401912, ...       | 2019 |
| Corn             | -90.5987         | 35.5765         | 484932, 1401912, ...       | 2019 |
+------------------+------------------+-----------------+----------------------------+------+
```

---

#### 🔧 Function Highlight: `sample_raster_data()`

**Purpose:**
Performs spatial downsampling on the raster to reduce the number of pixels, improving performance and efficiency.

- Uses a sampling `interval` to keep 1 of every N pixels.
- Converts pixel positions to geographic coordinates (lat/lon).
- Returns sampled raster values and their geographic positions.

```python
sampled, lon, lat = sample_raster_data(tif_bytes, interval=3)
```

**Why Downsample?**
> Sampling reduces processing time and disk usage by avoiding the extraction of every pixel in high-resolution raster files.
> This is especially useful for building efficient training datasets in large AOIs.

---

### Summary of CDL Processing

```text
1. Define bounding box (bbox) and year
    ↓
2. Download GeoTIFF from API (or fallback to local)
    ↓
3. Subsample raster using spatial interval
    ↓
4. Convert raster values to labels and coordinates
    ↓
5. Store output as partitioned Parquet file
```

Each sample contains:
- `CDL` crop label
- `lon`, `lat` coordinates
- `bbox`, `year` as metadata

➡️ Output files are stored inside: `data/CDL_samples.parquet` (or a custom filename)


---

## 2. Sentinel-2 Scene Retrieval

- **Objective**: Identify and download Sentinel-2 scenes intersecting target areas.

This step retrieves multispectral scenes from Sentinel-2 that intersect the training zones defined in the CDL data. These scenes are the source of the temporal observations (features `X`) that feed the crop classification model.

The CDL dataset is partitioned by `bbox` and `year`, and each chunk is used to call the Sentinel-2 STAC API to retrieve a list of candidate scenes.

Each result contains:
- Scene metadata (ID, timestamp, tile)
- Cloud and vegetation coverage statistics
- URLs to download specific bands (e.g., red, NIR, SCL, etc.)

Example STAC scene result:
```text
{
  'id': 'S2A_33UXP_20200715_0_L2A',
  'properties': {
    'datetime': '2020-07-15T10:15:25Z',
    's2:cloud_coverage': 2.4,
    's2:vegetation_percentage': 36.2,
    's2:not_vegetated_percentage': 5.1,
    ...
  },
  'assets': {
    'red': {'href': 'https://sentinel/33UXP_20200715/red.tif'},
    'nir': {'href': 'https://sentinel/33UXP_20200715/nir.tif'},
    'scl': {'href': 'https://sentinel/33UXP_20200715/scl.tif'},
    'B01': {'href': '...'},
    'B02': {'href': '...'},
    ...
  }
}
```

These results are retrieved for each partition in the CDL data like so:
```python
for el in df_partition_list:
    bbox = el['bbox']  # e.g., '484932, 1401912, 489035, 1405125'
    year = el['year']  # e.g., '2019'
    CDL_parts_path = os.path.join(CDL_path, f"bbox={bbox}", f"year={year}")

    bbox_tuple = tuple(map(int, bbox.split(',')))

    results = query_stac_api(
        bounds=bbox_tuple,
        epsg4326=False,
        start_date=f"{year}-01-01T00:00:00Z",
        end_date=f"{year}-12-31T23:59:59Z"
    )
```

---

#### Function Highlight: `query_stac_api()`

**Purpose**: Queries the Sentinel-2 STAC API to retrieve available scenes for a given bounding box and year.

- Converts local bounds to EPSG:4326
- Uses POST requests to paginate through results
- Returns a list of dictionaries, each containing metadata and asset URLs for a single Sentinel-2 scene

Each (bbox, year) pair triggers a request to the API:
```python
results = query_stac_api(
    bounds=(484932, 1401912, 489035, 1405125),
    epsg4326=False,
    start_date="2019-01-01T00:00:00Z",
    end_date="2019-12-31T23:59:59Z"
)
```

---

### 2.1. Scene Shape Transformations

The results from the `query_stac_api()` call are used to enrich the previously sampled lat/lon points extracted from the Cropland Data Layer (CDL). This process involves a series of transformations, which are detailed below:

#### 🔧 Function Highlight: `process_result()`

**Purpose**: Converts a single Sentinel-2 scene into training-ready data by joining imagery with previously sampled CDL ground truth points.

##### Summary of Processing

1. ✅ Loads CDL training points (lat/lon) from `CDL_parts_path`, corresponding to a specific `bbox` and `year`
2. 🛰️ For each asset (e.g., red, nir, scl), uses `sample_geotiff()` to extract the reflectance or category value at each CDL point location
3. 🚫 Applies filtering using SCL values (e.g., to remove clouds or water)
4. 💾 Appends valid samples to a partitioned Parquet dataset for training

*Each sampled point is enriched with:*
```text
+ lon, lat
+ CDL crop type
+ scene_date, tile
+ reflectance values for each band (e.g., [red, nir, ...])
+ SCL (scene classification label)
```

> 📂 Output partitions: by `bbox`, `year`, `tile`, and `scene_date`

---

#### Function Highlight: `sample_geotiff()`

**Purpose**: Retrieves pixel values from a remote raster (GeoTIFF) at exact lat/lon locations derived from CDL training points.

- Reprojects coordinates from EPSG:4326 to the raster's native CRS
- Retrieves the pixel value at the corresponding location from a **monoband raster**
- Returns a vector of reflectance or category values (one per input coordinate)

Internally, the line:
```python
values = [val[0] for val in dataset.sample(coords)]
```
uses `rasterio.DatasetReader.sample()` to extract values at specific coordinates. Each `val` is a NumPy array with one element (e.g., `[1372]`), so `val[0]` returns the actual value at that pixel.

This process queries the raster precisely at known lat/lon points defined in the CDL DataFrame.

---

#### 🔢 Example Output Table (from `process_result()` write step)

```text
+-----------+------------+------+------------+----------+----------+--------------+-----+------+------+------+
|    bbox   |    year    | tile | scene_date |   lon    |   lat    |     CDL      | SCL |  red |  nir |  ... |
+-----------+------------+------+------------+----------+----------+--------------+-----+------+------+------+
|484932,... |    2019    | 33UXP| 2020-07-15 | -90.59.. | 35.57... |   Soybeans   |  4  | 1324 | 2127 |  ... |
|484932,... |    2019    | 33UXP| 2020-07-15 | -90.58.. | 35.57... |   Soybeans   |  4  | 1299 | 2056 |  ... |
|484932,... |    2019    | 33UXP| 2020-07-15 | -90.57.. | 35.57... |   Corn       |  1  | 1220 | 1901 |  ... |
+-----------+------------+------+------------+----------+----------+--------------+-----+------+------+------+
```

Each row corresponds to a sampled CDL training point enriched with reflectance values from Sentinel-2 spectral bands (e.g., red, nir, etc.), the crop type (CDL), and SCL classification.

Each lat/lon location from the CDL sample points is treated as a unique spatial unit. All observations from different dates (scenes) are grouped per point into temporal sequences. This results in one row per unique location containing all values over time.

#### 🔧 Function Highlight: `agg_to_time_series()`

**Purpose**: Aggregates all samples from Sentinel-2 scenes into time series per training point. Each lat/lon point is treated as a unique entity, and values from all temporal scenes are grouped and encoded row-wise by feature.

This aggregation is handled in the following steps:

1. **`create_image_map()`**: Builds a key-value map where each `scene_date` points to all band values, the SCL class, and tile ID.
2. **`group_time_series()`**: Groups all scenes for a given lat/lon and crop label (`CDL`) into a list.
3. **`flatten_time_series()`**: Converts these grouped values into string arrays (one per band/time slot).
4. **`convert_to_binary()`**: Efficiently encodes the arrays into compact byte representations.

The transformation reshapes the dataset from a flat scene-wise structure into a single-row-per-location format. 
Each output row now contains a temporal sequence of band values associated with a unique lat/lon and crop type.

#### 🔢 Example Output Table (abbreviated)
```text
+----------+----------+---------+-------------+-------------------+-------------+------------------------+-------------+--------------+------+
|   lon    |   lat    |   CDL   | num_images  | bands             |   tiles     | img_dates             | scl_vals   |     bbox     | year |
+----------+----------+---------+-------------+-------------------+-------------+------------------------+-------------+--------------+------+
| -90.58.. | 35.57... |  Corn   |     2       | [01 6E 02 24 (...)]| [33UXP,...] | [2020-07-15,...]       | [4, 5]     | 484932,...   | 2019 |
| -90.57.. | 35.56... |Soybeans |     3       | [01 91 02 55 (...)]| [33UXP,...] | [2020-07-15,...]       | [4, 4, 5]   | 484932,...   | 2019 |
| -90.56.. | 35.55... |  Corn   |     2       | [01 2A 01 F3 (...)]| [33UXP,...] | [2020-07-15,...]       | [4, 3]     | 484932,...   | 2019 |
+----------+----------+---------+-------------+-------------------+-------------+------------------------+-------------+--------------+------+
```

📦 The final result is written to Parquet with one row per CDL sample point, now transformed into a byte-encoded temporal vector ready for training.

#### 🧬 Scene-wise to Time-Series
```python
# Input (multiple rows per location across scenes)
lat, lon, scene_date, red, nir, scl, CDL
↓
# Group and encode by lat/lon
lon, lat, CDL, [red_1, red_2, ...], [nir_1, nir_2, ...], [scl_1, scl_2, ...]
```

#### 🧩 Visual Example

```text
From this:
+ lat   | lon   | date       | red  | nir  | CDL
+-------+-------+------------+------+------+
| 35.57 | -90.58| 2020-07-15 | 1324 | 2127 | Corn
| 35.57 | -90.58| 2020-08-14 | 1450 | 2180 | Corn
↓
To this:
+ lat   | lon   | CDL  | bands             | img_dates              |
+-------+-------+------+--------------------+-------------------------+
| 35.57 | -90.58| Corn | [1324, 1450 (...)]| [2020-07-15, 2020-08-14]|
```

The final time-series dataset is exported in Parquet format to the target directory, organized by `bbox` and `year` partitions:
```python
# Write it out
ts.write.partitionBy(['bbox', 'year']).mode("append").parquet(output_uri)
```

---

## 3. Time Series to Training Data

- **Objective**: Convert Parquet rows into model-ready TensorFlow batches.

### 3.1 Tensor Preparation Module: `dataloader.py`

Utility functions for preparing Sentinel-2 time series data for TensorFlow models.

This module includes:

- Parsing functions to transform raw Parquet-encoded rows into input/output tensors.
- Time series bucketing logic to aggregate imagery over a defined window.
- Label creation logic to assign crop categories based on CDL and SCL inputs. 
- Data filtering functions to remove ambiguous training samples (e.g., double croppings).

Each output sample has the following structure:
- **X**: Tensor of shape `(MAX_IMAGES_PER_SERIES, features)`
- **y**: One-hot encoded label vector

#### 🔧 Function Highlight: `make_dataset()`

**Purpose**  
Creates TensorFlow datasets (`train_ds`, `val_ds`) from raw Parquet files using a configurable preprocessing pipeline.

This is a wrapper that orchestrates the full pipeline:
- Loads Parquet data using either:
  - `"pandas"`: loads all data into memory using `pandas.read_parquet`
  - `"tensorflow"`: reads directly from disk via `tensorflow_io.IODataset.from_parquet`  
    ⚠️ May cause kernel crashes due to decoding issues. See `_read_parquet_file()` for details.
- Applies data filtering, parsing (`parse()`), normalization, batching, and prefetching.

Each row is processed using `parse()` to return:
- **X**: a time series tensor of shape `(max_images_per_series, 12)`
- **y**: a one-hot encoded label vector (based on `label_legend`)

**Returns**
```python
train_ds, val_ds = make_dataset(...)
# train_ds: tf.data.Dataset
# val_ds: tf.data.Dataset
```

```text
Parquet files
    ↓
Load files (method: "pandas" or "tensorflow_io")
    ↓
Filter ambiguous samples (e.g. double croppings)
    ↓
Parse rows into (X, y)
    ↓
Normalize bands (optional)
    ↓
Batch → Prefetch → Model-ready Datasets

```

#### 🔧 Function Highlight: `parse()`

**Purpose**:
Transforms a single row from a Sentinel-2 Parquet dataset into:

- **X**: a time series tensor with fixed temporal structure
- **y**: a one-hot encoded label

This function is applied to every row of the dataset (via `.map(parse, ...)`) and is a key step in preparing model-ready samples for supervised learning.

- `parse()` is responsible for transforming raw Parquet data into structured, normalized `(X, y)` pairs.
- It internally calls `bucket_timeseries()` to shape the time dimension and `create_label()` to construct labels.
- It ensures consistency across samples and supports optional normalization for training stability.

---

##### 📥 Input Format:

Each row is a dictionary of serialized tensors from the Parquet schema:

- `'bands'`: serialized bytes representing 12-band reflectance values
- `'scl_vals'`: serialized Scene Classification Layer (SCL) values
- `'img_dates'`: serialized acquisition dates (in days)
- `'CDL'`: crop type code
- `'num_images'`: the number of temporal observations

---

##### Summary of Processing

1. 🧩 **Decode Byte Fields**
   Serialized tensors are decoded into their original numeric arrays:

   ```python
   bands_decoded: shape (num_images, 12)
   scl_decoded: shape (num_images, 1)
   days_from_start_of_series: shape (num_images, 1)
   ```

2. 🌿 **Compute Spectral Indices**
   From Sentinel-2 bands, calculate vegetation and water indices:

   ```python
   # Normalized Difference Vegetation Index
   NDVI = ((nir - red) / (nir + red)) * 100
   
   # Enhanced Vegetation Index  
   EVI = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1.0)) * 100
   
   # Normalized Difference Water Index
   NDWI = ((nir - swir1) / (nir + swir1)) * 100
   
   # Normalized Difference Built-up Index
   NDBI = ((swir1 - nir) / (swir1 + nir)) * 100
   ```

3. 🔗 **Concatenate Features**
   Combine all per-image information into a single raw tensor:

   ```python
   raw_data: shape (num_images, 18)
   → [12 bands, NDVI, EVI, NDWI, NDBI, SCL, relative_day]
   ```

4. 📦 **Bucket Time Series**
   Apply `bucket_timeseries()` to select one observation per time bucket:

   ```python
   bucketed_data: shape (max_images_per_series, 18)
   ```

   This becomes the **feature vector X**.

5. 🏷️ **Create Label**
   Use `create_label()` to derive the label from the CDL value and SCL information:

   ```python
   y: one-hot vector, shape = (num_classes,)
   ```
   This becomes the **label vector y**.

6. 📊 **Optional Normalization**
   If `norm=True`, normalize the band values and spectral indices (first 16 columns of X):

   ```python
   X = bucketed_data[:, :16]  # Extract only bands + indices (exclude SCL, relative_day)
   X = (X - means) / stds → z-score
   → masked so padding does not affect the result
   ```

   **Note**: When `norm=True`, the final shape becomes `(max_images_per_series, 16)` instead of 18.

#### 🧾 Output Format Example

Below is a compact example of how the model input (`X`) and label (`y`) tensors look after parsing and batching:

```text
X
tf.Tensor(
[[[-0.41 -0.47 ...  0.49]
  [ 0.40  0.10 ...  0.83]
  ...
  [-0.00 -0.00 ... -0.00]]
 [[-0.85 -0.83 ... -0.65]
  [-0.00 -0.00 ... -0.00]
  ...
  [-0.00 -0.00 ... -0.00]]],
shape=(batch_size, max_images_per_series, 16), dtype=float32)  # 16 when norm=True, 18 otherwise
```

y
tf.Tensor(
[[0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 ...
 [0. 0. 0. 0. 1. 0. 0.]],
shape=(batch_size, num_classes), dtype=float32)
```

#### 🔧 Function Highlight: `bucket_timeseries()`

**Purpose:**
Reduces a raw time series of observations into a fixed-size, temporally-bucketed tensor.

Given the variability in the number and timing of available satellite observations, it's necessary to capture temporal dynamics in a uniform way. Temporal bucketing allows for the creation of fixed-length sequences by selecting one representative observation per time window. This provides a consistent and compressed view of the time series across all samples, which is crucial for training neural networks. Combined with normalization, this helps stabilize the learning process and ensures balanced input distributions across spectral bands.

##### Summary of Processing 

1. 🪟 **Temporal Slice**  
   Select a random time window of `days_in_series` and filter images inside it.

```python
start_day = random
filtered = data[start_day ≤ day ≤ start_day + days_in_series]
```

2. 🎲 **Sample & Normalize days**  
   Take up to `max_images_per_series` random samples. Make their days relative.

```python
rinput = random sample of filtered
normalized_days = rinput.day - first_day
```

3. 🗂️ **Bucket by Time Frame**  
   Group images into time buckets using `days_per_bucket`. Keep one per bucket.

```python
bucket_idx = normalized_days // days_per_bucket
→ only keep first image per bucket
```

4. 🧱 **Pad to Fixed Shape**  
   Scatter into a fixed-size tensor of shape `(max_images_per_series, 18)`.

```python
output[bucket_idx[i]] = image_i
→ missing buckets = padded with zeros
```

##### 🧪 Toy Output Example

Suppose after filtering and normalization we have:

```python
normalized_days = [0, 3, 9, 12, 25]
bucket_idx = [0, 0, 0, 1, 2]
```

Each row is a 18-length vector:  
`[12 bands, NDVI, EVI, NDWI, NDBI, SCL, relative_day]`

We keep only the **first image per bucket**:

```python
selected_rows = [row_0, row_3, row_4]
```

These become the **feature vector X** used in `parse()`:

```python
X = [
  row_0,  # → for bucket 0
  row_3,  # → for bucket 1
  row_4,  # → for bucket 2
  [0, 0, ..., 0],  # → padding
  ...
]  # shape = (max_images_per_series, 18) → (max_images_per_series, 16) when norm=True
```

> 🔍 Note: Only the first 16 columns of each row (12 bands + 4 indices) are used when `norm=True`.

📄 For a detailed breakdown with visuals, see [`doc/bucket_timeseries.md`](doc/bucket_timeseries.md)


#### 🔧 Function Highlight: `create_label()`

**Purpose:**  
Assigns an integer label to each training sample based on CDL crop class and SCL conditions.  
This function is used inside `parse()` to compute the **label `y`** before one-hot encoding.

---

##### 📥 Inputs

- `cdl`: Tensor with crop type (e.g., `tf.constant("Soybeans")`)
- `scl_val`: Tensor with SCL values across the time series (shape `(N,)`)
- `targeted_cultivated_crops_list`: List of crops to assign **individual** labels (e.g. `["Soybeans", "Rice", "Corn", "Cotton"]`)
- `other_cultivated_crops_list`: List of **generic cultivated** crops (e.g. `"Alfalfa"`, `"Sunflower"`...)

---

##### 🚩 Label Encoding Logic

```text
Label Value | Meaning
------------|-----------------------------
0           | Uncultivated / ignored
1           | Other cultivated crops
2           | Ambiguous: no vegetation detected in final frames
3           | Soybeans
4           | Rice
5           | Corn
6           | Cotton
```

If a `cdl` crop name matches an entry in `targeted_cultivated_crops_list`, its **position in that list + 3** is used as label.  
Otherwise, it’s mapped to 1 or 0 depending on whether it’s cultivated or not.

Then, if no vegetation is observed in the last 2 frames (`SCL != 4`), and the crop was previously marked as cultivated, it's reassigned to label `2` to mark ambiguity.

---

##### 🧪 Toy Output Example

```python
# Label legend for one-hot encoding
label_legend = [
    "Uncultivated",           # 0
    "Cultivated",             # 1
    "No Crop Growing",        # 2
    "Soybeans", "Rice", "Corn", "Cotton"  # 3–6
]

cdl = tf.constant("Soybeans")  # target crop
scl_val = tf.constant([4, 4, 0, 1, 4, 4, 3], dtype=tf.int32)  # includes vegetation

label = create_label(cdl, scl_val, ["Soybeans", "Rice", "Corn", "Cotton"], [...])
# label = 3 (position 0 in list + offset 3)

# Then in parse:
y = tf.one_hot(label, depth=len(label_legend))
# y = [0, 0, 0, 1, 0, 0, 0]  ← one-hot encoded for 7 classes
```

---

## 6. Training Workflow

### Dataset Generation & Class Balance

The training dataset generation process allows configuration of two key parameters that significantly impact class distribution:

- **Bucketing Strategy**: Controls how time series data is sampled
  - `random`: Balanced sampling across time periods (recommended)
  - `early_season`: Favors early growing season data
  - `late_season`: Favors late growing season data
  - `deterministic`: Fixed sampling pattern

- **Frames to Check**: Number of time steps to sample per series
  - Higher values increase dataset size but may affect class balance
  - Recommended: 2 for optimal balance

### Impact on Class Distribution

These parameters directly influence the proportion of "No Crop Growing" vs. cultivated crop pixels, affecting overall model performance and minority class representation. The `random` strategy with `frames_to_check=2` provides the best balance between dataset size and class distribution.

### Model Architecture

We tested several deep learning architectures to classify time series derived from Sentinel-2 temporal crop data, including 1D CNNs, ResNets, U-Nets, Transformers, and Inception variants. In time series classification, unlike in image processing, models learn to detect **temporal patterns**—such as peaks, drops, or repeating oscillations—rather than spatial patterns like edges or shapes.

### 📌 Key Insight:
- 👉 In image processing, filters learn spatial features like **edges**, **textures**, or **shapes**.
- 👉 In time series classification, filters learn **temporal dynamics** such as **pulses**, **trends**, or **seasonal cycles**.

---

### 🔍 Benchmarks Across Model Variants

| Model                                  | Params      | Accuracy | Macro F1 | Recall Cultivated | Notes |
|----------------------------------------|-------------|----------|----------|--------------------|-------|
| **Simple CNN**                         | ~50K        | 84.1%    | 0.709    | 0.055              | Baseline |
| Big CNN                                | ~150K       | 81.1%    | 0.680    | 0.074              | Overfitting |
| VGG1D Compact                          | ~90K        | 85.1%    | 0.713    | 0.016              | Best global accuracy, fails on 'Cultivated' |
| U-Net1D Light                          | ~100K       | 84.6%    | 0.705    | 0.000              | Similar to VGG1D |
| ResNet1D                               | ~130K       | 82.9%    | 0.694    | 0.060              | Slightly better than baseline |
| ResUNet1D                              | ~160K       | 85.3%    | 0.714    | 0.000              | Excellent globally, 0% 'Cultivated' |
| Inception1D                            | ~140K       | 84.2%    | 0.697    | 0.019              | Minimal gain over CNN |
| **Inception1D + SE + MixUp + Focal**   | ~160K       | 84.5%    | **0.737**| **0.264**          | 🔥 Best balance, only model with useful recall on 'Cultivated' |

> ✅ **Inception1D with Squeeze-Excite, MixUp, Focal Loss, Residuals and dynamic alpha (AnnealedAlpha) was the only model that successfully increased recall on 'Cultivated' while preserving global accuracy and macro F1.**  
> 🔍 Most other deep or transformer-based models consistently ignored this class, suggesting architectural complexity alone does not solve class imbalance in this task.

---

### 🧠 Best-Performing Architectures

#### 🔹 Input Format

The model expects sequences of shape:

```python
(input_shape) = (120, 16)
```

```text
           Feature dimension →
Time ↓   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
         │ f1 │ f2 │ f3 │ …  │    │    │    │    │    │ f16│
Day 1    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Day 2    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
...      ┆                                             ┆
Day 120  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
         ←────────── (16 features) ──────────→
```

---

### 🔹 Simple CNN: Layer-by-Layer Output Dimensions

| Layer Type         | Filters | Kernel Size | Output Shape |
|--------------------|---------|-------------|--------------|
| Input              | —       | —           | (120, 16)    |
| Conv1D             | 64      | 3           | (120, 64)    |
| MaxPooling1D       | —       | 2           | (60, 64)     |
| Dropout            | —       | —           | (60, 64)     |
| Conv1D             | 64      | 3           | (60, 64)     |
| MaxPooling1D       | —       | 2           | (30, 64)     |
| Dropout            | —       | —           | (30, 64)     |
| GlobalAvgPooling1D | —       | —           | (64,)        |
| Dense              | 128     | —           | (128,)       |
| Output (Softmax)   | N       | —           | (N,)         |

---

### 🧩 Feature Map Evolution through Simple CNN architecture:

```text
Input (120, 16)
  ▼ Conv1D(64) → MaxPool → Dropout
(60, 64)
  ▼ Conv1D(64) → MaxPool → Dropout
(30, 64)
  ▼ GlobalAvgPooling1D → Dense(128) → Softmax
```

#### 🟩 Input Layer  
- Raw temporal signal (120 time steps × 16 features).

#### 🟦 Conv Blocks  
- 2 convolutional stages with downsampling.  
- Each block reduces sequence length and increases abstraction.

#### 🟪 Classification Head  
- Temporal feature aggregation via GlobalAvgPooling.  
- Fully connected layer maps to class logits.

---

### 🔹 Inception1D + SE + Residual + MixUp + Focal: Overview

This model builds upon Inception1D with the following enhancements:

- **Squeeze-and-Excitation (SE)**: adaptively recalibrates feature maps to prioritize informative channels.
- **Residual connections**: mitigate vanishing gradients and help optimization.
- **Focal Loss + Annealed Alpha**: emphasizes hard-to-classify classes, especially 'Cultivated', using a dynamically annealed class weight.
- **MixUp augmentation**: regularizes the model and improves generalization under class imbalance.

---

### 🔹 Inception1D-Enhanced: Layer-by-Layer Output Dimensions

| Layer Type             | Filters | Kernel Size | Output Shape |
|------------------------|---------|-------------|--------------|
| Input                  | —       | —           | (120, 16)    |
| Inception Block 1      | [16,32] | 1/3/5        | (120, 64)    |
| Squeeze-Excite         | —       | —           | (120, 64)    |
| Residual Add + ReLU    | —       | —           | (120, 64)    |
| MaxPooling1D           | —       | 2           | (60, 64)     |
| Inception Block 2      | [32,64] | 1/3/5        | (60, 128)    |
| Squeeze-Excite         | —       | —           | (60, 128)    |
| Residual Add + ReLU    | —       | —           | (60, 128)    |
| MaxPooling1D           | —       | 2           | (30, 128)    |
| GlobalAvgPooling1D     | —       | —           | (128,)       |
| Dense + Dropout        | 128     | —           | (128,)       |
| Output (Softmax)       | N       | —           | (N,)         |

---

### 🧩 Feature Map Evolution through Inception1D-SE-Residual architecture:

```text
Input Time Series
(120, 16)

    ▼ Inception Block + SE + Residual
(120, 64)

    ▼ MaxPooling1D
(60, 64)

    ▼ Inception Block + SE + Residual
(60, 128)

    ▼ MaxPooling1D
(30, 128)

    ▼ GlobalAvgPooling1D → Dense(128) → Softmax
(128,) → (num_classes,)
```

#### 🟩 Input Layer  
- Sequence of 120 time steps with 16 spectral and index features.

#### 🟦 Inception Blocks  
- Multi-scale temporal filters (1×, 3×, 5×) extract diverse patterns (e.g., sudden changes, slow growth).

#### 🟦 Squeeze-and-Excite  
- Recalibrates filters dynamically to emphasize informative features.

#### 🟨 Residual Add  
- Combines input and output to facilitate gradient flow and better convergence.

#### 🟪 Classification Head  
- Aggregates temporal descriptors into dense vector.  
- Focal Loss + AnnealedAlpha ensures focus on hard classes.

> 🔥 This model was the only one that consistently improved **recall for 'Cultivated'** while maintaining competitive accuracy and F1-macro.  
> The combination of architectural, optimization, and augmentation enhancements proved necessary to address the class imbalance challenge.

---

## 7. Training & Validation

- Splits, batch generation, class weighting.

---

## 8. Evaluation

- Metrics: Accuracy, F1, Confusion Matrix, etc.

---

## 9. Deployment / Inference

- Applying the model to new data and reconstructing label maps.

