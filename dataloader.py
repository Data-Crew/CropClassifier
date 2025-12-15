"""
dataloader.py

Utility functions for preparing Sentinel-2 time series data for TensorFlow models.

This module includes:

- Parsing functions to transform raw Parquet-encoded rows into input/output tensors.
- Spectral indices calculation (NDVI, EVI, NDWI, NDBI) from Sentinel-2 bands.
- Time series bucketing logic to aggregate imagery over a defined window (X features).
- Label creation logic to assign crop categories based on CDL and SCL inputs (y labels).
- Data filtering functions to remove ambiguous training samples (e.g., double croppings).

The expected output format for training is a batched TensorFlow dataset where each element is:
    - X: A time series tensor of shape `(MAX_IMAGES_PER_SERIES, features)` containing 
      spectral bands and calculated vegetation indices
    - y: A one-hot encoded label vector

Intended to be used in pipelines that load data directly from Parquet files using TensorFlow I/O.
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_io as tfio

# Constants for feature dimensions
NUM_SENTINEL_BANDS = 12  # Number of Sentinel-2 bands
NUM_VEGETATION_INDICES = 4  # NDVI, EVI, NDWI, NDBI
NUM_FEATURES = NUM_SENTINEL_BANDS + NUM_VEGETATION_INDICES  # Default: 16 features (12 bands + 4 indices)
# Note: NUM_FEATURES can be overridden via command line arguments (12=bands only, 16=bands+indices)

def make_from_pandas(files):
        # Ensure files are sorted for consistent ordering
        if isinstance(files, list):
            files = sorted(files)
        df = pd.read_parquet(files)
        ds = tf.data.Dataset.from_tensor_slices(dict(df))
        return ds

def make_dataset(
    train_files: list[str],
    val_files: list[str],
    method: str = "pandas",
    batch_size: int = 32,
    means=None,
    stds=None,
    label_legend: dict[str, int] = None,
    targeted_cultivated_crops_list: list[str] = None,
    other_cultivated_crops_list: list[str] = None,
    days_in_series: int = 25,
    days_per_bucket: int = 1,
    max_images_per_series: int = 121,
    frames_to_check: int = 3,
    bucketing_strategy: str = "deterministic",
    num_features: int = NUM_FEATURES
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Builds training and validation datasets from a list of Parquet files.

    Parameters
    ----------
    train_files : list[str]
        List of Parquet file paths used for training.

    val_files : list[str]
        List of Parquet file paths used for validation.

    method : str, default="pandas"
        Data loading strategy:
        - "pandas": loads all data into memory using `pandas.read_parquet`.
        - "tensorflow": reads data directly from disk using `tensorflow_io.IODataset.from_parquet`.
          ⚠️ NOTE: This method may crash the kernel due to decoding issues related to TensorFlow I/O.

    batch_size : int, default=32
        Number of samples per batch.

    means : np.ndarray or None
        Per-channel mean values for band normalization.

    stds : np.ndarray or None
        Per-channel standard deviation values for band normalization.

    label_legend : dict[str, int]
        Mapping from crop label strings to class indices.

    targeted_cultivated_crops_list : list[str]
        List of target crops to classify as individual categories.

    other_cultivated_crops_list : list[str]
        List of crops grouped into a shared "other crops" class.

    days_in_series : int, default=25
        Number of days expected in each time series sample.

    days_per_bucket : int, default=1
        Temporal resolution to aggregate days into buckets.

    max_images_per_series : int, default=121
        Maximum number of images in a single time series sample.

    frames_to_check : int, default=3
        Number of last frames to check for vegetation presence when creating labels.
        Used to determine if a cultivated crop should be labeled as "no vegetation".

    bucketing_strategy : str, default="deterministic"
        Time window selection strategy for bucketing time series:
        - "deterministic": Hash-based deterministic selection (recommended)
        - "early_season": Focus on early growing season
        - "mid_season": Focus on mid growing season
        - "late_season": Focus on late growing season
        - "random": Random selection

    num_features : int, default=NUM_FEATURES
        Number of features to use:
        - 12: Only Sentinel-2 bands (no vegetation indices)
        - 16: Sentinel-2 bands + vegetation indices (NDVI, EVI, NDWI, NDBI)

    augment : bool, default=False
        Whether to apply data augmentation to the training dataset.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset]
        A tuple with (train_ds, val_ds), ready for model training and evaluation.
    """

    if method == "pandas":
        train_ds_raw = make_from_pandas(train_files)
        val_ds_raw = make_from_pandas(val_files)

    elif method == "tensorflow":
        train_files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        val_files_ds = tf.data.Dataset.from_tensor_slices(val_files)

        train_ds_raw = train_files_ds.interleave(
            _read_parquet_file,
            cycle_length=1,
            num_parallel_calls=1
        )

        val_ds_raw = val_files_ds.interleave(
            _read_parquet_file,
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        raise ValueError("Invalid method: choose either 'pandas' or 'tensorflow'")

    def apply_pipeline(ds, is_training: bool):
        ds = (
            ds
            .filter(filter_double_croppings)
            .shuffle(batch_size * 10)
            .map(lambda x: parse_with_error_handling(
                x, norm=True,
                means=means, stds=stds,
                label_legend_=label_legend,
                targeted_cultivated_crops_list=targeted_cultivated_crops_list,
                other_cultivated_crops_list=other_cultivated_crops_list,
                days_in_series=days_in_series,
                days_per_bucket=days_per_bucket,
                max_images_per_series=max_images_per_series,
                frames_to_check=frames_to_check,
                bucketing_strategy=bucketing_strategy,
                num_features=num_features
            ), num_parallel_calls=1)  # Reduced to 1 to see errors sequentially
        )



        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = apply_pipeline(train_ds_raw, is_training=True)
    val_ds = apply_pipeline(val_ds_raw, is_training=False)
    return train_ds, val_ds

# TODO: Investigate decoding crash when using tfio.IODataset.from_parquet

def _read_parquet_file(filepath: str) -> tf.data.Dataset:
    """
    Reads a Parquet file using TensorFlow I/O.

    Parameters:
    ------------
    filepath : str
        Path to a Parquet file.

    Returns:
    --------
    tf.data.Dataset
        TensorFlow dataset constructed directly from the Parquet file.

    TODO:
        This method currently causes kernel crashes when reading large or complex serialized
        byte strings (e.g. 'bands'). The issue appears to be in how TensorFlow handles decoding
        from disk using tfio.
    """
    columns = {
        'lon': tf.float64,
        'lat': tf.float64,
        'num_images': tf.int32,
        'bands': tf.string,
        'tiles': tf.string,
        'img_dates': tf.string,
        'scl_vals': tf.string,
        'CDL': tf.string
    }
    return tfio.IODataset.from_parquet(filepath, columns)


def parse_with_error_handling(
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
    bucketing_strategy: str = "random",
    start_day: int = None,
    num_features: int = NUM_FEATURES
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Wrapper around parse() that catches errors and provides debugging information.
    """
    try:
        return parse(row, norm, means, stds, label_legend_, targeted_cultivated_crops_list,
                    other_cultivated_crops_list, days_in_series, days_per_bucket,
                    max_images_per_series, frames_to_check, bucketing_strategy, start_day, num_features)
    except Exception as e:
        # Try to extract debugging info before failing
        try:
            if tf.executing_eagerly():
                num_imgs_val = tf.cast(row['num_images'], tf.int64).numpy()
                bands_size = tf.size(row['bands']).numpy()
                dates_size = tf.size(row['img_dates']).numpy()
                scl_size = tf.size(row['scl_vals']).numpy()
                
                print(f"❌ [PARSE ERROR] Exception in parse():")
                print(f"   - Error type: {type(e).__name__}")
                print(f"   - Error message: {str(e)}")
                print(f"   - num_images: {num_imgs_val}")
                print(f"   - bands size: {bands_size} bytes")
                print(f"   - dates size: {dates_size} bytes")
                print(f"   - scl_vals size: {scl_size} bytes")
                print(f"   - Expected bands size: {num_imgs_val * 12 * 2}")
                print(f"   - Expected dates size: {num_imgs_val * 2}")
                print(f"   - Expected scl size: {num_imgs_val * 1}")
        except:
            pass
        
        # Re-raise the original error
        raise

def parse(
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
    bucketing_strategy: str = "random",
    start_day: int = None,
    num_features: int = NUM_FEATURES
) -> tuple[tf.Tensor, tf.Tensor]:
    num_images = row['num_images']
    
    # Debug: Use tf.print to show information (works in graph mode)
    num_images_int = tf.cast(num_images, tf.int32)
    # For string tensors, use tf.strings.length to get byte length, not tf.size
    bands_size = tf.strings.length(row['bands'], unit='BYTE')
    dates_size = tf.strings.length(row['img_dates'], unit='BYTE')
    scl_size = tf.strings.length(row['scl_vals'], unit='BYTE')
    
    # Expected sizes (in bytes)
    expected_bands_size = num_images_int * 12 * 2  # num_images * 12 bands * 2 bytes per uint16
    expected_dates_size = num_images_int * 2  # num_images * 2 bytes per uint16
    expected_scl_size = num_images_int * 1  # num_images * 1 byte per uint8
    
    # Print when sizes don't match (this will show in logs)
    size_mismatch = tf.logical_or(
        tf.not_equal(bands_size, expected_bands_size),
        tf.logical_or(
            tf.not_equal(dates_size, expected_dates_size),
            tf.not_equal(scl_size, expected_scl_size)
        )
    )
    
    # Print when sizes don't match - both branches must return same type (bool)
    # DISABLED: Too verbose, only print actual errors
    _ = tf.cond(size_mismatch, 
        lambda: tf.constant(True),  # Just return True, don't print
        lambda: tf.constant(True))

    # Decode and reshape data
    scl_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['scl_vals'], out_type=tf.uint8), tf.int32), (num_images, 1))
    bands_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['bands'], out_type=tf.uint16, little_endian=False), tf.int32), shape=(num_images, 12))
    date_decoded = tf.reshape(tf.cast(tf.io.decode_raw(row['img_dates'], out_type=tf.uint16, little_endian=False), tf.int32), shape=(num_images, 1))
    
    # Debug: Print shape information BEFORE accessing indices
    date_shape = tf.shape(date_decoded)
    date_rank = tf.rank(date_decoded)
    num_images_actual = date_shape[0]
    
    # Print debug info for every record (will show in logs)
    # DISABLED: Too verbose, generates thousands of lines
    # tf.print("🔍 [PARSE DEBUG] Processing record - num_images:", num_images_int, 
    #          "date_decoded shape:", date_shape, 
    #          "date_decoded rank:", date_rank,
    #          "num_images_actual:", num_images_actual)
    
    # Check if shape is valid before accessing
    is_shape_valid = tf.logical_and(
        tf.equal(date_rank, 2),
        tf.greater(num_images_actual, 0)
    )
    
    def compute_days():
        # Print before accessing
        # DISABLED: Too verbose
        # tf.print("✅ [PARSE DEBUG] Computing days - accessing date_decoded[0,0] and date_decoded[:,0]")
        return tf.cast(date_decoded[:, 0] - date_decoded[0, 0], tf.int32)
    
    def return_zeros():
        # DISABLED: Too verbose
        # tf.print("⚠️ [PARSE DEBUG] Invalid shape - returning zeros. rank:", date_rank, "num_images_actual:", num_images_actual)
        return tf.zeros([num_images], dtype=tf.int32)
    
    days_from_start_of_series = tf.cond(is_shape_valid, compute_days, return_zeros)
    days_from_start_of_series = tf.reshape(days_from_start_of_series, shape=(num_images, 1))
    
    # Additional validation: ensure date_decoded has correct shape
    date_decoded_shape = tf.shape(date_decoded)
    date_decoded_rank = tf.rank(date_decoded)
    
    # Debug print if shape is wrong
    def debug_shape():
        if tf.executing_eagerly():
            try:
                shape_val = date_decoded_shape.numpy()
                rank_val = date_decoded_rank.numpy()
                num_imgs_val = tf.cast(num_images, tf.int64).numpy()
                print(f"⚠️ [PARSE DEBUG] date_decoded shape issue:")
                print(f"   - date_decoded shape: {shape_val}")
                print(f"   - date_decoded rank: {rank_val}")
                print(f"   - num_images: {num_imgs_val}")
                print(f"   - date_decoded actual shape: {date_decoded.shape}")
            except:
                pass
        return tf.constant(True)
    
    # Only print if there's an issue
    _ = tf.cond(
        tf.not_equal(date_decoded_rank, 2),
        debug_shape,
        lambda: tf.constant(False)
    )
    days_from_start_of_series = tf.reshape(days_from_start_of_series, shape=(num_images, 1))

    # Conditionally calculate vegetation indices based on num_features
    if num_features == NUM_SENTINEL_BANDS:
        # Use only 12 bands (no vegetation indices)
        raw_data = tf.concat([bands_decoded, scl_decoded, days_from_start_of_series], axis=1)  # shape=(num_images, 14)
    else:
        # Use 12 bands + 4 vegetation indices
        indices_stack = calculate_vegetation_indices(bands_decoded)
        raw_data = tf.concat([bands_decoded, indices_stack, scl_decoded, days_from_start_of_series], axis=1)  # shape=(num_images, 18)
    # Validate raw_data shape before bucketing
    raw_data_shape = tf.shape(raw_data)
    num_images_in_data = raw_data_shape[0]
    
    # Ensure we have valid data before bucketing
    def safe_bucket_and_label():
        bucketed_data = bucket_timeseries(
            raw_data, days_in_series, days_per_bucket, max_images_per_series, 
            strategy=bucketing_strategy, start_day=start_day, is_testing=(start_day is not None)
        )
        
        # Validate bucketed_data shape before accessing
        bucketed_shape = tf.shape(bucketed_data)
        num_bucketed_images = bucketed_shape[0]
        
        # Ensure we can safely access bucketed_data[:, -2]
        def create_label_safe():
            scl_col = bucketed_data[:, -2]
            label = tf.cast(
                create_label(row['CDL'], 
                scl_col, 
                targeted_cultivated_crops_list, 
                other_cultivated_crops_list, 
                frames_to_check
            ), tf.int32)
            # Ensure label is a scalar (squeeze any extra dimensions)
            label = tf.squeeze(label)
            # Debug: print CDL and label
            # DISABLED: Too verbose, generates thousands of lines
            # tf.print("🏷️ [LABEL DEBUG] CDL:", row['CDL'], "Label:", label)
            return label
        
        def return_default_label():
            # Return label 0 (Uncultivated) as default
            return tf.constant(0, dtype=tf.int32)
        
        y = tf.cond(
            tf.greater(num_bucketed_images, 0),
            create_label_safe,
            return_default_label
        )
        
        # Ensure y is a scalar before returning
        y = tf.squeeze(y)
        
        return bucketed_data, y
    
    bucketed_data, y = safe_bucket_and_label()

    if norm:
        X = bucketed_data[:, :num_features]  # Use dynamic num_features
    else:
        X = bucketed_data

    X = tf.cast(X, tf.float32)
    # Ensure y is a scalar before one_hot encoding
    y = tf.squeeze(y)
    y = tf.one_hot(y, len(label_legend_))
    y = tf.ensure_shape(y, (len(label_legend_),))

    if norm:
        mask = tf.where(tf.not_equal(X, 0), tf.ones_like(X), tf.zeros_like(X))
        X = (X - means) / stds
        X = X * mask

    return X, y

    
def create_label(
    cdl: tf.Tensor,
    scl_val: tf.Tensor,
    targeted_cultivated_crops_list: list[str],
    other_cultivated_crops_list: list[str],
    frames_to_check: int = 3
) -> tf.Tensor:
    """
    Assigns a numeric label to a sample based on the annual CDL crop type and the presence of vegetation
    in the selected time window, as determined by SCL (Scene Classification Layer) values.

    Although the CDL value is annual and fixed for each pixel, the true label assigned for each sample
    can change depending on the presence of vegetation in the last N frames of the selected time window.
    The date (window) and the vegetation status are determined by the SCL values for that window.
    This allows the label to reflect whether a crop is actually growing at that time, not just the annual dominant crop.

    Example:
        - If CDL = 'Corn' for a pixel, but there is no vegetation in the last frames of the window
          (e.g., after harvest, as indicated by SCL), the label will be 'No Crop Growing'.
        - If CDL = 'Corn' and there is vegetation in the last frames (as indicated by SCL), the label will be 'Corn'.
        - This way, the true label can change with the window (date), even though CDL is fixed.

    Parameters:
    -----------
    cdl : tf.Tensor
        Crop label from the CDL dataset (string, annual, fixed per pixel).
    scl_val : tf.Tensor
        Sentinel-2 SCL values across the time series (int, varies per window and defines vegetation status for the date).
    targeted_cultivated_crops_list : list of str
        Specific crops to be classified individually (e.g., 'Soybeans', 'Corn').
    other_cultivated_crops_list : list of str
        General cultivated crops grouped into a single label.
    frames_to_check : int, default=3
        Number of last frames to check for vegetation presence.

    Returns:
    --------
    tf.Tensor
        An integer label (int16) representing the crop class:
            0 = Uncultivated / ignored
            1 = Other cultivated crop
            2 = Crop but no vegetation in last N frames (No Crop Growing)
            3+ = Targeted crops (index offset by +3)
    """
    # Debug: print CDL value and lists
    # DISABLED: Too verbose, generates thousands of lines
    # tf.print("🔍 [CREATE_LABEL DEBUG] CDL:", cdl, "targeted_list:", targeted_cultivated_crops_list)
    
    # Check if CDL matches targeted crops
    targeted_match = tf.reduce_any(tf.math.equal(cdl, tf.constant(targeted_cultivated_crops_list)))
    other_match = tf.reduce_any(tf.math.equal(cdl, tf.constant(other_cultivated_crops_list)))
    
    # DISABLED: Too verbose
    # tf.print("🔍 [CREATE_LABEL DEBUG] targeted_match:", targeted_match, "other_match:", other_match)
    
    label = None
    if targeted_match:
        matched_idx = tf.where(tf.equal(cdl, tf.constant(targeted_cultivated_crops_list)))
        label_value = tf.squeeze(matched_idx) + 3
        label = tf.cast(label_value, dtype=tf.int16)
        # DISABLED: Too verbose
        # tf.print("✅ [CREATE_LABEL DEBUG] Matched targeted crop, label:", label)
    elif other_match:
        label = tf.constant(1, dtype=tf.int16)
        # DISABLED: Too verbose
        # tf.print("✅ [CREATE_LABEL DEBUG] Matched other crop, label:", label)
    else:
        label = tf.constant(0, dtype=tf.int16)
        # DISABLED: Too verbose
        # tf.print("⚠️ [CREATE_LABEL DEBUG] No match, default label:", label)

    # Configurable SCL checking - check if there's vegetation in any of the last N frames
    non_zero_scl = tf.gather(scl_val, tf.where(scl_val != 0))
    if tf.shape(non_zero_scl)[0] > 0:
        # Check last N frames if available, otherwise check all available
        frames_to_check_actual = tf.minimum(frames_to_check, tf.shape(non_zero_scl)[0])
        last_frames = non_zero_scl[-frames_to_check_actual:]
        has_vegetation = tf.reduce_any(tf.equal(last_frames, 4))
        
        if (label != 0) & ~has_vegetation:
            label = tf.constant(2, dtype=tf.int16)

    return label

def bucket_timeseries(
    data: tf.Tensor,
    days_in_series: int,
    days_per_bucket: int,
    max_images_per_series: int,
    strategy: str = "random",
    start_day: int = None,
    is_testing: bool = False
) -> tf.Tensor:
    """
    Reduces a raw time series of observations into a fixed-size, temporally-bucketed tensor.

    Parameters:
    -----------
    data : tf.Tensor
        Tensor of shape (num_images, 18), where each row contains 12 band values, 4 vegetation indices,
        SCL value, and the number of days from the start of the series.
    days_in_series : int
        The number of days to select as the time window for sampling.
    days_per_bucket : int
        The number of days grouped into one bucket.
    max_images_per_series : int
        Maximum number of time steps to keep (i.e., number of buckets).
    strategy : str
        Bucketing strategy (only used when start_day is None):
        - "deterministic": Hash-based deterministic selection
        - "early_season": Focus on early growing season (start from day 0)
        - "mid_season": Focus on mid growing season (start from middle of series)
        - "late_season": Focus on late growing season (start from end of series)
        - "random": Random selection of time window
    start_day : int, optional
        Specific starting day for the time window. If provided, overrides strategy.
        Use None for training (automatic selection) or specific value for testing.
    is_testing : bool, optional
        If True, uses original bucketing logic (no random sampling). If False, uses strategy-based selection.

    Returns:
    --------
    tf.Tensor
        Tensor of shape (max_images_per_series, 18) with one representative
        observation per bucket (or zero-padded if insufficient data).
    """
    # 1. extract absolute day-of-year column
    days = data[:, -1]  # Last column = absolute days since start of year
    max_day = tf.reduce_max(days)

    # 2. select interval of fixed length days_in_series
    if start_day is not None:
        # Use specific start_day (for testing)
        start_day = tf.constant(start_day, dtype=tf.int32)
    else:
        # Use automatic strategy (for training)
        if strategy == "deterministic":
            # Use a hash-based approach to make it deterministic but still varied across samples
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
        elif strategy == "random":
            # Random selection (original behavior)
            start_day = tf.random.uniform(
                shape=(), minval=0, maxval=tf.maximum(1, max_day - days_in_series), dtype=tf.int32
            )
        else:
            # Default to deterministic if strategy is not recognized
            sample_hash = tf.strings.to_hash_bucket_fast(tf.strings.as_string(tf.reduce_sum(days)), 1000)
            start_day = tf.cast(sample_hash % tf.cast(tf.maximum(1, max_day - days_in_series), tf.int64), tf.int32)
    
    end_day = start_day + days_in_series

    # 3. filter the observations within that interval
    mask = (days >= start_day) & (days <= end_day)
    mask_indices = tf.where(mask)
    
    # Ensure we have at least one element before gathering
    def gather_filtered():
        squeezed_indices = tf.squeeze(mask_indices, axis=1)
        # Handle case where squeeze might return scalar instead of 1D tensor
        squeezed_indices = tf.cond(
            tf.equal(tf.rank(squeezed_indices), 0),
            lambda: tf.expand_dims(squeezed_indices, 0),
            lambda: squeezed_indices
        )
        return tf.gather(data, squeezed_indices, axis=0)
    
    def return_empty_with_shape():
        # Return empty tensor with correct feature shape
        num_features = tf.shape(data)[1]
        return tf.zeros((0, num_features), dtype=tf.int32)
    
    series_in_time_range = tf.cond(
        tf.greater(tf.shape(mask_indices)[0], 0),
        gather_filtered,
        return_empty_with_shape
    )

    # 4. select up to max_images_per_series from the range
    # store the features of the images randomly chosen within the num_features interval
    # e.g. 12 bands + 4 vegetation indices + SCL + day (num_features = 18)
    num_features = tf.shape(series_in_time_range)[1]
    num_images_in_range = tf.shape(series_in_time_range)[0]
    
    # Use original logic for testing, strategy-based for training
    # Both paths need to handle empty tensors safely
    if is_testing:
        # Original logic: no random sampling, just take all data in range
        # But still need to ensure we have at least one row
        def return_test_data():
            return series_in_time_range
        
        def return_empty_test():
            return tf.zeros((0, num_features), dtype=tf.int32)
        
        rinput = tf.cond(
            tf.greater(num_images_in_range, 0),
            return_test_data,
            return_empty_test
        )
    else:
        # Use strategy-based selection (for training)
        def select_samples():
            idxs = tf.range(num_images_in_range)
            if strategy == "random":
                # Random selection (original behavior)
                # Ensure we don't try to shuffle empty tensor
                shuffled = tf.random.shuffle(idxs)
                num_to_select = tf.minimum(tf.shape(shuffled)[0], max_images_per_series)
                ridxs = tf.sort(shuffled[:num_to_select])
            else:
                # Deterministic selection - take first N images (for all other strategies)
                num_to_select = tf.minimum(num_images_in_range, max_images_per_series)
                ridxs = tf.sort(tf.gather(idxs, tf.range(num_to_select)))
            return tf.gather(series_in_time_range, ridxs)
        
        def return_empty():
            return tf.zeros((0, num_features), dtype=tf.int32)
        
        rinput = tf.cond(
            tf.greater(num_images_in_range, 0),
            select_samples,
            return_empty
        )

    # 5. Ensure we have at least one row before accessing indices
    rinput_size = tf.shape(rinput)[0]
    
    def process_with_data():
        # 6. make days relative to the first observed day in the interval
        normalized_days = tf.clip_by_value(rinput[:, -1] - rinput[0, -1], 0, 365)
        
        # 7. (optional) concatenate relative days back to full data
        # norm_days = tf.concat((rinput[:, 0:-1], tf.expand_dims(normalized_days, -1)), axis=1)

        # 8. assign each observation to a time bucket based on relative day
        indices = normalized_days // days_per_bucket

        # 9. keep only the first image for each bucket
        def unique_with_inverse(x):
            y, idx = tf.unique(x)
            num_segments = tf.shape(y)[0]
            num_elems = tf.shape(x)[0]
            return (y, idx, tf.math.unsorted_segment_min(tf.range(num_elems), idx, num_segments))

        unique_indices, _, idxs = unique_with_inverse(indices)  # unique buckets + first index img per bucket

        # 10. gather selected images and their relative day
        normalized_unique_days = tf.gather(normalized_days, idxs)
        rinput_unique = tf.gather(rinput, idxs)
        norm_days = tf.concat((rinput_unique[:, 0:-1], tf.expand_dims(normalized_unique_days, -1)), axis=1)

        # 11. scatter values into final padded output shape
        padded_data = tf.scatter_nd(
            tf.reshape(unique_indices, (-1, 1)),  # destination bucket positions
            norm_days,                            # values to insert
            [max_images_per_series, num_features]
        )
        return padded_data
    
    def return_zero_padded():
        # Return zero-padded tensor if no data
        return tf.zeros((max_images_per_series, num_features), dtype=tf.int32)
    
    # Only process if we have at least one row
    padded_data = tf.cond(
        tf.greater(rinput_size, 0),
        process_with_data,
        return_zero_padded
    )

    return padded_data  # shape: (max_images_per_series, num_features)


def filter_double_croppings(row: dict) -> tf.Tensor:
    # row: dict with key 'CDL' of type tf.Tensor
    # return: tf.Tensor (bool)
    return ~tf.reduce_any(tf.strings.regex_full_match(row['CDL'], '.*Dbl.*'))

def filter_valid_time_series(row: dict) -> tf.Tensor:
    """
    Filter out invalid time series that have num_images <= 0 or missing data.
    """
    num_images = tf.cast(row['num_images'], tf.int32)
    # Ensure num_images > 0 and data fields are not empty
    has_valid_images = tf.greater(num_images, 0)
    has_bands = tf.greater(tf.size(row['bands']), 0)
    has_dates = tf.greater(tf.size(row['img_dates']), 0)
    
    is_valid = tf.logical_and(tf.logical_and(has_valid_images, has_bands), has_dates)
    
    # Debug: Print information about filtered rows (only first few to avoid spam)
    def print_debug_info():
        num_imgs_val = tf.cast(num_images, tf.int64).numpy() if tf.executing_eagerly() else None
        bands_size = tf.size(row['bands']).numpy() if tf.executing_eagerly() else None
        dates_size = tf.size(row['img_dates']).numpy() if tf.executing_eagerly() else None
        is_valid_val = is_valid.numpy() if tf.executing_eagerly() else None
        
        if not is_valid_val:
            print(f"⚠️ [FILTER] Filtered out invalid time series:")
            print(f"   - num_images: {num_imgs_val}")
            print(f"   - bands size: {bands_size}")
            print(f"   - dates size: {dates_size}")
            print(f"   - has_valid_images: {has_valid_images.numpy() if tf.executing_eagerly() else 'N/A'}")
            print(f"   - has_bands: {has_bands.numpy() if tf.executing_eagerly() else 'N/A'}")
            print(f"   - has_dates: {has_dates.numpy() if tf.executing_eagerly() else 'N/A'}")
        return is_valid
    
    # Only print in eager mode and limit prints
    if tf.executing_eagerly():
        try:
            if not is_valid.numpy():
                print_debug_info()
        except:
            pass
    
    return is_valid

def calculate_vegetation_indices(bands_decoded: tf.Tensor) -> tf.Tensor:
    """
    Calculates vegetation indices from Sentinel-2 band data.
    
    Parameters:
    -----------
    bands_decoded : tf.Tensor
        Tensor of shape (num_images, 12) containing the 12 Sentinel-2 bands.
        
    Returns:
    --------
    tf.Tensor
        Tensor of shape (num_images, 4) containing the calculated indices:
        [NDVI, EVI, NDWI, NDBI]
    """
    # Extract individual bands
    BLUE = tf.cast(bands_decoded[:, 1], tf.float32)
    GREEN = tf.cast(bands_decoded[:, 2], tf.float32)
    RED = tf.cast(bands_decoded[:, 3], tf.float32)
    NIR = tf.cast(bands_decoded[:, 7], tf.float32)
    SWIR1 = tf.cast(bands_decoded[:, 10], tf.float32)
    
    # Calculate vegetation indices
    NDVI = tf.reshape(tf.clip_by_value((NIR - RED) / (NIR + RED + 1e-6), -1.0, 1.0), shape=(-1, 1))
    EVI = tf.reshape(2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1.0 + 1e-6), shape=(-1, 1))
    NDWI = tf.reshape((NIR - SWIR1) / (NIR + SWIR1 + 1e-6), shape=(-1, 1))
    NDBI = tf.reshape((SWIR1 - NIR) / (SWIR1 + NIR + 1e-6), shape=(-1, 1))
    
    # Stack all indices
    indices_stack = tf.cast(tf.concat([NDVI, EVI, NDWI, NDBI], axis=1) * 100.0, tf.int32)
    
    return indices_stack

def compute_normalization_stats(
    train_files: list[str],
    label_legend: list[str],
    targeted_cultivated_crops_list: list[str],
    other_cultivated_crops_list: list[str],
    days_in_series: int = 120,
    days_per_bucket: int = 5,
    max_images_per_series: int = 25,
    frames_to_check: int = 3,
    bucketing_strategy: str = "random",
    batch_size: int = 1028,
    num_features: int = NUM_FEATURES
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute normalization parameters (means and standard deviations) from training data.
    
    This function loads training data and computes mean and standard deviation
    for normalization, similar to what's done in process.py. It's designed to be
    reusable across different modules.
    
    Parameters:
    -----------
    train_files : list[str]
        List of training file paths
    label_legend : list[str]
        List of class labels
    targeted_cultivated_crops_list : list[str]
        List of targeted crops
    other_cultivated_crops_list : list[str]
        List of other cultivated crops
    days_in_series : int, default=120
        Number of days in time series
    days_per_bucket : int, default=5
        Days per bucket
    max_images_per_series : int, default=25
        Maximum images per series
    frames_to_check : int, default=3
        Frames to check for vegetation
    bucketing_strategy : str, default="deterministic"
        Bucketing strategy for time window selection
    batch_size : int, default=1028
        Batch size for processing
        
    Returns:
    --------
    tuple
        (means, stds) tensors for normalization of shape (num_features,)
        When num_features=16, applies separate normalization strategy for bands vs indices
    """
    # Load training data
    print(f"📊 [DEBUG] Loading {len(train_files)} training files...")
    train_files_ds = make_from_pandas(train_files)
    
    # Create dataset for statistics computation
    print(f"📊 [DEBUG] Applying filters and parsing data...")
    non_normed_ds = (
        train_files_ds
        .filter(filter_double_croppings)
        .filter(filter_valid_time_series)
        .map(lambda x: parse(
            x,
            norm=False,
            means=tf.zeros([num_features], dtype=tf.float32),
            stds=tf.ones([num_features], dtype=tf.float32),
            label_legend_=label_legend,
            targeted_cultivated_crops_list=targeted_cultivated_crops_list,
            other_cultivated_crops_list=other_cultivated_crops_list,
            days_in_series=days_in_series,
            days_per_bucket=days_per_bucket,
            max_images_per_series=max_images_per_series,
            frames_to_check=frames_to_check,
            bucketing_strategy=bucketing_strategy,
            start_day=0,  # Use start_day=0 for consistent statistics
            num_features=num_features
        ), num_parallel_calls=1)  # Reduced to 1 to see errors sequentially
        .batch(batch_size)
    )
    
    # Collect data for statistics
    all_non_normalized_data = []
    processed_count = 0
    error_count = 0
    
    print(f"📊 [DEBUG] Starting to iterate through dataset...")
    for batch_idx, batch in enumerate(non_normed_ds):
        try:
            data, label = batch
            all_non_normalized_data.append(data)
            batch_size_actual = tf.shape(data)[0].numpy() if tf.executing_eagerly() else 1
            processed_count += batch_size_actual
            
            if batch_idx == 0:
                print(f"📊 [DEBUG] First batch shape: data={tf.shape(data).numpy() if tf.executing_eagerly() else 'N/A'}, label={tf.shape(label).numpy() if tf.executing_eagerly() else 'N/A'}")
            
            if (batch_idx + 1) % 10 == 0:
                print(f"📊 [DEBUG] Processed {batch_idx + 1} batches ({processed_count} samples so far)")
        except Exception as e:
            error_count += 1
            print(f"❌ [DEBUG] Error processing batch {batch_idx}: {str(e)}")
            import traceback
            print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
            if error_count > 5:
                print(f"❌ [DEBUG] Too many errors ({error_count}), stopping iteration")
                break
    
    print(f"📊 [DEBUG] Finished iteration: {processed_count} samples processed, {error_count} errors")
    
    # Reshape for statistics
    all_non_normalized_data = tf.concat(all_non_normalized_data, axis=0)
    all_non_normalized_data = tf.reshape(all_non_normalized_data, shape=(-1, tf.shape(all_non_normalized_data)[-1]))
    
    if num_features == NUM_SENTINEL_BANDS:
        # For 12 features (bands only): use standard normalization
        all_non_normalized_data = all_non_normalized_data[:, :num_features]
        mask = tf.not_equal(all_non_normalized_data, 0)
        means = tf.math.reduce_mean(tf.ragged.boolean_mask(all_non_normalized_data, mask), axis=0)
        stds = tf.math.reduce_std(tf.ragged.boolean_mask(all_non_normalized_data, mask), axis=0)
    else:
        # For 16 features (bands + indices): apply separate normalization strategy
        bands_data = all_non_normalized_data[:, :NUM_SENTINEL_BANDS]  # First 12 columns (bands)
        indices_data = all_non_normalized_data[:, NUM_SENTINEL_BANDS:NUM_SENTINEL_BANDS+NUM_VEGETATION_INDICES]  # Columns 13-16 (indices)
        
        # Compute means and stds for bands (excluding padded zeros)
        bands_mask = tf.not_equal(bands_data, 0)
        bands_means = tf.math.reduce_mean(tf.ragged.boolean_mask(bands_data, bands_mask), axis=0)
        bands_stds = tf.math.reduce_std(tf.ragged.boolean_mask(bands_data, bands_mask), axis=0)
        
        # Compute means and stds for indices (excluding padded zeros)
        indices_mask = tf.not_equal(indices_data, 0)
        indices_means = tf.math.reduce_mean(tf.ragged.boolean_mask(indices_data, indices_mask), axis=0)
        indices_stds = tf.math.reduce_std(tf.ragged.boolean_mask(indices_data, indices_mask), axis=0)
        
        # Combine into single tensors
        means = tf.concat([bands_means, indices_means], axis=0)
        stds = tf.concat([bands_stds, indices_stds], axis=0)
    
    return means, stds


def test_parser(
    row: dict,
    norm: bool,
    start_day: int,
    means: tf.Tensor,
    stds: tf.Tensor,
    label_legend: list[str],
    targeted_cultivated_crops_list: list[str],
    other_cultivated_crops_list: list[str],
    days_in_series: int = 120,
    days_per_bucket: int = 5,
    max_images_per_series: int = 25,
    frames_to_check: int = 3,
    bucketing_strategy: str = "random",
    num_features: int = NUM_FEATURES
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Test parser that uses parse() with a specific start_day for testing.
    
    This function is a wrapper around parse() that adds the start_day parameter
    and returns additional metadata (lon, lat, CDL) needed for testing.
    
    Parameters:
    -----------
    row : dict
        Row from the dataset containing encoded data
    norm : bool
        Whether to normalize the data
    start_day : int
        Starting day for the time window (required for testing)
    means : tf.Tensor
        Mean values for normalization (num_features features)
    stds : tf.Tensor
        Standard deviation values for normalization
    label_legend : list[str]
        List of class labels
    targeted_cultivated_crops_list : list[str]
        List of targeted crops
    other_cultivated_crops_list : list[str]
        List of other cultivated crops
    days_in_series : int, default=120
        Number of days in the time series
    days_per_bucket : int, default=5
        Days per bucket for temporal aggregation
    max_images_per_series : int, default=25
        Maximum number of images per series
    frames_to_check : int, default=3
        Number of frames to check for vegetation
    bucketing_strategy : str, default="deterministic"
        Bucketing strategy (ignored when start_day is provided)
        
    Returns:
    --------
    tuple
        (X, y, lon, lat, CDL) where:
        - X: normalized features tensor
        - y: one-hot encoded labels
        - lon: longitude
        - lat: latitude  
        - CDL: raw CDL label
    """
    # Use parse() with the specific start_day
    X, y = parse(
        row=row,
        norm=norm,
        means=means,
        stds=stds,
        label_legend_=label_legend,
        targeted_cultivated_crops_list=targeted_cultivated_crops_list,
        other_cultivated_crops_list=other_cultivated_crops_list,
        days_in_series=days_in_series,
        days_per_bucket=days_per_bucket,
        max_images_per_series=max_images_per_series,
        frames_to_check=frames_to_check,
        bucketing_strategy=bucketing_strategy,
        start_day=start_day,
        num_features=num_features
    )
    
    # Return additional metadata needed for testing
    return X, y, row['lon'], row['lat'], row['CDL']
