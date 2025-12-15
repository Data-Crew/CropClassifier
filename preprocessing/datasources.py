from typing import Optional
import pandas as pd
import numpy as np
import rasterio
from collections import Counter

import requests
import xml.etree.ElementTree as ET
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

import pyproj
import random

from spark_session import spark
from pyspark.sql import functions as F

from geojson import Feature, Polygon

import time

###########################
##### CDL ACQUISITION #####
###########################
'''
1. GETTING DATA SECTION
'''
# match rgb color mapping schemes on CDL web interface
def map_cdl_codes_to_rgb_and_text() -> tuple[dict[int, tuple[int, int, int]], dict[int, str], pd.DataFrame]:
    """
    Retrieves and processes CDL class codes, RGB color mappings, and class names from an online Excel file.

    Returns:
    - tuple:
        - code_to_rgb_mapping (dict[int, tuple[int, int, int]]): Dictionary mapping CDL class codes to RGB color tuples (R, G, B).
        - code_to_text_mapping (dict[int, str]): Dictionary mapping CDL class codes to their corresponding class names.
        - df (pd.DataFrame): Raw DataFrame containing all CDL code information from the Excel file.
    """
    # Download the Excel file and read it into a pandas DataFrame
    excel_url = "https://www.nass.usda.gov/Research_and_Science/Cropland/docs/CDL_codes_names_colors.xlsx"
    df = pd.read_excel(excel_url, skiprows=3)
    # Filter rows with non-missing class names
    valid_rows = df.dropna(subset=['Class_Names'])

    # Create a dictionary to map codes to RGB values for valid rows
    code_to_rgb_mapping = dict(zip(valid_rows['Codes'], zip(valid_rows['ESRI_Red'], valid_rows['ESRI_Green'], valid_rows['ESRI_Blue'])))

    # Create a dictionary to map codes to class names for valid rows
    code_to_text_mapping = dict(zip(valid_rows['Codes'], valid_rows['Class_Names']))

    return code_to_rgb_mapping, code_to_text_mapping, df

def CDL_clip_retrieve(
    bbox: str = "130783,2203171,153923,2217961",
    year: int = 2018,
    local_path: Optional[str] = None
) -> bytes:
    """
    Retrieves a CDL GeoTIFF file either from an API or from local storage.

    Parameters:
    - bbox (str, optional): Bounding box coordinates in the format "xmin,ymin,xmax,ymax" (default: "130783,2203171,153923,2217961").
    - year (int, optional): Year of the CDL dataset (default: 2018).
    - local_path (Optional[str], optional): Directory path to retrieve the file locally if the API request fails (default: None).

    Returns:
    - bytes: The GeoTIFF file content as raw bytes.

    Raises:
    - FileNotFoundError: If the file cannot be retrieved from either the API or local storage.
    - Exception: If an error occurs during the API request.
    """


    url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
    params = {"year": year, "bbox": bbox}

    try:
        # Attempt to retrieve the file from the API
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200 and "<returnURL>" in response.text:
            root = ET.fromstring(response.content)
            cdl_file_url = root.find(".//returnURL").text

            tif_response = requests.get(cdl_file_url, timeout=30)
            if tif_response.status_code == 200:
                print("✅ File successfully downloaded from the API.")
                return tif_response.content  # Return GeoTIFF bytes

        print("⚠️ API unavailable or no data found, attempting to use local file...")

    except Exception as e:
        print(f"❌ API error: {e}")

    # Determine resolution based on year (2024 is 10m, all others are 30m)
    resolution = "10m" if year == 2024 else "30m"

    # Attempt to load the file from local storage if the API fails
    if local_path:
        local_file = os.path.join(local_path, f"{year}_{resolution}_cdls.tif")
        if os.path.exists(local_file):
            print(f"📂 Loading local file: {local_file}")
            with open(local_file, "rb") as f:
                return f.read()
        else:
            print(f"⚠️ Local file not found: {local_file}")

    raise FileNotFoundError("❌ Could not retrieve CDL file from either the API or local storage.")

def replace_matrix_elements(
    matrix: np.ndarray,
    color_dict: dict[int, tuple[int, int, int]],
    text_dict: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replaces numerical class labels in a raster matrix with RGB values and text descriptions.

    Parameters:
    - matrix (np.ndarray): Input matrix containing numerical class labels.
    - color_dict (dict[int, tuple[int, int, int]]): Mapping of class labels to RGB color tuples.
    - text_dict (dict[int, str]): Mapping of class labels to text descriptions.

    Returns:
    - tuple[np.ndarray, np.ndarray]: 
        - new_matrix (np.ndarray): Matrix where class labels are replaced with corresponding RGB values.
        - new_text_matrix (np.ndarray): Matrix where class labels are replaced with corresponding text descriptions.
    """
    rows, cols = matrix.shape
    new_matrix = np.zeros((rows, cols, len(next(iter(color_dict.values())))), dtype=int)
    new_text_matrix = np.empty((rows, cols), dtype=object)

    
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] in color_dict:
                new_matrix[i, j] = color_dict[matrix[i, j]]
                new_text_matrix[i, j] = text_dict[matrix[i, j]]
    return new_matrix, new_text_matrix


''' 2. SAMPLING SECTION
Taking all the data in our AOIs from the CDL will be very large and take a long time to process. 
We must downsample it to work with a subset that is sufficient for model building and testing. 
This function supports that.
'''
def sample_raster_data(tif_bytes: bytes, interval: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts a downsampled raster and corresponding geographic coordinates.

    Parameters:
    - tif_bytes (bytes): Raster image in TIFF format as a byte stream.
    - interval (int): Sampling interval (default is 3), selecting one pixel every 'interval' pixels.

    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray]: 
        - sampled_data (np.ndarray): Downsampled raster values.
        - longitude_matrix (np.ndarray): Longitudes of sampled pixels.
        - latitude_matrix (np.ndarray): Latitudes of sampled pixels.
    """
    with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(1)

            # Sample the grid at the center of each interval
            # start = interval // 2
            start = random.randint(0,interval-1)
            sampled_data = data[start::interval, start::interval]

            # Get the coordinates of the sampled pixels
            rows, cols = np.mgrid[start:data.shape[0]:interval, start:data.shape[1]:interval]
            x, y = np.array(dataset.xy(rows, cols))

            # Create a coordinate transformer from the dataset CRS to EPSG:4326
            transformer = pyproj.Transformer.from_crs(dataset.crs, rasterio.CRS.from_epsg(4326), always_xy=True)

            # Convert the coordinates to longitude and latitude
            longitudes, latitudes = transformer.transform(x.ravel(), y.ravel())
            longitude_matrix, latitude_matrix = longitudes.reshape(x.shape), latitudes.reshape(y.shape)           
            return sampled_data, longitude_matrix, latitude_matrix
        
def create_matrix_from_sampled_data(sampled_data: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """
    Expands downsampled raster data back into its original shape, keeping sampled pixels at their approximate positions.

    Parameters:
    - sampled_data (np.ndarray): Downsampled raster values.
    - original_shape (tuple[int, int]): Shape of the original raster (rows, cols).

    Returns:
    - np.ndarray: Sparse raster with sampled values placed at corresponding positions.
    """
    new_data = np.zeros(original_shape, dtype=sampled_data.dtype)
    
    # Calculate the interval from the shape of the original raster and sampled_data
    row_interval = round(original_shape[0] / sampled_data.shape[0])
    col_interval = round(original_shape[1] / sampled_data.shape[1])
    
    start = row_interval // 2

    for i in range(sampled_data.shape[0]):
        for j in range(sampled_data.shape[1]):
            row = start + i * row_interval
            col = start + j * col_interval
            # Check if row and col are within the bounds of new_data
            if row < new_data.shape[0] and col < new_data.shape[1]:
                new_data[row, col] = sampled_data[i, j]
    
    return new_data


'''
3. STATS TABLES SECTION
'''
def get_crop_stats_from_AOI(
    tif_bytes: bytes,
    code_to_rgb_mapping_dict: dict[int, tuple[int, int, int]],
    code_to_text_mapping_dict: dict[int, str]
) -> tuple[dict[str, int], dict[str, float], tuple[int, int]]:
    """
    Extracts crop classification statistics from a raster dataset and calculates class distribution.

    Parameters:
    - tif_bytes (bytes): GeoTIFF file content in byte format.
    - code_to_rgb_mapping_dict (dict[int, tuple[int, int, int]]): Mapping of class codes to RGB color values.
    - code_to_text_mapping_dict (dict[int, str]): Mapping of class codes to their corresponding text labels.

    Returns:
    - tuple:
        - dict[str, int]: Dictionary with class labels as keys and pixel counts as values.
        - dict[str, float]: Dictionary with class labels as keys and their percentage representation.
        - tuple[int, int]: Shape of the raster dataset (rows, cols).
    """
    with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(1)
            _, text_matrix = replace_matrix_elements(data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)

            fig = go.Figure()

            return *plot_counts_and_percents(text_matrix, fig, filter_percent=1, ret_values_only=True), data.shape

# Create table of % area of each land use/crop classification for any class with >=1% representation, for all AOIs
def display_table(list_of_lists):
    data = []
    for index, item in enumerate(list_of_lists):
        #dictionary = item[0] #counts
        dictionary = item[1] #percents
        size = item[2]
        year = item[3]
        dictionary_int = {key: int(value) for key, value in dictionary.items()}  # Cast dictionary values to integers
        row = {'Index': index, 'Size': size, 'Year': year}
        row.update(dictionary_int)  # Add the dictionary keys and their integer values as separate columns
        data.append(row)

    df = pd.DataFrame(data)
    return df

'''
4. PLOTTING UTILS 
'''
def plot_counts_and_percents(
    text_matrix: np.ndarray,
    fig: go.Figure,
    row: int = 1,
    col: int = 1,
    filter_percent: float = 1.0,
    ret_values_only: bool = False
) -> None | tuple[dict[str, int], dict[str, float]]:
    """
    Plots the class distribution from a raster dataset using a bar chart with class-specific colors.

    Parameters:
    - text_matrix (np.ndarray): Matrix containing class labels for each pixel.
    - fig (go.Figure): Plotly figure object where the bar chart will be added.
    - row (int, optional): Row position of the subplot in the figure (default: 1).
    - col (int, optional): Column position of the subplot in the figure (default: 1).
    - filter_percent (float, optional): Minimum percentage threshold for a class to be included in the plot (default: 1.0).
    - ret_values_only (bool, optional): If True, returns class counts and percentages instead of plotting (default: False).

    Returns:
    - None: If `ret_values_only` is False, the function updates `fig` with the bar plot.
    - tuple[dict[str, int], dict[str, float]]: If `ret_values_only` is True, returns:
        - filtered_value_counts (dict[str, int]): Dictionary with class names as keys and pixel counts as values.
        - filtered_value_percentages (dict[str, float]): Dictionary with class names as keys and pixel percentages as values.
    """
    flat_text_matrix = np.array(text_matrix).flatten().tolist()
    value_counts = Counter(flat_text_matrix)

    total_values = sum(value_counts.values())
    value_percentages = {key: (count / total_values) * 100 for key, count in value_counts.items()}

    filtered_value_counts = {key: val for key, val in value_counts.items() if value_percentages[key] >= filter_percent}
    filtered_value_percentages = {key: val for key, val in value_percentages.items() if val >= filter_percent}

    if ret_values_only:
        return filtered_value_counts, filtered_value_percentages
    
    # Convert class names to numerical codes for color mapping
    code_to_rgb_mapping_dict, code_to_text_mapping_dict, _ = map_cdl_codes_to_rgb_and_text()
    text_to_code = {v: k for k, v in code_to_text_mapping_dict.items()}  # Reverse mapping
    filtered_codes = [text_to_code[key] for key in filtered_value_counts.keys() if key in text_to_code]

    # Get RGB colors for each class
    colors = []
    for class_code in filtered_codes:
        if class_code in code_to_rgb_mapping_dict:
            rgb = code_to_rgb_mapping_dict[class_code]  # Get RGB tuple
            colors.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
        else:
            print(f"⚠️ Warning: No color found for class {class_code}. Assigning default black.")
            colors.append("rgb(0,0,0)")  # Assign black if no color is found

    # Add bar plot to the figure
    fig.add_trace(
        go.Bar(
            x=list(filtered_value_counts.keys()),
            y=list(filtered_value_counts.values()),
            text=[f'{v:.2f}%' for v in filtered_value_percentages.values()],
            textposition='auto',
            marker=dict(color=colors),  # Use assigned colors
            hovertemplate='Class: %{x}<br>Count: %{y}<br>Percentage: %{text}',
            name='Class Counts'
        ),
        row=row, col=col
    )

# Plot function to verify subsampling worked as intended
def plot_CDL_clipped_img(
    data: np.ndarray,
    rgb_image: np.ndarray,
    text_matrix: np.ndarray,
    fig: go.Figure,
    row: int = 1,
    col: int = 2
) -> None:
    """
    Plots a CDL raster image with its class distribution.

    Parameters:
    - data (np.ndarray): Raster data array representing CDL classifications.
    - rgb_image (np.ndarray): RGB image corresponding to CDL classes.
    - text_matrix (np.ndarray): Matrix containing class names for each pixel.
    - fig (go.Figure): Plotly figure object to embed the subplot.
    - row (int, optional): Row position of the subplot in the figure (default: 1).
    - col (int, optional): Column position of the subplot in the figure (default: 2).

    Returns:
    - None
    """

    fig.add_trace(go.Image(z=rgb_image), row=row, col=col)
    fig.add_trace(go.Heatmap(z=data, text=text_matrix, hoverinfo='text', opacity=0, showscale=False), row=row, col=col)


def read_bytes_plot_clipped_CDL_image(
    tif_bytes: bytes,
    window_size: int = 800,
    plot_counts_and_percents_only: bool = False,
    debug_window: bool = False,
    sample_region: bool = False
) -> None:
    """
    Reads raw GeoTIFF bytes, extracts crop classifications, and plots statistics.

    Parameters:
    - tif_bytes (bytes): GeoTIFF file content in byte format.
    - window_size (int, optional): Size of the sampled region (default: 800).
                                   If None, the entire dataset is read.
    - plot_counts_and_percents_only (bool, optional): If True, only plots the class distribution (default: False).
    - debug_window (bool, optional): If True, prints debug information about the sampled region (default: False).
    - sample_region (bool, optional): If True, selects a random region within the bounding box (default: False).

    Returns:
    - None: The function modifies the Plotly figure and displays plots.
    """
    with rasterio.io.MemoryFile(BytesIO(tif_bytes)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(1)  # Read entire dataset

            if window_size:
                if sample_region:
                    # Randomly sample a region
                    h, w = data.shape
                    x_start = np.random.randint(0, max(1, w - window_size))
                    y_start = np.random.randint(0, max(1, h - window_size))

                    print(f"🎯 Sampling region within bbox: X {x_start}-{x_start + window_size}, Y {y_start}-{y_start + window_size}")
                    data = data[y_start:y_start + window_size, x_start:x_start + window_size]
                else:
                    print("🔲 Limiting adjacent window size")
                    data = data[:window_size, :window_size] #limit size
            else:
                print("📂 Reading the entire dataset.")

            if debug_window:
                # Print unique class values before mapping
                unique_values, counts = np.unique(data, return_counts=True)
                print("🔍 Unique values in sampled region:")
                for value, count in zip(unique_values, counts):
                    print(f"Class {value}: {count} pixels")

            # Convert to RGB and text
            code_to_rgb_mapping_dict, code_to_text_mapping_dict, _ = map_cdl_codes_to_rgb_and_text()
            rgb_image, text_matrix = replace_matrix_elements(data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)

            # Create subplots with continuous axes
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Class Distribution", "CDL Image"], shared_yaxes=False)

            # Plot class distribution with RGB colors
            plot_counts_and_percents(text_matrix, fig, row=1, col=1)

            if not plot_counts_and_percents_only:
                plot_CDL_clipped_img(data, rgb_image, text_matrix, fig, row=1, col=2)

            # Show both plots together
            fig.update_layout(title="CDL Data Analysis", width=1200, height=600)
            fig.show()

'''
5. WRITTING DATA OUT
'''

def sample_data_into_dataframe_write_parquet(
    tif_bytes: bytes,
    bbox: str,
    year: int,
    filename: str = "CDL_samples.parquet",
    interval: int = 10
) -> None:
    """
    Extracts raster data, converts it into a PySpark DataFrame, and saves it in Parquet format within the 'data/' directory.

    Parameters:
    - tif_bytes (bytes): Raster image in TIFF format as a byte stream.
    - bbox (str): Bounding box representing the spatial extent of the data.
    - year (int): Year associated with the raster dataset.
    - filename (str, optional): Name of the Parquet file (default: "CDL_samples.parquet").
    - interval (int, optional): Sampling interval to reduce data density (default: 10).

    Returns:
    - None: The function processes and saves the data without returning a value.

    Side Effects:
    - Saves a partitioned Parquet file in the 'data/' directory, using "bbox" and "year" as partition keys.
    - Prints a confirmation message with the file path.

    Raises:
    - FileNotFoundError: If the 'data/' directory does not exist.
    """

    # Define path to data dir
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Use the filename as the full path if it's already a complete path
    if os.path.isabs(filename) or filename.startswith(('data/', '../data/', './data/')):
        file_path = filename
        # Create directory if it doesn't exist
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    else:
        # Create if it doesn't exist
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        # Construct full file path inside "data/" directory    
        file_path = os.path.join(DATA_DIR, filename)

    # Retrieve CDL color and text mappings
    code_to_rgb_mapping_dict, code_to_text_mapping_dict, _ = map_cdl_codes_to_rgb_and_text()

    # Perform raster subsampling
    sampled_data, longitudes, latitudes = sample_raster_data(tif_bytes, interval=interval)
    _, text_matrix = replace_matrix_elements(sampled_data, code_to_rgb_mapping_dict, code_to_text_mapping_dict)

    # Create a Pandas DataFrame
    pandas_df = pd.DataFrame({
        'CDL': text_matrix.flatten(), 
        'lon': longitudes.flatten(), 
        'lat': latitudes.flatten()
    })

    # Convert Pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)

    # Add additional columns
    spark_df = spark_df.withColumn('bbox', F.lit(bbox)).withColumn('year', F.lit(year))

    # Define columns to be saved
    columns_to_write = ['bbox', 'year', 'CDL', 'lon', 'lat']
    
    # Save as a Parquet file inside "data/" directory
    spark_df.select(columns_to_write).write.partitionBy("bbox", "year").mode("append").parquet(file_path)

    print(f"✅ File saved at: {file_path}")

################################
##### SENTINEL ACQUISITION #####
################################
import os
import gc
import random
import numpy as np
import pandas as pd
import requests
import rasterio
import pyproj
from io import BytesIO
from datetime import datetime
from contextlib import closing
import multiprocessing
from spark_session import spark
'''
1. GETTING DATA SECTION
'''
def get_transformed_bounds(
    bounds: tuple[float, float, float, float] = (426362, 1405686, 520508, 1432630),
    src_epsg: str = "EPSG:5070",
    dst_epsg: str = "EPSG:4326"
) -> tuple[float, float, float, float]:
    """
    Transforms bounding box coordinates from a source EPSG projection to a target EPSG projection.

    Parameters:
    - bounds (tuple, optional): Bounding box coordinates (min_x, min_y, max_x, max_y) in the source CRS.
                                Default: (426362, 1405686, 520508, 1432630).
    - src_epsg (str, optional): EPSG code of the source coordinate reference system. Default: "EPSG:5070".
    - dst_epsg (str, optional): EPSG code of the target coordinate reference system. Default: "EPSG:4326".

    Returns:
    - tuple: Transformed bounding box (min_longitude, min_latitude, max_longitude, max_latitude) in the target CRS.

    Raises:
    - pyproj.exceptions.CRSError: If the provided EPSG codes are invalid.
    """

    # Define the coordinate reference systems
    src_proj = pyproj.CRS(src_epsg)
    dst_proj = pyproj.CRS(dst_epsg)

    # Create a transformer for coordinate conversion
    transformer = pyproj.Transformer.from_proj(src_proj, dst_proj, always_xy=True)

    # Transform the bounds
    min_lon, min_lat = transformer.transform(bounds[0], bounds[1])
    max_lon, max_lat = transformer.transform(bounds[2], bounds[3])

    return min_lon, min_lat, max_lon, max_lat

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.io import MemoryFile


# Reutilizable: sesión con retry y timeout
def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['POST']),  # Para STAC
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def query_stac_api(
    bounds: tuple[float, float, float, float],
    epsg4326: bool = False,
    start_date: str = "2023-01-01T00:00:00Z",
    end_date: str = "2023-12-31T23:59:59Z",
    limit: int = 100,
    verbose: bool = True
) -> list:
    """
    Queries the Sentinel-2 STAC API for available imagery within a given area and timeframe.
    """
    
    # Convert bounds if needed
    if not epsg4326:
        min_lon, min_lat, max_lon, max_lat = get_transformed_bounds(bounds)
    else:
        min_lon, min_lat, max_lon, max_lat = bounds

    polygon = Feature(geometry=Polygon([[
        (min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat),
        (min_lon, max_lat), (min_lon, min_lat)
    ]]))

    stac_url = "https://earth-search.aws.element84.com/v1/search"
    all_results = []
    page = 1

    while True:
        query = {
            "datetime": f"{start_date}/{end_date}",
            "intersects": polygon.geometry,
            "collections": ["sentinel-2-l2a"],
            "limit": limit,
            "page": page
        }

        try:
            #response = requests.post(stac_url, json=query)
            response = requests_retry_session().post(stac_url, json=query, timeout=10)
            response.raise_for_status()
            results = response.json()

            if not results.get("features"):
                print(f"⚠️ No features returned on page {page}. Stopping pagination.")
                break
            
            features = results.get("features", [])

            if verbose:
                print(f"📦 Page {page} returned {len(features)} features.")

            if not features:
                break  # ✅ Detener el ciclo si no hay más resultados

            all_results.extend(features)
            page += 1

        except requests.exceptions.RequestException as e:
            print(f"❌ Error querying STAC API on page {page}: {e}")
            break

    return all_results


def list_folders_second_to_deepest_level(path: str, folders: list, current_depth: int, target_depth: int) -> list:
    """
    Lists folders that are at the second-to-deepest level within a given directory.

    Parameters:
    - path (str): Root directory path to search.
    - folders (list): Accumulative list of found folder paths.
    - current_depth (int): Current depth within the directory tree.
    - target_depth (int): Target depth (second-to-deepest level).

    Returns:
    - list: A list of folder paths found at the target depth.
    """
    if current_depth == target_depth:
        folders.append(path)
    else:
        try:
            subfolders = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            for folder in subfolders:
                list_folders_second_to_deepest_level(folder, folders, current_depth + 1, target_depth)
        except PermissionError:
            print(f"⚠️ Unable to access: {path}")

    return folders

def get_existing_data(file_path: str) -> dict:
    """
    Extracts existing Sentinel-2 scene dates from a partitioned Parquet directory structure.

    For each folder path at the specified depth, this function parses directory names 
    corresponding to 'bbox', 'year', 'tile', and 'scene_date' partition keys. It groups 
    available scene dates by unique (bbox, year, tile) combinations.

    Parameters:
    - file_path (str): Root path to the partitioned Parquet directory to inspect.

    Returns:
    - dict: Dictionary where keys are tuples of (bbox, year, tile), and values are 
            sorted lists of associated scene_date strings.
    """
    existing_s2_dates = {}

    try:
        for item in list_folders_second_to_deepest_level(file_path, [], 0, 4):
            parts = item.split('/')
            bbox = None
            year = None
            scene_date = None

            for part in parts:
                if part.startswith('bbox='):
                    bbox = part.split('=')[1]
                elif part.startswith('year='):
                    year = part.split('=')[1]
                elif part.startswith('tile='):
                    tile = part.split('=')[1]
                elif part.startswith('scene_date='):
                    scene_date = part.split('=')[1]

            if bbox and year and scene_date and tile:
                key = (bbox, year, tile)
                if key in existing_s2_dates:
                    existing_s2_dates[key].append(scene_date)
                else:
                    existing_s2_dates[key] = [scene_date]

        # Ordenar las fechas dentro de cada key
        for key in existing_s2_dates:
            existing_s2_dates[key].sort()

        return existing_s2_dates
    except:
        return {}


def get_directory_size(directory_path: str) -> int:
    """
    Recursively calculates the total size of all files in a directory.

    Parameters:
    - directory_path (str): Path to the directory.

    Returns:
    - int: Total size in bytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except FileNotFoundError:
                # Handle broken symlinks or deleted files during walk
                pass
    return total_size


def unique_indices(scene_ids: list, one_tile: bool = False) -> list:
    """
    Filters a list of scene metadata dictionaries to keep only the latest version for each base scene ID.

    For each scene, the base ID is inferred by removing the last two components (e.g., version suffix).
    Among scenes with the same base ID, only the one with the highest version number is retained.
    Optionally, the result can be limited to scenes from a single random tile.

    Parameters:
    - scene_ids (list): List of scene metadata dictionaries containing an 'id' field.
    - one_tile (bool): Whether to randomly select and keep scenes from a single tile only.

    Returns:
    - list: Filtered list of scene dictionaries.
    """
    scene_ids_ids = [x['id'] for x in scene_ids]
    unique_dict = {}

    for index, scene_id in enumerate(scene_ids_ids):
        base_id = scene_id.rsplit('_', 2)[0]
        number = int(scene_id.split('_')[-2])

        if base_id not in unique_dict:
            unique_dict[base_id] = {'index': index, 'number': number}
        elif number > unique_dict[base_id]['number']:
            unique_dict[base_id] = {'index': index, 'number': number}

    unique_indices_to_use = [item['index'] for item in unique_dict.values()]
    scene_ids = [scene_ids[ii] for ii in unique_indices_to_use]

    if one_tile:
        scene_ids_ids = [x['id'] for x in scene_ids]
        tiles = list(set([element.split('_')[1] for element in scene_ids_ids]))
        chosen_tile = random.choice(tiles)
        scene_ids = [scene_ids[index] for index, element in enumerate(scene_ids_ids) if chosen_tile in element]

    return scene_ids

def safe_download(url, timeout=60, retries=10):
    """
    Downloads a file using retry and timeout logic.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download: {url}")
        print(f"🔁 Error: {e}")
        return None


from rasterio.warp import transform

def sample_geotiff(x: pd.Series, y: pd.Series, geotiff_url: str) -> pd.Series:
    """
    Retrieves raster values from a remote GeoTIFF file at the given lat/lon locations,
    reprojecting from EPSG:4326 to the native CRS of the raster.
    """

    content = safe_download(geotiff_url, timeout=60, retries=10)

    if content is None:
        return pd.Series([np.nan] * len(x))

    try:
        with MemoryFile(content) as memfile:
            with memfile.open() as dataset:
                dst_crs = dataset.crs
                # Transform from EPSG:4326 to the raster's CRS
                xs, ys = transform("EPSG:4326", dst_crs, x.tolist(), y.tolist())
                coords = list(zip(xs, ys))
                values = [val[0] for val in dataset.sample(coords)]
                return pd.Series(values)
    except Exception as e:
        print(f"⚠️ Sampling failed for {geotiff_url}: {e}")
        return pd.Series([np.nan] * len(x))



def process_result(
    result: dict,
    existing_s2_dates: dict,
    CDL_parts_path: str,
    assets_list: list,
    scl_exclude_list: list,
    s2_file_path: str,
    bbox: str,
    year: str,
    lock: multiprocessing.Lock
) -> int:
    """
    Processes a single STAC scene: samples assets, filters training points, and writes to partitioned Parquet.

    If the scene is new (not in existing_s2_dates) and has enough valid area coverage, it:
    - Loads training points from `CDL_parts_path`
    - Samples values from all assets (GeoTIFFs) into the dataframe
    - Filters invalid pixels using SCL asset
    - Appends data to a Spark Parquet table partitioned by bbox, year, tile, and scene_date

    Parameters:
    - result (dict): STAC item metadata including ID, datetime, and assets.
    - existing_s2_dates (dict): Dict of processed scenes per (bbox, year, tile).
    - CDL_parts_path (str): Local path to training points for the given bbox/year.
    - assets_list (list): Names of assets (e.g., ['scl', 'nir', 'red']).
    - scl_exclude_list (list): Values to exclude from SCL asset (e.g., clouds).
    - s2_file_path (str): Output directory path for Parquet dataset.
    - bbox (str): Bbox string, used as a partition key.
    - year (str): Year string, used as a partition key.
    - lock (multiprocessing.Lock): Lock to synchronize concurrent writes.

    Returns:
    - int: Always returns 0 (used for multiprocessing stability).
    """
    props = result['properties']
    tile = result['id'].split('_')[1] + '_' + result['id'].split('_')[-2]

    try:
        not_these_tile_dates = existing_s2_dates[(bbox, year, tile)]
    except KeyError:
        not_these_tile_dates = []

    scene_date = props['datetime'].split('T')[0]
    if scene_date in not_these_tile_dates:
        return 0

    expected_keys = [
    's2:vegetation_percentage',
    's2:not_vegetated_percentage',
    's2:thin_cirrus_percentage',
    's2:cloud_shadow_percentage',
    's2:dark_features_percentage'
    ]

    # extracts key value or 0 if it doesnt exists
    valid_percent_area = sum(
    props.get(key, 0) for key in expected_keys
    )

    missing_keys = [key for key in expected_keys if key not in props]
    if missing_keys:
        print(f"⚠️ Worker {os.getpid()}: missing keys in scene {result['id']}: {missing_keys}")

    if valid_percent_area > 30:
        try:
            print(f"Started Worker ID: {os.getpid()}: {result['id']} at {datetime.now().strftime('%H:%M:%S')}")
            start_time = time.time()
            scene_id = result['id']
            print(f"🔄 [{os.getpid()}] Initializing scene: {scene_id} a las {datetime.now().strftime('%H:%M:%S')}")
            
            # Always create a new Spark session for each worker
            try:
                from pyspark.sql import SparkSession
                worker_spark = SparkSession.builder \
                    .appName(f"CDLClassifier_Worker_{os.getpid()}") \
                    .config("spark.driver.memory", "4g") \
                    .config("spark.executor.memory", "2g") \
                    .config("spark.driver.maxResultSize", "2g") \
                    .config("spark.sql.execution.arrow.enabled", "true") \
                    .config("spark.driver.host", "localhost") \
                    .config("spark.driver.bindAddress", "127.0.0.1") \
                    .config("spark.sql.adaptive.enabled", "false") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                    .getOrCreate()
                print(f"✅ [Worker {os.getpid()}] New Spark session initialized successfully")
            except Exception as spark_error:
                print(f"❌ [Worker {os.getpid()}] Failed to initialize Spark: {spark_error}")
                return 0
            
            t_load = time.time()
            df_train_subset = pd.read_parquet(CDL_parts_path)
            df_train_subset['bbox'] = bbox
            df_train_subset['year'] = year
            print(f"⏱️ [{scene_id}] Training points loaded in {time.time() - t_load:.2f} seconds")
            print(f"🧪 Loaded {len(df_train_subset)} training points from {CDL_parts_path}")
            print(f"🧪 First 3 rows:\n{df_train_subset.head(3)}")

            # --- bands sampling ---
            t_sample = time.time()
            nan_count = 0
            total_assets = len(assets_list)
            
            for ass in assets_list:
                geotiff_url = result['assets'][ass]['href']
                print(f"🎯 Sampling asset '{ass}' from: {geotiff_url}")
                
                try:
                    df_train_subset[ass] = sample_geotiff(df_train_subset["lon"], df_train_subset["lat"], geotiff_url)
                    print(f"✅ Sampled '{ass}' – first 5 values:\n{df_train_subset[ass].head().to_list()}")
                    
                    # Check for too many NaN values early
                    if df_train_subset[ass].isna().all():
                        nan_count += 1
                        print(f"⚠️ Asset '{ass}' is all NaN - likely timeout issues")
                        
                        # If more than 50% of assets are NaN, skip this scene
                        if nan_count > total_assets * 0.5:
                            print(f"🚫 [EARLY_SKIP] Scene {scene_id} has too many failed downloads ({nan_count}/{total_assets} assets are NaN) - skipping")
                            return 0
                            
                except Exception as e:
                    print(f"❌ Failed to sample asset '{ass}': {e}")
                    nan_count += 1
                    continue

                if ass == 'scl':
                    print(f"🧪 SCL unique values BEFORE filtering: {sorted(df_train_subset['scl'].unique())}")
                    print(f"🧪 SCL exclude list: {scl_exclude_list}")
                    print(f"🧪 SCL value counts: {df_train_subset['scl'].value_counts().to_dict()}")
                    df_train_subset = df_train_subset[~df_train_subset[ass].isin(scl_exclude_list)]
                    if df_train_subset.empty:
                        print("⚠️ All training points were excluded after SCL filtering.")

                df_train_subset.reset_index(drop=True, inplace=True)
            
            print(f"⏱️ [{scene_id}] Sampling completed in {time.time() - t_sample:.2f} seconds")

            print(f"🔍 [DEBUG] Scene {scene_id}: DataFrame shape before dropna: {df_train_subset.shape}")
            df_train_subset = df_train_subset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            print(f"🔍 [DEBUG] Scene {scene_id}: DataFrame shape after dropna: {df_train_subset.shape}")
            if df_train_subset.empty:
                print(f"⚠️ [WARNING] Scene {scene_id}: DataFrame is empty after dropna - skipping")
                return 0
            df_train_subset['scene_date'] = scene_date
            df_train_subset['tile'] = tile
            df_train_subset[assets_list] = df_train_subset[assets_list].astype('int32')

            print(f"🧪 Final sample preview for scene {scene_id} (before write):")
            print(df_train_subset[["lon", "lat", "scl"] + [b for b in assets_list if b != "scl"]].head())
            print(f"📏 Final shape: {df_train_subset.shape}")
            print(f"✅ Final value stats (non-null):")
            for b in assets_list:
                if b in df_train_subset.columns:
                    print(f"  • {b}: min={df_train_subset[b].min()}, max={df_train_subset[b].max()}, unique={df_train_subset[b].nunique()}")

            with lock:
                if df_train_subset.empty:
                    print(f"⚠️ [DEBUG] Skipping scene {result['id']} – DataFrame is empty after filtering.")
                    return 0

                temp_path = os.path.join("/tmp", f"s2_temp_{os.getpid()}.parquet")
                df_train_subset.to_parquet(temp_path, index=False)
                sdf = worker_spark.read.parquet(f"file://{temp_path}")
                
                print(f"💾 Writing to: {s2_file_path} for bbox={bbox}, year={year}, tile={tile}, scene_date={scene_date}")

                try:
                    sdf.write.mode("append").partitionBy("bbox", "year", "tile", "scene_date").parquet(s2_file_path)
                    print(f"✅ [SUCCESS] Successfully wrote scene {result['id']} to {s2_file_path}")
                    os.remove(temp_path)
                except Exception as write_error:
                    import traceback
                    print(f"❌ [WRITE_ERROR] Failed to write scene {result['id']}: {write_error}")
                    print(f"❌ [WRITE_ERROR] Write error type: {type(write_error).__name__}")
                    print(f"❌ [WRITE_ERROR] Write error traceback:")
                    traceback.print_exc()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return 0

            del df_train_subset
            del sdf
            worker_spark.catalog.clearCache()
            gc.collect()
            print(f"Finished Worker ID: {os.getpid()}: {result['id']} at {datetime.now().strftime('%H:%M:%S')}")
            print(f"✅ [{os.getpid()}] Terminó escena: {scene_id} en {time.time() - start_time:.2f} segundos")
            return 0

        except Exception as e:
            print(f"Exception: {e} - Worker {os.getpid()}: {result['id']}")
            import traceback
            print(f"❌ [ERROR] Exception in Worker {os.getpid()}: {result['id']}")
            print(f"❌ [ERROR] Exception type: {type(e).__name__}")
            print(f"❌ [ERROR] Exception message: {str(e)}")
            print(f"❌ [ERROR] Full traceback:")
            traceback.print_exc()
            return 0
    else:
        return 0
