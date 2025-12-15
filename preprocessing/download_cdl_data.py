import sys
import os
from datasources import (
    CDL_clip_retrieve, 
    map_cdl_codes_to_rgb_and_text, 
    sample_data_into_dataframe_write_parquet
    )

# Get the project root directory (one level up from preprocessing)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

'''
This script creates a table of "ground truth" values by clipping sections out from 
the cropland data layer and manipulating the rasters into tabular format with unique row identities
based on the CDL lat/lon location and year. 

See the great FAQ section at 
https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php for more information about the CDL.

Other relevant links:
API developer guide --> https://nassgeodata.gmu.edu/CropScape/devhelp/help.html
Draw boundaries and export them --> https://nassgeodata.gmu.edu/CropScape/
Use EPSG:5070 (CONUS Albers) for submitting bbox areas.
Manual download --> https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php
'''
# Load mappings (optional usage later)
map_cdl_codes_to_rgb_and_text()

# Check environment variables
BBOX = os.environ.get("BBOX")
YEAR = os.environ.get("YEAR")
BBOX_LIST = os.environ.get("BBOX_LIST")
YEAR_LIST = os.environ.get("YEAR_LIST")
print("DEBUG BBOX:", BBOX)
print("DEBUG YEAR:", YEAR)
print("DEBUG BBOX_LIST:", BBOX_LIST)
print("DEBUG YEAR_LIST:", YEAR_LIST)

if BBOX and YEAR:
    # Unique mode
    print(f"📦 Downloading unique bbox: {BBOX}, year: {YEAR}")
    tif_bytes_out = CDL_clip_retrieve(BBOX, int(YEAR))
    sample_data_into_dataframe_write_parquet(
    tif_bytes_out,
    BBOX,
    int(YEAR),
    filename=os.path.join(project_root, "data", "test", "CDL_unique_scene.parquet"),
    interval=1
    )

elif BBOX_LIST and YEAR_LIST:
    # Multiple mode
    bbox_list = BBOX_LIST.split('|')
    years = [int(y) for y in YEAR_LIST.split(',')]
    print(f"📦 Downloading multiple bboxes ({len(bbox_list)}), years: {years}")

    # Check which bbox/year combinations already exist
    cdl_output_path = os.path.join(project_root, "data", "valtrain", "CDL_multiple_scene.parquet")
    existing_combinations = set()
    if os.path.exists(cdl_output_path):
        try:
            for bbox_dir in os.listdir(cdl_output_path):
                if bbox_dir.startswith('bbox='):
                    bbox_value = bbox_dir.split('=')[1]
                    bbox_path = os.path.join(cdl_output_path, bbox_dir)
                    if os.path.isdir(bbox_path):
                        for year_dir in os.listdir(bbox_path):
                            if year_dir.startswith('year='):
                                year_value = year_dir.split('=')[1]
                                existing_combinations.add((bbox_value, year_value))
            print(f"🔒 [DEBUG] Found {len(existing_combinations)} existing CDL combinations: {existing_combinations}")
        except Exception as e:
            print(f"⚠️ [WARNING] Could not read existing CDL data: {e}")

    for year in years:
        for bbox in bbox_list:
            # Skip if this combination already exists
            if (bbox, str(year)) in existing_combinations:
                print(f"⏭️ [SKIP] CDL for bbox {bbox}, year {year} already exists - skipping")
                continue
                
            print(f"🔄 Processing bbox: {bbox}, year: {year}")
            tif_bytes_out = CDL_clip_retrieve(bbox, year)
            sample_data_into_dataframe_write_parquet(
            tif_bytes_out,
            bbox,
            year,
            filename=os.path.join(project_root, "data", "valtrain", "CDL_multiple_scene.parquet"),
            interval=15
            )

else:
    print("❌ No valid environment variables found. Please set BBOX and YEAR or BBOX_LIST and YEAR_LIST.")
    sys.exit(1)

print("✅ CDL data generation complete.")
