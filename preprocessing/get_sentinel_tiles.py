import os
import time
import multiprocessing
from datetime import timedelta
from datasources import (
    query_stac_api, 
    list_folders_second_to_deepest_level, 
    get_existing_data, 
    unique_indices, 
    process_result
)

'''
Sentinel-2 STAC scene refers to a specific satellite acquisition on a given date, 
covering a Sentinel-2 tile (a fixed 100 km × 100 km spatial grid).

Each scene has an ID like S2A_15SYV_20191127_1_L2A, which includes:

* S2A/S2B: the satellite (Sentinel-2A or Sentinel-2B),
* 15SYV: the Sentinel-2 tile (in this case, the same tile for all your scenes),
* 20191127: the date (November 27, 2019),
* 1_L2A: processing level (L2A = surface reflectance with atmospheric correction).

Note: If the CDL clipping applies to a single bounding box (bbox), only one Sentinel-2 tile is expected.
When using multiple bboxes, scenes may cover multiple tiles depending on geographic extent.
'''

# Determine mode from environment
MODE = os.environ.get("MODE", "unique")

# Define paths based on mode
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

if MODE == "multiple":
    CDL_path = os.path.join(data_dir, "valtrain", "CDL_multiple_scene.parquet/")
    s2_file_path = os.path.join(data_dir, "valtrain", "s2_multiple_scene.parquet/")
else:
    CDL_path = os.path.join(data_dir, "test", "CDL_unique_scene.parquet/")
    s2_file_path = os.path.join(data_dir, "test", "s2_unique_scene.parquet/")

# Load existing Sentinel-2 data
existing_s2_dates = get_existing_data(s2_file_path)
lock = multiprocessing.Lock()

# Read partition folders
folder_paths = list_folders_second_to_deepest_level(CDL_path, [], 0, 2)
print(f"🔍 [DEBUG] Found {len(folder_paths)} CDL partition folders")

# Check which bboxes are already complete in s2_multiple_scene.parquet
def is_bbox_complete(bbox, s2_base_path, configured_years):
    bbox_path = os.path.join(s2_base_path, f"bbox={bbox}")
    if not os.path.exists(bbox_path):
        return False
    
    # Get existing years for this bbox
    existing_years = set()
    for year_dir in os.listdir(bbox_path):
        if year_dir.startswith('year='):
            year = year_dir.replace('year=', '')
            existing_years.add(year)
    
    # Only complete if it has ALL configured years
    return configured_years.issubset(existing_years)

completed_bboxes = set()
if os.path.exists(s2_file_path):
    try:
        # Get configured years from environment
        configured_years = set(os.environ.get('YEAR_LIST', '').split(','))
        print(f"🔍 [DEBUG] Configured years: {configured_years}")
        
        s2_folders = list_folders_second_to_deepest_level(s2_file_path, [], 0, 2)
        potential_bboxes = set()
        
        # First, collect all potential bboxes
        for path in s2_folders:
            segments = path.split(os.sep)
            for segment in segments:
                if segment.startswith('bbox='):
                    bbox_value = segment.split('=')[1]
                    potential_bboxes.add(bbox_value)
        
        # Then check which ones are actually complete
        for bbox in potential_bboxes:
            if is_bbox_complete(bbox, s2_file_path, configured_years):
                completed_bboxes.add(bbox)
            else:
                print(f"⚠️ [DEBUG] Bbox {bbox} is incomplete - missing some years")
        
        print(f"🔒 [DEBUG] Found {len(completed_bboxes)} completed bboxes in s2_multiple_scene.parquet: {completed_bboxes}")
    except Exception as e:
        print(f"⚠️ [WARNING] Could not read existing s2 data: {e}")

df_train_partition_values_list = []
for path in folder_paths:
    segments = path.split(os.sep)
    partition_values = {}
    for segment in segments:
        if '=' in segment:
            key, value = segment.split('=')
            partition_values[key] = value
    
    # Skip if this bbox is already complete
    bbox_value = partition_values.get('bbox', '')
    if bbox_value in completed_bboxes:
        print(f"⏭️ [SKIP] Bbox {bbox_value} already complete - skipping")
        continue
    
    df_train_partition_values_list.append(partition_values)
    print(f"📁 [DEBUG] Added partition: {partition_values}")

print(f"🔍 [DEBUG] Total partitions to process: {len(df_train_partition_values_list)}")

assets_list = ['scl', 'coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'nir09', 'swir16', 'swir22']
scl_exclude_list = [0, 1, 7, 8, 9, 11]

start_time = time.time()
successful_scenes = 0
failed_scenes = 0

print(f"🔍 [DEBUG] Total bboxes to process: {len(df_train_partition_values_list)}")
for i, el in enumerate(df_train_partition_values_list):
    bbox = el['bbox']
    year = el['year']
    print(f"🎯 [DEBUG] Processing bbox {i+1}/{len(df_train_partition_values_list)}: {bbox} (year: {year})")
    CDL_parts_path = os.path.join(CDL_path, f"bbox={bbox}", f"year={year}")

    try:
        print(f"[INFO] Trying delimiter ', ': {bbox}")
        bbox_tuple = tuple([int(x) for x in bbox.split(', ')])
        print("[OK] Parsed bbox using delimiter ', '")
    except ValueError:
        print(f"[INFO] Trying delimiter ',': {bbox}")
        bbox_tuple = tuple([int(x) for x in bbox.split(',')])
        print("[OK] Parsed bbox using delimiter ','")

    print(f"🔍 [DEBUG] Querying STAC API for bbox {bbox} year {year}")
    results = query_stac_api(
        bounds=bbox_tuple,
        epsg4326=False,
        start_date=f"{year}-01-01T00:00:00Z",
        end_date=f"{year}-12-31T23:59:59Z"
    )

    results = unique_indices(results)
    print(f"🎯 [DEBUG] STAC API returned {len(results)} scenes for bbox {bbox} year {year}")
    if len(results) == 0:
        print(f"⚠️ [WARNING] No scenes found for bbox {bbox} year {year} - skipping")
        continue

    def process_results_in_parallel(result):
        status = process_result(result, existing_s2_dates, CDL_parts_path,
                                assets_list, scl_exclude_list,
                                s2_file_path, bbox, year, lock)
        return status

    use_mp = True
    if use_mp:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1) as pool:
            statuses = pool.map(process_results_in_parallel, results)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            statuses = list(executor.map(process_results_in_parallel, results))

    successful_scenes += sum(1 for s in statuses if s == 0)
    failed_scenes += sum(1 for s in statuses if s != 0)
    print(f"✅ [DEBUG] Completed bbox {i+1}/{len(df_train_partition_values_list)}: {bbox} (year: {year}) - Success: {sum(1 for s in statuses if s == 0)}, Failed: {sum(1 for s in statuses if s != 0)}")
    del results

total_time = timedelta(seconds=int(time.time() - start_time))

print("\n✅ Done.")
print(f"⏱️ Total time: {total_time}")
print(f"📸 Scenes processed: {successful_scenes}")
print(f"❌ Scenes failed: {failed_scenes}")