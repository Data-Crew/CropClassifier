'''
This module performs the following preprocessing steps to convert Sentinel-2 image data
into a format suitable for time series modeling and TensorFlow training:

1. Combine spectral bands into a dictionary-like structure indexed by scene date (create_image_map)
2. Group records by pixel coordinates and CDL class (group_time_series)
3. Chronologically sort observations and flatten band/date/tile/SCL data into string lists (flatten_time_series)
4. Convert all stringified lists into compact binary representations for efficient loading (convert_to_binary)
5. Run the full transformation pipeline and write partitioned output as Parquet (agg_to_time_series)

Examples:

🔹 Unique scene mode
python transform_input_data.py \
  --input ../data/test/s2_unique_scene.parquet/ \
  --output ../data/test/CDL_unique_scene_ts.parquet/ \
  --bbox "484932,1401912,489035,1405125" \
  --year 2019 \
  --write_parquet

🔸 Multiple scene mode
python transform_input_data.py \
  --input ../data/valtrain/s2_multiple_scene.parquet/ \
  --output ../data/valtrain/CDL_multiple_scene_ts.parquet/ \
  --bbox_list "426362,1405686,520508,1432630|390747,1195097,437820,1284288" \
  --year_list "2019,2020" \
  --write_parquet
'''
import os
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from datasources import get_existing_data
from spark_session import spark

# Step 1: Combine band values and map to scene date
def create_image_map(df: DataFrame) -> DataFrame:
    band_info_cols = [
        'coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3',
        'nir', 'nir08', 'nir09', 'swir16', 'swir22', 'scl', 'tile'
    ]
    return df.withColumn('band_values_and_labels', F.array(band_info_cols)) \
             .withColumn('image_map', F.create_map([F.col('scene_date'), F.col('band_values_and_labels')])) \
             .drop(*band_info_cols) \
             .drop('band_values_and_labels') \
             .drop('scene_date')

# Step 2: Group data by pixel and CDL into time series
def group_time_series(df: DataFrame) -> DataFrame:
    ts_ids = ['lon', 'lat', 'CDL']
    return df.groupBy(ts_ids) \
             .agg(F.collect_list('image_map').alias('image_dicts_list'),
                  F.count('image_map').alias('num_images'))

# Step 3: Sort scenes and flatten bands, tiles, dates, SCL
def get_sorted_input(dicts_list):
    sd = sorted(dicts_list, key=lambda x: [*x][0])
    only_nums, tiles, img_dates, scl_vals = [], [], [], []
    for item in sd:
        keyi = [*item][0]
        only_nums.extend(item.get(keyi)[:-2])
        scl_vals.append(item.get(keyi)[-2])
        tiles.append(item.get(keyi)[-1])
        img_dates.append(keyi.strftime('%Y-%m-%d'))

    return [
        ','.join(map(str, only_nums)),
        ','.join(map(str, tiles)),
        ','.join(img_dates),
        ','.join(map(str, scl_vals))
    ]

def flatten_time_series(df: DataFrame) -> DataFrame:
    df = df.withColumn('inputs_lists', get_sorted_input_udf(F.col('image_dicts_list'))).drop('image_dicts_list')
    df = df.withColumn('bands', F.col('inputs_lists')[0])
    df = df.withColumn('tiles', F.col('inputs_lists')[1])
    df = df.withColumn('img_dates', F.col('inputs_lists')[2])
    df = df.withColumn('scl_vals', F.col('inputs_lists')[3])
    return df.drop('inputs_lists')

# Step 4: Convert to binary for model input
def convert_bytes(bands_str: str) -> bytes:
    band_vals = b''
    for num in bands_str.split(','):
        band_vals += int(float(num)).to_bytes(2, 'big')
    return band_vals

def convert_bytes_scl_vals(scl_vals_in: str) -> bytes:
    scl_vals_bstr = b''
    for num in scl_vals_in.split(','):
        scl_vals_bstr += int(num).to_bytes(1, 'big')
    return scl_vals_bstr

def convert_string_utf8(s: str) -> bytes:
    return s.encode('UTF-8')

def date_array_2_int(date_arr: str) -> bytes:
    dates = date_arr.split(',')
    date_vals = b''
    for x in dates:
        days = (datetime.strptime(x, '%Y-%m-%d').date() - datetime(1970, 1, 1).date()).days
        date_vals += int(days).to_bytes(2, 'big')
    return date_vals

get_sorted_input_udf = F.udf(get_sorted_input, ArrayType(StringType()))
convert_bytes_udf = F.udf(convert_bytes, BinaryType())
convert_bytes_scl_vals_udf = F.udf(convert_bytes_scl_vals, BinaryType())
convert_string_utf8_udf = F.udf(convert_string_utf8, BinaryType())
date_array_2_int_udf = F.udf(date_array_2_int, BinaryType())

def convert_to_binary(df: DataFrame, bbox: str, year: str) -> DataFrame:
    return df \
        .withColumn('bands', convert_bytes_udf(F.col('bands'))) \
        .withColumn('img_dates', date_array_2_int_udf(F.col('img_dates'))) \
        .withColumn('tiles', convert_string_utf8_udf(F.col('tiles'))) \
        .withColumn('CDL', convert_string_utf8_udf(F.col('CDL'))) \
        .withColumn('scl_vals', convert_bytes_scl_vals_udf(F.col('scl_vals'))) \
        .withColumn('bbox', F.lit(bbox.encode('UTF-8'))) \
        .withColumn('year', F.lit(year))

# Master function
def agg_to_time_series(
    input_uri_: str,
    output_uri_: str,
    path_parts: list[str],
    spark: SparkSession,
    write_parquet: bool
) -> DataFrame | None:
    """
    Processes Sentinel-2 image data into a time series format suitable for TensorFlow models.

    Parameters:
    -----------
    input_uri_ : str
        Base input path where Parquet files are stored (expects partitioned folders by bbox/year).

    output_uri_ : str
        Output path where the processed time series data will be saved in Parquet format.

    path_parts : list[str]
        A list containing [bbox, year], used to construct the full path for reading and writing.

    spark : SparkSession
        Active Spark session used to read and process the data.

    write_parquet : bool
        If True, writes the resulting DataFrame to a Parquet file. If False, returns the DataFrame instead.

    Returns:
    --------
    pyspark.sql.DataFrame or None
        If write_parquet is False, returns the processed DataFrame. Otherwise, writes to Parquet and returns None.
    """
    df = spark.read.parquet(f"{input_uri_}bbox={path_parts[0]}/year={path_parts[1]}")
    df = create_image_map(df)
    df = group_time_series(df)
    df = flatten_time_series(df)
    df = convert_to_binary(df, path_parts[0], path_parts[1])
    if write_parquet:
        df.write.partitionBy(['bbox', 'year']).mode("append").parquet(output_uri_)
    else:
        return df
    
def get_existing_partitions_from_scenes(path: str) -> set[tuple[str, str]]:
    """
    Derives (bbox, year) partition keys from existing scene_date-level partitioned data.
    """
    scene_data = get_existing_data(path)
    return set((bbox, year) for (bbox, year, _) in scene_data.keys())

def log_success(bbox, year):
    print(f"✅ Wrote partition → bbox={bbox}, year={year} to {args.output}")

# CLI Support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform Sentinel-2 samples into time series.")
    parser.add_argument("--input", type=str, required=True, help="Input base path of Parquet files")
    parser.add_argument("--output", type=str, required=True, help="Output base path for processed files")
    parser.add_argument("--bbox", type=str, help="Single bounding box (xmin,ymin,xmax,ymax)")
    parser.add_argument("--year", type=str, help="Single year")
    parser.add_argument("--bbox_list", type=str, help="Pipe-separated list of bboxes")
    parser.add_argument("--year_list", type=str, help="Comma-separated list of years")
    parser.add_argument("--write_parquet", action="store_true", help="Whether to write output to Parquet")

    args = parser.parse_args()
    # Overwrite from environment variables if they exist
    args.bbox = os.environ.get("BBOX", args.bbox)
    args.year = os.environ.get("YEAR", args.year)
    args.bbox_list = os.environ.get("BBOX_LIST", args.bbox_list)
    args.year_list = os.environ.get("YEAR_LIST", args.year_list)

    existing_output = get_existing_partitions_from_scenes(args.output)
    
    if args.bbox and args.year:
        print(f"🟢 Running in unique mode → BBOX: {args.bbox}, YEAR: {args.year}")
        if (args.bbox, args.year) in existing_output:
            print(f"⏩ Skipping already processed partition: bbox={args.bbox}, year={args.year}")
        else:
            agg_to_time_series(
                input_uri_=args.input,
                output_uri_=args.output,
                path_parts=[args.bbox, args.year],
                spark=spark,
                write_parquet=args.write_parquet
            )
            log_success(args.bbox, args.year)

    elif args.bbox_list and args.year_list:
        bboxes = args.bbox_list.split("|")
        years = args.year_list.split(",")
        print(f"🟡 Running in multiple mode → {len(bboxes)} bboxes × {len(years)} years")

        for bbox in bboxes:
            for year in years:
                if (bbox, year) in existing_output:
                    print(f"⏩ Skipping already processed: bbox={bbox}, year={year}")
                    continue
                print(f"▶ Processing bbox={bbox} year={year}")
                agg_to_time_series(
                    input_uri_=args.input,
                    output_uri_=args.output,
                    path_parts=[bbox, year],
                    spark=spark,
                    write_parquet=args.write_parquet
                )
                log_success(bbox, year)
    else:
        raise ValueError("❌ You must specify either --bbox/--year or --bbox_list/--year_list.")