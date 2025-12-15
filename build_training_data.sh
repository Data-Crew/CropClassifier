#!/bin/bash

# build_training_data.sh
# Runs the full pipeline to download CDL and Sentinel-2 data.

# ==============================================================================================
# Usage:
# ==============================================================================================
# 🟢 Unique scene – for one bbox and one year:
#   bash build_training_data.sh unique              # run both steps (CDL + Sentinel)
#   bash build_training_data.sh unique 1            # run only CDL download step
#   bash build_training_data.sh unique cdl          # same as above
#   bash build_training_data.sh unique 2            # run only Sentinel-2 download step
#   bash build_training_data.sh unique sentinel     # same as above
#   bash build_training_data.sh unique 3            # run only Sentinel-2 transform step
#   bash build_training_data.sh unique transform    # same as above
#   bash build_training_data.sh unique all          # run CDL + Sentinel + Transform (all steps)

# 🟡 Multiple scenes – for multiple bboxes and years from config file:
#   bash build_training_data.sh multiple 1          # run only CDL download step
#   bash build_training_data.sh multiple cdl        # same as above
#   bash build_training_data.sh multiple 2          # run only Sentinel-2 download step
#   bash build_training_data.sh multiple sentinel   # same as above
#   bash build_training_data.sh multiple 3          # run only Sentinel-2 transform step
#   bash build_training_data.sh multiple transform  # same as above
#   bash build_training_data.sh multiple all        # run CDL + Sentinel + Transform (all steps)

# Requires:
#   - A config file named 'bbox_config.txt' with bbox and year info
# ==============================================================================================
CONFIG_FILE="config/bbox_config.txt"
MODE=$1
STEP=${2:-both}  # default is 'both' if not provided

# Validate CONFIG file
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Error: Config file '$CONFIG_FILE' not found"
  exit 1
fi

# Note: GPU setup is NOT needed for preprocessing scripts (download_cdl_data.py, get_sentinel_tiles.py, transform_input_data.py)
# These scripts only use Spark/PySpark, not TensorFlow.
# GPU setup will be done automatically when needed by TensorFlow scripts (train.py, test.py, predict.py)

if [ "$MODE" == "unique" ]; then
  echo "🔹 Mode: Unique scene"

  # Read first non-comment line from config
  line=$(grep -v '^#' "$CONFIG_FILE" | head -n 1)
  IFS="|" read -r name BBOX_raw YEAR_raw <<< "$line"

  export MODE="unique"
  export BBOX="$BBOX_raw"
  export YEAR=$(echo "$YEAR_raw" | cut -d',' -f1)

  echo "Using BBOX: $BBOX"
  echo "Using YEAR: $YEAR"

elif [ "$MODE" == "multiple" ]; then
  echo "🔸 Mode: Multiple scenes"

  # Aggregate all BBOX and YEAR entries from config
  BBOX_LIST=$(grep -v '^#' "$CONFIG_FILE" | cut -d'|' -f2 | paste -sd "|")
  YEAR_LIST=$(grep -v '^#' "$CONFIG_FILE" | cut -d'|' -f3 | tr ',' '\n' | sort -u | paste -sd ",")

  export MODE="multiple"
  export BBOX_LIST="$BBOX_LIST"
  export YEAR_LIST="$YEAR_LIST"

  echo "Using BBOX_LIST: $BBOX_LIST"
  echo "Using YEAR_LIST: $YEAR_LIST"


else
  echo "❌ Error: Mode must be 'unique' or 'multiple'"
  exit 1
fi

if [ "$STEP" == "1" ] || [ "$STEP" == "cdl" ] || [ "$STEP" == "all" ]; then
  echo "🚀 Running CDL data download..."
  PYTHONPATH=./preprocessing python preprocessing/download_cdl_data.py
fi

if [ "$STEP" == "2" ] || [ "$STEP" == "sentinel" ] || [ "$STEP" == "all" ]; then
  echo "📡 Running Sentinel-2 tile generation..."
  PYTHONPATH=./preprocessing python preprocessing/get_sentinel_tiles.py
fi

if [ "$STEP" == "3" ] || [ "$STEP" == "transform" ] || [ "$STEP" == "all" ]; then
  echo "🔄 Running Sentinel-2 input transformation..."

  if [ "$MODE" == "unique" ]; then
    PYTHONPATH=./preprocessing python preprocessing/transform_input_data.py \
      --input ./data/test/s2_unique_scene.parquet/ \
      --output ./data/test/CDL_unique_scene_ts.parquet/ \
      --write_parquet

  elif [ "$MODE" == "multiple" ]; then
    PYTHONPATH=./preprocessing python preprocessing/transform_input_data.py \
      --input ./data/valtrain/s2_multiple_scene.parquet/ \
      --output ./data/valtrain/CDL_multiple_scene_ts.parquet/ \
      --write_parquet
    
  fi
fi