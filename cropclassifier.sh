#!/bin/bash

# cropclassifier.sh
# Runs the crop classification pipeline with different steps.

# ==============================================================================================
# Usage:
# ==============================================================================================
# 🟢 Individual actions:
#   bash cropclassifier.sh -action process                # prepare training and validation datasets
#   bash cropclassifier.sh -action train                  # train model (requires datasets to be prepared)
#   bash cropclassifier.sh -action test                   # run testing + model assessment
#   bash cropclassifier.sh -action predict                # run prediction + prediction assessment

# 🟡 Combined actions:
#   bash cropclassifier.sh -action "process train"        # run process + train 
#   bash cropclassifier.sh -action "process train test"   # run process + train + test (complete pipeline)
#   bash cropclassifier.sh -action "predict"              # run predict on new data  

# 🟠 Override config parameters:
#   bash cropclassifier.sh -action process -days-in-series 90 -days-per-bucket 3 -frames-to-check 5
#   bash cropclassifier.sh -action train -model resnet1d -epochs 100 -batch-size 512
#   bash cropclassifier.sh -action test -test-path 'data/custom_test.parquet/*/*[TESTYEAR]*/*.parquet'

#   bash cropclassifier.sh -action predict \
#       -model simplecnn \
#       -input-path "data/[INPUT_NAME].parquet/*/*[YEAR]*/*.parquet" \
#       -output-path "results/dense_test_quiet" \
#       -save-probabilities \
#       -pred-year [YEAR] \
#       -quiet

# Available models:
#   simplecnn, bigcnn, vgg1d, vgg1d_compact, unet1d, unet1d_light
#   resnet1d, resunet1d, tcn, transformer1d, cnn_transformer1d
#   efficientnet1d, inception1d, inception1d_se_augmented, inception1d_se_mixup_focal_attention_residual


# Options:
#   -quiet: Suppress TensorFlow/GPU verbose logs
#   -verbose: Show all logs (default)

# Requires:
#   - A config file named 'config/dataloader.txt' with hyperparameters and crop lists
# ==============================================================================================

# Quick help function
show_quick_help() {
    echo "🔧 CropClassifier - Quick Help"
    echo "=============================="
    echo ""
    echo "🧪 TESTING:"
echo "   bash cropclassifier.sh -action test -model simplecnn"
echo "   bash cropclassifier.sh -action test -model inception1d_se_mixup_focal_attention_residual"
echo "   bash cropclassifier.sh -action test -model simplecnn -test-path 'data/custom_test.parquet/*/*.parquet'"
echo "   # Runs test.py + model_assessment.py (config from dataloader.txt or command line args)"
    echo ""
    echo "🔮 PREDICTION:"
    echo "   bash cropclassifier.sh -action predict -model simplecnn -input-path ./data/s2_unique_time_series.parquet/"
    echo "   bash cropclassifier.sh -action predict -model simplecnn -input-path ./data/s2_final.parquet/"
    echo "   bash cropclassifier.sh -action predict -model simplecnn -input-path ./data/new_data.parquet -output-path results/custom_predictions -save-probabilities -pred-year 2024"
    echo "   # Runs predict.py + model_assessment.py (prediction mode)"
    echo "   # Always uses the same date range as test.py for consistency"
    echo ""
    echo "📊 TRAINING:"
    echo "   bash cropclassifier.sh -action train -model simplecnn"
    echo "   bash cropclassifier.sh -action \"process train\" -model resnet1d"
    echo ""
    echo "📁 DATA PROCESSING:"
    echo "   bash cropclassifier.sh -action process"
    echo ""
    echo "For detailed help, run: bash cropclassifier.sh -action invalid_action"
    echo ""
}

set -e  # exit on error

CONFIG_FILE="config/dataloader.txt"
ACTIONS=""
MODEL=""
BATCH_SIZE=""
DAYS_IN_SERIES=""
DAYS_PER_BUCKET=""
FRAMES_TO_CHECK=""
BUCKETING_STRATEGY=""
EPOCHS=""
ES_PATIENCE=""
NUM_FEATURES=""
INPUT_PATH=""
OUTPUT_PATH=""
SAVE_PROBABILITIES=false
PRED_YEAR=""
QUIET_MODE=false

# Show quick help if no arguments provided
if [ $# -eq 0 ]; then
    show_quick_help
    exit 0
fi

# Validate CONFIG file
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Error: Config file '$CONFIG_FILE' not found"
  exit 1
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -action)
      ACTIONS="$2"
      shift
      shift
      ;;
    -model)
      MODEL="$2"
      shift
      shift
      ;;
    -batch-size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    -days-in-series)
      DAYS_IN_SERIES="$2"
      shift
      shift
      ;;
    -days-per-bucket)
      DAYS_PER_BUCKET="$2"
      shift
      shift
      ;;
    -frames-to-check)
      FRAMES_TO_CHECK="$2"
      shift
      shift
      ;;
    -bucketing-strategy)
      BUCKETING_STRATEGY="$2"
      shift
      shift
      ;;
    -epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    -es-patience)
      ES_PATIENCE="$2"
      shift
      shift
      ;;
    -num-features)
      NUM_FEATURES="$2"
      shift
      shift
      ;;
    -input-path)
      INPUT_PATH="$2"
      shift
      shift
      ;;
    -output-path)
      OUTPUT_PATH="$2"
      shift
      shift
      ;;
    -save-probabilities)
      SAVE_PROBABILITIES=true
      shift
      ;;
    -pred-year)
      PRED_YEAR="$2"
      shift
      shift
      ;;
    -test-path)
      TEST_PATH="$2"
      shift
      shift
      ;;
    -quiet)
      QUIET_MODE=true
      shift
      ;;
    -verbose)
      QUIET_MODE=false
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Load configuration from dataloader.txt
echo "📋 Loading configuration from $CONFIG_FILE..."

# Read paths (use command line override if provided, otherwise use config)
CONFIG_TRAIN_PATH=$(grep -A 3 '\[paths\]' "$CONFIG_FILE" | grep 'train_path' | cut -d'=' -f2)
CONFIG_VAL_PATH=$(grep -A 3 '\[paths\]' "$CONFIG_FILE" | grep 'val_path' | cut -d'=' -f2)
CONFIG_TEST_PATH=$(grep -A 3 '\[paths\]' "$CONFIG_FILE" | grep 'test_path' | cut -d'=' -f2)

# Use command line override if provided, otherwise use config
TRAIN_PATH=${TRAIN_PATH:-$CONFIG_TRAIN_PATH}
VAL_PATH=${VAL_PATH:-$CONFIG_VAL_PATH}
TEST_PATH=${TEST_PATH:-$CONFIG_TEST_PATH}

# Read hyperparameters
CONFIG_MODEL_NAME=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'model_name' | cut -d'=' -f2)
CONFIG_BATCH_SIZE=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'batch_size' | cut -d'=' -f2)
CONFIG_DAYS_IN_SERIES=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'days_in_series' | cut -d'=' -f2)
CONFIG_DAYS_PER_BUCKET=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'days_per_bucket' | cut -d'=' -f2)
CONFIG_FRAMES_TO_CHECK=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'frames_to_check' | cut -d'=' -f2)
CONFIG_BUCKETING_STRATEGY=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'bucketing_strategy' | cut -d'=' -f2)
CONFIG_NUM_FEATURES=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'num_features' | cut -d'=' -f2)
CONFIG_MAX_EPOCHS=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'max_epochs' | cut -d'=' -f2)
CONFIG_ES_PATIENCE=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'es_patience' | cut -d'=' -f2)
CONFIG_TEST_YEAR=$(grep -A 10 '\[hyperparams\]' "$CONFIG_FILE" | grep 'test_year' | cut -d'=' -f2)

# Read crop lists
TARGETED_CROPS=$(sed -n '/^\[targeted_crops\]$/,/^\[/p' "$CONFIG_FILE" | grep -v '^\[targeted_crops\]' | grep -v '^\[other_crops\]' | grep -v '^\s*$' | paste -sd',' - | sed 's/,$//')
OTHER_CROPS=$(sed -n '/^\[other_crops\]$/,/^\[/p' "$CONFIG_FILE" | grep -v '^\[other_crops\]' | grep -v '^\[label_legend\]' | grep -v '^\s*$' | paste -sd',' - | sed 's/,$//')
LABEL_LEGEND=$(sed -n '/^\[label_legend\]$/,/^\[/p' "$CONFIG_FILE" | grep -v '^\[label_legend\]' | grep -v '^\s*$' | paste -sd',' - | sed 's/,$//')

# Use command line arguments if provided, otherwise use config values
MODEL=${MODEL:-$CONFIG_MODEL_NAME}
BATCH_SIZE=${BATCH_SIZE:-$CONFIG_BATCH_SIZE}
DAYS_IN_SERIES=${DAYS_IN_SERIES:-$CONFIG_DAYS_IN_SERIES}
DAYS_PER_BUCKET=${DAYS_PER_BUCKET:-$CONFIG_DAYS_PER_BUCKET}
FRAMES_TO_CHECK=${FRAMES_TO_CHECK:-$CONFIG_FRAMES_TO_CHECK}
BUCKETING_STRATEGY=${BUCKETING_STRATEGY:-$CONFIG_BUCKETING_STRATEGY}
NUM_FEATURES=${NUM_FEATURES:-$CONFIG_NUM_FEATURES}
MAX_EPOCHS=${EPOCHS:-$CONFIG_MAX_EPOCHS}
ES_PATIENCE=${ES_PATIENCE:-$CONFIG_ES_PATIENCE}
TEST_YEAR=${CONFIG_TEST_YEAR}

echo "📊 Configuration loaded:"
echo "   - Model: $MODEL"
echo "   - Train path: $TRAIN_PATH"
echo "   - Val path: $VAL_PATH"
echo "   - Test path: $TEST_PATH"
echo "   - Test year: $TEST_YEAR"
echo "   - Pred year: ${PRED_YEAR:-'not specified (using test year)'}"
echo "   - Input path: $INPUT_PATH"
echo "   - Batch size: $BATCH_SIZE"
echo "   - Days in series: $DAYS_IN_SERIES"
echo "   - Days per bucket: $DAYS_PER_BUCKET"
echo "   - Frames to check: $FRAMES_TO_CHECK"
echo "   - Bucketing strategy: $BUCKETING_STRATEGY"
echo "   - Num features: $NUM_FEATURES"
echo "   - Max epochs: $MAX_EPOCHS"
echo "   - ES patience: $ES_PATIENCE"
echo "   - Targeted crops: $TARGETED_CROPS"
echo "   - Other crops: $OTHER_CROPS"
echo "   - Label legend: $LABEL_LEGEND"

# Build command arguments
PROCESS_ARGS=""
TRAIN_ARGS=""

# Common arguments shared between process and train
COMMON_ARGS="--days-in-series $DAYS_IN_SERIES --days-per-bucket $DAYS_PER_BUCKET --label-legend '$LABEL_LEGEND'"

# Process-specific arguments (including frames-to-check which is only for process.py)
PROCESS_ARGS="$PROCESS_ARGS --batch-size $BATCH_SIZE"
PROCESS_ARGS="$PROCESS_ARGS --train-path \"$TRAIN_PATH\""
PROCESS_ARGS="$PROCESS_ARGS --val-path \"$VAL_PATH\""
PROCESS_ARGS="$PROCESS_ARGS --targeted-crops '$TARGETED_CROPS'"
PROCESS_ARGS="$PROCESS_ARGS --other-crops '$OTHER_CROPS'"
PROCESS_ARGS="$PROCESS_ARGS --frames-to-check $FRAMES_TO_CHECK"
PROCESS_ARGS="$PROCESS_ARGS --bucketing-strategy $BUCKETING_STRATEGY"
PROCESS_ARGS="$PROCESS_ARGS --num-features $NUM_FEATURES"
PROCESS_ARGS="$PROCESS_ARGS $COMMON_ARGS"

# Train-specific arguments (frames-to-check is not needed for train.py)
TRAIN_ARGS="$TRAIN_ARGS --model $MODEL"
TRAIN_ARGS="$TRAIN_ARGS --epochs $MAX_EPOCHS"
TRAIN_ARGS="$TRAIN_ARGS --es-patience $ES_PATIENCE"
TRAIN_ARGS="$TRAIN_ARGS --num-features $NUM_FEATURES"
TRAIN_ARGS="$TRAIN_ARGS $COMMON_ARGS"

# Configure GPU once at the beginning
echo "🔧 Configuring GPU..."
if [ "$QUIET_MODE" = true ]; then
  # Suppress TensorFlow/GPU verbose logs
  export TF_CPP_MIN_LOG_LEVEL=2  # Only show ERROR logs
  export CUDA_VISIBLE_DEVICES=0  # Ensure GPU is visible
  python config/gpu/setup_gpu.py 2>/dev/null
else
  python config/gpu/setup_gpu.py
fi


# Main dispatcher
for ACTION in $ACTIONS; do
    case $ACTION in

  process)
    echo "🧮 Running data processing pipeline..."
    echo "📊 Step 1: Prepare training and validation datasets"
    if [ "$QUIET_MODE" = true ]; then
      # Suppress TensorFlow/GPU verbose logs
      export TF_CPP_MIN_LOG_LEVEL=2
      eval python process.py $PROCESS_ARGS 2>/dev/null
    else
      eval python process.py $PROCESS_ARGS
    fi
    echo "✅ Data processing completed!"
    ;;

  train)
    echo "🎯 Running training pipeline..."
    echo "📂 Step 1: Load prepared datasets"
    echo "🏗️ Step 2: Train $MODEL model"
    if [ "$QUIET_MODE" = true ]; then
      # Suppress TensorFlow/GPU verbose logs
      export TF_CPP_MIN_LOG_LEVEL=2
      eval python train.py $TRAIN_ARGS 2>/dev/null
    else
      eval python train.py $TRAIN_ARGS
    fi
    echo "✅ Training completed!"
    ;;

  test)
    echo "🧪 Running test pipeline..."
    echo "📊 Step 1: Load trained model"
    echo "🔮 Step 2: Run predictions on test data"
    echo "📈 Step 3: Calculate and save metrics"
    if [ "$QUIET_MODE" = true ]; then
      # Suppress TensorFlow/GPU verbose logs
      export TF_CPP_MIN_LOG_LEVEL=2
      python test.py -model_name $MODEL -days_in_series $DAYS_IN_SERIES -batch_size $BATCH_SIZE -num_features $NUM_FEATURES -days_per_bucket $DAYS_PER_BUCKET -frames_to_check $FRAMES_TO_CHECK -bucketing_strategy $BUCKETING_STRATEGY -targeted_crops "$TARGETED_CROPS" -other_crops "$OTHER_CROPS" -label_legend "$LABEL_LEGEND" -train_path "$TRAIN_PATH" -test_path "$TEST_PATH" 2>/dev/null
    else
      python test.py -model_name $MODEL -days_in_series $DAYS_IN_SERIES -batch_size $BATCH_SIZE -num_features $NUM_FEATURES -days_per_bucket $DAYS_PER_BUCKET -frames_to_check $FRAMES_TO_CHECK -bucketing_strategy $BUCKETING_STRATEGY -targeted_crops "$TARGETED_CROPS" -other_crops "$OTHER_CROPS" -label_legend "$LABEL_LEGEND" -train_path "$TRAIN_PATH" -test_path "$TEST_PATH"
    fi
    echo "✅ Testing completed!"
    
    echo "📊 Step 4: Generate model assessment and visualizations"
    if [ "$QUIET_MODE" = true ]; then
      PYTHONPATH=./utils python utils/model_assessment.py -model_name $MODEL --days_in_series $DAYS_IN_SERIES --year $TEST_YEAR 2>/dev/null
    else
              PYTHONPATH=./utils python utils/model_assessment.py -model_name $MODEL --days_in_series $DAYS_IN_SERIES --year $TEST_YEAR
    fi
    echo "✅ Model assessment completed!"
    ;;

  predict)
    echo "🔮 Running prediction pipeline..."
    echo "📊 Step 1: Load trained model"
    echo "🔮 Step 2: Load input data"
    echo "📈 Step 3: Generate predictions and confidence scores"
    
    # Check if input path is provided
    if [ -z "$INPUT_PATH" ]; then
      echo "❌ Error: Input path is required for prediction. Use -input_path argument."
      echo "Example: bash cropclassifier.sh -action predict -model simplecnn -input_path data/new_data.parquet"
      exit 1
    fi
    
    # Build predict command with optional parameters
    PREDICT_CMD="python predict.py -model_name $MODEL -input_path \"$INPUT_PATH\" -days_in_series $DAYS_IN_SERIES -batch_size $BATCH_SIZE -num_features $NUM_FEATURES -days_per_bucket $DAYS_PER_BUCKET -frames_to_check $FRAMES_TO_CHECK -bucketing_strategy $BUCKETING_STRATEGY -targeted_crops \"$TARGETED_CROPS\" -other_crops \"$OTHER_CROPS\" -label_legend \"$LABEL_LEGEND\" -train_path \"$TRAIN_PATH\""
    
    # Add optional parameters if provided
    if [ -n "$OUTPUT_PATH" ]; then
      PREDICT_CMD="$PREDICT_CMD -output_path \"$OUTPUT_PATH\""
    fi
    
    if [ "$SAVE_PROBABILITIES" = true ]; then
      PREDICT_CMD="$PREDICT_CMD -save_probabilities"
    fi
    
    if [ "$QUIET_MODE" = true ]; then
      # Suppress TensorFlow/GPU verbose logs
      export TF_CPP_MIN_LOG_LEVEL=2
      eval $PREDICT_CMD 2>/dev/null
    else
      eval $PREDICT_CMD
    fi
    echo "✅ Prediction completed!"
    
    echo "📊 Step 4: Generate prediction assessment and visualizations"
    # Use PRED_YEAR if provided, otherwise use TEST_YEAR as fallback
    ASSESSMENT_YEAR=${PRED_YEAR:-$TEST_YEAR}
    
    # Determine the results file path based on output directory
    if [ -n "$OUTPUT_PATH" ]; then
      RESULTS_FILE_PATH="$OUTPUT_PATH/predictions_${MODEL}_${DAYS_IN_SERIES}days.parquet"
    else
      RESULTS_FILE_PATH="results/predictions/predictions_${MODEL}_${DAYS_IN_SERIES}days.parquet"
    fi
    
    if [ "$QUIET_MODE" = true ]; then
      PYTHONPATH=./utils python utils/model_assessment.py -model_name $MODEL --days_in_series $DAYS_IN_SERIES --year $ASSESSMENT_YEAR --prediction_mode --results_path "$RESULTS_FILE_PATH" 2>/dev/null
    else
              PYTHONPATH=./utils python utils/model_assessment.py -model_name $MODEL --days_in_series $DAYS_IN_SERIES --year $ASSESSMENT_YEAR --prediction_mode --results_path "$RESULTS_FILE_PATH"
    fi
    echo "✅ Prediction assessment completed!"
    ;;

    *)
    echo "❌ Unknown action: $ACTION"
    echo ""
    echo "Usage: $0 -action {process|train|test|predict|process train|pt} [OPTIONS]"
    echo ""
    echo "See the documentation at the top of this script for detailed usage examples."
    echo "Available actions: process, train, test, predict, process train, pt"
    exit 1
    ;;
    esac
done
