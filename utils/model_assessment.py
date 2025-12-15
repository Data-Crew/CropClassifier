import os
# Set matplotlib to use non-interactive backend to prevent crashes
# This prevents crashes in Docker/headless environments while preserving notebook functionality
import matplotlib
# Force Agg backend unless we're explicitly in a working Jupyter notebook
# This is more robust than checking DISPLAY, which can be set but non-functional
try:
    # Try to detect if we're in a Jupyter notebook or IPython
    from IPython import get_ipython
    ipython_instance = get_ipython()
    # Only skip Agg if we're in a notebook AND it's not a script execution
    if ipython_instance is not None and hasattr(ipython_instance, 'kernel'):
        # We're in a real Jupyter notebook kernel - don't change backend
        pass
    else:
        # Not in a real notebook - force Agg backend
        try:
            matplotlib.use('Agg', force=True)
        except TypeError:
            # Older matplotlib versions don't support force parameter
            matplotlib.use('Agg')
except (ImportError, NameError, AttributeError):
    # IPython not available - force Agg backend
    try:
        matplotlib.use('Agg', force=True)
    except TypeError:
        # Older matplotlib versions don't support force parameter
        matplotlib.use('Agg')
import pandas as pd
from report_utils import (
    plot_and_save_confusion_matrix,
    plot_and_save_accuracy_by_date,
    plot_crop_classification_comparison,
    plot_crop_classification_series,
    plot_cdl_vs_predictions,
    plot_predictions_only_series,
    plot_predictions_with_confidence,
    make_gif_from_pngs,
    get_results_predictions_dir
)
from fpdf import FPDF
import glob
import datetime
import argparse
import re

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Model assessment and report generation.")
parser.add_argument("-model_name", required=True, help="Name of the model to assess")
parser.add_argument('--days_in_series', type=int, default=120, help='Number of days in the series (default: 120)')
parser.add_argument('--year', type=int, default=2019, help='Reference year for prediction_date (default: 2019)')
parser.add_argument('--prediction_mode', action='store_true', help='Run in prediction mode (different file names)')
parser.add_argument('--results_path', type=str, help='Path to results file (overrides default path)')
args = parser.parse_args()
MODEL_NAME = args.model_name
DAYS_IN_SERIES = args.days_in_series
YEAR = args.year
PREDICTION_MODE = args.prediction_mode

# --- Configuration ---
EXPORT_PNGS = True
EXPORT_GIFS = True

# Choose results path based on mode
if PREDICTION_MODE:
    if args.results_path:
        RESULTS_PATH = args.results_path
    else:
        RESULTS_PATH = f"results/predictions/{MODEL_NAME}_{DAYS_IN_SERIES}days.parquet"
    OUTPUT_SUFFIX = "_prediction"
    REPORT_TITLE = "Crop Classification Prediction Report"
else:
    if args.results_path:
        RESULTS_PATH = args.results_path
    else:
        RESULTS_PATH = f"results/test/{MODEL_NAME}_{DAYS_IN_SERIES}days_results.parquet"
    OUTPUT_SUFFIX = ""
    REPORT_TITLE = "Crop Classification Model Report"

print(f"Using model_name={MODEL_NAME}")
print(f"Using DAYS_IN_SERIES={DAYS_IN_SERIES}")
print(f"Using year={YEAR}")
print(f"Loading results from: {RESULTS_PATH}")

# --- Output directory ---
if PREDICTION_MODE:
    output_dir = "results/predictions"
else:
    output_dir = "results/test"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- 1. Load results ---
results = pd.read_parquet(RESULTS_PATH)
results['prediction_day'] = results.start_day + DAYS_IN_SERIES
results['year'] = YEAR
results['prediction_date'] = pd.to_datetime(results.year * 1000 + results.prediction_day, format='%Y%j')

# Handle prediction mode - rename columns to match expected format
if PREDICTION_MODE:
    if 'predicted_class' in results.columns:
        results['predictions'] = results['predicted_class']
    if 'predicted_label' in results.columns:
        results['predicted_labels'] = results['predicted_label']
    # For prediction mode, we don't have true labels, so we'll skip accuracy calculations

# --- 2. Confusion Matrix ---
# Read label_legend from config file to ensure consistency
def read_label_legend_from_config():
    config_file = "config/dataloader.txt"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        # Extract label_legend section
        start = content.find('[label_legend]')
        if start != -1:
            end = content.find('[', start + 1)
            if end == -1:
                end = len(content)
            legend_section = content[start:end]
            # Parse the labels
            labels = []
            for line in legend_section.split('\n')[1:]:  # Skip [label_legend]
                line = line.strip()
                if line and not line.startswith('['):
                    labels.append(line)
            return labels
    # Fallback to hardcoded version if config file not found
    return ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']

label_legend = read_label_legend_from_config()
print(f"Using label_legend: {label_legend}")
print("\n[1/6] Confusion Matrix:")
if not PREDICTION_MODE:
    plot_and_save_confusion_matrix(results, label_legend, export_png=EXPORT_PNGS, output_dir=output_dir)
    cm_path = os.path.join(output_dir, f'confusion_matrix{OUTPUT_SUFFIX}.png')
else:
    print("Skipping confusion matrix for prediction mode (no ground truth available)")
    cm_path = os.path.join(output_dir, f'confusion_matrix{OUTPUT_SUFFIX}.png')

# --- 3. Accuracy by Date ---
print("\n[2/6] Accuracy by Prediction Date:")
if not PREDICTION_MODE:
    plot_and_save_accuracy_by_date(results, export_png=EXPORT_PNGS, output_dir=output_dir)
    acc_path = os.path.join(output_dir, f'accuracy_by_date{OUTPUT_SUFFIX}.png')
else:
    print("Skipping accuracy by date for prediction mode (no ground truth available)")
    acc_path = os.path.join(output_dir, f'accuracy_by_date{OUTPUT_SUFFIX}.png')

# --- 4. Best vs Worst Day Comparison ---
print("\n[3/6] Best vs Worst Day Comparison:")
if not PREDICTION_MODE:
    import sklearn.metrics
    accuracy_by_date = results.groupby('prediction_date').apply(
        lambda x: sklearn.metrics.accuracy_score(x['true_label'], x['predictions'])
    )
    worst_date = accuracy_by_date.idxmin()
    best_date = accuracy_by_date.idxmax()
    worst_start_day = results[results['prediction_date'] == worst_date]['start_day'].iloc[0]
    best_start_day = results[results['prediction_date'] == best_date]['start_day'].iloc[0]
    fig = plot_crop_classification_comparison(results, worst_start_day, best_start_day, title_1="Worst Day", title_2="Best Day")
    bestworst_path = os.path.join(output_dir, f'best_vs_worst_day{OUTPUT_SUFFIX}.png')
    if EXPORT_PNGS and fig is not None:
        fig.savefig(bestworst_path)
        print(f"Saved: {bestworst_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
else:
    print("Skipping best vs worst day comparison for prediction mode (no ground truth available)")
    bestworst_path = os.path.join(output_dir, f'best_vs_worst_day{OUTPUT_SUFFIX}.png')

# --- 4b. Early vs Late Season Comparison ---
print("\n[3b/6] Early vs Late Season Comparison:")
if not PREDICTION_MODE:
    fig = plot_crop_classification_comparison(results, 30, 120, "Early Season", "Late Season")
    earlylate_path = os.path.join(output_dir, f'early_vs_late_season{OUTPUT_SUFFIX}.png')
    if EXPORT_PNGS and fig is not None:
        fig.savefig(earlylate_path)
        print(f"Saved: {earlylate_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
else:
    print("Skipping early vs late season comparison for prediction mode (no ground truth available)")
    earlylate_path = os.path.join(output_dir, f'early_vs_late_season{OUTPUT_SUFFIX}.png')

# --- 5. True vs Pred and CDL vs Pred for three representative days ---
print("\n[4/6] True vs Pred and CDL vs Pred for three representative days:")
all_days = sorted(results['start_day'].unique())
if len(all_days) >= 3:
    rep_days = [all_days[0], all_days[len(all_days)//2], all_days[-1]]
else:
    rep_days = all_days
print(f"Representative start_days: {rep_days}")

if not PREDICTION_MODE:
    # True vs Pred for three days
    plot_crop_classification_series(results[results['start_day'].isin(rep_days)], export_pngs=EXPORT_PNGS, output_dir=output_dir)
    # CDL vs Pred for three days
    plot_cdl_vs_predictions(results, start_days=rep_days, export_pngs=EXPORT_PNGS, output_dir=output_dir)
else:
    # For prediction mode, generate prediction-only visualizations
    print("Generating prediction-only visualizations for representative days...")
    # Generate predictions with confidence scores
    plot_predictions_with_confidence(results[results['start_day'].isin(rep_days)], export_pngs=EXPORT_PNGS, output_dir=output_dir)
    # Generate predictions only (without confidence)
    plot_predictions_only_series(results[results['start_day'].isin(rep_days)], export_pngs=EXPORT_PNGS, output_dir=output_dir)

# Rename files if in prediction mode
if PREDICTION_MODE and EXPORT_PNGS:
    print("Renaming files for prediction mode...")
    # Rename pred_confidence files
    for d in rep_days:
        old_name = os.path.join(output_dir, f"pred_confidence_{d}.png")
        new_name = os.path.join(output_dir, f"pred_confidence_{d}{OUTPUT_SUFFIX}.png")
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
    
    # Rename pred_only files
    for d in rep_days:
        old_name = os.path.join(output_dir, f"pred_only_{d}.png")
        new_name = os.path.join(output_dir, f"pred_only_{d}{OUTPUT_SUFFIX}.png")
        if os.path.exists(old_name):
            os.rename(old_name, new_name)

# --- 6. Optionally, generate GIFs for the full series ---
if EXPORT_GIFS:
    print("\n[5/6] Generating PNGs and GIFs for the full series:")
    if not PREDICTION_MODE:
        all_days = sorted(results['start_day'].unique())
        plot_crop_classification_series(results, export_pngs=True, output_dir=output_dir)
        plot_cdl_vs_predictions(results, start_days=all_days, export_pngs=True, output_dir=output_dir)
    else:
        print("Generating prediction-only visualizations for the full series...")
        all_days = sorted(results['start_day'].unique())
        # Generate predictions with confidence scores for all days
        plot_predictions_with_confidence(results, export_pngs=True, output_dir=output_dir)
        # Generate predictions only for all days
        plot_predictions_only_series(results, export_pngs=True, output_dir=output_dir)
    
    # Rename all generated files if in prediction mode
    if PREDICTION_MODE:
        print("Checking for files to rename for prediction mode...")
        # Rename all pred_confidence files
        pred_confidence_files = glob.glob(os.path.join(output_dir, "pred_confidence_*.png"))
        if pred_confidence_files:
            print(f"Renaming {len(pred_confidence_files)} pred_confidence files...")
            for old_file in pred_confidence_files:
                if not old_file.endswith(OUTPUT_SUFFIX + ".png"):
                    new_file = old_file.replace(".png", f"{OUTPUT_SUFFIX}.png")
                    os.rename(old_file, new_file)
        
        # Rename all pred_only files
        pred_only_files = glob.glob(os.path.join(output_dir, "pred_only_*.png"))
        if pred_only_files:
            print(f"Renaming {len(pred_only_files)} pred_only files...")
            for old_file in pred_only_files:
                if not old_file.endswith(OUTPUT_SUFFIX + ".png"):
                    new_file = old_file.replace(".png", f"{OUTPUT_SUFFIX}.png")
                    os.rename(old_file, new_file)
    
    # Generate GIFs based on mode
    if PREDICTION_MODE:
        # Predictions with confidence GIF
        def extract_start_day_pred_confidence(path):
            match = re.search(r'pred_confidence_(\d+)' + re.escape(OUTPUT_SUFFIX) + r'\.png', os.path.basename(path))
            return int(match.group(1)) if match else -1
        png_paths = glob.glob(os.path.join(output_dir, f"pred_confidence_*{OUTPUT_SUFFIX}.png"))
        if png_paths:
            png_paths = sorted(png_paths, key=extract_start_day_pred_confidence)
            gif1 = os.path.join(output_dir, f"animation_predictions_confidence{OUTPUT_SUFFIX}.gif")
            make_gif_from_pngs(png_paths, gif1)
            print(f"Generated GIF: {gif1}")
        else:
            gif1 = None
            print("No pred_confidence files found for GIF generation")
        
        # Predictions only GIF
        def extract_start_day_pred_only(path):
            match = re.search(r'pred_only_(\d+)' + re.escape(OUTPUT_SUFFIX) + r'\.png', os.path.basename(path))
            return int(match.group(1)) if match else -1
        png_paths = glob.glob(os.path.join(output_dir, f"pred_only_*{OUTPUT_SUFFIX}.png"))
        if png_paths:
            png_paths = sorted(png_paths, key=extract_start_day_pred_only)
            gif2 = os.path.join(output_dir, f"animation_predictions_only{OUTPUT_SUFFIX}.gif")
            make_gif_from_pngs(png_paths, gif2)
            print(f"Generated GIF: {gif2}")
        else:
            gif2 = None
            print("No pred_only files found for GIF generation")
    else:
        # Test mode GIFs
        # True vs Pred GIF
        def extract_start_day_true_vs_pred(path):
            match = re.search(r'true_vs_pred_(\d+)' + re.escape(OUTPUT_SUFFIX) + r'\.png', os.path.basename(path))
            return int(match.group(1)) if match else -1
        png_paths = glob.glob(os.path.join(output_dir, f"true_vs_pred_*{OUTPUT_SUFFIX}.png"))
        if png_paths:
            png_paths = sorted(png_paths, key=extract_start_day_true_vs_pred)
            gif1 = os.path.join(output_dir, f"animation_true_vs_pred{OUTPUT_SUFFIX}.gif")
            make_gif_from_pngs(png_paths, gif1)
            print(f"Generated GIF: {gif1}")
        else:
            gif1 = None
            print("No true_vs_pred files found for GIF generation")
        
        # CDL vs Pred GIF
        def extract_start_day_cdl_vs_pred(path):
            match = re.search(r'cdl_vs_pred_(\d+)' + re.escape(OUTPUT_SUFFIX) + r'\.png', os.path.basename(path))
            return int(match.group(1)) if match else -1
        png_paths = glob.glob(os.path.join(output_dir, f"cdl_vs_pred_*{OUTPUT_SUFFIX}.png"))
        if png_paths:
            png_paths = sorted(png_paths, key=extract_start_day_cdl_vs_pred)
            gif2 = os.path.join(output_dir, f"animation_cdl_vs_pred{OUTPUT_SUFFIX}.gif")
            make_gif_from_pngs(png_paths, gif2)
            print(f"Generated GIF: {gif2}")
        else:
            gif2 = None
            print("No cdl_vs_pred files found for GIF generation")
else:
    gif1 = gif2 = None

# --- 7. Build PDF Report ---
print("\n[6/6] Compiling PDF report...")
pdf = FPDF()

# Page 1: Confusion Matrix and Accuracy by Date (smaller, both on one page)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, REPORT_TITLE, ln=True, align='C')
pdf.set_font("Arial", '', 12)
pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
pdf.ln(8)
if os.path.exists(cm_path):
    # Centered, smaller
    x = (210 - 120) / 2  # A4 width is 210mm
    pdf.image(cm_path, x=x, w=120)
pdf.ln(5)
if os.path.exists(acc_path):
    x = (210 - 120) / 2
    pdf.image(acc_path, x=x, w=120)
pdf.ln(5)

# Page 2: Best vs Worst Day (large)
pdf.add_page()
if os.path.exists(bestworst_path):
    x = (210 - 200) / 2
    pdf.image(bestworst_path, x=x, w=200)

# Page 3: Early vs Late Season (large)
pdf.add_page()
if os.path.exists(earlylate_path):
    x = (210 - 200) / 2
    pdf.image(earlylate_path, x=x, w=200)

# Page 4: True vs Pred for three representative days (stacked vertically)
if not PREDICTION_MODE:
    pdf.add_page()
    for d in rep_days:
        png = os.path.join(output_dir, f"true_vs_pred_{d}{OUTPUT_SUFFIX}.png")
        if os.path.exists(png):
            pdf.ln(5)
            x = (210 - 180) / 2
            pdf.image(png, x=x, w=180)

# Page 5: CDL vs Pred for three representative days (stacked vertically)
    pdf.add_page()
    for d in rep_days:
        png = os.path.join(output_dir, f"cdl_vs_pred_{d}{OUTPUT_SUFFIX}.png")
        if os.path.exists(png):
            pdf.ln(5)
            x = (210 - 180) / 2
            pdf.image(png, x=x, w=180)
else:
    # For prediction mode, add prediction visualizations
    # Page 4: Predictions with confidence for three representative days
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Predictions with Confidence Scores", ln=True, align='C')
    pdf.ln(5)
    for d in rep_days:
        png = os.path.join(output_dir, f"pred_confidence_{d}{OUTPUT_SUFFIX}.png")
        if os.path.exists(png):
            pdf.ln(5)
            x = (210 - 180) / 2
            pdf.image(png, x=x, w=180)
    
    # Page 5: Predictions only for three representative days
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Predictions Only", ln=True, align='C')
    pdf.ln(5)
    for d in rep_days:
        png = os.path.join(output_dir, f"pred_only_{d}{OUTPUT_SUFFIX}.png")
        if os.path.exists(png):
            pdf.ln(5)
            x = (210 - 180) / 2
            pdf.image(png, x=x, w=180)
    
    # Page 6: Prediction results summary
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Prediction Results Summary", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Model: {MODEL_NAME}", ln=True)
    pdf.cell(0, 10, f"Total predictions: {len(results)}", ln=True)
    pdf.cell(0, 10, f"Prediction year: {YEAR}", ln=True)
    pdf.cell(0, 10, f"Days in series: {DAYS_IN_SERIES}", ln=True)
    pdf.ln(10)
    
    # Add confidence statistics
    if 'confidence_score' in results.columns:
        avg_confidence = results['confidence_score'].mean()
        min_confidence = results['confidence_score'].min()
        max_confidence = results['confidence_score'].max()
        pdf.cell(0, 10, f"Average confidence: {avg_confidence:.3f}", ln=True)
        pdf.cell(0, 10, f"Min confidence: {min_confidence:.3f}", ln=True)
        pdf.cell(0, 10, f"Max confidence: {max_confidence:.3f}", ln=True)
        pdf.ln(10)
    
    # Add class distribution
    if 'predicted_label' in results.columns:
        pdf.cell(0, 10, "Predicted Class Distribution:", ln=True)
        class_counts = results['predicted_label'].value_counts()
        for label, count in class_counts.head(10).items():  # Show top 10
            percentage = (count / len(results)) * 100
            pdf.cell(0, 10, f"  {label}: {count} ({percentage:.1f}%)", ln=True)
    
    pdf.ln(10)
    pdf.cell(0, 10, "Note: Ground truth comparisons are not available for prediction mode.", ln=True)
    pdf.cell(0, 10, "Only prediction results and confidence scores are generated.", ln=True)

# --- 8. Export PDF ---
report_path = os.path.join(output_dir, f'model_report{OUTPUT_SUFFIX}.pdf')
pdf.output(report_path)
print(f"\nPDF report saved to {report_path}")

print("\nModel assessment complete. All outputs saved in:", output_dir) 