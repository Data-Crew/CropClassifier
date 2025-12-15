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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import imageio

import geopandas as gpd
import contextily
import matplotlib.colors as cls
import matplotlib.patches as mpatches
import sklearn.metrics

# Helper function to detect if we're in a notebook environment
def _is_notebook():
    """Check if we're running in a Jupyter notebook or IPython."""
    try:
        from IPython import get_ipython
        ipython_instance = get_ipython()
        return ipython_instance is not None
    except (ImportError, NameError, AttributeError):
        return False

# Helper function to conditionally close figures (don't close in notebooks)
def _close_figure_if_needed(fig=None):
    """Close figure only if not in a notebook environment."""
    if not _is_notebook():
        if fig is not None:
            plt.close(fig)
        else:
            plt.close()

def get_results_predictions_dir():
    # Walk up the directory tree until 'results/predictions' is found
    current = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(current, "results", "predictions")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("Could not find 'results/predictions' directory in any parent folder.")
        current = parent

def plot_and_save_confusion_matrix(results, label_legend, export_png=False, output_dir=None):
    """
    Plot and optionally save the confusion matrix for the model predictions, and print classification metrics.
    Args:
        results (pd.DataFrame): DataFrame with 'true_label' and 'predictions' columns.
        label_legend (list): List of label names for axis ticks and reports.
        export_png (bool): If True, saves the confusion matrix PNG to the results predictions directory.
        output_dir (str): Directory to save the PNG file. If None, uses get_results_predictions_dir().
    Returns:
        None.
    """
    
    true = results.true_label
    pred = results.predictions
    confusion_matrix = sklearn.metrics.confusion_matrix(true, pred, normalize=None)

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        xticklabels=label_legend,
        yticklabels=label_legend,
        cmap='Blues',
        cbar_kws={'label': 'Number of samples'},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('True Labels', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Crop Classification', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    exported = None
    if export_png:
        if output_dir is None:
            output_dir = get_results_predictions_dir()
        export_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(export_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {export_path}")
        _close_figure_if_needed()
        exported = export_path
    else:
        # If not exporting, close the figure only if not in notebook (to allow display in notebooks)
        _close_figure_if_needed()
    
    # Removed plt.show() to prevent crashes in headless environments
    # In notebooks, figures will display automatically if not closed

    print("\n📊 CONFUSION MATRIX STATISTICS:")
    print("=" * 50)
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    print("\n📈 CLASSIFICATION REPORT:")
    print(classification_report(true, pred, target_names=label_legend))
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    f1 = f1_score(true, pred, average='weighted')
    print(f"\n🎯 OVERALL METRICS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\n📋 SAMPLE DISTRIBUTION:")
    for i, label in enumerate(label_legend):
        count = np.sum(true == i)
        print(f"{label}: {count:,} samples")
    

def plot_and_save_accuracy_by_date(results, export_png=False, output_dir=None):
    """
    Plot and optionally save the model accuracy by prediction date.
    Args:
        results (pd.DataFrame): DataFrame with 'prediction_date', 'true_label', and 'predictions'.
        export_png (bool): If True, saves the plot as 'accuracy_by_date.png' in the results predictions directory.
        output_dir (str): Directory to save the PNG file. If None, uses get_results_predictions_dir().
    Returns:
        str or None: Path to the saved PNG file if exported, else None.
    """
    accuracy_by_time_of_year = results.groupby('prediction_date').apply(
        lambda x: sklearn.metrics.accuracy_score(x['true_label'], x['predictions'])
    )
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_by_time_of_year.index, accuracy_by_time_of_year.values, 
             marker='o', linewidth=2.5, markersize=8, color='#2E86AB', alpha=0.8)
    plt.fill_between(accuracy_by_time_of_year.index, accuracy_by_time_of_year.values, 
                     alpha=0.3, color='#2E86AB')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Prediction Date', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy by Prediction Date', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    for i, (date, acc) in enumerate(zip(accuracy_by_time_of_year.index, accuracy_by_time_of_year.values)):
        plt.annotate(f'{acc:.3f}', (date, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    exported = None
    if export_png:
        if output_dir is None:
            output_dir = get_results_predictions_dir()
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, 'accuracy_by_date.png')
        plt.savefig(export_path, bbox_inches='tight')
        print(f"Accuracy by date plot saved to {export_path}")
        exported = export_path
    # Removed plt.show() to prevent crashes in headless environments
    # In notebooks, figures will display automatically if not closed
    _close_figure_if_needed()
    return exported

def plot_crop_classification_comparison(results, start_day_1, start_day_2=None, title_1="Day 1", title_2="Day 2"):
    """
    Plot crop classification results for comparison between two days.
    
    Parameters:
    -----------
    results : pandas.DataFrame
        DataFrame containing the results with columns: start_day, true_label, predictions, 
        longitude, latitude, prediction_date
    start_day_1 : int
        First start_day to visualize
    start_day_2 : int, optional
        Second start_day to visualize. If None, only plots start_day_1
    title_1 : str, optional
        Title for the first day (default: "Day 1")
    title_2 : str, optional
        Title for the second day (default: "Day 2")
    """
    
    # Create the discrete labels using CDL color scheme
    label_names = ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
    colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), 
              cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), 
              cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), 
              cls.rgb2hex((255/255, 38/255, 38/255))]
    patches = [mpatches.Patch(color=c) for c in colors]
    CDLcmp = cls.ListedColormap(colors, name='CDL')
    
    # Determine number of rows based on whether we have one or two days
    n_rows = 1 if start_day_2 is None else 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(24, 8*n_rows))
    
    # If only one day, make axs 2D for consistent indexing
    if start_day_2 is None:
        axs = axs.reshape(1, -1)
    
    # Function to get accuracy for a specific start_day
    def get_accuracy_for_start_day(start_day):
        day_results = results[results.start_day == start_day]
        if len(day_results) > 0:
            return sklearn.metrics.accuracy_score(day_results['true_label'], day_results['predictions'])
        return 0.0
    
    # Function to plot a single day
    def plot_single_day(start_day, row_idx, title):
        day_results = results[results.start_day == start_day]
        if len(day_results) == 0:
            print(f"No data found for start_day {start_day}")
            return
            
        gdf = gpd.GeoDataFrame(day_results, 
                              geometry=gpd.points_from_xy(day_results.longitude, day_results.latitude, crs="epsg:4326")).to_crs("epsg:3857")
        
        prediction_date = day_results['prediction_date'].iloc[0]
        accuracy = get_accuracy_for_start_day(start_day)
        
        # Ground Truth
        #gdf.plot('true_label', s=20, ax=axs[row_idx,0], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.5)
        gdf.plot('true_label', s=15, ax=axs[row_idx,0], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        basemap = contextily.providers.USGS.USImagery
        contextily.add_basemap(axs[row_idx,0], source=basemap, alpha=0.7)
        axs[row_idx,0].set_title(f'{title} - Ground Truth ({prediction_date.strftime("%B %d")})', 
                                fontsize=14, fontweight='bold', pad=15)
        axs[row_idx,0].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[row_idx,0].set_ylabel('Latitude', fontsize=11, fontweight='bold')
        
        # Predictions
        #gdf.plot('predictions', s=20, ax=axs[row_idx,1], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.5)
        gdf.plot('predictions', s=15, ax=axs[row_idx,1], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        basemap = contextily.providers.USGS.USImagery
        contextily.add_basemap(axs[row_idx,1], source=basemap, alpha=0.7)
        axs[row_idx,1].set_title(f'{title} - Predictions (Accuracy: {accuracy:.3f})', 
                                fontsize=14, fontweight='bold', pad=15)
        axs[row_idx,1].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[row_idx,1].set_ylabel('Latitude', fontsize=11, fontweight='bold')
        
        print(f"{title}: start_day={start_day}, date={prediction_date.strftime('%B %d, %Y')}, accuracy={accuracy:.4f}")
    
    # Plot first day
    plot_single_day(start_day_1, 0, title_1)
    
    # Plot second day if provided
    if start_day_2 is not None:
        plot_single_day(start_day_2, 1, title_2)
    
    # Create legend using axs[0, 1].legend() - bbox_to_anchor is relative to the right axis
    # (1.02, 0.5) means: x=1.02 (just right of the axis), y=0.5 (vertically centered)
    legend = axs[0, 1].legend(patches, label_names, 
                       loc='center left', 
                       bbox_to_anchor=(1.02, 0.5),
                       title='Crop Classes',
                       title_fontsize=12,
                       fontsize=11,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       borderpad=1)
    legend.get_title().set_fontweight('bold')
    
    # Main title
    if start_day_2 is None:
        main_title = f'Crop Classification: {title_1}'
    else:
        main_title = f'Crop Classification Comparison: {title_1} vs {title_2}'
    
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout to fit everything nicely
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    return fig

def plot_crop_classification_series(results, export_pngs=False, figsize=(24, 8), output_dir=None):
    """
    Plots the full crop classification time series for all available days.
    For each day, shows Ground Truth and Predictions side by side. Always displays the plot on screen.
    If output_dir is specified, exports one PNG per day named true_vs_pred_{start_day}.png.
    
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with columns: start_day, true_label, predictions, longitude, latitude, prediction_date
    export_pngs : bool, optional
        if True, saves PNGs in results/predictions, optional.
    figsize : tuple, optional
        Figure size for each day's plot.
    output_dir : str, optional
        Directory to save PNG files. If None, uses get_results_predictions_dir().
    """

    # Define colors and legend
    label_names = ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
    colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), 
              cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), 
              cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), 
              cls.rgb2hex((255/255, 38/255, 38/255))]
    patches = [mpatches.Patch(color=c) for c in colors]
    CDLcmp = cls.ListedColormap(colors, name='CDL')

    # Get unique, sorted days
    days = sorted(results['start_day'].unique())

    for start_day in days:
        day_results = results[results.start_day == start_day]
        if len(day_results) == 0:
            continue
        gdf = gpd.GeoDataFrame(
            day_results,
            geometry=gpd.points_from_xy(day_results.longitude, day_results.latitude, crs="epsg:4326")
        ).to_crs("epsg:3857")
        prediction_date = day_results['prediction_date'].iloc[0]
        accuracy = sklearn.metrics.accuracy_score(day_results['true_label'], day_results['predictions'])

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # Ground Truth
        gdf.plot('true_label', s=15, ax=axs[0], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        contextily.add_basemap(axs[0], source=contextily.providers.USGS.USImagery, alpha=0.7)
        axs[0].set_title(f'Day {start_day} - Ground Truth ({prediction_date.strftime("%B %d")})', fontsize=14, fontweight='bold', pad=15)
        axs[0].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[0].set_ylabel('Latitude', fontsize=11, fontweight='bold')

        # Predictions
        gdf.plot('predictions', s=15, ax=axs[1], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        contextily.add_basemap(axs[1], source=contextily.providers.USGS.USImagery, alpha=0.7)
        axs[1].set_title(f'Day {start_day} - Predictions (Accuracy: {accuracy:.3f})', fontsize=14, fontweight='bold', pad=15)
        axs[1].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[1].set_ylabel('Latitude', fontsize=11, fontweight='bold')

        # Create legend using axs[1].legend() - bbox_to_anchor is relative to the right axis
        # (1.02, 0.5) means: x=1.02 (just right of the axis), y=0.5 (vertically centered)
        legend = axs[1].legend(patches, label_names, 
                           loc='center left', 
                           bbox_to_anchor=(1.02, 0.5),
                           title='Crop Classes',
                           title_fontsize=12,
                           fontsize=11,
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           borderpad=1)
        legend.get_title().set_fontweight('bold')

        # Export if requested (inside the loop)
        if export_pngs:
            if output_dir is None:
                output_dir = get_results_predictions_dir()
            png_path = os.path.join(output_dir, f"true_vs_pred_{start_day}.png")
            # Use bbox_inches='tight' to include the legend that's outside the axes
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")

        # Close the plot only if not in notebook (to allow display in notebooks)
        _close_figure_if_needed(fig)

def plot_cdl_vs_predictions(results, *, start_days=None, date_range=None, export_pngs=False, output_dir=None):
    """
    Plots CDL ground truth and model predictions for given start_days or date_range.
    Optionally exports each frame as a PNG in results/predictions.
    Returns a list of exported PNG paths (if any).
    
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with columns: start_day, predictions, longitude, latitude, prediction_date, 'Raw CDL Label'
    start_days : list, optional
        List of start_days to plot. If None, plots all available days.
    date_range : list, optional
        List of dates to plot. If None, uses start_days.
    export_pngs : bool, optional
        If True, saves PNGs in the output directory.
    output_dir : str, optional
        Directory to save PNG files. If None, uses get_results_predictions_dir().
    """
   
    # --- Color and label configuration for CDL ground truth ---
    # Use ordered list to ensure consistent mapping between labels and colors
    # Handle both bytes and string formats for CDL labels
    # IMPORTANT: Order matters! Each class maps to its index in the colormap
    CDL_classes_ordered_bytes = [
        b'Peanuts',
        b'Rice',                                    # Index 1 - Light blue (celeste)
        b'Other Hay/Non Alfalfa',                  # Index 2 - Light green
        b'Winter Wheat',
        b'Corn',
        b'Developed/Med Intensity',
        b'Developed/Open Space',
        b'Fallow/Idle Cropland',
        b'Developed/Low Intensity',
        b'Dbl Crop WinWht/Soybeans',
        b'Background',
        b'Cotton',
        b'Soybeans',
        b'Herbaceous Wetlands',
        b'Woody Wetlands',
        # Add missing classes found in warnings
        b'Grass/Pasture',
        b'Open Water',
        b'Pop or Orn Corn',
        b'Oats',
        b'Developed/High Intensity',
        b'Aquaculture',
        b'Deciduous Forest',
        b'Sunflower',
        b'Pecans'
    ]
    CDL_classes_ordered_str = [k.decode('utf-8') if isinstance(k, bytes) else k for k in CDL_classes_ordered_bytes]
    CDL_colors_ordered = [
        cls.rgb2hex((112/255,168/255,0/255)),      # Peanuts (dark olive green)
        cls.rgb2hex((0/255, 169/255, 230/255)),    # Rice (light blue/cyan) - INDEX 1
        cls.rgb2hex((165/255,245/255,141/255)),   # Other Hay/Non Alfalfa (light green) - INDEX 2
        cls.rgb2hex((168/255,112/255,0/255)),      # Winter Wheat (brown)
        cls.rgb2hex((255/255, 212/255, 0/255)),    # Corn (yellow)
        cls.rgb2hex((156/255, 156/255, 156/255)), # Developed/Med Intensity (gray)
        cls.rgb2hex((156/255, 156/255, 156/255)), # Developed/Open Space (gray)
        cls.rgb2hex((191/255, 191/255,122/255)),  # Fallow/Idle Cropland (light brown)
        cls.rgb2hex((156/255, 156/255, 156/255)), # Developed/Low Intensity (gray)
        cls.rgb2hex((115/255,115/255,0)),          # Dbl Crop WinWht/Soybeans (dark olive)
        cls.rgb2hex((0,0,0)),                      # Background (black)
        cls.rgb2hex((255/255, 38/255, 38/255)),   # Cotton (red)
        cls.rgb2hex((38/255, 115/255, 0/255)),     # Soybeans (dark green)
        cls.rgb2hex((128/255,179/255,179/255)),    # Herbaceous Wetlands (light blue/cyan)
        cls.rgb2hex((128/255,179/255,179/255)),    # Woody Wetlands (light blue/cyan)
        # Colors for missing classes (using official CDL colors where known)
        cls.rgb2hex((208/255,209/255,23/255)),     # Grass/Pasture (yellow-green)
        cls.rgb2hex((0/255,197/255,255/255)),      # Open Water (cyan/blue)
        cls.rgb2hex((255/255,255/255,0/255)),      # Pop or Orn Corn (yellow)
        cls.rgb2hex((171/255,112/255,41/255)),     # Oats (brown)
        cls.rgb2hex((104/255,104/255,104/255)),    # Developed/High Intensity (dark gray)
        cls.rgb2hex((102/255,204/255,255/255)),    # Aquaculture (light cyan)
        cls.rgb2hex((0/255,100/255,0/255)),         # Deciduous Forest (dark green)
        cls.rgb2hex((255/255,211/255,127/255)),    # Sunflower (light orange/yellow)
        cls.rgb2hex((112/255,168/255,0/255))       # Pecans (olive green, same as Peanuts)
    ]
    
    # Create encoder mapping: label -> index (ensuring consistent order)
    # Support both bytes and string formats
    CDL_value_encoder = {}
    for idx, (label_bytes, label_str) in enumerate(zip(CDL_classes_ordered_bytes, CDL_classes_ordered_str)):
        CDL_value_encoder[label_bytes] = idx
        CDL_value_encoder[label_str] = idx
        # Also handle numpy bytes arrays
        if hasattr(label_bytes, 'item'):
            CDL_value_encoder[label_bytes.item()] = idx
    
    label_names_cdl = CDL_classes_ordered_str.copy()
    colors_cdl = CDL_colors_ordered.copy()  # Use the ordered list directly
    patches_cdl = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_cdl, label_names_cdl)]
    base_CDLcmp = cls.ListedColormap(colors_cdl, name='CDL')

    # --- Color and label configuration for model predictions ---
    label_names_pred = ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
    colors_pred = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), cls.rgb2hex((255/255, 38/255, 38/255))]
    patches_pred = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_pred, label_names_pred)]
    pred_CDLcmp = cls.ListedColormap(colors_pred, name='CDL')

    exported_pngs = []

    # Determine which days to use
    if date_range is not None:
        days_to_plot = date_range
    elif start_days is not None:
        days_to_plot = start_days
    else:
        days_to_plot = sorted(results['start_day'].unique())

    if export_pngs:
        if output_dir is None:
            output_dir = get_results_predictions_dir()
        os.makedirs(output_dir, exist_ok=True)

    for start_day in days_to_plot:
        results_sel_doy = results[results.start_day == start_day]
        if results_sel_doy.empty:
            print(f"Warning: No data for start_day {start_day}")
            continue
        gdf = gpd.GeoDataFrame(
            results_sel_doy, 
            geometry=gpd.points_from_xy(results_sel_doy.longitude, results_sel_doy.latitude, crs="epsg:4326")
        ).to_crs("epsg:3857")
        # Map CDL labels to indices, handling missing values
        # Convert bytes to string if needed for consistent mapping
        def normalize_cdl_label(label):
            """Normalize CDL label to string format for consistent mapping."""
            if isinstance(label, bytes):
                return label.decode('utf-8')
            elif isinstance(label, str):
                return label
            elif hasattr(label, 'item'):  # numpy bytes array
                return label.item().decode('utf-8') if isinstance(label.item(), bytes) else str(label.item())
            else:
                return str(label)
        
        # Normalize all CDL labels before mapping
        gdf['normalized_CDL'] = gdf['Raw CDL Label'].apply(normalize_cdl_label)
        gdf['encoded_CDL_val'] = gdf['normalized_CDL'].map(CDL_value_encoder)
        
        # Fill NaN values with a default index (Background = 10) for unmapped classes
        if gdf['encoded_CDL_val'].isna().any():
            unmapped = gdf[gdf['encoded_CDL_val'].isna()]['normalized_CDL'].unique()
            print(f"⚠️  Warning: {len(unmapped)} unmapped CDL classes found: {unmapped}")
            # Use the last index (Background) for unmapped classes
            background_idx = len(CDL_classes_ordered_str) - 1
            gdf['encoded_CDL_val'] = gdf['encoded_CDL_val'].fillna(background_idx)
        
        # Ensure encoded values are integers and within valid range [0, num_classes-1]
        gdf['encoded_CDL_val'] = gdf['encoded_CDL_val'].astype(int)
        max_valid_idx = len(CDL_colors_ordered) - 1
        if (gdf['encoded_CDL_val'] > max_valid_idx).any():
            print(f"⚠️  Warning: Some encoded values exceed colormap range. Clamping to {max_valid_idx}")
            gdf['encoded_CDL_val'] = gdf['encoded_CDL_val'].clip(0, max_valid_idx)
        
        # Debug: Print sample of mappings to verify Rice is correctly mapped
        if start_day == days_to_plot[0]:  # Only print for first day to avoid spam
            rice_samples = gdf[gdf['normalized_CDL'] == 'Rice']['encoded_CDL_val'].head(3)
            if len(rice_samples) > 0:
                print(f"🔍 Debug: Rice samples mapped to index: {rice_samples.values}")
                print(f"   Expected index for Rice: {CDL_value_encoder.get('Rice', 'NOT FOUND')}")
                print(f"   Color at index 1 (Rice): {CDL_colors_ordered[1]}")
                print(f"   Color at index 2 (Other Hay): {CDL_colors_ordered[2]}")
        prediction_date = gdf['prediction_date'].iloc[0]

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))

        # --- Plot CDL ground truth (left) ---
        # Use vmin and vmax to ensure ListedColormap maps indices correctly
        vmin_cdl = 0
        vmax_cdl = len(CDL_colors_ordered) - 1
        gdf.plot('encoded_CDL_val', s=15, ax=axs[0], cmap=base_CDLcmp, marker='s', edgecolor='white', linewidth=0.2,
                 vmin=vmin_cdl, vmax=vmax_cdl)
        contextily.add_basemap(axs[0], source=contextily.providers.USGS.USImagery, alpha=0.7)
        axs[0].set_title("CDL Ground Truth\n(Reference Year)", fontsize=13, fontweight='bold', pad=25)
        axs[0].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[0].set_ylabel('Latitude', fontsize=11, fontweight='bold')
        # --- Plot model predictions (right) ---
        gdf.plot('predictions', s=15, ax=axs[1], cmap=pred_CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        contextily.add_basemap(axs[1], source=contextily.providers.USGS.USImagery, alpha=0.7)
        axs[1].set_title(f'Model Predictions\n(start_day={start_day}, {prediction_date.strftime("%Y-%m-%d")})', fontsize=13, fontweight='bold', pad=25)
        axs[1].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[1].set_ylabel('Latitude', fontsize=11, fontweight='bold')

        # Adjust layout to reserve space for legends
        # Use subplots_adjust - leave ~30% space on right for legends (0.70 for plots)
        # Increase wspace significantly to create gap between plots for left legend
        plt.subplots_adjust(left=0.05, right=0.70, top=0.95, bottom=0.08, wspace=0.50)
        
        # Now add legends AFTER layout adjustment, positioned relative to their axes
        # Left plot legend: positioned to the right of left plot, in the gap between plots
        # Using bbox_to_anchor relative to left axis (1.0 = right edge of left plot)
        # The wspace=0.50 creates enough gap for this legend
        legend_cdl = axs[0].legend(handles=patches_cdl, title='CDL Classes', title_fontsize=10, fontsize=8, 
                                   loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=True)
        legend_cdl.get_title().set_fontweight('bold')
        
        # Right plot legend: positioned to the right of right plot
        # Using bbox_to_anchor relative to right axis (1.0 = right edge of right plot)
        legend_pred = axs[1].legend(handles=patches_pred, title='Model Classes', title_fontsize=10, fontsize=8,
                                     loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=True)
        legend_pred.get_title().set_fontweight('bold')
        
        # Ensure axes have equal aspect ratio and are aligned
        axs[0].set_aspect('equal', adjustable='box')
        axs[1].set_aspect('equal', adjustable='box')
        
        # Set same xlim and ylim for both plots to ensure alignment across frames
        xlims = [min(axs[0].get_xlim()[0], axs[1].get_xlim()[0]), max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])]
        ylims = [min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]), max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])]
        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)
        axs[1].set_xlim(xlims)
        axs[1].set_ylim(ylims)

        # --- Export PNG if requested ---
        if export_pngs:
            png_path = os.path.join(output_dir, f"cdl_vs_pred_{start_day}.png")
            fig.savefig(png_path, bbox_inches='tight', dpi=150)
            print(f"Saved: {png_path}")
            exported_pngs.append(png_path)
        # Removed plt.show() to prevent crashes in headless environments
        # In notebooks, figures will display automatically if not closed
        _close_figure_if_needed(fig)
    return exported_pngs

def make_gif_from_pngs(png_paths, gif_path, fps=1, loop=0):
    """
    Creates a GIF from a list of PNG paths.
    - gif_path: full path where the GIF will be saved.
    - fps: frames per second.
    - loop: number of loops (0 = infinite).
    
    Note: If images have different sizes, they will be resized to match the first image.
    """
    from PIL import Image
    import numpy as np
    
    if not png_paths:
        print(f"Warning: No PNG files provided for GIF creation")
        return
    
    # Read all images
    pil_images = [Image.open(png) for png in png_paths]
    
    # Get target size from the first image
    target_size = pil_images[0].size  # (width, height)
    
    # Check if all images have the same size
    sizes = [img.size for img in pil_images]
    if len(set(sizes)) > 1:
        print(f"Note: Resizing {len([s for s in sizes if s != target_size])} images to {target_size[0]}x{target_size[1]} for GIF compatibility")
        # Resize images that don't match
        resized_images = []
        for img in pil_images:
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_images.append(np.array(img))
        images = resized_images
    else:
        images = [np.array(img) for img in pil_images]
    
    # Close PIL images to free memory
    for img in pil_images:
        img.close()
    
    imageio.mimsave(gif_path, images, format='GIF', fps=fps, loop=loop)
    print(f"GIF saved to {gif_path}")

def plot_predictions_only_series(results, export_pngs=False, figsize=(16, 8), output_dir=None):
    """
    Plots only predictions (no ground truth) for prediction mode.
    For each day, shows only the model predictions. Always displays the plot on screen.
    If output_dir is specified, exports one PNG per day named pred_only_{start_day}.png.
    
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with columns: start_day, predicted_label, longitude, latitude, prediction_date
    export_pngs : bool, optional
        if True, saves PNGs in results/predictions, optional.
    figsize : tuple, optional
        Figure size for each day's plot.
    output_dir : str, optional
        Directory to save PNG files. If None, uses get_results_predictions_dir().
    """

    # Define colors and legend
    label_names = ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
    colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), 
              cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), 
              cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), 
              cls.rgb2hex((255/255, 38/255, 38/255))]
    patches = [mpatches.Patch(color=c) for c in colors]
    CDLcmp = cls.ListedColormap(colors, name='CDL')

    # Get unique, sorted days
    days = sorted(results['start_day'].unique())

    for start_day in days:
        day_results = results[results.start_day == start_day]
        if len(day_results) == 0:
            continue
        gdf = gpd.GeoDataFrame(
            day_results,
            geometry=gpd.points_from_xy(day_results.longitude, day_results.latitude, crs="epsg:4326")
        ).to_crs("epsg:3857")
        prediction_date = day_results['prediction_date'].iloc[0]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        # Predictions only
        gdf.plot('predicted_class', s=15, ax=ax, cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        contextily.add_basemap(ax, source=contextily.providers.USGS.USImagery, alpha=0.7)
        ax.set_title(f'Day {start_day} - Model Predictions ({prediction_date.strftime("%B %d")})', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')

        fig.suptitle(f'Crop Classification Predictions: Day {start_day}', fontsize=18, fontweight='bold', y=0.98)

        # Create legend using ax.legend() - bbox_to_anchor is relative to the axes
        # (1.02, 0.5) means: x=1.02 (just right of the axis), y=0.5 (vertically centered)
        legend = ax.legend(patches, label_names, 
                          loc='center left', 
                          bbox_to_anchor=(1.02, 0.5),
                          title='Crop Classes',
                          title_fontsize=12,
                          fontsize=11,
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          borderpad=1)
        legend.get_title().set_fontweight('bold')

        # Export if requested (inside the loop)
        if export_pngs:
            if output_dir is None:
                output_dir = get_results_predictions_dir()
            png_path = os.path.join(output_dir, f"pred_only_{start_day}.png")
            # Use bbox_inches='tight' to include the legend that's outside the axes
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")

        # Removed plt.show() to prevent crashes in headless environments
        # In notebooks, figures will display automatically if not closed
        _close_figure_if_needed(fig)

def plot_predictions_with_confidence(results, export_pngs=False, figsize=(20, 8), output_dir=None):
    """
    Plots predictions with confidence scores for prediction mode.
    For each day, shows predictions and confidence scores side by side.
    If output_dir is specified, exports one PNG per day named pred_confidence_{start_day}.png.
    
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with columns: start_day, predicted_label, confidence_score, longitude, latitude, prediction_date
    export_pngs : bool, optional
        if True, saves PNGs in results/predictions, optional.
    figsize : tuple, optional
        Figure size for each day's plot.
    output_dir : str, optional
        Directory to save PNG files. If None, uses get_results_predictions_dir().
    """

    # Define colors and legend for predictions
    label_names = ['Uncultivated', 'Cultivated', 'No Crop Growing', 'Soybeans', 'Rice', 'Corn', 'Cotton']
    colors = [cls.rgb2hex((156/255,156/255,156/255)), cls.rgb2hex((0/255, 175/255, 77/255)), 
              cls.rgb2hex((0/255, 0/255, 0/255)), cls.rgb2hex((38/255, 115/255, 0/255)), 
              cls.rgb2hex((0/255, 169/255, 230/255)), cls.rgb2hex((255/255, 212/255, 0/255)), 
              cls.rgb2hex((255/255, 38/255, 38/255))]
    patches = [mpatches.Patch(color=c) for c in colors]
    CDLcmp = cls.ListedColormap(colors, name='CDL')

    # Get unique, sorted days
    days = sorted(results['start_day'].unique())

    for start_day in days:
        day_results = results[results.start_day == start_day]
        if len(day_results) == 0:
            continue
        gdf = gpd.GeoDataFrame(
            day_results,
            geometry=gpd.points_from_xy(day_results.longitude, day_results.latitude, crs="epsg:4326")
        ).to_crs("epsg:3857")
        prediction_date = day_results['prediction_date'].iloc[0]

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # Predictions (left) - with crop class legend
        gdf.plot('predicted_class', s=15, ax=axs[0], cmap=CDLcmp, marker='s', edgecolor='white', linewidth=0.2)
        contextily.add_basemap(axs[0], source=contextily.providers.USGS.USImagery, alpha=0.7)
        axs[0].set_title(f'Day {start_day} - Predictions ({prediction_date.strftime("%B %d")})', fontsize=14, fontweight='bold', pad=15)
        axs[0].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[0].set_ylabel('Latitude', fontsize=11, fontweight='bold')
        
        # Add crop class legend to predictions plot (left)
        legend = axs[0].legend(patches, label_names, 
                              loc='upper left',
                              title='Crop Classes',
                              title_fontsize=10,
                              fontsize=9,
                              frameon=True,
                              fancybox=True,
                              framealpha=0.9)
        legend.get_title().set_fontweight('bold')

        # Confidence scores (right) - with colorbar
        # Use legend=False and create a proper colorbar instead
        scatter = gdf.plot('confidence_score', s=15, ax=axs[1], cmap='viridis', marker='s', 
                          edgecolor='white', linewidth=0.2, legend=False)
        contextily.add_basemap(axs[1], source=contextily.providers.USGS.USImagery, alpha=0.7)
        axs[1].set_title(f'Day {start_day} - Confidence Scores', fontsize=14, fontweight='bold', pad=15)
        axs[1].set_xlabel('Longitude', fontsize=11, fontweight='bold')
        axs[1].set_ylabel('Latitude', fontsize=11, fontweight='bold')
        
        # Create a proper colorbar for confidence scores
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=gdf['confidence_score'].min(), 
                                                     vmax=gdf['confidence_score'].max()))
        sm._A = []  # Required for ScalarMappable
        cbar = fig.colorbar(sm, ax=axs[1], shrink=0.8, pad=0.02)
        cbar.set_label('Confidence', fontsize=11, fontweight='bold')

        fig.suptitle(f'Crop Classification Predictions with Confidence: Day {start_day}', fontsize=18, fontweight='bold', y=0.98)

        # Export if requested (inside the loop)
        if export_pngs:
            if output_dir is None:
                output_dir = get_results_predictions_dir()
            png_path = os.path.join(output_dir, f"pred_confidence_{start_day}.png")
            # Use bbox_inches='tight' to include the legend that's outside the axes
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")

        # Removed plt.show() to prevent crashes in headless environments
        # In notebooks, figures will display automatically if not closed
        _close_figure_if_needed(fig)