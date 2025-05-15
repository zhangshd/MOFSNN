#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to summarize model performance metrics from all versions in the 
results/cgcnn_models_opt directory and rank them by validation metrics.
"""

import os
import sys
from pathlib import Path

# Add src/cgcnn to Python path to resolve yaml loading issues with custom classes
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "src" / "cgcnn"))
sys.path.append(str(ROOT_DIR / "src"))

import pandas as pd
import numpy as np
import yaml
import time
import argparse
from datetime import datetime

def get_best_model_val_metric(checkpoint_dir):
    """
    Extract the validation metric from the best model checkpoint filename.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        float: The validation metric value
    """
    best_model_files = list(checkpoint_dir.glob('best*.ckpt'))
    if not best_model_files:
        return None, None
    
    best_model_file = best_model_files[0]
    try:
        # Extract val_Metric from the filename (example: best-epoch=130-val_Metric=0.595.ckpt)
        val_metric = float(best_model_file.name.split('=')[-1].replace('.ckpt', ''))
        epoch_num = int(best_model_file.name.split('=')[1].split('-')[0])
        return val_metric, epoch_num
    except (ValueError, IndexError):
        return None, None


def collect_model_metrics(results_dir):
    """
    Collect metrics from all model versions in the specified directory.
    
    Args:
        results_dir: Path to the root directory containing model results
        
    Returns:
        DataFrame: Combined metrics from all models
    """
    results_dir = Path(results_dir)
    dfs = []
    
    # Collect task directories
    task_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.endswith('.db')]
    
    print(f"Found {len(task_dirs)} task directories")
    
    for task_dir in sorted(task_dirs):
        print(f"Processing {task_dir.name}")
        version_count = 0
        
        # Process each version directory
        for version_dir in sorted(task_dir.glob('version_*'), key=lambda x: int(x.name.split('_')[-1])):
            if not version_dir.is_dir():
                continue

            if version_count >= 50:
                print(f"  - Skipping {version_dir.name} (more than 50 versions)")
                break
                
            # Check if test metrics exist
            if not (version_dir / 'test_metrics.csv').exists():
                continue
                
            # Read test metrics
            try:
                df_test = pd.read_csv(version_dir / 'test_metrics.csv')
            except pd.errors.ParserError:
                print(f"Error parsing test metrics in {version_dir}")
                continue
                
            # Get validation metrics
            val_metric = None
            best_epoch = None
            df_val = None
            checkpoint_dir = version_dir / 'checkpoints'
            if checkpoint_dir.exists():
                val_metric, best_epoch = get_best_model_val_metric(checkpoint_dir)
            if (version_dir / 'val_metrics.csv').exists():
                try:
                    df_val = pd.read_csv(version_dir / 'val_metrics.csv')
                    df_val.rename(columns={'Unnamed: 0': 'Epoch'}, inplace=True)
                    df_val = df_val[df_val['Epoch'] == best_epoch]
                except (pd.errors.ParserError, IndexError):
                    print(f"Error parsing validation metrics in {version_dir}")

            # If still no validation metric, skip this version
            if val_metric is None or df_val is None:
                continue
                
            # Read hyperparameters
            hparams_file = version_dir / 'hparams.yaml'
            if hparams_file.exists():
                try:
                    with open(hparams_file, 'r') as f:
                        hparams = yaml.load(f, Loader=yaml.Loader)  # Use yaml.Loader to handle custom classes
                except Exception as e:
                    print(f"Error loading hyperparameters from {hparams_file}: {e}")
                    hparams = {}
            else:
                hparams = {}
            
            # Select important hyperparameters
            params_needed = [
                'atom_fea_len', 'h_fea_len', 'n_conv', 'n_h', 'dropout_prob', 'extra_fea_len', 
                'use_extra_fea', 'use_cell_params', 'att_S', 'loss_aggregation', 'dl_sampler', 
                'atom_layer_norm', 'lr', 'augment', 'lr_mult', 'group_lr', 'optim_config', 
                'patience', 'att_pooling', 'task_norm', 'reconstruct', 'max_graph_len', 
                'max_epochs', 'batch_size', 'task_weights'
            ]
            selected_hparams = {k: hparams.get(k) for k in params_needed if k in hparams}
            
            # Skip versions with augmentation or extra features if needed
            if selected_hparams.get("augment") is True or selected_hparams.get("use_extra_fea") is True:
                continue
            
            df_test.rename(columns={'Unnamed: 0': 'Epoch'}, inplace=True)
            if df_test.empty:
                print(f"Warning: Empty test metrics for {version_dir}")
            metric_cols = [col for col in df_test.columns if col.endswith('R2Score') or col.endswith('_Accuracy')]
            test_metric_mean = df_test[metric_cols].mean(axis=1).iloc[0] if len(df_test) > 0 else float('nan')
            
            # Add metadata to the dataframe
            df_test.insert(1, "TestMetric", test_metric_mean)
            df_test.insert(1, "ValMetric", val_metric)
            df_test.insert(1, "Parameters", str(selected_hparams))
            df_test.insert(1, "Version", version_dir.name)
            df_test.insert(1, "Model", task_dir.name.split("_seed42_")[-1])
            df_test.insert(1, "Task", task_dir.name.split("_seed42_")[0])
            df_test["Path"] = str(version_dir)
            if not df_val.empty:
                for col in df_val.columns:
                    if col.endswith('R2Score') or col.endswith('_Accuracy') or \
                        col.endswith('MeanAbsoluteError') or col == 'Epoch':
                        df_test[col] = df_val[col].values
            
            dfs.append(df_test)
            version_count += 1
        
        print(f"  - Found {version_count} valid versions")
    
    # Combine all dataframes
    if not dfs:
        print("No valid model results found!")
        return None
        
    return pd.concat(dfs)


def save_results_to_excel(df, output_file):
    """
    Save model results to Excel with one sheet per task.
    
    Args:
        df: DataFrame containing all model results
        output_file: Path to save the Excel file
    """
    # Standardize task names
    df["Task"] = df["Task"].apply(lambda x: x.replace(
        "WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling", "WS24_all"))
    
    # Sort by validation metric within each task
    df = df.sort_values(by=["Task", "ValMetric"], ascending=[True, False])
    
    # Select and order columns
    important_cols = ["Task", "Model", "Version", "Epoch", "ValMetric", "TestMetric", "Parameters"]
    metric_cols = [col for col in df.columns if col.endswith('R2Score') or 
                  col.endswith('_Accuracy') or col.endswith('MeanAbsoluteError')]
    val_metric_cols  = [col for col in metric_cols if "val_" in col]
    test_metric_cols = [col for col in metric_cols if "test_" in col]
    other_cols = [col for col in df.columns if col not in metric_cols and col not in important_cols+["Path"]]

    df = df[important_cols + test_metric_cols + val_metric_cols + other_cols + ["Path"]]

    # Get unique tasks
    tasks = df["Task"].unique()
    
    # Save to Excel with one sheet per task
    with pd.ExcelWriter(output_file) as writer:
        for task in sorted(tasks):
            sub_df = df[df["Task"] == task]
            # Ensure sheet name is valid for Excel (max 31 chars, no special chars)
            sheet_name = task[:31].replace('/', '_').replace('\\', '_')
            if not sheet_name:  # If task name is empty after cleaning
                sheet_name = f"Task_{tasks.index(task)}"
            sub_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Add an overview sheet with all models
        df.to_excel(writer, sheet_name="All Models", index=False)
    
    print(f"Results saved to {output_file}")
    return df


def format_excel_file(excel_file):
    """
    Format the Excel file with colors for good/excellent performance.
    
    Args:
        excel_file: Path to the Excel file to format
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print("openpyxl not installed. Excel formatting skipped.")
        return
    
    wb = load_workbook(excel_file)
    font = Font(name='Arial')
    yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')  # Excellent
    green_fill = PatternFill(start_color='00B050', end_color='00B050', fill_type='solid')  # Good
    
    # Define thresholds for different metrics
    metric_thresholds = {
        'TSD/test_R2Score': {'excellent': 0.485, 'good': 0.455},
        'TSD/test_MeanAbsoluteError': {'excellent': 42.5, 'good': 44.5, 'is_lower_better': True},
        'SSD/test_Accuracy': {'excellent': 0.775, 'good': 0.755},
        'WS24_water/test_Accuracy': {'excellent': 0.795, 'good': 0.725},
        'WS24_water4/test_Accuracy': {'excellent': 0.645, 'good': 0.595},
        'WS24_acid/test_Accuracy': {'excellent': 0.805, 'good': None},
        'WS24_base/test_Accuracy': {'excellent': 0.745, 'good': None},
        'WS24_boiling/test_Accuracy': {'excellent': 0.805, 'good': 0.725}
    }
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"Formatting sheet: {sheet_name}")
        
        # Set font for all cells and adjust column widths
        for row in ws.iter_rows():
            for cell in row:
                if not hasattr(cell, 'font'):  # Skip merged cells
                    continue
                cell.font = font
                
        # Process columns for width adjustment and conditional formatting
        header_row = next(ws.rows, None)
        if not header_row:
            continue  # Empty sheet
            
        # Process each column based on header
        for i, cell in enumerate(header_row):
            if cell.value is None:
                continue
                
            # Get column letter and adjust width
            col_idx = cell.column_letter
            col_name = str(cell.value)
            
            # Set column width
            ws.column_dimensions[col_idx].width = min(len(col_name) + 2, 20)
            
            # Apply conditional formatting if this is a metric column
            if col_name in metric_thresholds:
                thresholds = metric_thresholds[col_name]
                is_lower_better = thresholds.get('is_lower_better', False)
                
                # Get data cells in this column (skip header)
                for data_row in list(ws.rows)[1:]:
                    data_cell = data_row[i]
                    
                    # Skip empty cells or cells without value attribute
                    if not hasattr(data_cell, 'value') or data_cell.value is None:
                        continue
                    
                    # Apply conditional formatting based on value
                    if is_lower_better:
                        if thresholds['excellent'] is not None and data_cell.value < thresholds['excellent']:
                            data_cell.fill = yellow_fill
                        elif thresholds['good'] is not None and data_cell.value < thresholds['good']:
                            data_cell.fill = green_fill
                    else:
                        if thresholds['excellent'] is not None and data_cell.value >= thresholds['excellent']:
                            data_cell.fill = yellow_fill
                        elif thresholds['good'] is not None and data_cell.value >= thresholds['good']:
                            data_cell.fill = green_fill
    
    wb.save(excel_file)
    print(f"Excel formatting completed: {excel_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Summarize model performance metrics from CGCNN model results.'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=str(ROOT_DIR/"results/cgcnn_models_opt/rand0"),
        help='Directory containing model results (default: results/cgcnn_models_opt/rand0)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to save the Excel output file (default: <results_dir>/<date>_model_results_summary.xlsx)'
    )
    
    parser.add_argument(
        '--skip-formatting',
        action='store_true',
        help='Skip Excel formatting (useful if openpyxl is not installed)'
    )
    
    return parser.parse_args()


def main():
    """Main function to execute the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define the directory containing model results
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist.")
        sys.exit(1)
    
    # Current timestamp for output filename
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Use provided output file path or generate default
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = results_dir / f"{timestamp}_model_results_summary.xlsx"
    
    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect model metrics
    print(f"Collecting model metrics from {results_dir}")
    df = collect_model_metrics(results_dir)
    
    if df is None or df.empty:
        print("No valid model results found.")
        sys.exit(1)
    
    # Save results to Excel
    print(f"Saving results to {output_file}")
    save_results_to_excel(df, output_file)
    
    # Format Excel file
    if not args.skip_formatting:
        print("Formatting Excel file")
        format_excel_file(output_file)
    
    print(f"Summary completed! Results saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
