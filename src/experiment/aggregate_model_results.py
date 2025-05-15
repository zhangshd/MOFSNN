#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to aggregate model performance metrics across multiple randomization runs
and create a summary of the best models with mean and standard deviation.
"""

import os
import sys
from pathlib import Path
import re

# Add src/cgcnn to Python path to resolve yaml loading issues with custom classes
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "src" / "cgcnn"))
sys.path.append(str(ROOT_DIR / "src"))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import yaml

from sum_model_results import collect_model_metrics, save_results_to_excel, format_excel_file

model_configs = {
    "model_dirs_map": {
        "Baseline-TSD": {
            "Path": [
                "results/ml_models/TSD/RAC_and_zeo_features_with_id_prop/Label",
                "results/ml_models/TSD/RAC_and_zeo_features_with_id_prop_rand0/Label",
                "results/ml_models/TSD/RAC_and_zeo_features_with_id_prop_rand1/Label",
                "results/ml_models/TSD/RAC_and_zeo_features_with_id_prop_rand2/Label",
                "results/ml_models/TSD/RAC_and_zeo_features_with_id_prop_rand3/Label",
            ],
            "DisplayName": "Baseline",
            "Model": "RandomForestRegressor"
        },
        "Baseline-SSD": {
            "Path": [
                "results/ml_models/SSD/RAC_and_zeo_features_with_id_prop/Label",
                "results/ml_models/SSD/RAC_and_zeo_features_with_id_prop_rand0/Label",
                "results/ml_models/SSD/RAC_and_zeo_features_with_id_prop_rand1/Label",
                "results/ml_models/SSD/RAC_and_zeo_features_with_id_prop_rand2/Label",
                "results/ml_models/SSD/RAC_and_zeo_features_with_id_prop_rand3/Label",
            ],
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "Baseline-WS24_water": {
            "Path": [
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/water_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand0/water_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand1/water_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand2/water_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand3/water_label",
            ],
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "Baseline-WS24_water4": {
            "Path": [
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/water4_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand0/water4_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand1/water4_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand2/water4_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand3/water4_label",
            ],
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "Baseline-WS24_acid": {
            "Path": [
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/acid_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand0/acid_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand1/acid_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand2/acid_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand3/acid_label",
            ],
            "DisplayName": "Baseline",
            "Model": "GaussianProcessClassifier"
        },
        "Baseline-WS24_base": {
            "Path": [
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/base_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand0/base_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand1/base_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand2/base_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand3/base_label",
            ],
            "DisplayName": "Baseline",
            "Model": "GaussianProcessClassifier"
        },
        "Baseline-WS24_boiling": {
            "Path": [
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/boiling_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand0/boiling_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand1/boiling_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand2/boiling_label",
                "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop_rand3/boiling_label",
            ],
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "TSD": {
            "Path": [
                "results/cgcnn_models/TSD_seed42_cgcnn_raw/version_29",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "SSD": {
            "Path": [
                "results/cgcnn_models/SSD_seed42_cgcnn_raw/version_4",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_water": {
            "Path": [
                "results/cgcnn_models/WS24_water_seed42_cgcnn_raw/version_29",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_water4": {
            "Path": [
                "results/cgcnn_models/WS24_water4_seed42_cgcnn_raw/version_3",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_acid": {
            "Path": [
                "results/cgcnn_models/WS24_acid_seed42_cgcnn_raw/version_47",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_base": {
            "Path": [
                "results/cgcnn_models/WS24_base_seed42_cgcnn_raw/version_0",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_boiling": {
            "Path": [
                "results/cgcnn_models/WS24_boiling_seed42_cgcnn_raw/version_7",
            ],
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "TSD_SSD_WS24_all": {
            "Path": [
                "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_cgcnn_raw/version_24",
            ],
            "DisplayName": "CGCNN_MT",
            "Model": "cgcnn_raw"
        },
        "TSD_SSD_WS24_all_attn": {
            "Path": [
                "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43",
            ],
            "DisplayName": "MOFSNN",
            "Model": "att_cgcnn"
        }
    },
    "tasks": [
        "TSD",
        "SSD",
        "WS24_water",
        "WS24_water4",
        "WS24_acid",
        "WS24_base",
        "WS24_boiling"
    ],
    "task_types": {
        "TSD": "regression",
        "SSD": "classification",
        "WS24_water": "classification",
        "WS24_water4": "classification",
        "WS24_acid": "classification",
        "WS24_base": "classification",
        "WS24_boiling": "classification"
    },
    "comparison_groups": {
        "standard": [
            "Baseline-TSD",
            "Baseline-SSD", 
            "Baseline-WS24_water",
            "Baseline-WS24_water4",
            "Baseline-WS24_acid",
            "Baseline-WS24_base",
            "Baseline-WS24_boiling",
            "TSD",
            "SSD",
            "WS24_water",
            "WS24_water4",
            "WS24_acid",
            "WS24_base",
            "WS24_boiling",
            "TSD_SSD_WS24_all",
            "TSD_SSD_WS24_all_attn"
        ]
    }
}


def find_rand_directories(base_dir):
    """
    Find all rand directories in the base directory.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        list: List of rand directory paths
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base directory {base_path} does not exist.")
        return []
    
    # Look for directories named rand0, rand1, etc.
    rand_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and re.match(r'rand\d+', item.name):
            rand_dirs.append(item)
    
    return sorted(rand_dirs)


def process_single_directory(rand_dir, timestamp):
    """
    Process a single rand directory to collect and summarize model metrics.
    
    Args:
        rand_dir: Path to the rand directory
        timestamp: Current timestamp for filename
        
    Returns:
        DataFrame: Processed model metrics dataframe
    """
    print(f"\nProcessing directory: {rand_dir}")
    
    # Collect model metrics
    df = collect_model_metrics(rand_dir)
    if df is None or df.empty:
        print(f"No valid model results found in {rand_dir}")
        return None
    
    # Save results to Excel
    output_file = rand_dir / f"{timestamp}_model_results_summary.xlsx"
    df = save_results_to_excel(df, output_file)
    
    # Format Excel file
    format_excel_file(output_file)
    
    print(f"Results for {rand_dir} saved to: {output_file}")
    
    return df


def select_best_models(df):
    """
    For each Task and Model combination, select the version with the highest ValMetric.
    
    Args:
        df: DataFrame containing model metrics
        
    Returns:
        DataFrame: DataFrame with only the best versions for each Task/Model
    """
    # Standardize task names first
    df["Task"] = df["Task"].apply(lambda x: x.replace(
        "WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling", "WS24_all"))
    best_models = []
    # Group by Task and Model, select the row with highest ValMetric
    for group_keys, group_df in df.groupby(['Task', 'Model']):
        # Sort by ValMetric in descending order
        group_df = group_df.sort_values(by='ValMetric', ascending=False)
        
        # Select the first row (highest ValMetric)
        best_row = group_df.iloc[0]
        best_models.append(best_row)
        
    best_models = pd.DataFrame(best_models).reset_index(drop=True)
    
    return best_models


def aggregate_across_rands(all_best_models, rand_dirs):
    """
    Aggregate metrics for best models across all rand directories.
    
    Args:
        all_best_models: Dictionary of DataFrames with best models for each rand
        rand_dirs: List of rand directories
        
    Returns:
        DataFrame: Combined DataFrame with individual models and mean/std metrics
    """
    # List to hold all individual best models and their aggregates
    all_models = []
    
    if not all_best_models:
        print("No data to aggregate")
        return None
    
    # Get the first DataFrame to identify all possible Task/Model combinations
    first_df = next(iter(all_best_models.values()))
    task_model_combinations = []
    for _, row in first_df.iterrows():
        task_model_combinations.append((row['Task'], row['Model']))
    
    # Process each Task/Model combination
    for task, model in task_model_combinations:
        # Collect models and metrics from each rand directory
        model_rows = []
        model_metrics = {}
        
        # First, collect all individual best models
        for rand_name, df in all_best_models.items():
            # Find the row for this Task/Model
            matching_rows = df[(df['Task'] == task) & (df['Model'] == model)]
            if matching_rows.empty:
                continue
                
            # Get the full row of data for this model
            row_data = matching_rows.iloc[0].to_dict()
            row_data['RandomSeed'] = rand_name  # Add RandomSeed column
            model_rows.append(row_data)
            
            # Collect metrics for averaging
            metric_cols = [col for col in row_data.keys() if col.endswith(('R2Score', '_Accuracy', 'MeanAbsoluteError'))]
            for col in metric_cols:
                if col not in model_metrics:
                    model_metrics[col] = []
                model_metrics[col].append(row_data[col])
        
        # Add all individual models to the results
        all_models.extend(model_rows)
        
        # Create aggregated metrics row
        mean_row = {
            'Task': task,
            'Model': model,
            'RandomSeed': 'Mean',
            'Epoch': None,
            'Path': None,
            'Version': None,
            'Parameters': None,
            'RandCount': len(model_rows)
        }
        
        # Calculate mean and std for ValMetric and TestMetric
        val_metrics = [row['ValMetric'] for row in model_rows if not pd.isna(row['ValMetric'])]
        test_metrics = [row['TestMetric'] for row in model_rows if not pd.isna(row['TestMetric'])]
        
        if val_metrics:
            mean_row['ValMetric'] = f"{np.mean(val_metrics):.4f} ± {np.std(val_metrics):.4f}"
        else:
            mean_row['ValMetric'] = None
            
        if test_metrics:
            mean_row['TestMetric'] = f"{np.mean(test_metrics):.4f} ± {np.std(test_metrics):.4f}"
        else:
            mean_row['TestMetric'] = None
        
        # Calculate mean and std for all metrics
        for metric, values in model_metrics.items():
            values = [v for v in values if not pd.isna(v)]
            if values:
                mean_row[metric] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
            else:
                mean_row[metric] = None
        
        # Add aggregate row
        all_models.append(mean_row)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(all_models)
    
    # Define column order
    important_cols = ["Task", "Model", "RandomSeed", "RandCount", "Version", "Epoch", 
                     "ValMetric", "TestMetric", "Parameters", "Path"]
    
    # Get metric columns (those ending with R2Score, Accuracy, MeanAbsoluteError)
    metric_cols = [col for col in combined_df.columns 
                  if col not in important_cols and 
                  col.endswith(('R2Score', '_Accuracy', 'MeanAbsoluteError'))]
    
    # Get any other columns
    other_cols = [col for col in combined_df.columns 
                 if col not in important_cols and col not in metric_cols]
    
    # Sort first by Task, then by Model, then put Mean rows first, then sort by ValMetric
    combined_df['SortOrder'] = combined_df['RandomSeed'].apply(lambda x: 0 if x == 'Mean' else 1)
    combined_df = combined_df.sort_values(
        by=["Task", "Model", "SortOrder", "ValMetric"], 
        ascending=[True, True, True, False]
    )
    combined_df = combined_df.drop(columns=['SortOrder'])
    
    # Reorder columns
    all_cols = [col for col in important_cols if col in combined_df.columns]
    all_cols += sorted(metric_cols)
    all_cols += [col for col in other_cols if col not in all_cols]
    
    combined_df = combined_df[all_cols]
    
    return combined_df


def save_aggregated_results(combined_df, output_file):
    """
    Save aggregated results to Excel with one sheet per task.
    
    Args:
        combined_df: DataFrame containing individual models and aggregated metrics
        output_file: Path to save the Excel output file
    """
    if combined_df is None or combined_df.empty:
        print("No results to save.")
        return

    # Save to Excel with one sheet per task
    with pd.ExcelWriter(output_file) as writer:
        tasks = combined_df["Task"].unique()
        for task in sorted(tasks):
            # Ensure sheet name is valid for Excel (max 31 chars, no special chars)
            sheet_name = task[:31].replace('/', '_').replace('\\', '_')
            if not sheet_name:  # If task name is empty after cleaning
                sheet_name = f"Task_{list(sorted(tasks)).index(task)}"
                
            sub_df = combined_df[combined_df["Task"] == task]
            sub_df.to_excel(writer, sheet_name=sheet_name, index=False)

            if sheet_name == "All Models":
                continue
            elif sheet_name == "TSD_SSD_WS24_all":
                for model_name in sub_df["Model"].unique():
                    model_paths = sub_df.loc[(sub_df["Model"] == model_name)&(sub_df["Version"].notna())]["Path"].tolist()
                    if model_name == "att_cgcnn":
                        model_configs["model_dirs_map"]["TSD_SSD_WS24_all_attn"]["Path"] += [
                            os.path.relpath(path, start=ROOT_DIR) for path in model_paths]
                    elif model_name == "cgcnn_raw":
                        model_configs["model_dirs_map"]["TSD_SSD_WS24_all"]["Path"] += [
                            os.path.relpath(path, start=ROOT_DIR) for path in model_paths]
            else:
                for model_name in sub_df["Model"].unique():
                    model_paths = sub_df.loc[(sub_df["Model"] == model_name)&(sub_df["Version"].notna())]["Path"].tolist()
                    model_configs["model_dirs_map"][sheet_name]["Path"] += [
                        os.path.relpath(path, start=ROOT_DIR) for path in model_paths]
        
        # Add an overview sheet with all models
        combined_df.to_excel(writer, sheet_name="All Models", index=False)
        # save model configurations to yaml file
        with open(output_file.with_suffix('.yaml'), 'w') as yaml_file:
            yaml.dump(model_configs, yaml_file)

    print(f"Combined results saved to {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate model metrics across multiple rand directories.'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default=str(ROOT_DIR/"results/cgcnn_models_opt"),
        help='Base directory containing rand0, rand1, etc. (default: results/cgcnn_models_opt)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to save the aggregated Excel file (default: <base_dir>/<date>_aggregated_model_results.xlsx)'
    )
    
    return parser.parse_args()


def main():
    """Main function to execute the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define the base directory containing rand directories
    base_dir = Path(args.base_dir)
    
    # Find all rand directories
    rand_dirs = find_rand_directories(base_dir)
    if not rand_dirs:
        print(f"No rand directories found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {len(rand_dirs)} rand directories: {[d.name for d in rand_dirs]}")
    # Current timestamp for output filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Process each rand directory
    all_best_models = {}
    for rand_dir in rand_dirs:
        df = process_single_directory(rand_dir, timestamp)
        if df is not None:
            print("Number of models in this directory:", len(df))
            # Select best models for each Task/Model combination
            best_models = select_best_models(df)
            # print("best_models", best_models)
            all_best_models[rand_dir.name] = best_models
    
    # Aggregate results across all rand directories
    print("\nAggregating results across all rand directories...")
    agg_df = aggregate_across_rands(all_best_models, rand_dirs)
    
    # Define output file path for aggregated results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = base_dir / f"aggregated_model_results.xlsx"
    
    # Save aggregated results
    save_aggregated_results(agg_df, output_file)
    
    print("\nAggregation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
