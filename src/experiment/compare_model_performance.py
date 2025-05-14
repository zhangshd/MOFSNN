#!/usr/bin/env python
'''
Author: zhangsd
Date: 2024-06-01
Description: Compare the performance of different models on test datasets
and save summarized results to Excel.
'''

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score,
    balanced_accuracy_score, roc_auc_score
)
from typing import Dict, List, Optional, Any, Tuple
from matplotlib.figure import Figure

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory of the project (two levels up from the script directory)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Add the src directory to the Python path
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Default paths
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results/model_comparison")

def read_results_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV results file and return the DataFrame.

    Args:
        file_path: Path to the CSV file with model results

    Returns:
        DataFrame with the results data or None if file not found
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def calculate_metrics(df: pd.DataFrame, task_type: str = "regression") -> Dict[str, float]:
    """
    Calculate performance metrics from results DataFrame.

    Args:
        df: DataFrame with 'GroundTruth' and 'Predicted' columns
        task_type: Type of task - 'regression' or 'classification'

    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}

    try:
        # Common preprocessing
        y_true = df["GroundTruth"].values
        y_pred = df["Predicted"].values

        if task_type == "regression":
            # Calculate regression metrics
            metrics["R2"] = r2_score(y_true, y_pred)
            metrics["MAE"] = mean_absolute_error(y_true, y_pred)

        else:  # classification
            # Calculate classification metrics
            metrics["ACC"] = accuracy_score(y_true, y_pred)
            metrics["BACC"] = balanced_accuracy_score(y_true, y_pred)

            # Calculate AUROC if probabilities are available
            if "Prob" in df.columns:
                probs = df["Prob"].values
                # Check if probabilities are stored as strings (lists)
                if isinstance(probs[0], str):
                    try:
                        # Try to parse probability strings
                        parsed_probs = []
                        for p in probs:
                            p_val = eval(p)
                            if isinstance(p_val, list):
                                parsed_probs.append(p_val)
                            else:
                                parsed_probs.append([1-p_val, p_val])  # Binary case
                        probs = np.array(parsed_probs)
                    except Exception:
                        print("Warning: Could not parse probability strings")
                        probs = None

                if probs is not None:
                    try:
                        # Check if this is a multi-class problem
                        if len(probs.shape) > 1 and probs.shape[1] > 2:
                            # Multi-class case (including 4-class tasks)
                            metrics["AUROC"] = roc_auc_score(y_true, probs, multi_class='ovo', average='macro')
                        elif len(probs.shape) > 1 and probs.shape[1] == 2:
                            # Binary case with 2-column probabilities
                            metrics["AUROC"] = roc_auc_score(y_true, probs[:, 1])
                        else:
                            # Binary case with 1-column probabilities
                            metrics["AUROC"] = roc_auc_score(y_true, probs)
                    except Exception as e:
                        print(f"Warning: Could not calculate AUROC: {e}")
                        metrics["AUROC"] = np.nan

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        if task_type == "regression":
            metrics = {"R2": np.nan, "MAE": np.nan}
        else:
            metrics = {"ACC": np.nan, "BACC": np.nan, "AUROC": np.nan}

    return metrics

def process_model_results(model_path: str, display_name: str, actual_ml_model_type: Optional[str],
                         model_key: str,
                         tasks: List[str], task_types: Dict[str, str],
                         split: str = "test") -> List[Dict[str, Any]]:
    """
    Process results for a single model across multiple tasks.

    Args:
        model_path: Path to the model results directory
        display_name: Display name for the model (used in output 'Model' column)
        actual_ml_model_type: The actual ML algorithm string (e.g., "RandomForestRegressor")
                              if this is a baseline ML model (derived from config's "Model" field).
                              For DL models, this is also their "Model" field from config (e.g., "att_cgcnn").
        model_key: The unique key for the model from the configuration (e.g., "Baseline-TSD", "MOFSNN")
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types ('regression' or 'classification')
        split: Data split to analyze ('test' or 'external_test')

    Returns:
        List of dictionaries with metrics for each task
    """
    results = []

    for task in tasks:
        found_for_current_task = False
        task_metrics_dict = {}

        is_baseline_model_by_key = model_key.startswith("Baseline-")

        if is_baseline_model_by_key:
            if not actual_ml_model_type: # Baseline models must have the 'Model' field in config specifying the algorithm
                print(f"Warning: Baseline model '{model_key}' is missing the 'Model' (algorithm type) field in its configuration. Skipping task '{task}'.")
                continue

            expected_task_for_baseline = None
            # Infer task from model_key like "Baseline-TaskName"
            parts = model_key.split('-', 1) # model_key starts with "Baseline-"
            if len(parts) > 1:
                expected_task_for_baseline = parts[1]
            
            # If this baseline model is task-specific, only process its designated task.
            if expected_task_for_baseline and task != expected_task_for_baseline:
                continue  # Skip if the current global task is not the one for this specific baseline model

            # Construct file path for baseline model results
            # e.g., test_predicted_RandomForestRegressor.csv
            result_file_path = os.path.join(model_path, f"{split}_predicted_{actual_ml_model_type}.csv")
            df = read_results_file(result_file_path)

            if df is not None:
                found_for_current_task = True
                current_task_type = task_types.get(task, "classification")
                calculated_metrics = calculate_metrics(df, current_task_type)
                
                task_metrics_dict = {"Task": task, "Model": display_name, **calculated_metrics}
                results.append(task_metrics_dict)
                print(f"Processed {task} for baseline model '{model_key}' (algo: {actual_ml_model_type}) using file '{result_file_path}'")
            else:
                # Warning if the specific file for the baseline model's designated task was not found
                if not expected_task_for_baseline or task == expected_task_for_baseline: # Only warn if we were expecting this file
                    print(f"Warning: Results file not found for baseline model '{model_key}' (algo: {actual_ml_model_type}) for task '{task}' at '{result_file_path}'")
        
        else:  # This is a DL model (model_key does not start with "Baseline-")
            # Check if the DL model's key itself is a recognized task name (e.g., model_key="TSD")
            # task_types.keys() provides all defined task names.
            is_dl_model_for_a_specific_task_only = model_key in task_types.keys()

            if is_dl_model_for_a_specific_task_only:
                # If this DL model is named after a specific task (e.g. model_key="TSD"),
                # it should only be processed for that particular task.
                if model_key != task: # 'task' is the current task from the outer loop
                    continue # Skip if current_task_from_outer_loop is not the specific task this model is for.
            
            # If we reach here, it's either:
            # 1. A multi-task DL model (e.g. model_key="MOFSNN_all_tasks") -> process current 'task'
            # 2. A single-task DL model AND current 'task' IS its designated task. -> process it.

            potential_dirs = [
                os.path.join(model_path, "lightning_logs/version_0"),
                os.path.join(model_path),  # Check model_path directly
                os.path.join(model_path, "evaluation")  # Check evaluation subdir
            ]

            for dir_path in potential_dirs:
                if not os.path.exists(dir_path):
                    continue

                result_file_path = os.path.join(dir_path, f"{split}_results_{task}.csv")
                df = read_results_file(result_file_path)

                if df is not None:
                    found_for_current_task = True
                    current_task_type = task_types.get(task, "classification")
                    calculated_metrics = calculate_metrics(df, current_task_type)
                    task_metrics_dict = {"Task": task, "Model": display_name, **calculated_metrics}
                    results.append(task_metrics_dict)
                    print(f"Processed {task} for DL model '{model_key}' (DisplayName: {display_name}) from '{result_file_path}'")
                    break  # Found results for this task for this DL model
            
            if not found_for_current_task:
                 print(f"Warning: No results found for task '{task}' for DL model '{model_key}' (DisplayName: {display_name}) in checked directories: {potential_dirs}")

    return results

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    file_ext = os.path.splitext(config_file)[1].lower()

    try:
        if file_ext in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                print(f"Loaded YAML configuration from {config_file}")
                return config
        elif file_ext in ['.json']:
            with open(config_file, 'r') as f:
                config = json.load(f)
                print(f"Loaded JSON configuration from {config_file}")
                return config
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}. Use .json, .yaml, or .yml")
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_file}: {e}")

def filter_models_by_group(model_dirs_map: Dict[str, Dict[str, str]],
                          comparison_groups: Dict[str, List[str]],
                          group_name: str) -> Dict[str, Dict[str, str]]:
    """
    Filter models based on a comparison group and preserve the order defined in the group.

    Args:
        model_dirs_map: Dictionary mapping model names to their configuration
        comparison_groups: Dictionary of comparison group definitions
        group_name: Name of the comparison group to use

    Returns:
        Filtered and ordered model directory mapping
    """
    if not group_name or group_name not in comparison_groups:
        print(f"No valid group specified or group '{group_name}' not found. Using all models.")
        return model_dirs_map

    # Get the list of model keys in this group
    group_models = comparison_groups[group_name]
    print(f"Using comparison group '{group_name}' with models: {group_models}")

    # Filter the model_dirs_map and preserve the order from group_models
    filtered_map = {}
    for model_key in group_models:
        if model_key in model_dirs_map:
            filtered_map[model_key] = model_dirs_map[model_key]

    if not filtered_map:
        print(f"Warning: No matching models found for group '{group_name}'. Using all models.")
        return model_dirs_map

    return filtered_map

def compare_model_performance(
    model_dirs_map: Dict[str, Dict[str, str]],
    tasks: List[str],
    task_types: Dict[str, str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    split: str = "test",
    output_filename: Optional[str] = None,
    model_order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Compare model performance across specified tasks and save results.

    Args:
        model_dirs_map: Dictionary mapping model keys to their configuration
                        (Path, DisplayName, and optionally Model for baselines)
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types
        output_dir: Directory to save output Excel file and plots
        split: Data split to analyze ('test' or 'external_test')
        output_filename: Optional custom name for the output Excel file
        model_order: Optional list of model keys to set the order in the output

    Returns:
        DataFrame with summarized results or None if no results processed
    """
    all_results = []

    for model_key, model_config in model_dirs_map.items():
        model_specific_path = model_config["Path"]
        model_display_name = model_config["DisplayName"]
        # Get the actual ML model type (e.g., "RandomForestRegressor") if specified (for baselines)
        actual_ml_model = model_config.get("Model")

        # Determine tasks for this specific model.
        # The existing logic for 'tasks_for_this_model' was in main();
        # Here, process_model_results is called with the global 'tasks' list,
        # and it internally filters for baselines if model_key indicates a specific task.
        # For DL models (non-baseline), it will attempt all tasks and report if files are found.

        model_task_results = process_model_results(
            model_path=model_specific_path,
            display_name=model_display_name,
            actual_ml_model_type=actual_ml_model,
            model_key=model_key,
            tasks=tasks, # Pass the global/group-filtered list of tasks
            task_types=task_types,
            split=split
        )
        all_results.extend(model_task_results)

    if not all_results:
        print("No results processed. Exiting.")
        return None

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    if df_results.empty:
        print("DataFrame is empty after processing results. Exiting.")
        return None

    # 1. Prepare Task column for ordered sorting
    # 'tasks' argument is config["tasks"], which has the desired order.
    df_results["Task"] = df_results["Task"].astype(str).str.strip() # Clean task names
    task_order_categories = tasks 
    df_results["Task"] = pd.Categorical(
        df_results["Task"],
        categories=task_order_categories,
        ordered=True
    )

    # 2. Prepare Model column (DisplayName) for ordered sorting
    # The 'Model' column contains DisplayNames.
    # 'model_order' argument is a list of model KEYS (e.g., "Baseline-TSD", "MOFSNN") from comparison group.
    # 'model_dirs_map' argument is the map (model_key -> config) that was processed;
    # this map is already ordered if a comparison group was used.

    ordered_model_display_names = []
    seen_display_names = set()

    if model_order: # Prioritize model_order (list of model KEYS from comparison group)
        for key in model_order:
            if key in model_dirs_map: # Check if the key from model_order is in the models processed
                display_name = model_dirs_map[key]["DisplayName"]
                if display_name not in seen_display_names:
                    ordered_model_display_names.append(display_name)
                    seen_display_names.add(display_name)
    else:
        # Fallback: use the order from model_dirs_map.keys()
        # model_dirs_map is already ordered if a comparison group was used.
        for key in model_dirs_map.keys():
            display_name = model_dirs_map[key]["DisplayName"]
            if display_name not in seen_display_names:
                ordered_model_display_names.append(display_name)
                seen_display_names.add(display_name)
    
    # Filter this master ordered list to include only those display names actually present in df_results["Model"]
    # This ensures categories in pd.Categorical are relevant to the data.
    final_model_categories = [dn for dn in ordered_model_display_names if dn in df_results["Model"].unique()]
    
    # Add any DisplayNames from df_results that were not captured by the above logic
    # (e.g., if a model was processed but its DisplayName wasn't in the derived order for some reason)
    # This places them at the end of the sort order for models.
    for dn_in_df in df_results["Model"].unique():
        if dn_in_df not in final_model_categories:
            final_model_categories.append(dn_in_df)

    if final_model_categories: # Only set categorical if there are categories to set
        df_results["Model"] = pd.Categorical(
            df_results["Model"],
            categories=final_model_categories,
            ordered=True
        )
    
    # 3. Sort the DataFrame: first by Task (already categorical and ordered), then by Model (now categorical and ordered)
    df_results.sort_values(by=["Task", "Model"], inplace=True)

    # 4. Set the multi-index
    df_results.set_index(["Task", "Model"], inplace=True)

    # Round numeric results
    numeric_cols = ["R2", "MAE", "ACC", "BACC", "AUROC"]
    for col in numeric_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].apply(lambda x: round(float(x), 2) if pd.notnull(x) else x)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to Excel
    if output_filename is None:
        output_filename = f"model_performance_comparison_{split}.xlsx"

    output_path = os.path.join(output_dir, output_filename)
    df_results.to_excel(output_path)
    print(f"Results saved to {output_path}")

    return df_results

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for model comparison, reflecting the 'standard'
    group from the model_comparison_config.yaml file.

    Returns:
        Dictionary with default configuration
    """
    # Default model directories mapping based on 'standard' group
    model_dirs_map = {
        "Baseline-TSD": {
            "Path": "results/ml_models/TSD/RAC_and_zeo_features_with_id_prop/Label",
            "DisplayName": "Baseline",
            "Model": "RandomForestRegressor"
        },
        "Baseline-SSD": {
            "Path": "results/ml_models/SSD/RAC_and_zeo_features_with_id_prop/Label",
            "DisplayName": "Baseline",
            "Model": "SVC"
        },
        "Baseline-WS24_water": {
            "Path": "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/water_label",
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "Baseline-WS24_water4": {
            "Path": "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/water4_label",
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "Baseline-WS24_acid": {
            "Path": "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/acid_label",
            "DisplayName": "Baseline",
            "Model": "GaussianProcessClassifier"
        },
        "Baseline-WS24_base": {
            "Path": "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/base_label",
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "Baseline-WS24_boiling": {
            "Path": "results/ml_models/WS24/RAC_and_zeo_features_with_id_prop/boiling_label",
            "DisplayName": "Baseline",
            "Model": "RandomForestClassifier"
        },
        "TSD": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/TSD_seed42_cgcnn_raw/version_29",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "SSD": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/SSD_seed42_cgcnn_raw/version_4",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_water": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/WS24_water_seed42_cgcnn_raw/version_29",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_water4": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/WS24_water4_seed42_cgcnn_raw/version_3",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_acid": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/WS24_acid_seed42_cgcnn_raw/version_47",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_base": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/WS24_base_seed42_cgcnn_raw/version_0",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "WS24_boiling": { # Representative for CGCNN Single-Task
            "Path": "results/cgcnn_models/WS24_boiling_seed42_cgcnn_raw/version_7",
            "DisplayName": "CGCNN_SG",
            "Model": "cgcnn_raw"
        },
        "TSD_SSD_WS24_all": { # CGCNN Multi-Task
            "Path": "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_cgcnn_raw/version_24",
            "DisplayName": "CGCNN_MT",
            "Model": "cgcnn_raw_multi"
        },
        "TSD_SSD_WS24_all_attn": { # MOFSNN
            "Path": "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43",
            "DisplayName": "MOFSNN",
            "Model": "att_cgcnn"
        }
    }

    # Default tasks
    tasks = [
        "TSD", "SSD", "WS24_water", "WS24_water4",
        "WS24_acid", "WS24_base", "WS24_boiling"
    ]

    # Default task types
    task_types = {
        "TSD": "regression",
        "SSD": "classification",
        "WS24_water": "classification",
        "WS24_water4": "classification",
        "WS24_acid": "classification",
        "WS24_base": "classification",
        "WS24_boiling": "classification"
    }

    # Default comparison group
    comparison_groups = {
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

    return {
        "model_dirs_map": model_dirs_map,
        "tasks": tasks,
        "task_types": task_types,
        "comparison_groups": comparison_groups
    }

def plot_bars(df: pd.DataFrame, figsize: Tuple[int, int]=(14, 8),
              label_size: int=16, tick_size: int=14, bar_width: float=0.5, **kwargs) -> Figure:
    """
    Create bar plots for model performance comparison.

    Args:
        df: DataFrame with columns 'Task', 'Model', and 'Performance'
        figsize: Figure size as (width, height)
        label_size: Font size for labels
        tick_size: Font size for ticks
        bar_width: Width of bars
        **kwargs: Additional keyword arguments:
            - mae_lim: Tuple for MAE y-axis limits (min, max)
            - annotate_size: Font size for annotations

    Returns:
        matplotlib Figure object
    """
    mae_lim = kwargs.pop('mae_lim', (10, 60))
    acc_lim = kwargs.pop('acc_lim', (0, 1.0))
    annotate_size = kwargs.pop('annotate_size', tick_size-1)

    # Set plot style
    sns.set_style("whitegrid")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)
    df = df[df.duplicated(subset=['Task'], keep=False)]  # Remove those groups with only one model

    if "TSD" in df['Task'].unique():
        # Plot MAE bar chart for TSD
        tsd_plot = sns.barplot(
            data=df[df['Task'] == 'TSD'],
            x='Task',
            y='Performance',
            hue='Model',
            dodge=True,
            ax=ax1,
            palette='Blues',
            width=bar_width
        )

        # Annotate bars with their values
        for p in tsd_plot.patches:
            if p.get_height() == 0:
                continue
            height = p.get_height()
            tsd_plot.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                          fontsize=annotate_size, color='blue')

        # Set left axis label
        ax1.set_xlabel('Task', fontsize=label_size, fontweight='bold')
        ax1.set_ylabel('MAE(←)', color='tab:blue', fontsize=label_size, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=tick_size)
        ax1.tick_params(axis='x', labelsize=tick_size)
        handles1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(loc='upper left', fontsize=tick_size-1)
        ax1.set_ylim(*mae_lim)

        # Create second y-axis
        ax2 = ax1.twinx()
    else:
        ax2 = ax1

    # Plot ACC bar chart for other tasks
    acc_plot = sns.barplot(
        data=df[df['Task'] != 'TSD'],
        x='Task',
        y='Performance',
        hue='Model',
        dodge=True,
        ax=ax2,
        palette='Greens',
        width=bar_width
    )

    # Annotate bars with their values
    for p in acc_plot.patches:
        if p.get_height() == 0:
            continue
        height = p.get_height()
        acc_plot.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                      fontsize=annotate_size, color='green')

    # Set right axis label
    ax2.set_xlabel('Task', fontsize=label_size, fontweight='bold')
    ax2.set_ylabel('ACC(→)', color='tab:green', fontsize=label_size, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=tick_size)
    ax2.tick_params(axis='x', labelsize=tick_size)
    ax2.set_ylim(*acc_lim)
    if ax2 is not ax1:
        ax2.grid(False)

    # Handle legend
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(loc='upper right', fontsize=tick_size-1)

    plt.tight_layout()
    return fig

def generate_visualization(df_results: pd.DataFrame,
                        fig_dir: Optional[str] = None,
                        split: str = "test",
                        fig_format: str = "both",
                        fig_dpi: int = 200,
                        **kwargs) -> Optional[Figure]:
    """
    Generate and save visualization for model performance comparison.

    Args:
        df_results: DataFrame with performance results
        fig_dir: Directory to save figures (if None, figures won't be saved)
        split: Data split name for filename
        fig_format: Format to save figures ("tif", "svg", "both", or "png")
        fig_dpi: DPI for saved figures
        model_order: Optional list specifying the order of models for the plot
        **kwargs: Additional parameters to pass to plot_bars function

    Returns:
        matplotlib Figure object or None if no visualization created
    """
    if df_results is None or df_results.empty:
        print("Error: No results available for visualization")
        return None

    # Create a copy and reset index for plotting
    df_plot = df_results.reset_index().copy()

    # Preprocess data for visualization - separate TSD and other tasks
    tsd_data = pd.DataFrame()
    other_tasks_data = pd.DataFrame()

    # For TSD (regression task), use MAE as performance metric
    if 'TSD' in df_plot['Task'].unique() and 'MAE' in df_results.columns:
        tsd_data = df_plot.loc[df_plot['Task'] == 'TSD', ['Task', 'Model', 'MAE']]
        tsd_data = tsd_data.rename(columns={'MAE': 'Performance'}).dropna()

    # For classification tasks, use ACC as performance metric
    if 'ACC' in df_results.columns:
        other_tasks_data = df_plot.loc[df_plot['Task'] != 'TSD', ['Task', 'Model', 'ACC']]
        other_tasks_data = other_tasks_data.rename(columns={'ACC': 'Performance'}).dropna()

    # Combine performance metrics for visualization
    combined_data = pd.concat([tsd_data, other_tasks_data])

    # Clean data
    combined_data = combined_data.dropna()

    if combined_data.empty:
        print("Error: No valid data for visualization")
        return None

    # Default visualization parameters
    viz_params = {
        'figsize': (14, 8),
        'label_size': 16,
        'tick_size': 14,
        'bar_width': 0.5,
        'mae_lim': (10, 60),
        'acc_lim': (0, 1.0),
        'annotate_size': 11
    }

    # Update with any provided kwargs
    viz_params.update(kwargs)

    # Generate visualization
    fig = plot_bars(combined_data, **viz_params)

    # Save figure if directory is specified
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

        base_filename = f"model_comparison_vis_{split}"

        # Save in specified format(s)
        if fig_format in ["tif", "both"]:
            tif_path = os.path.join(fig_dir, f"{base_filename}.tif")
            fig.savefig(tif_path, dpi=96)
            print(f"Figure saved as {tif_path}")

        if fig_format in ["svg", "both"]:
            svg_path = os.path.join(fig_dir, f"{base_filename}.svg")
            fig.savefig(svg_path, dpi=fig_dpi, transparent=True)
            print(f"Figure saved as {svg_path}")

        if fig_format == "png":
            png_path = os.path.join(fig_dir, f"{base_filename}.png")
            fig.savefig(png_path, dpi=fig_dpi)
            print(f"Figure saved as {png_path}")

    # Always close figure without showing
    plt.close(fig)

    return fig

def main():
    parser = ArgumentParser(description="Compare model performance across tasks and save results to Excel")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f"Directory to save comparison results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--split", type=str, default="test", choices=["test", "external_test"],
                      help="Data split to analyze (default: test)")
    parser.add_argument("--config_file", type=str,
                      help="Optional JSON/YAML file with model paths and configurations")
    parser.add_argument("--output_filename", type=str, default=None,
                      help="Custom filename for the output Excel file")
    parser.add_argument("--model_group", type=str, default=None,
                      help="Specify a comparison group from the config file")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualization of model performance")
    parser.add_argument("--fig_dir", type=str, default=None,
                      help="Directory to save visualization figures")
    parser.add_argument("--fig_format", type=str, default="both",
                      choices=["tif", "svg", "both", "png"],
                      help="Format to save visualization figures (default: both tif and svg)")
    parser.add_argument("--fig_dpi", type=int, default=300,
                      help="DPI for saved figures (default: 300)")
    parser.add_argument("--mae_min", type=float, default=20,
                      help="Minimum value for MAE y-axis")
    parser.add_argument("--mae_max", type=float, default=70,
                      help="Maximum value for MAE y-axis")
    parser.add_argument("--acc_min", type=float, default=0.0,
                      help="Minimum value for ACC y-axis")
    parser.add_argument("--acc_max", type=float, default=1.0,
                      help="Maximum value for ACC y-axis")
    parser.add_argument("--annotate_size", type=int, default=8,
                        help="Font size for annotations in the plot")
    parser.add_argument("--bar_width", type=float, default=0.8,
                      help="Width of bars in the plot")

    args = parser.parse_args()

    # Get configuration
    if args.config_file:
        try:
            config = load_config(args.config_file)
            print(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            print(f"Error: {e}")
            print("Using default configuration")
            config = get_default_config()
    else:
        print("No config file specified. Using default configuration.")
        config = get_default_config()

    # Extract model order from comparison group if specified
    if args.model_group and "comparison_groups" in config:
        if args.model_group in config["comparison_groups"]:
            model_order = config["comparison_groups"][args.model_group]
            print(f"Using model order from comparison group '{args.model_group}': {model_order}")
    else:
        args.model_group = "standard"

    # Filter models by group if specified
    model_dirs_map = config["model_dirs_map"]
    if args.model_group and "comparison_groups" in config:
        model_dirs_map = filter_models_by_group(
            model_dirs_map,
            config["comparison_groups"],
            args.model_group
        )

    # Run comparison with model_order
    results_df = compare_model_performance(
        model_dirs_map,
        config["tasks"],
        config["task_types"],
        os.path.join(args.output_dir, args.model_group) if args.model_group else args.output_dir,
        args.split,
        args.output_filename,
    )

    # Print summary
    if results_df is not None:
        print("\nResults Summary:")
        print(results_df)

        # Generate visualization if requested
        if args.visualize:
            # Set figure directory if not specified
            fig_dir = args.fig_dir if args.fig_dir else os.path.join(args.output_dir, args.model_group, "figures")

            # Set visualization parameters
            viz_params = {
                'mae_lim': (args.mae_min, args.mae_max),
                'acc_lim': (args.acc_min, args.acc_max),
                'bar_width': args.bar_width,
                'annotate_size': args.annotate_size
            }

            # Generate and save visualization with model_order
            generate_visualization(
                results_df,
                fig_dir=fig_dir,
                split=args.split,
                fig_format=args.fig_format,
                fig_dpi=args.fig_dpi,
                **viz_params
            )

if __name__ == "__main__":
    main()
