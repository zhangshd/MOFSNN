#!/usr/bin/env python
'''
Author: zhangsd
Date: 2024-06-12
Description: Compare the performance of different models on test datasets with support for multiple model paths.
Enhanced version of compare_model_performance.py that supports Path as a list for calculating means and standard deviations.
'''

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from typing import Dict, List, Optional, Any, Union, Tuple

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory of the project (two levels up from the script directory)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Add the src directory to the Python path
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import reusable functions from compare_model_performance.py
from experiment.compare_model_performance import (
    read_results_file, calculate_metrics, load_config, generate_visualization,
    get_default_config, filter_models_by_group
)

# Default paths
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results/model_comparison/repeat")

def process_model_results_multi_path(
    model_paths: Union[str, List[str]], 
    display_name: str, 
    actual_ml_model_type: Optional[str],
    model_key: str,
    tasks: List[str], 
    task_types: Dict[str, str],
    split: str = "test", 
    include_std: bool = True
) -> List[Dict[str, Any]]:
    """
    Process results for a single model across multiple tasks and multiple paths.

    Args:
        model_paths: Path or list of paths to the model results directories
        display_name: Display name for the model (used in output 'Model' column)
        actual_ml_model_type: The actual ML algorithm string (e.g., "RandomForestRegressor")
                            if this is a baseline ML model.
                            For DL models, this is also their "Model" field from config.
        model_key: The unique key for the model from the configuration
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types ('regression' or 'classification')
        split: Data split to analyze ('test' or 'external_test')
        include_std: Whether to include standard deviation metrics when multiple directories are processed

    Returns:
        List of dictionaries with metrics for each task, including mean and std if multiple paths
    """
    # Convert model_paths to a list if it's a string
    if isinstance(model_paths, str):
        model_paths = [model_paths]

    # Dictionary to store metrics by task
    task_metrics = {}

    # Flag to indicate if this is a baseline model
    is_baseline_model_by_key = model_key.startswith("Baseline-")

    for task in tasks:
        # Skip tasks that don't match baseline-specific task
        if is_baseline_model_by_key:
            # Extract expected task from model_key like "Baseline-TaskName"
            parts = model_key.split('-', 1)
            expected_task_for_baseline = parts[1] if len(parts) > 1 else None
            
            # Skip if this baseline model is for a specific task and current task doesn't match
            if expected_task_for_baseline and task != expected_task_for_baseline:
                continue
        
        # For DL models with specific task names
        is_dl_model_for_a_specific_task_only = model_key in task_types.keys()
        if is_dl_model_for_a_specific_task_only and model_key != task:
            continue

        # Process each path for the current task
        path_metrics = []
        for path_index, model_path in enumerate(model_paths):
            # Handle baselines
            if is_baseline_model_by_key:
                if not actual_ml_model_type:
                    print(f"Warning: Baseline model '{model_key}' is missing the 'Model' field. Skipping task '{task}'.")
                    continue
                
                # Construct file path for baseline model results
                result_file_path = os.path.join(model_path, f"{split}_predicted_{actual_ml_model_type}.csv")
                df = read_results_file(result_file_path)
                
                if df is not None:
                    current_task_type = task_types.get(task, "classification")
                    metrics = calculate_metrics(df, current_task_type)
                    path_metrics.append(metrics)
                    print(f"Processed {task} for baseline model '{model_key}' (algo: {actual_ml_model_type}) path {path_index} using file '{result_file_path}'")
            
            # Handle DL models
            else:
                # For DL models, check several potential directories and use the correct file naming format
                found_for_current_path = False
                potential_dirs = [
                    os.path.join(model_path, "lightning_logs/version_0"),
                    os.path.join(model_path),  # Check model_path directly
                    os.path.join(model_path, "evaluation")  # Check evaluation subdir
                ]
                
                for dir_path in potential_dirs:
                    if not os.path.exists(dir_path):
                        continue
                    
                    # CGCNN models use format "{split}_results_{task}.csv"
                    result_file_path = os.path.join(dir_path, f"{split}_results_{task}.csv")
                    df = read_results_file(result_file_path)
                    
                    if df is not None:
                        found_for_current_path = True
                        current_task_type = task_types.get(task, "classification")
                        metrics = calculate_metrics(df, current_task_type)
                        path_metrics.append(metrics)
                        print(f"Processed {task} for model '{model_key}' path {path_index} using file '{result_file_path}'")
                        break  # Found results for this task in this path
                
                if not found_for_current_path:
                    print(f"Warning: No results found for task '{task}' for model '{model_key}' in path {model_path}")
        
        # Aggregate metrics across paths for the current task
        if path_metrics:
            task_entry = {"Task": task, "Model": display_name}
            
            # Calculate mean and std for each metric if multiple paths
            if len(path_metrics) > 1 and include_std:
                # Get all metric keys
                all_metric_keys = set()
                for metrics_dict in path_metrics:
                    all_metric_keys.update(metrics_dict.keys())
                
                # Calculate mean and std for each metric
                for metric in all_metric_keys:
                    values = [m.get(metric, np.nan) for m in path_metrics]
                    task_entry[metric] = np.nanmean(values)
                    task_entry[f"{metric}_std"] = np.nanstd(values)
            
            else:
                # Only one path or std not requested
                if len(path_metrics) > 1:
                    # Average metrics across paths
                    for metric in path_metrics[0].keys():
                        values = [m.get(metric, np.nan) for m in path_metrics]
                        task_entry[metric] = np.nanmean(values)
                else:
                    # Single path metrics
                    task_entry.update(path_metrics[0])
            
            # Add task entry to results
            if task not in task_metrics:
                task_metrics[task] = []
            task_metrics[task].append(task_entry)
    
    # Flatten dictionary of tasks to list
    results = []
    for task_list in task_metrics.values():
        results.extend(task_list)
    
    return results

def compare_model_performance_repeat(
    model_dirs_map: Dict[str, Dict[str, Any]],
    tasks: List[str],
    task_types: Dict[str, str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    split: str = "test",
    output_filename: Optional[str] = None,
    model_order: Optional[List[str]] = None,
    include_std: bool = True
) -> Optional[pd.DataFrame]:
    """
    Compare model performance across specified tasks with support for multiple paths per model.

    Args:
        model_dirs_map: Dictionary mapping model keys to their configuration
                        (Path (string or list), DisplayName, and optionally Model for baselines)
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types
        output_dir: Directory to save output Excel file and plots
        split: Data split to analyze ('test' or 'external_test')
        output_filename: Optional custom name for the output Excel file
        model_order: Optional list of model keys to set the order in the output
        include_std: Whether to include standard deviation metrics when multiple paths are processed

    Returns:
        DataFrame with summarized results or None if no results processed
    """
    all_results = []

    for model_key, model_config in model_dirs_map.items():
        model_specific_paths = model_config["Path"]
        model_display_name = model_config["DisplayName"]
        actual_ml_model = model_config.get("Model")

        model_task_results = process_model_results_multi_path(
            model_paths=model_specific_paths,
            display_name=model_display_name,
            actual_ml_model_type=actual_ml_model,
            model_key=model_key,
            tasks=tasks,
            task_types=task_types,
            split=split,
            include_std=include_std
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

    # Prepare Task column for ordered sorting
    df_results["Task"] = df_results["Task"].astype(str).str.strip()
    task_order_categories = tasks
    df_results["Task"] = pd.Categorical(
        df_results["Task"],
        categories=task_order_categories,
        ordered=True
    )

    # Prepare Model column for ordered sorting
    ordered_model_display_names = []
    seen_display_names = set()

    if model_order:
        for key in model_order:
            if key in model_dirs_map:
                display_name = model_dirs_map[key]["DisplayName"]
                if display_name not in seen_display_names:
                    ordered_model_display_names.append(display_name)
                    seen_display_names.add(display_name)
    else:
        for key in model_dirs_map.keys():
            display_name = model_dirs_map[key]["DisplayName"]
            if display_name not in seen_display_names:
                ordered_model_display_names.append(display_name)
                seen_display_names.add(display_name)
    
    final_model_categories = [dn for dn in ordered_model_display_names if dn in df_results["Model"].unique()]
    
    for dn_in_df in df_results["Model"].unique():
        if dn_in_df not in final_model_categories:
            final_model_categories.append(dn_in_df)
    
    df_results["Model"] = pd.Categorical(
        df_results["Model"],
        categories=final_model_categories,
        ordered=True
    )

    # Sort by Task and Model
    df_results = df_results.sort_values(["Task", "Model"])

    # Save results to Excel
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        if output_filename:
            results_file = os.path.join(output_dir, output_filename)
        else:
            results_file = os.path.join(output_dir, f"model_performance_results_{split}.xlsx")
        
        df_results.to_excel(results_file, index=False)
        print(f"Results saved to {results_file}")

    return df_results

def main():
    parser = ArgumentParser(description="Compare model performance across tasks with support for multiple model paths")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f"Directory to save comparison results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--split", type=str, default="test", choices=["test", "external_test"],
                      help="Data split to analyze (default: test)")
    parser.add_argument("--config_file", type=str, required=True,
                      help="JSON/YAML file with model paths and configurations")
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
    parser.add_argument("--mae_min", type=float, default=10,
                      help="Minimum value for MAE y-axis")
    parser.add_argument("--mae_max", type=float, default=60,
                      help="Maximum value for MAE y-axis")
    parser.add_argument("--bar_width", type=float, default=0.8,
                      help="Width of bars in the plot")
    parser.add_argument("--no_std", action="store_true",
                      help="Disable calculation of standard deviations for multiple paths")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config_file)
        print(f"Loaded configuration from {args.config_file}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Extract model order from comparison group if specified
    if args.model_group and "comparison_groups" in config:
        if args.model_group in config["comparison_groups"]:
            model_order = config["comparison_groups"][args.model_group]
            print(f"Using model order from comparison group '{args.model_group}': {model_order}")
    else:
        model_order = None
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
    results_df = compare_model_performance_repeat(
        model_dirs_map,
        config["tasks"],
        config["task_types"],
        os.path.join(args.output_dir, args.model_group) if args.model_group else args.output_dir,
        args.split,
        args.output_filename,
        model_order,
        not args.no_std
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
                'bar_width': args.bar_width
            }

            # Generate and save visualization
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
