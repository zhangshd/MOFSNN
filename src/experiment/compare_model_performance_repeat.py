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
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from typing import Dict, List, Optional, Any, Union, Tuple
from matplotlib.figure import Figure

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

def plot_bars_with_error(df: pd.DataFrame, figsize: Tuple[int, int]=(14, 8),
              label_size: int=16, tick_size: int=14, bar_width: float=0.5, **kwargs) -> Figure:
    """
    Create bar plots for model performance comparison with error bars.

    Args:
        df: DataFrame with columns 'Task', 'Model', 'Performance', and optionally 'Performance_std'
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
    annotate = kwargs.pop('annotate', False)


    # Set plot style
    sns.set_style("whitegrid")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)
    df = df[df.duplicated(subset=['Task'], keep=False)]  # Remove those groups with only one model

    if "TSD" in df['Task'].unique():
        # Plot MAE bar chart for TSD
        tsd_data = df[df['Task'] == 'TSD']
        
        # Extract standard deviation if available
        if 'Performance_std' in tsd_data.columns:
            yerr = tsd_data['Performance_std'].values
        else:
            yerr = None
        
        tsd_plot = sns.barplot(
            data=tsd_data,
            x='Task',
            y='Performance',
            hue='Model',
            dodge=True,
            ax=ax1,
            palette='Blues',
            width=bar_width
        )
        
        # Add error bars if standard deviation is available
        if yerr is not None:
            # Get all unique tasks in the original dataframe to determine correct positions
            all_unique_tasks = df['Task'].unique()
            models = tsd_data['Model'].unique()
            num_models = len(models)
            
            # Calculate positions for each bar
            width = bar_width / num_models
            offsets = np.linspace(-bar_width/2 + width/2, bar_width/2 - width/2, num_models)
            
            for task in tsd_data['Task'].unique():
                # Find the correct position on the x-axis
                task_idx = np.where(all_unique_tasks == task)[0][0]
                task_data = tsd_data[tsd_data['Task'] == task]
                
                for i, model in enumerate(models):
                    model_task_data = task_data[task_data['Model'] == model]
                    if not model_task_data.empty:
                        x_pos = task_idx + offsets[i]
                        yerr_val = model_task_data['Performance_std'].values[0] if 'Performance_std' in model_task_data.columns else 0
                        ax1.errorbar(
                            x=x_pos,
                            y=model_task_data['Performance'].values[0],
                            yerr=yerr_val,
                            fmt='none',
                            ecolor='gray',
                            capsize=5
                        )
                        if 'Performance_std' in model_task_data.columns:
                            print(f"TSD error bar: task={task}, model={model}, x={x_pos}, y={model_task_data['Performance'].values[0]}, yerr={yerr_val}")

        # Annotate bars with their values
        if annotate:
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
    acc_data = df[df['Task'] != 'TSD']
    
    # Extract standard deviation if available
    if 'Performance_std' in acc_data.columns:
        yerr = acc_data['Performance_std'].values
    else:
        yerr = None
    
    acc_plot = sns.barplot(
        data=acc_data,
        x='Task',
        y='Performance',
        hue='Model',
        dodge=True,
        ax=ax2,
        palette='Greens',
        width=bar_width
    )
    
    # Add error bars if standard deviation is available
    if yerr is not None:
        # Get all unique tasks in the original dataframe to determine correct positions
        all_unique_tasks = df['Task'].unique()
        models = acc_data['Model'].unique()
        num_models = len(models)
        
        # For each classification task
        for task in acc_data['Task'].unique():
            # Find the correct position on the x-axis (considering TSD might be before it)
            task_idx = np.where(all_unique_tasks == task)[0][0]
            task_data = acc_data[acc_data['Task'] == task]
            
            # Calculate positions for each bar
            width = bar_width / num_models
            offsets = np.linspace(-bar_width/2 + width/2, bar_width/2 - width/2, num_models)
            
            for i, model in enumerate(models):
                model_task_data = task_data[task_data['Model'] == model]
                if not model_task_data.empty:
                    x_pos = task_idx + offsets[i]
                    yerr_val = model_task_data['Performance_std'].values[0] if 'Performance_std' in model_task_data.columns else 0
                    ax2.errorbar(
                        x=x_pos,
                        y=model_task_data['Performance'].values[0],
                        yerr=yerr_val,
                        fmt='none',
                        ecolor='gray',
                        capsize=5
                    )
                    if 'Performance_std' in model_task_data.columns:
                        print(f"ACC error bar: task={task}, model={model}, x={x_pos}, y={model_task_data['Performance'].values[0]}, yerr={yerr_val}")

    # Annotate bars with their values
    if annotate:
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

def generate_visualization_with_error(df_results: pd.DataFrame,
                              fig_dir: Optional[str] = None,
                              split: str = "test",
                              fig_format: str = "both",
                              fig_dpi: int = 200,
                              **kwargs) -> Optional[Figure]:
    """
    Generate and save visualization for model performance comparison with error bars.

    Args:
        df_results: DataFrame with performance results
        fig_dir: Directory to save figures (if None, figures won't be saved)
        split: Data split name for filename
        fig_format: Format to save figures ("tif", "svg", "both", or "png")
        fig_dpi: DPI for saved figures
        **kwargs: Additional parameters to pass to plot_bars_with_error function

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
        tsd_columns = ['Task', 'Model', 'MAE']
        if 'MAE_std' in df_results.columns:
            tsd_columns.append('MAE_std')
        tsd_data = df_plot.loc[df_plot['Task'] == 'TSD', tsd_columns].copy()
        
        # Rename columns for unified processing
        rename_dict = {'MAE': 'Performance'}
        if 'MAE_std' in df_results.columns:
            rename_dict['MAE_std'] = 'Performance_std'
        tsd_data = tsd_data.rename(columns=rename_dict).dropna(subset=['Performance'])

    # For classification tasks, use ACC as performance metric
    if 'ACC' in df_results.columns:
        acc_columns = ['Task', 'Model', 'ACC']
        if 'ACC_std' in df_results.columns:
            acc_columns.append('ACC_std')
        other_tasks_data = df_plot.loc[df_plot['Task'] != 'TSD', acc_columns].copy()
        
        # Rename columns for unified processing
        rename_dict = {'ACC': 'Performance'}
        if 'ACC_std' in df_results.columns:
            rename_dict['ACC_std'] = 'Performance_std'
        other_tasks_data = other_tasks_data.rename(columns=rename_dict).dropna(subset=['Performance'])

    # Combine performance metrics for visualization
    combined_data = pd.concat([tsd_data, other_tasks_data])

    # Clean data
    combined_data = combined_data.dropna(subset=['Performance'])

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
    fig = plot_bars_with_error(combined_data, **viz_params)

    # Save figure if directory is specified
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

        base_filename = f"model_comparison_vis_with_error_{split}"

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
    parser.add_argument("--acc_min", type=float, default=0.0,
                      help="Minimum value for ACC y-axis")
    parser.add_argument("--acc_max", type=float, default=1.0,
                      help="Maximum value for ACC y-axis")
    parser.add_argument("--annotate", action="store_true", 
                        help="Whether to annotate bars with their values")
    parser.add_argument("--annotate_size", type=int, default=8,
                        help="Font size for annotations in the plot")
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
                'acc_lim': (args.acc_min, args.acc_max),
                'bar_width': args.bar_width,
                'annotate_size': args.annotate_size
            }

            # Generate and save visualization with error bars
            generate_visualization_with_error(
                results_df,
                fig_dir=fig_dir,
                split=args.split,
                fig_format=args.fig_format,
                fig_dpi=args.fig_dpi,
                **viz_params
            )
            
            # Also generate standard visualization without error bars for comparison
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
