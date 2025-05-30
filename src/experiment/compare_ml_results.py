#!/usr/bin/env python
'''
Author: zhangsd
Date: 2025-06-05 (Updated: 2025-06-12)
Description: Process ML model results across tasks and save summarized results to Excel,
compatible with compare_model_performance.py for comparing with CGCNN results.
Supports multiple result directories per task for computing means and standard deviations.
Enhanced with visualization functionality from compare_model_performance_repeat.py.
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
    read_results_file, calculate_metrics, load_config, generate_visualization
)

# Import visualization functions from compare_model_performance_repeat.py
from experiment.compare_model_performance_repeat import (
    plot_bars_with_error, generate_visualization_with_error
)

# Default paths
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results/model_comparison/ml")
DEFAULT_MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "configs/ml_model_config.json")
ALGORITHM_MAP = {
    "GaussianProcessClassifier": "GP",
    "RandomForestClassifier": "RF",
    "GaussianProcessRegressor": "GP",
    "RandomForestRegressor": "RF",
    "SVR": "SVM",
    "SVC": "SVM",
}

def extract_model_name_from_filename(filename: str) -> str:
    """
    Extract model name from result filename.

    Args:
        filename: Filename of results file (e.g. test_predicted_GaussianProcessClassifier.csv)

    Returns:
        Model name (e.g. GaussianProcessClassifier)
    """
    # Extract model name from the filename
    parts = filename.split('_')
    if len(parts) >= 3:
        model_name = '_'.join(parts[2:]).replace('.csv', '')
        return model_name
    return "Unknown"

def process_ml_results_directory(results_dirs: Union[str, List[str]], task: str, task_type: str, 
                               split: str = "test", include_std: bool = True) -> List[Dict[str, Any]]:
    """
    Process ML model result files in one or more directories for a specific task.

    Args:
        results_dirs: Path to directory or list of directories containing ML model result files
        task: Name of the task
        task_type: Type of task ('regression' or 'classification')
        split: Data split to analyze ('test', 'validation' or 'external_test')
        include_std: Whether to include standard deviation metrics when multiple directories are processed

    Returns:
        List of dictionaries with metrics for each model, including mean and std if multiple directories
    """
    # Convert to list if a single path is provided
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    
    # Dictionary to store metrics by model
    model_metrics = {}
    
    # Look for result files that match the pattern
    prefix = f"{split}_predicted_"
    
    # Process each directory
    for dir_idx, results_dir in enumerate(results_dirs):
        # Check if directory exists
        if not os.path.exists(results_dir):
            print(f"Warning: Results directory not found - {results_dir}")
            continue
        
        # Find result files in this directory
        result_files = [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith(".csv")]
        
        if not result_files:
            print(f"Warning: No result files found in {results_dir} with pattern {prefix}*.csv")
            continue
        
        for result_file in result_files:
            model_name = extract_model_name_from_filename(result_file)
            file_path = os.path.join(results_dir, result_file)
            
            df = read_results_file(file_path)
            if df is not None:
                # Get metrics as dictionary
                task_metrics = calculate_metrics(df, task_type)
                
                # Initialize dictionary for this model if not already present
                if model_name not in model_metrics:
                    model_metrics[model_name] = {
                        "metrics": [],
                        "files": []
                    }
                
                # Add metrics from this directory
                model_metrics[model_name]["metrics"].append(task_metrics)
                model_metrics[model_name]["files"].append(file_path)
                
                print(f"Processed {task} for model {model_name} from {file_path}")
    
    # Convert metrics to final result with mean and std
    results = []
    for model_name, data in model_metrics.items():
        metrics_list = data["metrics"]
        
        if not metrics_list:
            continue
        
        # Calculate mean and std for each metric across multiple directories
        if len(metrics_list) > 1 and include_std:
            # Create dictionary to hold aggregates
            mean_metrics = {}
            std_metrics = {}
            
            # Get all metric keys from the first result
            metric_keys = list(metrics_list[0].keys())
            
            # Calculate mean and std for each metric
            for metric in metric_keys:
                values = [m.get(metric, np.nan) for m in metrics_list]
                mean_metrics[metric] = np.nanmean(values)
                std_metrics[f"{metric}_std"] = np.nanstd(values)
            
            # Create result entries
            result_entry = {
                "Task": task,
                "Model": ALGORITHM_MAP[model_name],
                "DisplayName": f"Baseline",
                # Include mean metrics and their standard deviations
                **mean_metrics,
                **std_metrics
            }
        else:
            # Only one directory or std not requested - use metrics directly
            if len(metrics_list) > 1:
                # Multiple results but no std requested - average them
                mean_metrics = {}
                for metric in metrics_list[0].keys():
                    values = [m.get(metric, np.nan) for m in metrics_list]
                    mean_metrics[metric] = np.nanmean(values)
                
                result_entry = {
                    "Task": task,
                    "Model": ALGORITHM_MAP[model_name],
                    "DisplayName": f"Baseline",
                    **mean_metrics
                }
            else:
                # Only one directory, use metrics directly
                result_entry = {
                    "Task": task,
                    "Model": ALGORITHM_MAP[model_name],
                    "DisplayName": f"Baseline",
                    **metrics_list[0]
                }
        
        results.append(result_entry)
    
    return results

def get_reference_results() -> List[Dict[str, Any]]:
    """
    Get hardcoded reference results from previous papers.
    
    Returns:
        List of dictionaries with reference metrics
    """
    # Reference results from Nandy et al. 2021(10.1021/jacs.1c07217)
    nandy_results = [
        {
            "Task": "TSD",
            "Model": "GP",
            "DisplayName": "Reference",
            "R2": 0.46,
            "MAE": 44
        },
        {
            "Task": "SSD",
            "Model": "GP", 
            "DisplayName": "Reference",
            "ACC": 0.76,
            "AUROC": 0.81
        }
    ]
    
    # Reference results from Terrones et al. 2024(10.1021/jacs.4c05879)
    terrones_results = [
        {
            "Task": "WS24_water",
            "Model": "RF",
            "DisplayName": "Reference",
            "ACC": 0.768493,
            "BACC": 0.742705,
            "AUROC": 0.828500
        },
        {
            "Task": "WS24_water4",
            "Model": "RF",
            "DisplayName": "Reference",
            "ACC": 0.655251,
            "BACC": 0.512478,
            "AUROC": 0.816146
        },
        {
            "Task": "WS24_acid",
            "Model": "RF",
            "DisplayName": "Reference",
            "ACC": 0.780556,
            "BACC": 0.780556,
            "AUROC": 0.850772
        },
        {
            "Task": "WS24_base",
            "Model": "RF",
            "DisplayName": "Reference",
            "ACC": 0.704000,
            "BACC": 0.703526,
            "AUROC": 0.780769
        },
        {
            "Task": "WS24_boiling",
            "Model": "RF",
            "DisplayName": "Reference",
            "ACC": 0.648148,
            "BACC": 0.648901,
            "AUROC": 0.693956
        }
    ]
    
    # Combine all reference results
    return nandy_results + terrones_results

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for ML model comparison.

    Returns:
        Dictionary with default configuration
    """
    # Default ML results directories mapping
    # Define both single-directory and multi-directory paths for each task
    base_dir = os.path.join(ROOT_DIR, "results/ml_models")
    
    # Define patterns for random seeds
    rand_seeds = ["", "_rand0", "_rand1", "_rand2", "_rand3"]
    
    # Default ML results directories mapping with multiple paths for each task
    ml_results_dirs = {
        "TSD": [
            os.path.join(base_dir, f"TSD/RAC_and_zeo_features_with_id_prop{seed}/Label")
            for seed in rand_seeds
        ],
        "SSD": [
            os.path.join(base_dir, f"SSD/RAC_and_zeo_features_with_id_prop{seed}/Label") 
            for seed in rand_seeds
        ],
        "WS24_water": [
            os.path.join(base_dir, f"WS24/RAC_and_zeo_features_with_id_prop{seed}/water_label")
            for seed in rand_seeds
        ],
        "WS24_water4": [
            os.path.join(base_dir, f"WS24/RAC_and_zeo_features_with_id_prop{seed}/water4_label")
            for seed in rand_seeds
        ],
        "WS24_acid": [
            os.path.join(base_dir, f"WS24/RAC_and_zeo_features_with_id_prop{seed}/acid_label")
            for seed in rand_seeds
        ],
        "WS24_base": [
            os.path.join(base_dir, f"WS24/RAC_and_zeo_features_with_id_prop{seed}/base_label")
            for seed in rand_seeds
        ],
        "WS24_boiling": [
            os.path.join(base_dir, f"WS24/RAC_and_zeo_features_with_id_prop{seed}/boiling_label")
            for seed in rand_seeds
        ]
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

    return {
        "ml_results_dirs": ml_results_dirs,
        "tasks": tasks,
        "task_types": task_types
    }

def process_ml_results(
    ml_results_dirs: Dict[str, Union[str, List[str]]],
    tasks: List[str],
    task_types: Dict[str, str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    split: str = "test",
    output_filename: Optional[str] = None,
    include_reference: bool = True,
    include_std: bool = True
) -> Optional[pd.DataFrame]:
    """
    Process ML model results across specified tasks.

    Args:
        ml_results_dirs: Dictionary mapping task names to result directories (str or list of str)
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types ('regression' or 'classification')
        output_dir: Directory to save output files
        split: Data split to analyze ('test', 'validation', or 'external_test')
        output_filename: Optional custom filename for the output Excel file
        include_reference: Whether to include reference results from literature
        include_std: Whether to include standard deviations when multiple results are available

    Returns:
        DataFrame with summarized results
    """
    all_results = []

    # Process each task and its models
    for task in tasks:
        if task not in ml_results_dirs:
            print(f"Warning: No results directory specified for task {task}")
            continue
        
        results_dir = ml_results_dirs[task]  # This can be a single string or a list of strings
        task_type = task_types.get(task, "classification")
        
        # Process ML model results
        task_results = process_ml_results_directory(
            results_dir,
            task,
            task_type,
            split,
            include_std
        )
        
        all_results.extend(task_results)

    # Include reference results if requested
    if include_reference:
        reference_results = get_reference_results()
        # Filter reference results to include only specified tasks
        reference_results = [r for r in reference_results if r["Task"] in tasks]
        all_results.extend(reference_results)

    # Convert to DataFrame
    if not all_results:
        print("Error: No results found!")
        return None

    df_results = pd.DataFrame(all_results)

    # Round numeric results and format standard deviations
    numeric_cols = ["R2", "MAE", "ACC", "BACC", "AUROC"]
    std_cols = []
    
    # Round all numeric columns (both metrics and their standard deviations)
    for col in df_results.columns:
        # Process standard metric columns
        if (col in numeric_cols) and (col in df_results.columns):
            df_results[col] = df_results[col].apply(lambda x: round(float(x), 4) if pd.notnull(x) else x)
            
        # Process standard deviation columns
        elif col.endswith('_std') and col.replace('_std', '') in numeric_cols:
            std_cols.append(col)
            df_results[col] = df_results[col].apply(lambda x: round(float(x), 4) if pd.notnull(x) else x)
    
    # Create new columns that combine mean ± std for better readability
    for col in numeric_cols:
        std_col = f"{col}_std"
        if std_col in df_results.columns:
            # Create a column with formatted values like "0.75 ± 0.02"
            formatted_col = f"{col}_formatted"
            df_results[formatted_col] = df_results.apply(
                lambda row: f"{row[col]:.2f}±{row[std_col]:.2f}" if pd.notnull(row[col]) and pd.notnull(row[std_col]) else row[col],
                axis=1
            )
            
            # If include_std is False, remove the std columns but keep formatted columns
            if not include_std:
                std_cols_to_drop = [c for c in df_results.columns if c.endswith('_std')]
                df_results = df_results.drop(columns=std_cols_to_drop)

    # Set Task and Model as index
    df_results["Task"] = df_results["Task"].apply(lambda x: str(x).strip())

    # Format for compatibility with model comparison framework
    df_final = df_results.copy()
    # Create a model identifier that combines the type and algorithm
    df_final["Model"] = df_final["DisplayName"] + "-" + df_final["Model"]
    
    # Convert Task column to categorical type with categories ordered as in the tasks list
    # This ensures tasks appear in the same order as in the original list
    df_final['Task'] = pd.Categorical(df_final['Task'], categories=tasks, ordered=True)
    
    # Sort first by Task (in original order), then keep original order within each task
    df_final['original_index'] = range(len(df_final))
    df_final = df_final.sort_values(['Task', 'original_index'])
    df_final = df_final.drop(columns=['original_index'])
    
    # Set only Task as index to group all models for same task together
    df_final.set_index(["Task", "Model"], inplace=True)
    df_final = df_final.drop(columns=["DisplayName"], errors='ignore')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to Excel
    if output_filename is None:
        output_filename = f"ml_model_performance_{split}.xlsx"

    output_path = os.path.join(output_dir, output_filename)
    df_final.to_excel(output_path)
    print(f"Results saved to {output_path}")

    # Generate compatible format for CGCNN comparison
    # Create test_results_{task}.csv files in the format expected by compare_model_performance.py
    for task in tasks:
        if task not in ml_results_dirs:
            continue
            
        task_results = df_results[df_results["Task"] == task]
        if task_results.empty:
            continue
            
        # Find the best performing ML model for this task
        best_metric = "R2" if task_types.get(task) == "regression" else "ACC"
        if best_metric not in task_results.columns:
            continue
            
        if task_types.get(task) == "regression":
            best_idx = task_results[best_metric].idxmax()
        else:
            best_idx = task_results[best_metric].idxmax()
            
        if pd.isna(best_idx):
            continue
            
        best_model = task_results.loc[best_idx, "Model"]
        
        # Handle results directory which could be a single path or a list of paths
        results_paths = ml_results_dirs[task]
        if not isinstance(results_paths, list):
            results_paths = [results_paths]  # Convert single path to a list
        
        # Try to find the model file in any of the provided paths
        model_file = None
        for path in results_paths:
            candidate_file = os.path.join(path, f"{split}_predicted_{best_model}.csv")
            if os.path.exists(candidate_file):
                model_file = candidate_file
                break
        
        if model_file and os.path.exists(model_file):
            df = pd.read_csv(model_file)
            # Save in CGCNN format
            cgcnn_result_file = os.path.join(output_dir, f"{split}_results_{task}.csv")
            df.to_csv(cgcnn_result_file, index=False)
            print(f"Saved best ML model results for task {task} to {cgcnn_result_file} (from {model_file})")

    return df_final

def main():
    parser = ArgumentParser(description="Process ML model results across tasks and save summary to Excel")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f"Directory to save comparison results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--split", type=str, default="test", choices=["test", "external_test"],
                      help="Data split to analyze (default: test)")
    parser.add_argument("--config_file", type=str, default=DEFAULT_MODEL_CONFIG_FILE,
                      help="Optional JSON/YAML file with model paths and configurations")
    parser.add_argument("--output_filename", type=str, default=None,
                      help="Custom filename for the output Excel file")
    parser.add_argument("--no_reference", action="store_true",
                      help="Exclude reference results from literature")
    parser.add_argument("--ml_dir", type=str, default=None,
                      help="Base directory for ML model results (overrides config)")
    parser.add_argument("--no_std", action="store_true",
                      help="Do not include standard deviation columns in output")
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

    args = parser.parse_args()

    # Get configuration
    if args.config_file:
        try:
            config = load_config(args.config_file)
            print(f"Loaded configuration from {args.config_file}")
            
            # Check if config has ml_results_dirs section
            if "ml_results_dirs" not in config:
                print("Warning: 'ml_results_dirs' not found in config, using default paths")
                default_config = get_default_config()
                config["ml_results_dirs"] = default_config["ml_results_dirs"]
                
            # Make sure we have task types
            if "task_types" not in config:
                default_config = get_default_config()
                config["task_types"] = default_config["task_types"]
                
            # Make sure we have tasks defined
            if "tasks" not in config:
                default_config = get_default_config()
                config["tasks"] = default_config["tasks"]
        except Exception as e:
            print(f"Error: {e}")
            print("Using default configuration")
            config = get_default_config()
    else:
        print("No config file specified. Using default configuration.")
        config = get_default_config()

    # Override ML base directory if specified
    if args.ml_dir:
        print(f"Using specified ML base directory: {args.ml_dir}")
        ml_results_dirs = {}
        for task in config["tasks"]:
            # Extract the task-specific subdirectory from the default paths
            default_dirs = config["ml_results_dirs"][task]
            
            # Handle both single path and list of paths
            if isinstance(default_dirs, list):
                new_paths = []
                for default_dir in default_dirs:
                    rel_path = os.path.relpath(default_dir, ROOT_DIR)
                    # Construct a new path with the specified base directory
                    new_paths.append(os.path.join(args.ml_dir, rel_path))
                ml_results_dirs[task] = new_paths
            else:
                # Single path case
                rel_path = os.path.relpath(default_dirs, ROOT_DIR)
                ml_results_dirs[task] = os.path.join(args.ml_dir, rel_path)
                
        config["ml_results_dirs"] = ml_results_dirs

    # Process ML model results
    results_df = process_ml_results(
        config["ml_results_dirs"],
        config["tasks"],
        config["task_types"],
        args.output_dir,
        args.split,
        args.output_filename,
        not args.no_reference,
        not args.no_std
    )

    # Print summary
    if results_df is not None:
        print("\nResults Summary:")
        print(results_df)

        # Generate visualization if requested
        if args.visualize:
            # Set figure directory if not specified
            fig_dir = args.fig_dir if args.fig_dir else os.path.join(args.output_dir, "figures")

            # Set visualization parameters
            viz_params = {
                'mae_lim': (args.mae_min, args.mae_max),
                'acc_lim': (args.acc_min, args.acc_max),
                'bar_width': args.bar_width,
                'annotate_size': args.annotate_size,
                'annotate': args.annotate
            }

            # Generate and save visualization with error bars if std data is available
            if not args.no_std and any(col.endswith('_std') for col in results_df.columns):
                generate_visualization_with_error(
                    results_df,
                    fig_dir=fig_dir,
                    split=args.split,
                    fig_format=args.fig_format,
                    fig_dpi=args.fig_dpi,
                    **viz_params
                )
                print(f"Visualization with error bars saved to {fig_dir}")
            else:
                # Generate standard visualization without error bars
                generate_visualization(
                    results_df,
                    fig_dir=fig_dir,
                    split=args.split,
                    fig_format=args.fig_format,
                    fig_dpi=args.fig_dpi,
                    **viz_params
                )
                print(f"Visualization saved to {fig_dir}")

if __name__ == "__main__":
    main()
