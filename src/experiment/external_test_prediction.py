#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-05-12
Description: Integrate ML and CGCNN model predictions on external test sets.
This script extracts ML model predictions from notebooks/06_ML_predicition_of_external_test_set.ipynb
and CGCNN model predictions from src/cgcnn/predict.py, then consolidates them
for comparison. The output follows the same format as src/experiment/compare_model_performance.py.
'''

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score,
    balanced_accuracy_score, roc_auc_score, 
    matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Add the src directory to the Python path
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import ML modules
from ml.module import ClassificationModel, RegressionModel
from ml.module import plot_roc_curve, plot_scatter, plot_confusion_matrix

# Import CGCNN modules
from cgcnn.utils import load_model_from_dir, MODEL_NAME_TO_DATASET_CLS
from torch.utils.data import DataLoader
import torch
import pickle
from cgcnn.module.module_utils import calculate_lse_from_tree, calculate_lsv_from_tree

# Default paths
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs/model_comparison_config.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "results/model_comparison")
DEFAULT_ML_MODEL_DIR = os.path.join(ROOT_DIR, "results/ml_models")
DEFAULT_CGCNN_MODEL_DIR = os.path.join(ROOT_DIR, "results/cgcnn_models")

# Define external test data paths
TS_EXTERNAL_CSV_ML = os.path.join(ROOT_DIR, "data/ml_data/TS_external_test/RAC_and_zeo_features_with_id_prop.csv")
WS_EXTERNAL_CSV_ML = os.path.join(ROOT_DIR, "data/ml_data/WS24v2_external_test/RAC_and_zeo_features_with_id_prop.csv")
TS_EXTERNAL_DIR_CGCNN = os.path.join(ROOT_DIR, "data/cgcnn_data/TS_external_test")
WS_EXTERNAL_DIR_CGCNN = os.path.join(ROOT_DIR, "data/cgcnn_data/WS24v2_external_test")


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded YAML configuration from {config_file}")
            return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_file}: {e}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None, 
                     task_type: str = "regression") -> Dict[str, float]:
    """
    Calculate performance metrics based on task type.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_prob: Prediction probabilities (for classification tasks)
        task_type: Type of task - 'regression' or 'classification'

    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}

    if task_type == "regression":
        # Calculate regression metrics
        metrics["R2"] = r2_score(y_true, y_pred)
        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    else:  # classification
        # Calculate classification metrics
        metrics["ACC"] = accuracy_score(y_true, y_pred)
        metrics["BACC"] = balanced_accuracy_score(y_true, y_pred)
        
        if y_prob is not None:
            # Calculate AUROC if probabilities are available
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
                    # Multi-class case
                    metrics["AUROC"] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
                elif len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    # Binary case with 2-column probabilities
                    metrics["AUROC"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Binary case with 1-column probabilities
                    metrics["AUROC"] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                print(f"Warning: Could not calculate AUROC: {e}")
                metrics["AUROC"] = np.nan

    return metrics


def ml_predict_external_test(model_dirs_map: Dict[str, Dict[str, Any]], 
                           tasks: List[str], task_types: Dict[str, str],
                           ):
    """
    Predict external test set using ML models.
    
    Args:
        model_dirs_map: Dictionary mapping model keys to their configuration
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types
        output_dir: Directory to save output results
    
    Returns:
        Dictionary with model results for each task
    """
    
    sys.path.append(os.path.join(ROOT_DIR, "src/ml"))

    # Load external test data
    df_ts = pd.read_csv(TS_EXTERNAL_CSV_ML)
    df_ws = pd.read_csv(WS_EXTERNAL_CSV_ML)
    
    print(f"Loaded external test data: TS shape={df_ts.shape}, WS shape={df_ws.shape}")
    
    # Define column to task mappings
    col2tasks = [
        {"ts_label": "TSD", "ss_label": "SSD"},
        {"water_label": "WS24_water", "water4_label": "WS24_water4", 
         "acid_label": "WS24_acid", "base_label": "WS24_base", 
         "boiling_label": "WS24_boiling"}
    ]
    
    # Define task to column mappings
    task2col = {"TSD": "Label", "SSD": "Label"}
    task2col.update({
        t: t.split("_")[-1] + "_label" 
        for t in ["WS24_water", "WS24_water4", "WS24_acid", "WS24_base", "WS24_boiling"]
    })
    
    # Process baseline ML models
    results = {}
    split = "external_test"
    
    for model_key, model_config in model_dirs_map.items():
        if not model_key.startswith("Baseline-"):
            continue
            
        model_specific_paths = model_config["Path"]
        model_display_name = model_config["DisplayName"]
        actual_ml_model = model_config.get("Model")
        
        # Convert model_paths to a list if it's a string
        if isinstance(model_specific_paths, str):
            model_specific_paths = [model_specific_paths]
        
        if not actual_ml_model:
            print(f"Warning: 'Model' field missing for {model_key}. Skipping.")
            continue
            
        # Extract task from model key (e.g., "Baseline-TSD" -> "TSD")
        task = model_key.split('-', 1)[1]
        
        if task not in tasks:
            print(f"Warning: Task {task} not in the tasks list. Skipping.")
            continue
            
        # Determine task type
        task_type = task_types.get(task, "classification")
        
        # Select appropriate dataset based on task
        if task.startswith("TSD") or task.startswith("SSD"):
            df = df_ts
        else:  # WS24 tasks
            df = df_ws
            
        # Get label column
        label_col = next((col for col, t in col2tasks[0 if task.startswith(("TSD", "SSD")) else 1].items() 
                         if t == task), None)
        
        if not label_col:
            print(f"Warning: Could not find label column for task {task}. Skipping.")
            continue
        
        # Process each model path
        path_metrics = []
        
        for path_index, model_specific_path in enumerate(model_specific_paths):
            # Find the model file
            model_dir = Path(model_specific_path)
            model_files = list(model_dir.glob(f"total_model_*_{actual_ml_model}_*.model"))
            
            if not model_files:
                print(f"Warning: No model files found for {model_key} in {model_dir}. Skipping path {path_index}.")
                continue
                
            model_path = model_files[0]  # Use the first matching model file
            print(f"Using model ({path_index+1}/{len(model_specific_paths)}): {model_path}")
            
            # Initialize model and make predictions
            if task_type == "regression":
                model = RegressionModel(random_state=0)
                model.load_total_model(model_path)
                # model.train(model.model, model.params)
                y_pred = model.predict(df.loc[:, "Di":]).squeeze() # this method will use k-fold models (self.models, not self.model) to make prediction and take average values. 
                y_true = df[label_col].values.astype(np.float64)  # Ensure numpy array type
                
                # Calculate metrics
                metrics = calculate_metrics(y_true, y_pred, task_type=task_type)
                path_metrics.append(metrics)
                
                # Save predictions
                df_pred = pd.DataFrame({
                    "MofName": df["MofName"].values, 
                    "GroundTruth": y_true, 
                    "Predicted": y_pred, 
                    "Error": np.abs(y_true - y_pred)
                })
                
                # Create plots only for the first path to avoid clutter
                if path_index == 0:
                    img_file = model_dir / f"{split}_scatter_{task}.png"
                    plot_scatter(y_true, y_pred, title=f"{split}/{task}", 
                               metrics=metrics, outfile=str(img_file))
                
            else:  # classification
                model = ClassificationModel(random_state=0, n_class=2)
                model.load_total_model(model_path)
                # model.train(model.model, model.params)
                y_pred = model.predict(df.loc[:, "Di":], return_prob=False).squeeze()
                y_prob = model.predict(df.loc[:, "Di":], return_prob=True)
                y_true = df[label_col].values.astype(np.int32)  # Ensure numpy array type
                
                # For WS24_water4, adjust labels (1-4 to 0-3)
                if task == "WS24_water4":
                    y_true = y_true.astype(np.int32) - 1
                
                # Calculate metrics
                metrics = calculate_metrics(y_true, y_pred, y_prob, task_type=task_type)
                path_metrics.append(metrics)
                
                # Save predictions
                if y_prob.shape[1] == 2:
                    y_prob_final = y_prob[:, 1]
                else:
                    y_prob_final = y_prob
                    
                df_pred = pd.DataFrame({
                    "MofName": df["MofName"].values, 
                    "GroundTruth": y_true, 
                    "Predicted": y_pred, 
                    "Prob": y_prob_final.tolist()
                })
                
                # Create visualizations only for the first path to avoid clutter
                if path_index == 0:
                    # Confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    img_file = model_dir / f"{split}_confusion_matrix_{task}.png"
                    plot_confusion_matrix(y_true, y_pred, title=f"{split}/{task}", 
                                        outfile=str(img_file))
                    
                    # ROC curve for binary classification
                    if y_prob.shape[1] == 2:
                        from sklearn.metrics import roc_curve
                        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1], drop_intermediate=False)
                        img_file = model_dir / f"{split}_roc_curve_{task}.png"
                        plot_roc_curve(fpr, tpr, metrics["AUROC"], title=f"{split}/{task}", 
                                      outfile=str(img_file))
            
            # Save predictions to model directory
            model_csv_file = model_dir / f"{split}_predicted_{actual_ml_model}.csv"
            df_pred.to_csv(model_csv_file, index=False)
            print(f"Saved predictions to model directory: {model_csv_file}")
        
        # Aggregate metrics across paths
        if path_metrics:
            # Create task results with averaged metrics
            task_results = {"Task": task, "Model": model_display_name}
            
            # Calculate mean for each metric
            all_metric_keys = set()
            for metrics_dict in path_metrics:
                all_metric_keys.update(metrics_dict.keys())
                
            for metric in all_metric_keys:
                values = [m.get(metric, np.nan) for m in path_metrics]
                task_results[metric] = np.nanmean(values)
                
                # Add standard deviation for multiple paths
                if len(path_metrics) > 1:
                    task_results[f"{metric}_std"] = np.nanstd(values)
            
            # Add to overall results
            results[f"{task}_Baseline"] = task_results
    
    sys.path.remove(os.path.join(ROOT_DIR, "src/ml"))
    
    return results


def cgcnn_predict_external_test(model_dirs_map: Dict[str, Dict[str, Any]], 
                                tasks: List[str],
                                task_types: Dict[str, str],
                                ):
    """
    Predict external test set using CGCNN models.
    
    Args:
        model_dirs_map: Dictionary mapping model keys to their configuration
        tasks: List of task names to process
        task_types: Dictionary mapping task names to task types
        output_dir: Directory to save output results
    
    Returns:
        Dictionary with model results for each task
    """
    
    # Define column to task mappings for external test sets
    col2tasks = [
        {"ts_label": "TSD", "ss_label": "SSD"},
        {"water_label": "WS24_water", "water4_label": "WS24_water4", 
         "acid_label": "WS24_acid", "base_label": "WS24_base", 
         "boiling_label": "WS24_boiling"}
    ]
    
    # Define data directories for external test sets
    data_dirs = [TS_EXTERNAL_DIR_CGCNN, WS_EXTERNAL_DIR_CGCNN]
    
    split = "external_test"
    results = {}
    
    # Process CGCNN models (all non-Baseline models)
    for model_key, model_config in model_dirs_map.items():
        if model_key.startswith("Baseline-") or model_key.startswith("Reference"):
            continue
            
        model_specific_paths = model_config["Path"]
        model_display_name = model_config["DisplayName"]
        cgcnn_model_type = model_config.get("Model")
        
        # Convert model_paths to a list if it's a string
        if isinstance(model_specific_paths, str):
            model_specific_paths = [model_specific_paths]
        
        if not cgcnn_model_type:
            print(f"Warning: 'Model' field missing for {model_key}. Skipping.")
            continue
            
        # Check if this is a task-specific model
        is_specific_task_model = model_key in task_types.keys()
        
        # Process each model path
        all_task_metrics = {}
        
        for path_index, model_specific_path in enumerate(model_specific_paths):
            # Load the model
            model_dir = Path(model_specific_path)
            model, trainer = load_model_from_dir(model_dir)
            hparams = model.hparams
            print(f"Successfully loaded model ({path_index+1}/{len(model_specific_paths)}) from {model_dir}")
            print("Model hyperparameters:")
            for k, v in hparams.items():
                if isinstance(v, (str, int, float, bool)):
                    print(f"{k}: {v}")
                    
            # Process each external test set
            for col2task, data_dir in zip(col2tasks, data_dirs):
                for col, task in col2task.items():
                    # Skip if this is a task-specific model and task doesn't match
                    if is_specific_task_model and model_key != task:
                        continue
                        
                    # Skip if task is not in model's task list
                    if "tasks" in hparams and task not in hparams["tasks"]:
                        print(f"Task {task} not found in model tasks: {hparams['tasks']}. Skipping.")
                        continue
                        
                    print(f"Predicting {task} using {model_key} model (path {path_index+1}/{len(model_specific_paths)})...")
                    
                    
                    # Get task information
                    task_id = hparams["tasks"].index(task)
                    task_tp = hparams["task_types"][task_id]
                    dataset_cls = MODEL_NAME_TO_DATASET_CLS[hparams["model_name"]]
                    
                    # Create a copy of hparams to modify safely
                    hparams_copy = dict(hparams)
                    for k in ["data_dir", "split", "task_id", "prop_cols", "csv_file_name"]:
                        if k in hparams_copy:
                            del hparams_copy[k]
                    
                    # Create dataset and dataloader
                    dataset = dataset_cls(
                        data_dir, 
                        split=split, 
                        task_id=task_id,
                        prop_cols=[col], 
                        csv_file_name="RAC_and_zeo_features_with_id_prop.csv",
                        **hparams_copy
                    )
                    
                    dataloader = DataLoader(
                        dataset, 
                        batch_size=min(len(dataset), hparams["batch_size"]), 
                        num_workers=hparams.get("num_workers", 2), 
                        shuffle=False,
                        collate_fn=dataset_cls.collate
                    )
                    
                    # Make predictions
                    outputs = trainer.predict(model, dataloader)
                    
                    # Gather predictions and targets
                    targets = torch.stack([d["targets"] for d in dataset]).cpu().numpy()
                    cif_ids = [d["cif_id"] for d in dataset]
                    predictions = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy()
                    last_layer_fea = torch.cat([d[f"{task}_last_layer_fea"] for d in outputs], dim=0).cpu().numpy()
                    
                    # Calculate uncertainty if trees are available
                    uncertainties = None
                    if "UncertaintyTreesPath" in model_config:
                        uncertainty_trees_path = model_config["UncertaintyTreesPath"]
                        try:
                            print(f"Loading uncertainty trees from {uncertainty_trees_path}")
                            with open(uncertainty_trees_path, 'rb') as f:
                                uncertainty_trees = pickle.load(f)
                            
                            if task in uncertainty_trees:
                                print(f"Calculating uncertainty for {task}...")
                                # Calculate uncertainty based on task type
                                if "classification" in task_tp:
                                    uncertainties = calculate_lse_from_tree(
                                        uncertainty_trees[task],
                                        last_layer_fea,
                                        k=uncertainty_trees[task].get("k", 5)
                                    )
                                else:
                                    uncertainties = calculate_lsv_from_tree(
                                        uncertainty_trees[task],
                                        last_layer_fea,
                                        k=uncertainty_trees[task].get("k", 5)
                                    )
                                print(f"Uncertainty shape: {uncertainties.shape if hasattr(uncertainties, 'shape') else 'unknown'}")
                            else:
                                print(f"No uncertainty trees found for task {task}")
                        except Exception as e:
                            print(f"Error loading or using uncertainty trees: {e}")
                    
                    if "classification" in task_tp:
                        # Process classification outputs
                        probabilities = torch.cat([d[f"{task}_prob"] for d in outputs], dim=0).cpu().numpy()
                        
                        # Save predictions
                        results_dict = {
                            "CifId": cif_ids,
                            "GroundTruth": np.concatenate(targets).astype(np.int8).tolist(),
                            "Predicted": np.concatenate(predictions).astype(np.int8).tolist(),
                        }
                        
                        # Add probability values
                        if probabilities.shape[1] == 2:
                            results_dict["Prob"] = probabilities[:, 1].tolist()
                        else:
                            results_dict["Prob"] = probabilities.tolist()
                        
                        # Add uncertainty if available
                        if uncertainties is not None:
                            results_dict["Uncertainty"] = uncertainties.tolist()
                            
                        df_results = pd.DataFrame(results_dict)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(
                            np.concatenate(targets), 
                            np.concatenate(predictions),
                            probabilities,
                            task_type="classification"
                        )
                        
                        img_file = model_dir / f"{split}_confusion_matrix_{task}.png"
                        plot_confusion_matrix(
                            np.concatenate(targets), 
                            np.concatenate(predictions),
                            title=f"{split}/{task}",
                            outfile=str(img_file)
                        )
                        
                        if probabilities.shape[1] == 2:
                            from sklearn.metrics import roc_curve
                            fpr, tpr, thresholds = roc_curve(
                                np.concatenate(targets),
                                probabilities[:, 1],
                                drop_intermediate=False
                            )
                            img_file = model_dir / f"{split}_roc_curve_{task}.png"
                            plot_roc_curve(
                                fpr, tpr, metrics["AUROC"], 
                                title=f"{split}/{task}",
                                outfile=str(img_file)
                            )
                    else:
                        # Process regression outputs
                        results_dict = {
                            "CifId": cif_ids,
                            "GroundTruth": np.concatenate(targets),
                            "Predicted": np.concatenate(predictions),
                        }
                        
                        # Add uncertainty if available
                        if uncertainties is not None:
                            results_dict["Uncertainty"] = uncertainties
                        
                        df_results = pd.DataFrame(results_dict)
                        df_results["Error"] = (df_results["GroundTruth"] - df_results["Predicted"]).abs()
                        
                        # Calculate metrics
                        metrics = calculate_metrics(
                            np.concatenate(targets), 
                            np.concatenate(predictions),
                            task_type="regression"
                        )
                        
                        # Create scatter plot only for the first path to avoid clutter
                        if path_index == 0:
                            img_file = model_dir / f"{split}_scatter_{task}.png"
                            plot_scatter(
                                np.concatenate(targets),
                                np.concatenate(predictions),
                                title=f"{split}/{task}",
                                metrics=metrics,
                                outfile=str(img_file)
                            )
                    
                    # Save predictions to model directory
                    model_csv_file = model_dir / f"{split}_results_{task}.csv"
                    df_results.to_csv(model_csv_file, index=False)
                    print(f"Saved predictions to model directory: {model_csv_file}")

                    # Save last layer features in model directory
                    np.savez(
                        model_dir / f"{split}_last_layer_fea_{task}.npz", 
                        last_layer_fea
                    )
                    
                    # Store metrics for this task and path
                    task_key = f"{task}_{model_display_name}"
                    if task_key not in all_task_metrics:
                        all_task_metrics[task_key] = []
                    all_task_metrics[task_key].append(metrics)
                    
                    print(f"Successfully processed {task} with metrics: {metrics}")
        
        # Aggregate metrics across paths for each task
        for task_key, metrics_list in all_task_metrics.items():
            if metrics_list:
                task_name = task_key.split('_')[0]  # Extract task name from the key
                
                # Create aggregated task results
                task_results = {"Task": task_name, "Model": model_display_name}
                
                # Calculate mean and std for each metric
                if len(metrics_list) > 1:
                    # Get all unique metric keys
                    all_metric_keys = set()
                    for m in metrics_list:
                        all_metric_keys.update(m.keys())
                    
                    # Calculate mean and std for each metric
                    for metric in all_metric_keys:
                        values = [m.get(metric, np.nan) for m in metrics_list]
                        task_results[metric] = np.nanmean(values)
                        task_results[f"{metric}_std"] = np.nanstd(values)
                else:
                    # Only one path, use metrics directly
                    task_results.update(metrics_list[0])
                
                # Add to overall results
                results[task_key] = task_results
    
    return results


def combine_results(ml_results: Dict[str, Dict[str, Any]], 
                   cgcnn_results: Dict[str, Dict[str, Any]],
                   tasks: List[str],
                   output_dir: str = DEFAULT_OUTPUT_DIR,
                   output_file: str = "external_test_results.xlsx"):
    """
    Combine ML and CGCNN model results and save to Excel file.
    
    Args:
        ml_results: Dictionary with ML model results
        cgcnn_results: Dictionary with CGCNN model results
        output_dir: Directory to save output Excel file
        output_file: Name of the output Excel file
    
    Returns:
        DataFrame with combined results
    """
    # Combine results
    all_results = []
    
    for results_dict in [ml_results, cgcnn_results]:
        for key, result in results_dict.items():
            all_results.append(result)
    
    # Convert to DataFrame
    if not all_results:
        print("No results to combine. Exiting.")
        return None
        
    df_results = pd.DataFrame(all_results)
    df_results["Task"] = df_results["Task"].astype(str).str.strip() # Clean task names
    task_order_categories = tasks 
    df_results["Task"] = pd.Categorical(
        df_results["Task"],
        categories=task_order_categories,
        ordered=True
    )
    
    # Make paths
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    # Save to Excel
    df_results.set_index(["Task", "Model"], inplace=True)
    df_results.to_excel(output_path)
    print(f"Saved combined results to {output_path}")
    
    return df_results


def main(config_path: str = DEFAULT_CONFIG_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        comparison_group: Optional[str] = None):
    """
    Main function to run the external test prediction.
    
    Args:
        config_path: Path to the configuration file
        output_dir: Directory to save output Excel file
        comparison_group: Optional comparison group to use
    Returns:
        None
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration
    model_dirs_map = config["model_dirs_map"]
    tasks = config["tasks"]
    task_types = config["task_types"]
    
    # Filter by comparison group if specified
    if comparison_group and "comparison_groups" in config:
        comparison_groups = config["comparison_groups"]
        if comparison_group in comparison_groups:
            group_models = comparison_groups[comparison_group]
            filtered_map = {k: model_dirs_map[k] for k in group_models if k in model_dirs_map}
            model_dirs_map = filtered_map
            print(f"Using comparison group '{comparison_group}' with models: {list(model_dirs_map.keys())}")
            output_dir = os.path.join(output_dir, comparison_group)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Run ML model predictions
    print("Running ML model predictions on external test sets...")
    sys.path.remove(os.path.join(ROOT_DIR, "src/cgcnn"))
    ml_results = ml_predict_external_test(model_dirs_map, tasks, task_types)
    
    # Run CGCNN model predictions
    print("Running CGCNN model predictions on external test sets...")
    sys.path.append(os.path.join(ROOT_DIR, "src/cgcnn"))
    cgcnn_results = cgcnn_predict_external_test(model_dirs_map, tasks, task_types)
    
    # Combine and save results
    print("Combining and saving results...")
    combine_results(ml_results, cgcnn_results, tasks, output_dir, "model_performance_comparison_external_test.xlsx")
    
    print("External test prediction complete!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Predict and evaluate models on external test sets")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the model comparison configuration file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save output Excel file")
    parser.add_argument("--comparison_group", type=str, default="standard",
                        help="Optional comparison group to use from the config file")
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        output_dir=args.output_dir,
        comparison_group=args.comparison_group,
    )
