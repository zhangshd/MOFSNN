'''
Author: zhangshd
Date: 2025-04-22
Description: This script visualizes the uncertainty in latent space using dimensionality reduction for the MOFSNN model.
It predicts all samples in the training, validation, and test sets, saves the latent vectors,
calculates uncertainty, and creates visualizations with uncertainty/target labels as color labels.
Multiple dimensionality reduction methods are supported: t-SNE, UMAP, and PCA.
'''

import os
import sys
# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory of the project (two levels up from the script directory)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Add the src directory to the Python path
sys.path.append(os.path.dirname(SCRIPT_DIR))
from argparse import ArgumentParser
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap
    import warnings
    # Filter UMAP specific warnings about n_jobs and random_state
    warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with 'pip install umap-learn' to use UMAP dimensionality reduction.")
from matplotlib.colors import Normalize
from tqdm import tqdm
import pickle
import json

from cgcnn.utils import load_model_from_dir
from cgcnn.module.module_utils import calculate_lse_from_tree, calculate_lsv_from_tree

# Class label mapping dictionary for classification tasks
# Maps numerical class labels to meaningful text labels for better visualization
TARGETS_MAP = {
    "TSD": None,  # TSD is a regression task
    "SSD": {0: "unstable", 1: "stable"},
    "WS24_water": {0: "unstable", 1: "stable"},
    "WS24_water4": {0: "unstable", 1: "low kinetic stability", 2: "high kinetic stability", 3: "thermodynamic stable"},
    "WS24_acid": {0: "unstable", 1: "stable"},
    "WS24_base": {0: "unstable", 1: "stable"},
    "WS24_boiling": {0: "unstable", 1: "stable"},
}

# Dimension reduction method display names
DIM_REDUCTION_METHODS = {
    "tsne": "t-SNE",
    "umap": "UMAP",
    "pca": "PCA"
}

def get_dataloader_from_datamodule(model, data_module, split='train'):
    """
    Get dataloader from a specific split of the data module.
    This function ensures the data module is properly set up before requesting a dataloader.
    
    Args:
        model: The loaded model
        data_module: The data module instance
        split: Which split to get ('train', 'val', or 'test')
        
    Returns:
        The appropriate dataloader for the requested split
    """
    # First complete proper setup of the data module for the specific stage
    print(f"Setting up data module for '{split}' split...")
    data_module.setup(stage=split)
    
    # Check if the required dataset attribute exists
    dataset_attr = f"{split}set"
    if not hasattr(data_module, dataset_attr) or getattr(data_module, dataset_attr) is None:
        print(f"Warning: {dataset_attr} not properly initialized in data module. Trying to force setup...")
        # Try to force a more complete setup
        setup_stage = 'fit' if split in ['train', 'val'] else 'test'
        data_module.setup(stage=setup_stage)
        if not hasattr(data_module, dataset_attr) or getattr(data_module, dataset_attr) is None:
            raise AttributeError(f"Data module does not have '{dataset_attr}' attribute even after forced setup.")
    
    dataloader_method = f"{split}_dataloader"
    return getattr(data_module, dataloader_method)()

def predict_and_collect_features(model, dataloader, device):
    """
    Run model prediction on dataloader and collect latent features, predictions, and targets.
    """
    model.eval()
    all_features = {}
    all_preds = {}
    all_targets = {}
    all_cif_ids = {}
    all_extra_features = {}
    
    for task_id, task in enumerate(model.hparams.tasks):
        all_features[task] = []
        all_preds[task] = []
        all_targets[task] = []
        all_cif_ids[task] = []
        all_extra_features[task] = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batch"):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs, last_layer_feas = model.model(**batch)
            
            for task_id, task in enumerate(model.hparams.tasks):
                # Get task mask
                task_mask = (batch['task_id'] == task_id)
                if not torch.any(task_mask):
                    continue
                
                # Get task-specific data
                task_targets = batch['targets'][task_mask].cpu().numpy()
                task_cif_ids = np.array(batch['cif_id'])[task_mask.cpu().numpy()]
                task_extra_features = batch['extra_fea'][task_mask].cpu().numpy()

                # Get predictions
                if model.hparams.task_types[task_id] == 'regression':
                    task_preds = model.denormalize(outputs[task_id][task_mask], task_id).cpu().numpy()
                else:
                    task_preds = torch.argmax(outputs[task_id][task_mask], dim=1).cpu().numpy()
                
                # Get latent features
                task_features = last_layer_feas[task_id][task_mask].cpu().numpy()
                
                # Append to lists
                all_features[task].append(task_features)
                all_preds[task].append(task_preds)
                all_targets[task].append(task_targets)
                all_cif_ids[task].extend(task_cif_ids)
                all_extra_features[task].append(task_extra_features)
    
    # Concatenate all batches
    for task in model.hparams.tasks:
        if len(all_features[task]) > 0:
            all_features[task] = np.concatenate(all_features[task], axis=0)
            all_preds[task] = np.concatenate(all_preds[task], axis=0)
            all_targets[task] = np.concatenate(all_targets[task], axis=0)
            all_extra_features[task] = np.concatenate(all_extra_features[task], axis=0)

    return all_features, all_preds, all_targets, all_cif_ids, all_extra_features

def calculate_uncertainty(latent_vectors, uncertainty_trees, task, task_type):
    """
    Calculate uncertainty for each sample based on its latent vector.
    """
    if task not in uncertainty_trees:
        print(f"Warning: Task {task} not found in uncertainty trees. Skipping uncertainty calculation.")
        return None
    
    if "classification" in task_type:
        return calculate_lse_from_tree(uncertainty_trees[task], latent_vectors, 
                                       k=uncertainty_trees[task]["k"], scale=True)
    else:
        return calculate_lsv_from_tree(uncertainty_trees[task], latent_vectors, 
                                       k=uncertainty_trees[task]["k"], scale=True)

def apply_dimensionality_reduction(features, method="tsne", random_state=42):
    """
    Apply dimensionality reduction to feature vectors.
    
    Args:
        features: Feature vectors to reduce (numpy array)
        method: Dimensionality reduction method ('tsne', 'umap', or 'pca')
        random_state: Random state for reproducibility
        
    Returns:
        2D reduced vectors as a numpy array
    """
    n_samples = features.shape[0]
    
    if method.lower() == "tsne":
        # Adjust perplexity to avoid issues with small datasets
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        reduced_vectors = reducer.fit_transform(features)
        
    elif method.lower() == "umap" and UMAP_AVAILABLE:
        # UMAP typically needs more samples than t-SNE to be effective
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
        reduced_vectors = reducer.fit_transform(features)
        
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_vectors = reducer.fit_transform(features)
        
    else:
        if method.lower() == "umap" and not UMAP_AVAILABLE:
            print("Warning: UMAP not available. Falling back to t-SNE.")
        else:
            print(f"Warning: Unknown method '{method}'. Falling back to t-SNE.")
        
        # Fallback to t-SNE
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        reduced_vectors = reducer.fit_transform(features)
    
    return reduced_vectors

def identify_error_samples(predictions, targets, task_type):
    """
    Identify samples with prediction errors based on task type.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task ('regression' or 'classification_*')
        
    Returns:
        Boolean mask where True indicates error samples
    """
    # Ensure predictions and targets have the same shape
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    if 'regression' in task_type:
        # For regression tasks, find samples with >20% relative error
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        relative_error = np.abs(predictions - targets) / (np.abs(targets) + epsilon)
        error_mask = relative_error > 0.2  # >20% error
        # error_mask = np.abs(predictions - targets) > 45.2
    else:
        # For classification tasks, find samples with mismatched predictions
        error_mask = predictions != targets
    
    return error_mask

def create_scatter_plot(ax, reduced_vectors, color_values, colormap, norm=None, alpha=0.7, s=20, is_discrete=False):
    """
    Create a scatter plot with given data and color settings.
    
    Args:
        ax: Matplotlib axis to plot on
        reduced_vectors: 2D array of points to plot
        color_values: Values used for coloring points
        colormap: Colormap to use
        norm: Normalization for colormap
        alpha: Transparency of points
        s: Size of points
        is_discrete: Whether the coloring is discrete or continuous
        
    Returns:
        scatter plot object
    """
    if is_discrete:
        # For discrete coloring, create a separate scatter for each class
        unique_values = np.unique(color_values)
        scatter = None
        for idx, val in enumerate(unique_values):
            mask = (color_values == val)
            scatter = ax.scatter(
                reduced_vectors[mask, 0], reduced_vectors[mask, 1],
                color=colormap(idx % colormap.N),
                alpha=alpha, s=s
            )
    else:
        # For continuous coloring, create a single scatter with colormap
        scatter = ax.scatter(
            reduced_vectors[:, 0], reduced_vectors[:, 1],
            c=color_values, cmap=colormap, norm=norm,
            alpha=alpha, s=s
        )
    
    return scatter

def plot_error_samples(ax, reduced_vectors, error_mask, label=None):
    """
    Plot error samples on the given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        reduced_vectors: 2D array of all points
        error_mask: Boolean mask indicating error samples
        label: Label for the error samples in the legend
        
    Returns:
        The scatter plot object or None if no error samples
    """
    if np.any(error_mask):
        error_points = reduced_vectors[error_mask]
        scatter = ax.scatter(
            error_points[:, 0], error_points[:, 1],
            color='black', alpha=0.5, s=30, marker='x',
            label=label
        )
        return scatter
    return None

def setup_class_legend(ax, targets, task, cmap):
    """
    Set up legend for classification tasks.
    
    Args:
        ax: Matplotlib axis
        targets: Target values 
        task: Task name for TARGETS_MAP lookup
        cmap: Colormap used
    """
    unique_targets = np.unique(targets)
    handles = []
    labels = []
    
    for class_idx, class_val in enumerate(unique_targets):
        # Use TARGETS_MAP for class labels if available
        if task in TARGETS_MAP and TARGETS_MAP[task] is not None and int(class_val) in TARGETS_MAP[task]:
            class_label = TARGETS_MAP[task][int(class_val)]
        else:
            class_label = f'Class {int(class_val)}'
        
        # Create a proxy artist for the legend
        handle = plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=cmap(class_idx % cmap.N), 
                            markersize=8)
        handles.append(handle)
        labels.append(class_label)
    
    ax.legend(handles, labels, loc='best', fontsize=8)

def create_visualization_figure(task, features, preds, targets, uncertainties, task_type, 
                               dim_reduction_methods, output_dir, figsize=(20, 12)):
    """
    Create visualization figure for a task with multiple dimensionality reduction methods
    and both target and uncertainty coloring.
    
    Args:
        task: Task name
        features: Feature vectors
        preds: Model predictions
        targets: Ground truth targets
        uncertainties: Uncertainty values
        task_type: Type of task ('regression' or 'classification_*')
        dim_reduction_methods: List of dimensionality reduction methods to use
        output_dir: Directory to save figure
        figsize: Size of figure
        
    Returns:
        Path to saved figure
    """
    # Ensure targets and predictions are flattened arrays
    targets = np.array(targets).flatten()
    preds = np.array(preds).flatten()
    uncertainties = np.array(uncertainties).flatten()
    
    # Identify error samples
    error_mask = identify_error_samples(preds, targets, task_type)
    print(f"Task {task}: {np.sum(error_mask)} error samples out of {len(error_mask)} ({np.sum(error_mask)/len(error_mask)*100:.1f}%)")
    
    # Cache for dimensionality reduction results
    reduced_vectors_cache = {}
    
    # Create figure with 2 rows (target and uncertainty) and len(dim_reduction_methods) columns
    fig, axs = plt.subplots(2, len(dim_reduction_methods), figsize=figsize)
    if len(dim_reduction_methods) == 1:
        axs = axs.reshape(-1, 1)  # Ensure 2D shape
    
    is_classification = 'classification' in task_type
    
    # Process each dimensionality reduction method
    for col_idx, method in enumerate(dim_reduction_methods):
        # Skip unavailable methods
        if (method == "umap" and not UMAP_AVAILABLE):
            print(f"Warning: UMAP not available. Skipping {method} visualization for task {task}.")
            continue
        
        # Apply dimensionality reduction
        if method not in reduced_vectors_cache:
            reduced_vectors_cache[method] = apply_dimensionality_reduction(
                features, method=method)
        
        reduced_vectors = reduced_vectors_cache[method]
        method_display_name = DIM_REDUCTION_METHODS.get(method.lower(), method)
        
        # --- First row: target visualization ---
        ax_target = axs[0, col_idx]
        
        if is_classification:
            # For classification tasks, create separate scatter plots for each class
            unique_targets = np.unique(targets)
            class_scatters = []
            
            # Create colormap
            cmap_name = 'tab10' if len(unique_targets) <= 10 else 'tab20'
            cmap_target = plt.colormaps[cmap_name]
            
            # Plot each class separately with explicit labels
            for idx, val in enumerate(unique_targets):
                mask = (targets == val)
                if task in TARGETS_MAP and TARGETS_MAP[task] is not None and int(val) in TARGETS_MAP[task]:
                    class_label = TARGETS_MAP[task][int(val)]
                else:
                    class_label = f'Class {int(val)}'
                
                scatter = ax_target.scatter(
                    reduced_vectors[mask, 0], reduced_vectors[mask, 1],
                    color=cmap_target(idx % cmap_target.N),
                    alpha=0.7, s=20, label=class_label
                )
                class_scatters.append(scatter)
        else:
            # Create continuous colormap for regression
            norm_target = Normalize(vmin=np.min(targets), vmax=np.max(targets))
            cmap_target = plt.colormaps['viridis_r']
            
            scatter_target = create_scatter_plot(ax_target, reduced_vectors, targets, 
                                               cmap_target, norm=norm_target)
            
            # Add colorbar
            cbar_target = plt.colorbar(scatter_target, ax=ax_target, fraction=0.046, pad=0.04)
            cbar_target.set_label('$T_d$')
        
        # --- Second row: uncertainty visualization ---
        ax_uncertainty = axs[1, col_idx]
        
        # Create uncertainty visualization
        norm_uncertainty = Normalize(vmin=np.min(uncertainties), vmax=np.max(uncertainties))
        cmap_uncertainty = plt.colormaps['viridis_r']  # Reverse viridis for low=blue
        
        scatter_uncertainty = create_scatter_plot(
            ax_uncertainty, reduced_vectors, uncertainties, 
            cmap_uncertainty, norm=norm_uncertainty
        )
        
        # Add colorbar
        cbar_uncertainty = plt.colorbar(scatter_uncertainty, ax=ax_uncertainty, fraction=0.046, pad=0.04)
        cbar_uncertainty.set_label('Uncertainty')
        
        # Add error markers to both plots
        error_label = f'error samples ({np.sum(error_mask)})'
        error_scatter_target = plot_error_samples(ax_target, reduced_vectors, error_mask, error_label)
        error_scatter_uncertainty = plot_error_samples(ax_uncertainty, reduced_vectors, error_mask, error_label)
        
        # Create legends
        if is_classification:
            # For target plot with classification: combine class and error labels
            if error_scatter_target:
                # Get handles and labels before adding error samples
                handles, labels = ax_target.get_legend_handles_labels()
                
                # Add error samples to legend
                ax_target.legend(loc='best', fontsize=8)
            else:
                ax_target.legend(loc='best', fontsize=8)
            
            # For uncertainty plot: add just error samples
            if error_scatter_uncertainty:
                ax_uncertainty.legend(loc='best', fontsize=10)
        else:
            # For regression tasks: add error samples legend to both plots
            if error_scatter_target:
                ax_target.legend(loc='best', fontsize=8)
            
            if error_scatter_uncertainty:
                ax_uncertainty.legend(loc='best', fontsize=10)
        
        # Set labels
        ax_target.set_title(f"{method_display_name} - Target Values")
        ax_uncertainty.set_title(f"{method_display_name} - Uncertainty")
        
        ax_target.set_xlabel(f"{method_display_name} 1")
        ax_target.set_ylabel(f"{method_display_name} 2")
        ax_uncertainty.set_xlabel(f"{method_display_name} 1")
        ax_uncertainty.set_ylabel(f"{method_display_name} 2")
    
    # Add super title
    plt.suptitle(f"Latent Space Visualization for {task} Task", fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = output_dir / f"{task}_combined_visualization.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved visualization for task {task} to {output_path}")
    return output_path, reduced_vectors_cache

def create_combined_visualization(results, tasks, model_hparams, output_dir, dim_reduction_method="tsne", 
                                  figsize=(20, 15), base_font_size=12):
    """
    Create combined visualization for all tasks with a specific dimensionality reduction method.
    Two figures are created: one colored by uncertainty and one by target values.
    
    Args:
        results: Dictionary of results
        tasks: List of task names
        model_hparams: Model hyperparameters
        output_dir: Output directory
        dim_reduction_method: Dimensionality reduction method
        figsize: Figure size
        
    Returns:
        Tuple of paths to saved figures (uncertainty_fig_path, target_fig_path)
    """
    # Create subplots
    n_tasks = len(tasks)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    # Create figures
    fig_uncertainty, axes_uncertainty = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig_target, axes_target = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle different subplot configurations
    if n_rows > 1 and n_cols > 1:
        axes_uncertainty = axes_uncertainty.flatten()
        axes_target = axes_target.flatten()
    elif n_rows > 1:
        axes_uncertainty = axes_uncertainty.reshape(-1)
        axes_target = axes_target.reshape(-1)
    elif n_cols > 1:
        pass  # Already 2D
    else:
        axes_uncertainty = np.array([axes_uncertainty])
        axes_target = np.array([axes_target])
    
    method_display_name = DIM_REDUCTION_METHODS.get(dim_reduction_method.lower(), dim_reduction_method)
    
    # Process each task
    for i, task in enumerate(tasks):
        if i >= len(axes_uncertainty):
            print(f"Warning: Not enough subplots for task {task}. Skipping.")
            continue
            
        task_id = model_hparams.tasks.index(task)
        task_type = model_hparams.task_types[task_id]
        
        # Check if data exists
        if task not in results:
            print(f"Warning: No data for task {task}. Skipping.")
            continue
            
        ax_uncertainty = axes_uncertainty[i]
        ax_target = axes_target[i]
        
        # Combine data from all splits
        all_features = []
        all_preds = []
        all_targets = []
        all_uncertainties = []
        
        for split in ['train', 'val', 'test']:
            if split in results[task]:
                all_features.append(results[task][split]['features'])
                all_preds.append(results[task][split]['preds'])
                all_targets.append(results[task][split]['targets'])
                all_uncertainties.append(results[task][split]['uncertainties'])
        
        if not all_features:
            print(f"No data for task {task}. Skipping.")
            continue
        
        # Concatenate data
        features = np.concatenate(all_features, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0).flatten()
        uncertainties = np.concatenate(all_uncertainties, axis=0).flatten()
        
        # Apply dimensionality reduction
        reduced_vectors = apply_dimensionality_reduction(features, method=dim_reduction_method)
        
        # Identify error samples
        error_mask = identify_error_samples(preds, targets, task_type)
        error_label = f'error samples ({np.sum(error_mask)})'
        
        # ----- Uncertainty plot -----
        norm_uncertainty = Normalize(vmin=np.min(uncertainties), vmax=np.max(uncertainties))
        cmap_uncertainty = plt.colormaps['viridis_r']
        
        scatter_uncertainty = create_scatter_plot(
            ax_uncertainty, reduced_vectors, uncertainties, 
            cmap_uncertainty, norm=norm_uncertainty
        )
        
        # Add error samples to uncertainty plot with label
        error_scatter_uncertainty = plot_error_samples(ax_uncertainty, reduced_vectors, error_mask, error_label)

        is_classification = 'classification' in task_type

        # Add colorbar for uncertainty
        fig_uncertainty.colorbar(scatter_uncertainty, ax=ax_uncertainty, 
                                label='LSE' if is_classification else 'LSV', fraction=0.046, pad=0.04)

        # Add legend for error samples on uncertainty plot
        if error_scatter_uncertainty:
            ax_uncertainty.legend(loc='best', fontsize=8)
        
        # ----- Target plot -----
        
        if is_classification:
            # Classification tasks - create separate scatter for each class
            unique_targets = np.unique(targets)
            cmap_name = 'tab10' if len(unique_targets) <= 10 else 'tab20'
            cmap_target = plt.colormaps[cmap_name]
            
            # Plot each class with its own label
            for idx, val in enumerate(unique_targets):
                mask = (targets == val)
                if task in TARGETS_MAP and TARGETS_MAP[task] is not None and int(val) in TARGETS_MAP[task]:
                    class_label = TARGETS_MAP[task][int(val)]
                else:
                    class_label = f'Class {int(val)}'
                
                scatter = ax_target.scatter(
                    reduced_vectors[mask, 0], reduced_vectors[mask, 1],
                    color=cmap_target(idx % cmap_target.N),
                    alpha=0.7, s=20, label=class_label
                )
        else:
            # Regression tasks - continuous colormap
            norm_target = Normalize(vmin=np.min(targets), vmax=np.max(targets))
            cmap_target = plt.colormaps['viridis_r']
            
            scatter_target = create_scatter_plot(
                ax_target, reduced_vectors, targets, 
                cmap_target, norm=norm_target
            )
            
            # Add colorbar for target values
            fig_target.colorbar(scatter_target, ax=ax_target, 
                              label='$T_d$', fraction=0.046, pad=0.04)
        
        # Add error samples to target plot with label
        error_scatter_target = plot_error_samples(ax_target, reduced_vectors, error_mask, error_label)
        
        # Add legend for target plot
        if is_classification or error_scatter_target:
            ax_target.legend(loc='best', fontsize=base_font_size)
        
        # Set titles and labels
        ax_uncertainty.set_title(f"{task}", fontsize=base_font_size+2, fontweight='bold')
        ax_target.set_title(f"{task}", fontsize=base_font_size+2, fontweight='bold')

        ax_uncertainty.set_xlabel(f"{method_display_name} 1", fontsize=base_font_size+2)
        ax_uncertainty.set_ylabel(f"{method_display_name} 2", fontsize=base_font_size+2)
        ax_target.set_xlabel(f"{method_display_name} 1", fontsize=base_font_size+2)
        ax_target.set_ylabel(f"{method_display_name} 2", fontsize=base_font_size+2)

    # Hide unused subplots
    for i in range(len(tasks), len(axes_uncertainty)):
        axes_uncertainty[i].axis('off')
    for i in range(len(tasks), len(axes_target)):
        axes_target[i].axis('off')
    
    # Add super titles
    # fig_uncertainty.suptitle(f"Latent Space Visualization using {method_display_name} Colored by Uncertainty", 
    #                        fontsize=base_font_size+6, fontweight='bold')
    # fig_target.suptitle(f"Latent Space Visualization using {method_display_name} Colored by Target Values", 
    #                   fontsize=base_font_size+6, fontweight='bold')

    # Adjust layout
    fig_uncertainty.tight_layout(rect=[0, 0, 1, 1])
    fig_target.tight_layout(rect=[0, 0, 1, 1])
    
    # Save figures
    uncertainty_path = output_dir / f"all_tasks_{dim_reduction_method.lower()}_uncertainty_visualization.tif"
    target_path = output_dir / f"all_tasks_{dim_reduction_method.lower()}_target_visualization.tif"
    
    fig_uncertainty.savefig(uncertainty_path, dpi=300, bbox_inches='tight')
    fig_target.savefig(target_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig_uncertainty)
    plt.close(fig_target)
    
    print(f"Saved combined visualizations to:\n- {uncertainty_path}\n- {target_path}")
    return uncertainty_path, target_path

def collect_and_process_data(model, data_module, device, uncertainty_trees, output_dir):
    """
    Collect and process data from all data splits.
    
    Args:
        model: Trained model
        data_module: Data module
        device: Computation device
        uncertainty_trees: Dictionary of uncertainty trees
        output_dir: Output directory
        
    Returns:
        Dictionary of results by task and split
    """
    results = {task: {} for task in model.hparams.tasks}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"Processing {split} set")
        try:
            dataloader = get_dataloader_from_datamodule(model, data_module, split)
            
            if not dataloader:
                print(f"No data in {split} set. Skipping.")
                continue
            # Predict and collect features
            features, preds, targets, cif_ids, extra_features = predict_and_collect_features(model, dataloader, device)

            # Process each task
            for task_id, task in enumerate(model.hparams.tasks):
                if task not in features or len(features[task]) == 0:
                    print(f"No data for task {task} in {split} set. Skipping.")
                    continue
                
                print(f"Processing task: {task}")
                
                # Calculate uncertainty
                task_type = model.hparams.task_types[task_id]
                uncertainties = calculate_uncertainty(features[task], uncertainty_trees, task, task_type)
                
                if uncertainties is None:
                    print(f"Could not calculate uncertainty for task {task}. Skipping.")
                    continue
                
                # Store results
                results[task][split] = {
                    'features': features[task],
                    'preds': preds[task],
                    'targets': targets[task],
                    'uncertainties': uncertainties,
                    'cif_ids': cif_ids[task],
                    'extra_features': extra_features[task] ## input extra features
                }
                
                # Save results to CSV
                # df = pd.DataFrame({
                #     'cif_id': cif_ids[task],
                #     'prediction': preds[task].squeeze(),
                #     'target': targets[task].squeeze(),
                #     'uncertainty': uncertainties.squeeze()
                # })
                
                # df.to_csv(output_dir / f"{task}_{split}_results.csv", index=False)
        except Exception as e:
            print(f"Error processing {split} set: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def create_data_module(model):
    """
    Create a data module from model hyperparameters.
    
    Args:
        model: Trained model with hyperparameters
        
    Returns:
        Configured data module
    """
    from cgcnn.datamodule.data_interface import DInterface
    
    # Extract parameters from model hparams
    hparams = model.hparams
    hparams_dict = dict(hparams)
    
    # Update data directory to standard location
    data_dir = os.path.join(ROOT_DIR, "data/cgcnn_data")
    print(f"Using data directory: {data_dir}")
    hparams_dict['data_dir'] = data_dir
    
    # Try to create data module with full parameters
    try:
        data_module = DInterface(**hparams_dict)
    except Exception as e:
        print(f"Error creating data module with full parameters: {str(e)}")
        # Fall back to minimal parameters
        min_params = {
            'data_dir': data_dir,
            'tasks': hparams.tasks,
            'task_types': hparams.task_types,
            'batch_size': getattr(hparams, 'batch_size', 32),
            'num_workers': getattr(hparams, 'num_workers', 2),
            'dl_sampler': getattr(hparams, 'dl_sampler', 'random')
        }
        print("Using minimal parameters instead")
        data_module = DInterface(**min_params)
    
    # Log key parameters
    print(f"Data module parameters:")
    print(f"  - tasks: {data_module.tasks}")
    print(f"  - task_types: {data_module.task_types}")
    print(f"  - batch_size: {data_module.batch_size}")
    print(f"  - data_dir: {data_module.root_dir}")
    
    return data_module

import scipy.stats as stats  # Add this import at the top of the file

def create_uncertainty_error_histograms(results, tasks, model_hparams, output_dir, figsize=(20, 15), bins=20):
    """
    Create mirror plots showing uncertainty distribution and error rates.
    
    For each task, creates a subplot with:
    - Top part (y > 0): KDE showing distribution of uncertainty values across train/val/test sets
    - Bottom part (y < 0): Histogram showing error rate within each uncertainty bin
    
    Args:
        results: Dictionary of results by task and split
        tasks: List of task names
        model_hparams: Model hyperparameters
        output_dir: Output directory
        figsize: Figure size
        bins: Number of bins for histograms
        
    Returns:
        Path to saved figure
    """
    # Create subplots
    n_tasks = len(tasks)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Ensure axs is 2D array even with single row or column
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Define colors for different splits
    split_colors = {
        'train': 'blue',
        'val': 'orange',
        'test': 'green'
    }
    
    # Process each task
    for i, task in enumerate(tasks):
        if i >= n_rows * n_cols:
            print(f"Warning: Not enough subplots for task {task}. Skipping.")
            continue
            
        # Determine subplot indices
        row = i // n_cols
        col = i % n_cols
        
        # Get the axis for this task
        ax = axs[row, col]
        
        # Check if data exists
        if task not in results:
            print(f"Warning: No data for task {task}. Skipping.")
            ax.axis('off')
            continue
            
        # Get task info
        task_id = model_hparams.tasks.index(task)
        task_type = model_hparams.task_types[task_id]
        
        # Calculate uncertainty range for consistent x-axis
        all_uncertainties = []
        for split in ['train', 'val', 'test']:
            if split in results[task]:
                all_uncertainties.append(results[task][split]['uncertainties'])
        
        if not all_uncertainties:
            print(f"No data for task {task}. Skipping.")
            ax.axis('off')
            continue
            
        all_uncertainties = np.concatenate(all_uncertainties)
        uncertainty_min = np.min(all_uncertainties)
        uncertainty_max = np.max(all_uncertainties)
        
        # Add a small buffer to the min/max range
        padding = (uncertainty_max - uncertainty_min) * 0.05
        x_min = uncertainty_min - padding
        x_max = uncertainty_max + padding
        
        # Create bin edges for histograms
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create evaluation points for KDE curves
        x_grid = np.linspace(x_min, x_max, 200)  # More points for smoother KDE
        
        # Track max density for axis scaling
        max_density = 0
        max_error_rate = 0  # Track max error rate for y-axis scaling
        
        # Plot top density curves: uncertainty distribution (y > 0)
        for split, color in split_colors.items():
            if split in results[task]:
                uncertainties = results[task][split]['uncertainties'].flatten()
                if len(uncertainties) > 10:  # Need sufficient samples for KDE
                    try:
                        # Calculate KDE
                        kde = stats.gaussian_kde(uncertainties)
                        density = kde(x_grid)
                        if split != "train": 
                            max_density = max(max_density, np.max(density))
                        # Plot KDE curve
                        ax.plot(x_grid, density, color=color, alpha=0.8, 
                               label=f'{split} ({len(uncertainties)} samples)', 
                               linewidth=2)
                        # Fill under the curve
                        ax.fill_between(x_grid, density, alpha=0.3, color=color)
                    except Exception as e:
                        print(f"Error calculating KDE for {task}, {split}: {str(e)}")
        
        # Plot bottom part: error rate within each uncertainty bin
        for split, color in split_colors.items():
            if split in results[task]:
                uncertainties = results[task][split]['uncertainties'].flatten()
                preds = results[task][split]['preds'].flatten()
                targets = results[task][split]['targets'].flatten()
                
                # Identify error samples
                error_mask = identify_error_samples(preds, targets, task_type)
                total_errors = np.sum(error_mask)
                
                # Calculate error rate in each bin
                bin_error_rates = []
                bin_counts = []
                
                for j in range(len(bin_edges) - 1):
                    # Get samples in this bin
                    bin_mask = (uncertainties >= bin_edges[j]) & (uncertainties < bin_edges[j+1])
                    bin_count = np.sum(bin_mask)
                    bin_counts.append(bin_count)
                    
                    # Calculate error rate for this bin
                    if bin_count > 0:
                        bin_error_count = np.sum(error_mask & bin_mask)
                        error_rate = bin_error_count / bin_count
                    else:
                        error_rate = 0
                    
                    bin_error_rates.append(error_rate)
                    max_error_rate = max(max_error_rate, error_rate)
                
                # Convert to numpy arrays for plotting
                bin_error_rates = np.array(bin_error_rates)
                
                # Plot error rates as bars
                # Negative values for mirroring below x-axis
                ax.bar(bin_centers, -bin_error_rates, width=bin_width*0.8, color=color, alpha=0.7,
                      label=f'{split} error rates ({total_errors} errors)')
                
                # Optional: Add text with actual bin counts for bins with samples
                for j, (count, rate) in enumerate(zip(bin_counts, bin_error_rates)):
                    if count > 0 and rate > 0.05:  # Only show text for bins with enough samples and visible error rate
                        ax.text(bin_centers[j], -rate-0.05, f'{count}', 
                               ha='center', va='top', fontsize=6, color=color)
        
        # Set axis properties
        ax.set_title(f"{task}")
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Density (top) / Error Rate (bottom)')
        
        # Adjust y-limits
        if max_density > 0:
            density_buffer = max_density * 0.1
            error_buffer = max(0.1, max_error_rate * 0.1)
            # Make error rate scale up to 1.0 if any bin has high error rate
            bottom_limit = -min(1.0, max_error_rate + error_buffer)
            ax.set_ylim(bottom_limit, max_density + density_buffer)
        
        # Set x-axis limits based on uncertainty range
        ax.set_xlim(x_min, x_max)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add legends with smaller font size and better positioning
        ax.legend(fontsize=7, loc='upper right')
    
    # Hide any unused subplots
    for i in range(len(tasks), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if row < len(axs) and col < len(axs[0]):
            axs[row, col].axis('off')
    
    # Add super title
    plt.suptitle("Uncertainty Distribution and Error Rates by Task\n(Top: Distribution, Bottom: Error Rate per Bin)", 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = output_dir / "uncertainty_error_rate_plots.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved uncertainty-error rate plots to {output_path}")
    return output_path

def find_nearest_neighbors(features, cif_ids, preds, targets, uncertainties, extra_features, reduced_vectors=None, k=5, df_info=None):
    """
    Find the k nearest neighbors for each sample in the feature space.
    
    Args:
        features: Feature vectors as numpy array [n_samples, n_features]
        cif_ids: List of CIF IDs for all samples
        preds: Model predictions as numpy array
        targets: Ground truth targets as numpy array
        uncertainties: Uncertainty values as numpy array
        reduced_vectors: Optional reduced vectors (e.g., from t-SNE, UMAP, PCA)
        k: Number of nearest neighbors to find (default: 5)
        
    Returns:
        Dictionary with nearest neighbor information for each sample
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # Ensure all inputs are numpy arrays
    features = np.array(features)
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    uncertainties = np.array(uncertainties).flatten()
    extra_features = np.array(extra_features)
    
    # Initialize nearest neighbors model on feature space
    nn_model = NearestNeighbors(n_neighbors=k+1)  # +1 because the sample itself is included
    nn_model.fit(features)
    
    # Find nearest neighbors (returns distances and indices)
    distances, indices = nn_model.kneighbors(features)
    
    # Create result dictionary
    nearest_neighbors_dict = {}
    
    # For each sample, collect neighbor information
    for i in range(len(features)):
        # Skip the first neighbor (which is the sample itself)
        neighbor_indices = indices[i, 1:k+1]
        neighbor_distances = distances[i, 1:k+1]
        
        # Collect neighbor information
        neighbors = []
        for j, (idx, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
            neighbor_info = {
                'cif_id': cif_ids[idx],
                'pred': float(preds[idx]),
                'target': float(targets[idx]),
                'uncertainty': float(uncertainties[idx]),
                'distance': float(dist),
                'extra_features': extra_features[idx].tolist()
            }
            if df_info is not None and len(df_info) > 0 and  'MofName' in df_info.columns and cif_ids[idx] in df_info['MofName'].values:
                neighbor_info["topology"] = df_info.loc[df_info['MofName'] == cif_ids[idx], 'topology'].values[0]
                neighbor_info["linkers"] = df_info.loc[df_info['MofName'] == cif_ids[idx], 'linkers'].values[0]
                neighbor_info["node_fomula"] = df_info.loc[df_info['MofName'] == cif_ids[idx], 'node_fomula'].values[0]


            
            # Add reduced vectors if available
            if reduced_vectors is not None:
                neighbor_info['reduced_vector'] = reduced_vectors[idx].tolist()
                
            neighbors.append(neighbor_info)
        
        # Create sample entry
        sample_info = {
            'cif_id': cif_ids[i],
            'pred': float(preds[i]),
            'target': float(targets[i]),
            'uncertainty': float(uncertainties[i]),
            'extra_features': extra_features[i].tolist()
        }
        if df_info is not None and len(df_info) > 0 and 'MofName' in df_info.columns and cif_ids[i] in df_info['MofName'].values:
            sample_info["topology"] = df_info.loc[df_info['MofName'] == cif_ids[i], 'topology'].values[0]
            sample_info["linkers"] = df_info.loc[df_info['MofName'] == cif_ids[i], 'linkers'].values[0]
            sample_info["node_fomula"] = df_info.loc[df_info['MofName'] == cif_ids[i], 'node_fomula'].values[0]
        
        # Add reduced vector if available
        if reduced_vectors is not None:
            sample_info['reduced_vector'] = reduced_vectors[i].tolist()
            
        # Add neighbors information
        sample_info['neighbors'] = neighbors
        
        # Add to dictionary with CIF ID as key
        nearest_neighbors_dict[cif_ids[i]] = sample_info
    
    return nearest_neighbors_dict

def run_analysis(model_dir, uncertainty_trees_file, output_dir, dim_reduction_methods=["tsne"], use_data_module=None):
    """
    Run the full analysis pipeline.
    
    Args:
        model_dir: Path to the directory containing the model checkpoint
        uncertainty_trees_file: Path to the file containing uncertainty trees
        output_dir: Directory where output files will be saved
        dim_reduction_methods: List of dimensionality reduction methods to use
        use_data_module: Optional pre-configured data module to use
        
    Returns:
        Dictionary containing the results of the analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {model_dir}")
    model, trainer = load_model_from_dir(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load uncertainty trees
    print(f"Loading uncertainty trees from {uncertainty_trees_file}")
    with open(uncertainty_trees_file, 'rb') as f:
        uncertainty_trees = pickle.load(f)
    
    # Get data module
    data_module = use_data_module if use_data_module else create_data_module(model)
    
    # Collect and process data
    results = collect_and_process_data(model, data_module, device, uncertainty_trees, output_dir)

    # Load additional information (nodes, linkers, and topologys) from Excel file
    info_file = Path(ROOT_DIR)/"data/raw_data/CoREMOF2019_mofs_addtion_processed.xlsx"
    if info_file.exists():
        print(f"Loading additional information from {info_file}")
        df_info_ = pd.read_excel(info_file)
        df_info_.rename(columns={"name": "MofName"}, inplace=True)
    else:
        df_info_ = pd.DataFrame([], columns=["MofName", "linkers", "node_fomula", "topology"])

    # Create task-specific visualizations
    print("\nCreating visualizations by task...")
    for task in model.hparams.tasks:
        if task not in results or not results[task]:
            print(f"No data for task {task}. Skipping visualization.")
            continue
        df_info = df_info_.copy()

        # for tasks that are not TSD or SSD, remove "_clean" from MofName
        if task not in ["TSD", "SSD"]:
            df_info["MofName"] = df_info["MofName"].apply(lambda x: x.replace("_clean", ""))

        # Combine data from all splits
        all_features = []
        all_preds = []
        all_targets = []
        all_uncertainties = []
        all_splits = []
        all_cif_ids = []
        all_extra_features = []
        
        for split in ['train', 'val', 'test']:
            if split in results[task]:
                all_features.append(results[task][split]['features'])
                all_preds.append(results[task][split]['preds'])
                all_targets.append(results[task][split]['targets'])
                all_uncertainties.append(results[task][split]['uncertainties'])
                all_splits.extend([split ]*len(results[task][split]['targets']))
                all_cif_ids.extend(results[task][split]['cif_ids'])
                all_extra_features.append(results[task][split]['extra_features'])

        if not all_features:
            continue
            
        # Get task info
        task_id = model.hparams.tasks.index(task)
        task_type = model.hparams.task_types[task_id]
        
        # Concatenate data
        features = np.concatenate(all_features, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        extra_features = np.concatenate(all_extra_features, axis=0)

        # Create visualization
        _, reduced_vectors_cache = create_visualization_figure(
            task, features, preds, targets, uncertainties, 
            task_type, dim_reduction_methods, output_dir
        )
        
        # Find nearest neighbors in feature space
        print(f"Finding nearest neighbors for task {task} in feature space...")

        nearest_neighbors_dict = find_nearest_neighbors(
            features=features,
            cif_ids=all_cif_ids,
            preds=preds,
            targets=targets,
            uncertainties=uncertainties,
            extra_features=extra_features,
            reduced_vectors=reduced_vectors_cache.get(dim_reduction_methods[0]) if dim_reduction_methods else None,
            k=10,
            df_info=df_info
        )

        # Save nearest neighbors information
        with open(output_dir / f"{task}_nearest_neighbors.json", 'w') as f:
            json.dump(nearest_neighbors_dict, f, indent=2)
        print(f"Saved nearest neighbors information for task {task} to {output_dir / f'{task}_nearest_neighbors.json'}")

        # save reduced vectors cache and results to csv
        df_results = pd.DataFrame({
            'MofName': all_cif_ids,
            'Partition': all_splits,
            'Predicted': preds.flatten(),
            'GroudTruth': targets.flatten(),
            'Uncertainty': uncertainties.flatten()
        })
        if task_type == 'regression':
            df_results['RelativeError'] = np.abs(preds.flatten() - targets.flatten())/(targets.flatten()+1e-6)
        else:
            df_results['IsError'] = preds.flatten() != targets.flatten()

        for method, reduced_vectors in reduced_vectors_cache.items():
            df_results[f'{method.upper()}1'] = reduced_vectors[:, 0]
            df_results[f'{method.upper()}2'] = reduced_vectors[:, 1]

        if df_info is not None:
            try:
                df_results = df_results.merge(df_info[["MofName", "linkers", "node_fomula", "topology"]], on='MofName', how='left')
            except KeyError as e:
                print(f"Error merging additional information: {e}")
        df_results.to_csv(output_dir / f"{task}_reduced_vectors.csv", index=False)
        print(f"Saved reduced vectors for task {task} to {output_dir / f'{task}_reduced_vectors.csv'}")


    # Create combined visualizations
    print("\nCreating combined visualizations for each dimensionality reduction method...")
    for method in dim_reduction_methods:
        print(f"Creating combined visualizations for {method}...")
        create_combined_visualization(
            results, model.hparams.tasks, model.hparams, 
            output_dir, dim_reduction_method=method
        )
    
    # Create uncertainty-error histograms
    print("\nCreating uncertainty-error histograms...")
    create_uncertainty_error_histograms(results, model.hparams.tasks, model.hparams, output_dir)
    
    print("Analysis complete!")
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, 
                        default=os.path.join(ROOT_DIR, "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43"),
                        help="Path to the model directory")
    parser.add_argument("--uncertainty_trees_file", type=str, 
                        default=os.path.join(ROOT_DIR, "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43/uncertainty_trees.pkl"),
                        help="Path to the uncertainty trees file")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(ROOT_DIR, "results/uncertainty_visualization"),
                        help="Directory where visualization outputs will be saved")
    parser.add_argument("--dim_reduction", type=str, nargs='+',
                        default=["tsne"],
                        choices=["tsne", "pca", "umap"],
                        help="Dimensionality reduction method(s) to use. Can specify multiple methods.")
    
    args = parser.parse_args()
    
    # Process paths
    model_dir = Path(args.model_dir)
    uncertainty_trees_file = Path(args.uncertainty_trees_file)
    output_dir = Path(args.output_dir)
    
    print("Starting uncertainty visualization in latent space...")
    print(f"Project root directory: {ROOT_DIR}")
    print(f"Model directory: {model_dir}")
    print(f"Uncertainty trees file: {uncertainty_trees_file}")
    print(f"Output directory: {output_dir}")
    print(f"Dimensionality reduction methods: {args.dim_reduction}")
    
    run_analysis(model_dir, uncertainty_trees_file, output_dir, dim_reduction_methods=args.dim_reduction)