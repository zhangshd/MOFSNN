"""
Feature-Target Relationship Analysis for CGCNN Models

This script analyzes the relationship between target variables and extra_fea features
in CGCNN models. It creates visualizations showing how each feature correlates with
the target variable for both classification and regression tasks.

Author: zhangshd
Date: 2025-05-27
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import torch
from sklearn.metrics import mutual_info_score
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(SCRIPT_DIR).parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from cgcnn.utils import load_model_from_dir, MODEL_NAME_TO_DATASET_CLS


def analyze_feature_target_relationships(model_dir, output_dir=None, split="train"):
    """
    Analyze the relationship between target variables and extra_fea features.
    
    Args:
        model_dir (str): Path to the trained model directory
        output_dir (str, optional): Directory to save output plots. If None, saves to model_dir
        split (str): Dataset split to analyze ('train', 'val', 'test')
    """
    
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "feature_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model and get hyperparameters
    print(f"Loading model from {model_dir}")
    model, trainer = load_model_from_dir(model_dir)
    hparams = model.hparams
    
    print(f"Model: {hparams.get('model_name', 'Unknown')}")
    print(f"Tasks: {hparams.get('tasks', [])}")
    print(f"Task types: {hparams.get('task_types', [])}")
    print(f"Use extra features: {hparams.get('use_extra_fea', False)}")
    
    # Check if model uses extra features
    if not hparams.get('use_extra_fea', False) and not hparams.get('use_cell_params', False):
        print("WARNING: Model does not use extra features. Analysis may not be meaningful.")
    
    # Get dataset class and data directory
    dataset_cls = MODEL_NAME_TO_DATASET_CLS[hparams["model_name"]]
    
    data_dirs = [
        ROOT_DIR/"data/cgcnn_data/TSD",
        ROOT_DIR/"data/cgcnn_data/SSD",
        ROOT_DIR/"data/cgcnn_data/WS24",
    ]
    col2tasks = [
        {"Label": "TSD"},
        {"Label": "SSD"},
        {"water_label": "WS24_water", "water4_label": "WS24_water4", "acid_label": "WS24_acid", "base_label": "WS24_base", "boiling_label": "WS24_boiling"},
        ]
    
    # Analyze each task
    for col2task, data_dir in zip(col2tasks, data_dirs):
        for col, task in col2task.items():
            if task not in hparams["tasks"]:
                continue
            task_id = hparams["tasks"].index(task)
            print(f"Predicting {task}...")
            task_tp = hparams["task_types"][task_id]
            print(f"\nAnalyzing task: {task}")
            
            task_type = hparams["task_types"][task_id]
            
            # Determine property column for this task
            if task in ["TSD", "SSD"]:
                prop_col = "Label"
            elif task.startswith("WS24"):
                prop_col = task.lower().replace("ws24_", "") + "_label"
            else:
                print(f"Unknown task {task}, skipping...")
                continue
            
            
            # Clean hparams for dataset creation
            dataset_hparams = dict(hparams)
            for k in ["data_dir", "split", "task_id", "prop_cols"]:
                if k in dataset_hparams:
                    del dataset_hparams[k]
            
            datasets = []
            for split in ["train", "val", "test"]:
                
                dataset = dataset_cls(
                    data_dir=data_dir,
                    split=split, 
                    task_id=task_id,
                    prop_cols=[prop_col],
                    **dataset_hparams
                )
                datasets.append(dataset)
            dataset = ConcatDataset(datasets)
            
            print(f"Created dataset with {len(dataset)} samples")
            
        
            # Extract features and targets
            extra_features = []
            targets = []
            cif_ids = []
            
            print("Extracting features and targets...")
            for i in range(len(dataset)):
                try:
                    sample = dataset[i]
                    if "extra_fea" in sample:
                        extra_fea = sample["extra_fea"]
                        if extra_fea.numel() > 0:  # Check if extra_fea is not empty
                            extra_features.append(extra_fea.numpy())
                            targets.append(sample["targets"].numpy())
                            cif_ids.append(sample["cif_id"])
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if len(extra_features) == 0:
                print(f"No extra features found for task {task}. Skipping analysis.")
                continue
            
            # Convert to numpy arrays
            extra_features = np.array(extra_features)
            targets = np.array(targets).flatten()
            
            print(f"Extracted {len(extra_features)} samples")
            print(f"Extra features shape: {extra_features.shape}")
            print(f"Targets shape: {targets.shape}")
            
            # Create feature names
            feature_names = ["a", "b", "c", "alpha", "beta", "gamma",]
            
            if extra_features.shape[1] > len(feature_names):
                # Extend feature names if necessary
                csv_file = data_dir / dataset.csv_file_name
                assert csv_file.exists(), f"CSV file not found: {csv_file}"
                df = pd.read_csv(csv_file, index_col=0)
                # Features start from "Di" column
                feature_cols = df.columns[df.columns.get_loc("Di"):].tolist()
                if len(feature_cols) >= extra_features.shape[1]:
                    feature_names = feature_cols[:extra_features.shape[1]-len(feature_names)]
                    print(f"Using feature names from CSV: {len(feature_names)} features")
            
            # Analyze relationships
            print("Analyzing feature-target relationships...")
            if "classification" in task_type:
                analyze_classification_relationships(
                    extra_features, targets, feature_names, task, output_dir
                )
            else:
                analyze_regression_relationships(
                    extra_features, targets, feature_names, task, output_dir
                )
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


def analyze_classification_relationships(features, targets, feature_names, task, output_dir):
    """
    Analyze relationships for classification tasks using violin/box plots.
    """
    n_features = features.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Feature-Target Relationships: {task} (Classification)', fontsize=16, y=0.98)
    
    # Convert targets to integers for classification
    targets_int = targets.astype(int)
    unique_classes = np.unique(targets_int)
    
    for i, feature_name in enumerate(feature_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        feature_values = features[:, i]
        
        # Create DataFrame for seaborn
        df_plot = pd.DataFrame({
            'Feature': feature_values,
            'Target': targets_int
        })
        
        # Create violin plot
        try:
            sns.violinplot(data=df_plot, x='Target', y='Feature', ax=ax)
            ax.set_title(f'{feature_name}', fontsize=12)
            ax.set_xlabel('Target Class')
            ax.set_ylabel('Feature Value')
            
            # Calculate and display mutual information
            mi_score = mutual_info_score(targets_int, 
                                       pd.cut(feature_values, bins=10, labels=False))
            ax.text(0.02, 0.98, f'MI: {mi_score:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{feature_name} (Error)', fontsize=12)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{task}_classification_relationships.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_stats = []
    for i, feature_name in enumerate(feature_names):
        feature_values = features[:, i]
        mi_score = mutual_info_score(targets_int, 
                                   pd.cut(feature_values, bins=10, labels=False))
        
        # Calculate class-wise statistics
        class_stats = {}
        for class_label in unique_classes:
            mask = targets_int == class_label
            if np.sum(mask) > 0:
                class_stats[f'Class_{class_label}_mean'] = np.mean(feature_values[mask])
                class_stats[f'Class_{class_label}_std'] = np.std(feature_values[mask])
        
        summary_stats.append({
            'Feature': feature_name,
            'Mutual_Information': mi_score,
            **class_stats
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / f'{task}_classification_summary.csv', index=False)
    print(f"Classification analysis saved for {task}")


def analyze_regression_relationships(features, targets, feature_names, task, output_dir):
    """
    Analyze relationships for regression tasks using scatter plots with correlation metrics.
    """
    n_features = features.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Feature-Target Relationships: {task} (Regression)', fontsize=16, y=0.98)
    
    summary_stats = []
    
    for i, feature_name in enumerate(feature_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        feature_values = features[:, i]
        
        try:
            # Create scatter plot
            ax.scatter(targets, feature_values, alpha=0.6, s=20)
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Feature Value')
            ax.set_title(f'{feature_name}', fontsize=12)
            
            # Calculate correlations
            pearson_r, pearson_p = pearsonr(targets, feature_values)
            spearman_r, spearman_p = spearmanr(targets, feature_values)
            
            # Add correlation info to plot
            corr_text = f'Pearson: {pearson_r:.3f}\nSpearman: {spearman_r:.3f}'
            ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add trend line if correlation is significant
            if abs(pearson_r) > 0.1:  # Threshold for showing trend line
                z = np.polyfit(targets, feature_values, 1)
                p = np.poly1d(z)
                ax.plot(targets, p(targets), "r--", alpha=0.8, linewidth=1)
            
            # Store statistics
            summary_stats.append({
                'Feature': feature_name,
                'Pearson_R': pearson_r,
                'Pearson_P': pearson_p,
                'Spearman_R': spearman_r,
                'Spearman_P': spearman_p,
                'Feature_Mean': np.mean(feature_values),
                'Feature_Std': np.std(feature_values),
                'Feature_Min': np.min(feature_values),
                'Feature_Max': np.max(feature_values)
            })
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{feature_name} (Error)', fontsize=12)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{task}_regression_relationships.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Pearson_R', key=abs, ascending=False)
    summary_df.to_csv(output_dir / f'{task}_regression_summary.csv', index=False)
    
    print(f"Regression analysis saved for {task}")
    print(f"Top 5 features by absolute Pearson correlation:")
    for idx, row in summary_df.head().iterrows():
        print(f"  {row['Feature']}: r={row['Pearson_R']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze feature-target relationships in CGCNN models')
    parser.add_argument('--model_dir', type=str, default=ROOT_DIR / 'results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43', 
                        help='Path to the trained model directory')
    parser.add_argument('--output_dir', type=str, default=ROOT_DIR / 'results' / 'feature_analysis', 
                       help='Output directory for plots (default: results/feature_analysis)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to analyze (default: train)')
    
    args = parser.parse_args()
    
    analyze_feature_target_relationships(args.model_dir, args.output_dir, args.split)


if __name__ == '__main__':
    main()
