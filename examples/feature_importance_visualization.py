'''
Author: zhangshd
Date: 2025-05-20
Description: Example script demonstrating feature importance visualization for crystal and extra features
'''

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from src.cgcnn.visualization.feature_visualizer import FeatureImportanceVisualizer
from src.cgcnn.module.att_cgcnn import CrystalGraphConvNet
from src.cgcnn.inference import InferenceDataset

def load_model(model_path):
    """Load a trained CGCNN model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get model arguments
    args = checkpoint.get('args', {})
    
    # Initialize model
    model = CrystalGraphConvNet(
        orig_atom_fea_len=args.get('atom_fea_len', 92),
        nbr_fea_len=args.get('nbr_fea_len', 41),
        orig_extra_fea_len=args.get('extra_fea_len', 0),
        atom_fea_len=args.get('atom_fea_len', 64),
        n_conv=args.get('n_conv', 3),
        h_fea_len=args.get('h_fea_len', 128),
        n_h=args.get('n_h', 1),
        task_types=args.get('task_types', ['regression']),
        att_pooling=args.get('att_pooling', False),
        task_att_type=args.get('task_att_type', 'self')
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"Loaded model with tasks: {model.task_types}")
    print(f"Model has extra features: {hasattr(model, 'embedding_extra')}")
    
    return model

def load_dataset(cif_path, saved_dir="./temp_inference"):
    """Load a CIF file as a dataset"""
    # Create temporary directory for processing
    temp_dir = Path(saved_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # Create inference dataset
    dataset = InferenceDataset(
        [cif_path], 
        radius=8.0, 
        max_num_nbr=12, 
        dmin=0, 
        step=0.2,
        saved_dir=temp_dir,
        use_cell_params=True  # Enable extra features
    )
    
    dataset.setup()
    return dataset

def visualize_feature_importance(model_path, cif_path, task_idx=0, apply_relu=True,
                              save_dir=None, compare_tasks=False, sample_idx=None):
    """
    Visualize feature importance for crystal and extra features
    
    Parameters
    ----------
    model_path : str
        Path to the model checkpoint
    cif_path : str
        Path to the CIF file
    task_idx : int, default=0
        Task index to visualize
    apply_relu : bool, default=True
        Whether to apply ReLU to importance scores
    save_dir : str, optional
        Directory to save visualizations
    compare_tasks : bool, default=False
        Whether to compare all tasks
    sample_idx : int, optional
        If provided, analyze a specific sample instead of the batch average
    """
    # Load model
    model = load_model(model_path)
    
    # Check if model has different tasks and supports compare mode
    if compare_tasks and len(model.task_types) <= 1:
        print("Model has only one task. Disabling task comparison.")
        compare_tasks = False
    
    # Load dataset
    dataset = load_dataset(cif_path)
    if len(dataset) == 0:
        print(f"Error: Could not load data from {cif_path}")
        return
    
    # Get data for the first structure
    data_dict = dataset[0]
    cif_id = data_dict["cif_id"]
    
    # Process into batch
    data_batch = dataset.collate([data_dict])
    atom_fea_batch = data_batch["atom_fea"]
    nbr_fea_batch = data_batch["nbr_fea"]
    nbr_fea_idx_batch = data_batch["nbr_fea_idx"]
    crystal_atom_idx_batch = data_batch["crystal_atom_idx"]
    extra_fea_batch = data_batch.get("extra_fea", None)
    
    # Create visualizer
    visualizer = FeatureImportanceVisualizer(model)
    
    # Determine file name and figure title components
    relu_status = "with_relu" if apply_relu else "without_relu"
    
    print(f"Analyzing structure: {cif_id}")
    print(f"Model: {model_path}")
    
    if compare_tasks:
        # Compare feature importance across all tasks
        results, fig = visualizer.compare_feature_importance_across_tasks(
            atom_fea_batch, nbr_fea_batch, nbr_fea_idx_batch, crystal_atom_idx_batch,
            extra_fea=extra_fea_batch, apply_relu=apply_relu,
            figsize=(12, 3 * len(model.task_types)),
            sample_idx=sample_idx
        )
        
        # Save visualization if requested
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            sample_suffix = f"_sample{sample_idx}" if sample_idx is not None else ""
            fig_path = save_path / f"{cif_id}_all_tasks_{relu_status}{sample_suffix}_feature_importance.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {fig_path}")
            
    else:
        # Visualize feature importance for a single task
        result, fig = visualizer.analyze_features(
            atom_fea_batch, nbr_fea_batch, nbr_fea_idx_batch, crystal_atom_idx_batch,
            extra_fea=extra_fea_batch, task_idx=task_idx, apply_relu=apply_relu,
            sample_idx=sample_idx
        )
        
        # Determine which data to use for statistics and filenames
        sample_suffix = f"_sample{sample_idx}" if sample_idx is not None else ""
        
        if sample_idx is not None and 'per_sample_importance' in result:
            crys_importance = result['per_sample_crys_importance'][sample_idx]
            extra_importance = result['per_sample_extra_importance'][sample_idx] if result['extra_fea_len'] > 0 else np.array([])
        else:
            crys_importance = result['crys_fea_importance'] 
            extra_importance = result['extra_fea_importance']
        
        # Print detailed statistics
        print("\nDetailed Statistics:")
        if result['crys_fea_len'] > 0:
            print("Top 5 most important crystal features:")
            crys_top_indices = np.argsort(-np.abs(crys_importance))[:5]
            for i, idx in enumerate(crys_top_indices):
                print(f"  {i+1}. Feature {idx}: {crys_importance[idx]:.4f}")
        
        if len(extra_importance) > 0:
            print("\nTop 5 most important extra features:")
            extra_top_indices = np.argsort(-np.abs(extra_importance))[:5]
            for i, idx in enumerate(extra_top_indices):
                print(f"  {i+1}. Feature {idx}: {extra_importance[idx]:.4f}")
        
        # Save visualization if requested
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            fig_path = save_path / f"{cif_id}_task{task_idx}_{relu_status}{sample_suffix}_feature_importance.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {fig_path}")
    
    # Display the visualization
    plt.show()

def main():
    """Parse arguments and run visualization"""
    parser = argparse.ArgumentParser(description="Visualize feature importance in crystal vs extra features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--cif_path", type=str, required=True, help="Path to CIF file")
    parser.add_argument("--task_idx", type=int, default=0, help="Task index to visualize")
    parser.add_argument("--no_relu", action="store_true", help="Disable ReLU activation (show negative contributions)")
    parser.add_argument("--compare_tasks", action="store_true", help="Compare all tasks in the model")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--sample_idx", type=int, default=None, help="Analyze a specific sample index in the batch (default: use batch average)")
    
    args = parser.parse_args()
    
    visualize_feature_importance(
        args.model_path, args.cif_path, args.task_idx, 
        apply_relu=not args.no_relu, save_dir=args.save_dir,
        compare_tasks=args.compare_tasks, sample_idx=args.sample_idx
    )

if __name__ == "__main__":
    main()