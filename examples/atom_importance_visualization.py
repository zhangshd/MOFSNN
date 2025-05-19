'''
Author: zhangshd
Date: 2024-05-16
Description: Example script demonstrating atom importance visualization for CrystalGraphConvNet models
'''

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from src.cgcnn.module.atom_visualizer import AtomImportanceVisualizer
from src.cgcnn.module.att_cgcnn import CrystalGraphConvNet
from src.cgcnn.inference import InferenceDataset
from src.cgcnn.datamodule.prepare_data import make_prepared_data

def load_model(model_path, model_args=None):
    """
    Load a trained CrystalGraphConvNet model from checkpoint
    
    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    model_args : dict, optional
        Model arguments to override checkpoint values
        
    Returns
    -------
    model : CrystalGraphConvNet
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get model arguments
    args = checkpoint.get('args', {})
    
    # Override with provided arguments if any
    if model_args is not None:
        for k, v in model_args.items():
            args[k] = v
    
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
    
    return model

def load_dataset(cif_path, radius=8.0, max_num_nbr=10, dmin=0, step=0.2):
    """
    Load data from a CIF file
    
    Parameters
    ----------
    cif_path : str
        Path to CIF file
    radius : float, optional
        Cutoff radius for neighbors
    max_num_nbr : int, optional
        Maximum number of neighbors
    dmin : float, optional
        Minimum interatomic distance
    step : float, optional
        Step size for discretizing distances
        
    Returns
    -------
    dataset : InferenceDataset
        Loaded dataset
    """
    # Create a temporary directory for data processing
    temp_dir = Path('./temp_inference')
    temp_dir.mkdir(exist_ok=True)
    
    # Create an inference dataset
    dataset = InferenceDataset(
        [cif_path], 
        radius=radius, 
        max_num_nbr=max_num_nbr, 
        dmin=dmin, 
        step=step,
        saved_dir=temp_dir,
        use_cell_params=False
    )
    
    # Setup the dataset to process CIF files
    dataset.setup()
    
    return dataset

def visualize_atom_importance(model, cif_path, task_idx=0, visualize=True, save_path=None,
                             highlight_threshold=0.7, colormap='plasma'):
    """
    Visualize atom importance for a MOF structure
    
    Parameters
    ----------
    model : CrystalGraphConvNet
        Trained model
    cif_path : str
        Path to CIF file
    task_idx : int, optional
        Index of the task to analyze
    visualize : bool, optional
        Whether to show plots
    save_path : str, optional
        Path to save visualizations
    highlight_threshold : float, optional
        Threshold for highlighting important atoms
    colormap : str, optional
        Matplotlib colormap name
        
    Returns
    -------
    result : dict
        Dictionary containing atom importance scores and other data
    """
    # Load dataset
    dataset = load_dataset(cif_path)
    if len(dataset) == 0:
        print(f"Error: Could not load data from {cif_path}")
        return None
    
    # Get data for the first (and only) structure
    data_dict = dataset[0]
    atom_fea = data_dict["atom_fea"] 
    nbr_fea = data_dict["nbr_fea"]
    nbr_fea_idx = data_dict["nbr_fea_idx"]
    extra_fea = data_dict["extra_fea"]
    cif_id = data_dict["cif_id"]
    
    # Get atom coordinates and elements from the CIF file
    from ase.io import read
    structure = read(cif_path)
    atom_coords = structure.get_positions()
    atom_elements = structure.get_chemical_symbols()
    
    # Process into a batch
    data_batch = dataset.collate([data_dict])
    atom_fea_batch = data_batch["atom_fea"]
    nbr_fea_batch = data_batch["nbr_fea"]
    nbr_fea_idx_batch = data_batch["nbr_fea_idx"]
    crystal_atom_idx_batch = data_batch["crystal_atom_idx"]
    
    # Create visualizer
    visualizer = AtomImportanceVisualizer(model)
    
    # Calculate atom importance
    result = visualizer.calculate_atom_importance(
        atom_fea_batch, nbr_fea_batch, nbr_fea_idx_batch, crystal_atom_idx_batch,
        extra_fea=data_batch["extra_fea"], task_idx=task_idx, return_gradients=True
    )
    
    # Store original data
    result['atom_coords'] = atom_coords
    result['atom_elements'] = atom_elements
    result['cif_id'] = cif_id
    
    if visualize:
        # Visualize atom importance
        fig, ax = visualizer.visualize_atom_importance(
            atom_coords, result['atom_importance'][0], 
            atom_elements=atom_elements,
            title=f"{cif_id} - Atom Importance (Task {task_idx})",
            highlight_threshold=highlight_threshold,
            colormap=colormap
        )
        
        # Save visualization if path is provided
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True, parents=True)
            fig_path = save_dir / f"{cif_id}_task{task_idx}_atom_importance.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {fig_path}")
            
        # Show plot
        plt.tight_layout()
        plt.show()
    
    return result

def main(model_path, cif_path, task_idx=0, save_dir=None):
    """
    Main function to visualize atom importance for a MOF structure
    
    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    cif_path : str
        Path to CIF file
    task_idx : int, optional
        Index of the task to analyze
    save_dir : str, optional
        Directory to save visualizations
    """
    # Load model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Model task types: {model.task_types}")
    
    # Visualize atom importance
    result = visualize_atom_importance(
        model, cif_path, task_idx=task_idx, visualize=True, save_path=save_dir
    )
    
    # Print summary
    importance = result['atom_importance'][0]
    top_indices = np.argsort(-importance)[:5]  # Get indices of top 5 important atoms
    
    print("\nTop 5 most important atoms:")
    for i, idx in enumerate(top_indices):
        element = result['atom_elements'][idx]
        score = importance[idx]
        print(f"  {i+1}. {element} (atom {idx}): score = {score:.4f}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize atom importance in MOF structures")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--cif_path", type=str, required=True, help="Path to CIF file")
    parser.add_argument("--task_idx", type=int, default=0, help="Task index to analyze")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    main(args.model_path, args.cif_path, args.task_idx, args.save_dir)
