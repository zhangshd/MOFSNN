'''
Author: zhangshd
Date: 2024-05-19
Description: Example script demonstrating comparison of different atom importance visualization methods
'''

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from src.cgcnn.visualization.atom_visualizer import AtomImportanceVisualizer
from src.cgcnn.module.att_cgcnn import CrystalGraphConvNet
from src.cgcnn.inference import InferenceDataset

def load_model(model_path):
    """Load a trained CGCNN model"""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    args = checkpoint.get('args', {})
    
    model = CrystalGraphConvNet(
        orig_atom_fea_len=args.get('atom_fea_len', 92),
        nbr_fea_len=args.get('nbr_fea_len', 41),
        orig_extra_fea_len=args.get('extra_fea_len', 0),
        atom_fea_len=args.get('atom_fea_len', 64),
        n_conv=args.get('n_conv', 3),
        h_fea_len=args.get('h_fea_len', 128),
        n_h=args.get('n_h', 1),
        task_types=args.get('task_types', ['regression']),
        att_pooling=args.get('att_pooling', False)
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def load_dataset(cif_path):
    """Load a CIF file as a dataset"""
    # Create temporary directory for processing
    temp_dir = Path('./temp_inference')
    temp_dir.mkdir(exist_ok=True)
    
    # Create inference dataset
    dataset = InferenceDataset(
        [cif_path], 
        radius=8.0, 
        max_num_nbr=12, 
        dmin=0, 
        step=0.2,
        saved_dir=temp_dir
    )
    
    dataset.setup()
    return dataset

def compare_visualization_methods(model_path, cif_path, task_idx=0, save_html=False, save_dir=None):
    """
    Compare different visualization methods for atom importance
    
    Parameters
    ----------
    model_path : str
        Path to the model checkpoint
    cif_path : str
        Path to the CIF file
    task_idx : int, default=0
        Task index to visualize
    save_html : bool, default=False
        Whether to save the visualization as HTML
    save_dir : str, optional
        Directory to save the visualization
    """
    # Load model and dataset
    model = load_model(model_path)
    dataset = load_dataset(cif_path)
    
    if len(dataset) == 0:
        print(f"Error: Could not load data from {cif_path}")
        return
    
    # Get data for the first structure
    data_dict = dataset[0]
    cif_id = data_dict["cif_id"]
    
    # Get atom structure
    from ase.io import read
    structure = read(cif_path)
    atom_elements = structure.get_chemical_symbols()
    
    # Process into batch
    data_batch = dataset.collate([data_dict])
    atom_fea_batch = data_batch["atom_fea"]
    nbr_fea_batch = data_batch["nbr_fea"]
    nbr_fea_idx_batch = data_batch["nbr_fea_idx"]
    crystal_atom_idx_batch = data_batch["crystal_atom_idx"]
    extra_fea_batch = data_batch.get("extra_fea", None)
    
    # Create visualizer
    visualizer = AtomImportanceVisualizer(model)
    
    print(f"Analyzing structure: {cif_id}")
    print(f"Model: {model_path}")
    print(f"Task index: {task_idx}")
    
    # Compare visualization methods
    views = visualizer.compare_visualization_methods(
        atom_fea_batch, nbr_fea_batch, nbr_fea_idx_batch, crystal_atom_idx_batch,
        atoms=structure,
        atom_elements=atom_elements,
        extra_fea=extra_fea_batch,
        task_idx=task_idx,
        colormap='coolwarm',
        show_legend=True,
        highlight_threshold=0.7
    )
    
    # Save visualizations if requested
    if save_html and views and save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        try:
            import nglview as nv
            for method_name, view in views.items():
                if hasattr(view, 'rendered'):
                    safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
                    html_path = save_path / f"{cif_id}_task{task_idx}_{safe_name}.html"
                    nv.write_html(str(html_path), view)
                    print(f"Saved {method_name} visualization to {html_path}")
        except Exception as e:
            print(f"Error saving HTML: {str(e)}")
    
    return views

def main():
    """Main function to parse arguments and run visualization"""
    parser = argparse.ArgumentParser(description='Compare atom importance visualization methods')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--cif_path', type=str, required=True, help='Path to CIF file')
    parser.add_argument('--task_idx', type=int, default=0, help='Task index to visualize')
    parser.add_argument('--save_html', action='store_true', help='Save visualizations as HTML')
    parser.add_argument('--save_dir', type=str, default='./results/atom_importance', 
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    compare_visualization_methods(
        args.model_path, args.cif_path, args.task_idx, args.save_html, args.save_dir
    )

if __name__ == '__main__':
    main()
