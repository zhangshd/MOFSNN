#!/usr/bin/env python
'''
Author: zhangshd
Date: 2024-05-17
Description: Example script for interactive atom importance visualization

This script demonstrates how to use the interactive atom importance visualization
tools provided by the MOFSNN package. It allows for visualizing atom importance
in MOF structures using NGLView.

Usage:
    python interactive_atom_visualization.py --model_path /path/to/model/ --cif_path /path/to/structure.cif \
        --task_idx 0 --save_dir results/atom_importance
'''

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import ase.io
from tqdm import tqdm
import time

# Import MOFSNN modules
from src.cgcnn.module.atom_visualizer import AtomImportanceVisualizer
from src.cgcnn.module.att_cgcnn import CrystalGraphConvNet
from src.cgcnn.inference import InferenceDataset
from src.cgcnn.utils import load_model_from_dir

# Check if NGLView is available
try:
    from src.cgcnn.visualization import HAS_NGLVIEW
except ImportError:
    HAS_NGLVIEW = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive atom importance visualization')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                      help='Path to the trained model directory')
    parser.add_argument('--cif_path', type=str, required=True, 
                      help='Path to the CIF file or directory containing CIF files')
    
    # Optional arguments
    parser.add_argument('--task_idx', type=int, default=0,
                      help='Index of the task to analyze (default: 0)')
    parser.add_argument('--method', type=str, default='nglview', choices=['nglview', 'matplotlib'],
                      help='Visualization method (default: nglview)')
    parser.add_argument('--save_dir', type=str, default='results/atom_importance',
                      help='Directory to save visualization results (default: results/atom_importance)')
    parser.add_argument('--highlight_threshold', type=float, default=0.7,
                      help='Threshold for highlighting important atoms (default: 0.7)')
    parser.add_argument('--save_image', action='store_true',
                      help='Save visualization as PNG image')
    parser.add_argument('--save_html', action='store_true',
                      help='Save interactive visualization as HTML (only for nglview)')
    parser.add_argument('--radius_scale', type=float, default=0.5,
                      help='Scale factor for atom radius (default: 0.5)')
    parser.add_argument('--colormap', type=str, default='viridis',
                      help='Matplotlib colormap name for atom importance (default: viridis)')
    
    return parser.parse_args()

def main():
    """Main function to run atom importance visualization."""
    args = parse_arguments()
    
    # Check if NGLView is available
    if args.method == 'nglview' and not HAS_NGLVIEW:
        print("Warning: NGLView is not installed. Falling back to matplotlib.")
        print("Install NGLView with: pip install nglview")
        args.method = 'matplotlib'
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model_from_dir(args.model_path)
    model.eval()
    
    # Determine dataset paths
    if os.path.isdir(args.cif_path):
        # If cif_path is a directory, use all CIF files in it
        cif_files = [os.path.join(args.cif_path, f) for f in os.listdir(args.cif_path) 
                    if f.endswith('.cif')]
    else:
        # Otherwise, use the single CIF file
        cif_files = [args.cif_path]
    
    # Create visualizer
    visualizer = AtomImportanceVisualizer(model)
    
    # Process each CIF file
    for cif_file in tqdm(cif_files, desc="Processing structures"):
        # Extract structure name from path
        structure_name = Path(cif_file).stem
        
        # Create dataset for this structure
        try:
            dataset = InferenceDataset(cif_file)
        except Exception as e:
            print(f"Error loading {cif_file}: {e}")
            continue
        
        # Get structure data
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea = dataset[0]
        
        # Add batch dimension
        atom_fea = atom_fea.unsqueeze(0)
        nbr_fea = nbr_fea.unsqueeze(0)
        nbr_fea_idx = nbr_fea_idx.unsqueeze(0)
        crystal_atom_idx = [crystal_atom_idx]
        
        if extra_fea is not None:
            extra_fea = extra_fea.unsqueeze(0)
            
        # Load ASE atoms object for visualization
        try:
            atoms = ase.io.read(cif_file)
        except Exception as e:
            print(f"Error loading atoms from {cif_file}: {e}")
            continue
            
        # Get element symbols
        atom_elements = atoms.get_chemical_symbols()
        
        # Visualization title
        if hasattr(model, 'task_types') and args.task_idx < len(model.task_types):
            task_name = model.task_types[args.task_idx]
            title = f"Structure: {structure_name}\nTask: {args.task_idx} ({task_name})"
        else:
            title = f"Structure: {structure_name}\nTask: {args.task_idx}"
        
        # Visualize using the specified method
        print(f"\nVisualizing {structure_name} using {args.method}...")
        
        if args.method == 'nglview':
            # Use NGLView visualization
            view = visualizer.visualize_atom_importance(
                atoms=atoms,
                atom_importance=visualizer.calculate_atom_importance(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                    extra_fea=extra_fea, task_idx=args.task_idx
                )['atom_importance'][0],
                atom_elements=atom_elements,
                title=title,
                colormap=args.colormap,
                highlight_threshold=args.highlight_threshold,
                radius_scale=args.radius_scale
            )
            
            # Save HTML if requested
            if args.save_html and view is not None:
                html_path = os.path.join(args.save_dir, f"{structure_name}_task{args.task_idx}.html")
                view._display_image()
                with open(html_path, 'w') as f:
                    f.write(view._repr_html_())
                print(f"Interactive visualization saved to {html_path}")
            
        # OVITO visualization has been removed to simplify codebase
            
        else:  # matplotlib
            # Use matplotlib visualization
            fig, ax = visualizer.visualize_atom_importance(
                atom_coords=np.array([atom.position for atom in atoms]),
                atom_importance=visualizer.calculate_atom_importance(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                    extra_fea=extra_fea, task_idx=args.task_idx
                )['atom_importance'][0],
                atom_elements=atom_elements,
                title=title,
                colormap=args.colormap,
                highlight_threshold=args.highlight_threshold
            )
            
            # Save image if requested
            if args.save_image:
                img_path = os.path.join(args.save_dir, f"{structure_name}_task{args.task_idx}.png")
                fig.savefig(img_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {img_path}")
            
            plt.close(fig)
        
        # For demo purposes in non-interactive environments, add delay
        time.sleep(0.5)
    
    print(f"\nVisualization complete. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
