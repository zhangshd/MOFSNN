'''
Author: zhangshd
Date: 2024-05-16
Description: A module for visualizing atom importance in CGCNN models using NGLView
'''

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from typing import Dict, List, Optional, Union, Tuple, Any

# Check if NGLView is available
try:
    import nglview as nv
    HAS_NGLVIEW = True
    # Only import NGLAtomVisualizer if NGLView is available
    try:
        from ..visualization.ngl_visualizer import NGLAtomVisualizer
    except ImportError:
        print("Warning: NGLAtomVisualizer not found. NGLView visualization will not be available.")
        NGLAtomVisualizer = None
except ImportError:
    HAS_NGLVIEW = False
    NGLAtomVisualizer = None
    print("Warning: NGLView not installed. Interactive 3D visualization will not be available.")
    print("Install with: pip install nglview")

class AtomImportanceVisualizer:
    """
    A class for visualizing atom importance in CGCNN models using Grad-CAM analysis.
    This implementation omits the final ReLU activation to preserve both positive and
    negative contribution scores from atoms.
    """
    def __init__(self, model):
        """
        Initialize the visualizer with a trained CGCNN model
        
        Parameters
        ----------
        model : CrystalGraphConvNet
            A trained CGCNN model
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.gradients = None
        self.atom_features = None
        
    def _save_gradients(self, module, grad_input, grad_output):
        """Hook function to save gradients during backpropagation"""
        # grad_output contains the gradients flowing backward from the next layer
        # We're interested in the gradients with respect to the output of the target layer
        self.gradients = grad_output[0].detach()
        
    def _save_features(self, module, input, output):
        """Hook function to save features during forward pass"""
        self.atom_features = output.detach()
    
    def _register_hooks(self):
        """Register hooks to capture atom features and gradients"""
        # Clear any existing hooks
        self.handles = []
        
        # Find the last convolutional layer to attach hooks
        target_layer = self.model.convs[-1]
        
        # Register forward hook to capture activations
        self.handles.append(target_layer.register_forward_hook(self._save_features))
        
        # Register backward hook to capture gradients
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradients))
        
    def calculate_atom_importance(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                                 extra_fea=None, task_idx=0, return_gradients=False):
        """
        Calculate atom importance scores using Grad-CAM
        
        This implementation follows Grad-CAM principles but omits the final ReLU activation
        to preserve both positive and negative contribution scores from atoms.
        
        Parameters
        ----------
        atom_fea : torch.Tensor
            Atom features
        nbr_fea : torch.Tensor
            Neighbor features
        nbr_fea_idx : torch.LongTensor
            Neighbor indices
        crystal_atom_idx : list of torch.LongTensor
            Mapping from crystal idx to atom idx
        extra_fea : torch.Tensor, optional
            Extra features
        task_idx : int, default=0
            Index of the task to analyze
        return_gradients : bool, default=False
            Whether to return raw gradients along with importance scores
            
        Returns
        -------
        dict
            Contains atom importance scores and optionally raw gradients
        """
        # Reset gradients and cached features/gradients
        self.model.zero_grad()
        self.gradients = None
        self.atom_features = None
        
        # Register hooks to capture feature maps and gradients
        self._register_hooks()
        
        # Forward pass with gradient tracking
        with torch.set_grad_enabled(True):
            if extra_fea is not None:
                outputs, _ = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
            else:
                outputs, _ = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, None)
            
            target = outputs[task_idx]
            print("Predicted output:", target)
            
            # For regression, we compute gradient of the output directly
            # For classification, we compute gradient of the specific class prediction
            if hasattr(self.model, 'task_types') and 'classification' in self.model.task_types[task_idx]:
                if target.shape[1] > 1:  # Multi-class
                    target = torch.max(target, dim=1)[0]
            
            # Compute gradients w.r.t the target layer's activations (not input features)
            target.mean().backward()
        
        # Make sure we have captured the features and gradients
        if self.gradients is None or self.atom_features is None:
            raise RuntimeError("Failed to capture features or gradients. Hooks may not be properly registered.")
            
        # Calculate importance scores using Grad-CAM
        importance_per_crystal = []
        gradients_per_crystal = []
        
        # Global average pooling of gradients to get weights
        # Shape: [num_features_in_conv_output]
        alpha_k_c = torch.mean(self.gradients, dim=0)
        
        # Weighted sum of feature maps
        # Shape: [num_atoms_in_batch]
        weighted_activations = self.atom_features * alpha_k_c.unsqueeze(0)
        atom_importance_scores = torch.sum(weighted_activations, dim=1)
        
        # Process importance scores for each crystal
        for idx_map in crystal_atom_idx:
            # Get importance scores for atoms in this crystal
            atom_importance = atom_importance_scores[idx_map].detach().cpu().numpy()
            
            # Store raw gradients if requested
            if return_gradients:
                crystal_gradients = self.gradients[idx_map].detach().cpu().numpy()
                gradients_per_crystal.append(crystal_gradients)
            
            # Normalize importance scores by the maximum absolute value to preserve signs
            max_abs_value = np.max(np.abs(atom_importance)) + 1e-10
            atom_importance = atom_importance / max_abs_value
            
            importance_per_crystal.append(atom_importance)
        
        # Clean up hooks
        for handle in getattr(self, 'handles', []):
            handle.remove()
            
        result = {
            'atom_importance': importance_per_crystal,
        }
        
        if return_gradients:
            result['gradients'] = gradients_per_crystal
            
        return result
    
    # Matplotlib visualization methods removed to simplify codebase
    
    def visualize_atom_importance(self, atoms, atom_importance, atom_elements=None,
                                    title=None, colormap='coolwarm',
                                    highlight_threshold=0.7, show_legend=True, 
                                    show_labels=False, show_cell=True,
                                    width="100%", height="500px", **kwargs):
        """
        Visualize atom importance scores in interactive 3D using NGLView
        
        Parameters
        ----------
        atoms : ase.Atoms 
            ASE Atoms object representing the crystal structure
        atom_importance : numpy.ndarray
            Atom importance scores, shape (n_atoms,). Can contain both positive and negative values.
        atom_elements : list, optional
            List of element names for each atom
        title : str, optional
            Plot title (defaults to "Atom Importance Visualization" if None)
        colormap : str, default='coolwarm'
            Matplotlib colormap name. 'coolwarm' is good for diverging data with positive/negative values.
        highlight_threshold : float, default=0.7
            Threshold for highlighting important atoms (based on absolute importance values)
        show_legend : bool, default=True
            Whether to show a color legend
        show_labels : bool, default=False
            Whether to show atom labels (default is False now, as labels have been simplified)
        show_cell : bool, default=True
            Whether to show the unit cell
        width : str, optional
            Width of the visualization widget
        height : str, optional
            Height of the visualization widget
        **kwargs
            Additional keyword arguments passed to the visualizer
            
        Returns
        -------
        view
            NGLView widget or None if NGLView is not available
        """
        if not HAS_NGLVIEW or NGLAtomVisualizer is None:
            print("NGLView or NGLAtomVisualizer is not available. Cannot create interactive visualization.")
            print("Install with: pip install nglview")
            return None
        
        # Set default title if None
        if title is None:
            title = "Atom Importance Visualization"
            
        # Create visualizer and generate view
        visualizer = NGLAtomVisualizer()
        view = visualizer.visualize_atom_importance(
            atoms=atoms,
            atom_importance=atom_importance,
            atom_elements=atom_elements,
            title=title,
            colormap=colormap,
            highlight_threshold=highlight_threshold,
            show_legend=show_legend,
            show_labels=show_labels,
            show_cell=show_cell,
            width=width,
            height=height,
            **kwargs
        )
        return view
    
    def compare_task_importance(self, atoms, task_importances, atom_elements=None, colormap='coolwarm', **kwargs):
        """
        Compare atom importance scores across different tasks/models with interactive visualization
        compare_task_importance
        Parameters
        ----------
        atoms : ase.Atoms
            ASE Atoms object representing the crystal structure
        task_importances : dict
            Dictionary mapping task names to atom importance arrays. 
            These arrays can contain both positive and negative values.
        atom_elements : list, optional
            List of element names for each atom
        colormap : str, default='coolwarm'
            Matplotlib colormap name. 'coolwarm' is good for diverging data with positive/negative values.
        **kwargs
            Additional keyword arguments passed to the NGLView visualizer
            
        Returns
        -------
        dict
            Dictionary mapping task names to NGLView widgets, or None if NGLView is not available
        """
        if not HAS_NGLVIEW or NGLAtomVisualizer is None:
            print("NGLView or NGLAtomVisualizer is not available. Cannot create interactive visualization.")
            print("Install with: pip install nglview")
            return None
        
        # Create visualizer and generate views
        visualizer = NGLAtomVisualizer()
        views = visualizer.visualize_comparison(
            atoms=atoms,
            importance_dict=task_importances,
            atom_elements=atom_elements,
            colormap=colormap,
            **kwargs
        )
        return views
    
    def analyze_mof(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, atoms,
                       atom_elements=None, extra_fea=None, task_idx=0, colormap='coolwarm', **kwargs):
        """
        Complete workflow to analyze and interactively visualize atom importance in a MOF
        
        Parameters
        ----------
        atom_fea : torch.Tensor
            Atom features
        nbr_fea : torch.Tensor
            Neighbor features
        nbr_fea_idx : torch.LongTensor
            Neighbor indices
        crystal_atom_idx : list of torch.LongTensor
            Mapping from crystal idx to atom idx
        atoms : ase.Atoms
            ASE Atoms object representing the crystal structure
        atom_elements : list, optional
            Element names for each atom
        extra_fea : torch.Tensor, optional
            Extra features
        task_idx : int or list, default=0
            Index of task(s) to analyze. If a list, will compare multiple tasks.
        colormap : str, default='coolwarm'
            Matplotlib colormap name. 'coolwarm' is good for diverging data with positive/negative values.
        **kwargs
            Additional visualization parameters
            
        Returns
        -------
        object or dict
            NGLView widget(s) for visualization
        """
        if not HAS_NGLVIEW or NGLAtomVisualizer is None:
            print("NGLView or NGLAtomVisualizer is not available. Cannot create interactive visualization.")
            print("Install with: pip install nglview")
            return None
            
        # If analyzing multiple tasks, generate task_importances dict
        if isinstance(task_idx, (list, tuple)):
            task_results = {}
            for idx in task_idx:
                task_name = f"Task {idx}"
                if hasattr(self.model, 'task_types') and idx < len(self.model.task_types):
                    task_name = f"Task {idx}: {self.model.task_types[idx]}"
                
                result = self.calculate_atom_importance(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                    extra_fea=extra_fea, task_idx=idx
                )
                task_results[task_name] = result['atom_importance'][0]
            
            # Return multi-task comparison
            return self.compare_task_importance(
                atoms=atoms,
                task_importances=task_results,
                atom_elements=atom_elements,
                colormap=colormap,
                **kwargs
            )
        else:
            # Calculate importance for single task
            result = self.calculate_atom_importance(
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                extra_fea=extra_fea, task_idx=task_idx
            )
            
            importance = result['atom_importance'][0]
            
            # Create title
            title = f"Task {task_idx} - Atom Importance (Grad-CAM)"
            if hasattr(self.model, 'task_types') and task_idx < len(self.model.task_types):
                title = f"Task {task_idx} ({self.model.task_types[task_idx]}) - Atom Importance (Grad-CAM)"
            
            # Return single task visualization
            return self.visualize_atom_importance(
                atoms=atoms,
                atom_importance=importance,
                atom_elements=atom_elements,
                title=title,
                colormap=colormap,
                **kwargs
            )