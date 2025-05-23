'''
Author: zhangshd
Date: 2024-05-16
Description: A module for visualizing atom importance in CGCNN models using NGLView
'''

import os
import numpy as np
import torch
import torch.nn as nn

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
    A class for visualizing atom importance in CGCNN models using various gradient-based
    analysis methods including Grad-CAM, Grad-CAM without ReLU, and Guided Grad-CAM.
    
    Different visualization methods:
    - Grad-CAM: Uses ReLU activation to focus on positive contributions only
    - Grad-CAM without ReLU: Preserves both positive and negative contributions
    - Guided Grad-CAM: Combines Guided Backpropagation with Grad-CAM for fine-grained visualization
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
        self.input_gradients = None  # For Guided Backpropagation
        
    def _save_gradients(self, module, grad_input, grad_output):
        """Hook function to save gradients during backpropagation"""
        # grad_output contains the gradients flowing backward from the next layer
        # We're interested in the gradients with respect to the output of the target layer
        self.gradients = grad_output[0].detach()
        
    def _save_features(self, module, input, output):
        """Hook function to save features during forward pass"""
        self.atom_features = output.detach()
    
    def _save_input_gradients(self, module, grad_input, grad_output):
        """Hook function to save input gradients for guided backpropagation"""
        if grad_input and len(grad_input) > 0 and grad_input[0] is not None:
            self.input_gradients = grad_input[0].detach()
    
    def _guided_backprop_relu_hook(self, module, grad_input, grad_output):
        """
        Hook for Guided Backpropagation - modifies gradients in ReLU during backprop
        Only positive gradients are allowed to flow back, and only through positive activations
        """
        if grad_input and len(grad_input) > 0 and grad_input[0] is not None:
            positive_grad_output = torch.clamp(grad_output[0], min=0.0)
            return (positive_grad_output,)
    
    def _register_hooks(self, method='grad_cam_no_relu'):
        """
        Register hooks to capture features, gradients, and implement guided backpropagation if needed
        
        Parameters
        ----------
        method : str
            Visualization method: 'grad_cam', 'grad_cam_no_relu', or 'guided_grad_cam'
        """
        # Clear any existing hooks
        self.handles = []
        
        # Find the last convolutional layer to attach hooks
        target_layer = self.model.convs[-1]
        
        # Register forward hook to capture activations
        self.handles.append(target_layer.register_forward_hook(self._save_features))
        
        # Register backward hook to capture gradients
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradients))
        
        # For Guided Grad-CAM, also register hooks for guided backpropagation
        if method == 'guided_grad_cam':
            # Save input gradients from the last conv layer
            self.handles.append(target_layer.register_full_backward_hook(self._save_input_gradients))
            
            # Replace ReLU backward hooks with guided backprop hooks
            for module in self.model.modules():
                if isinstance(module, nn.ReLU):
                    self.handles.append(module.register_full_backward_hook(self._guided_backprop_relu_hook))
        
    def calculate_atom_importance(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                                 extra_fea=None, task_idx=0, return_gradients=False,
                                 method='grad_cam_no_relu'):
        """
        Calculate atom importance scores using various gradient-based visualization methods
        
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
        method : str, default='grad_cam_no_relu'
            Visualization method to use. Options:
            - 'grad_cam': Standard Grad-CAM with ReLU activation (positive contributions only)
            - 'grad_cam_no_relu': Grad-CAM without ReLU (preserves positive and negative contributions)
            - 'guided_grad_cam': Guided Grad-CAM (combines Grad-CAM with Guided Backpropagation)
            - 'guided_grad_cam_no_relu': Guided Grad-CAM without ReLU (preserves both positive and negative contributions)
            
        Returns
        -------
        dict
            Contains atom importance scores and optionally raw gradients
        """
        if method not in ['grad_cam', 'grad_cam_no_relu', 'guided_grad_cam', 'guided_grad_cam_no_relu']:
            raise ValueError(f"Unsupported method: {method}. Choose from 'grad_cam', 'grad_cam_no_relu', 'guided_grad_cam', or 'guided_grad_cam_no_relu'")
        
        # Store original atom features for guided backpropagation
        original_atom_fea = atom_fea.clone().detach().requires_grad_(True) if method == 'guided_grad_cam' else None
        
        # Reset gradients and cached features/gradients
        self.model.zero_grad()
        self.gradients = None
        self.atom_features = None
        self.input_gradients = None
        
        # Register hooks according to the selected method
        self._register_hooks(method=method)
        
        # Forward pass with gradient tracking
        with torch.set_grad_enabled(True):
            if extra_fea is not None:
                outputs, _ = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
            else:
                outputs, _ = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, None)
            
            target = outputs[task_idx]
            print(f"Predicted output [{method}]:", target)
            
            # For regression, we compute gradient of the output directly
            # For classification, we compute gradient of the specific class prediction
            if hasattr(self.model, 'task_types') and 'classification' in self.model.task_types[task_idx]:
                if target.shape[1] > 1:  # Multi-class
                    # target = torch.max(target, dim=1)[0]
                    # Use the positive class index for the gradient
                    target = target[:, 1]  # Assuming binary classification
                elif hasattr(self.model, 'task_types') and self.model.task_types[task_idx] == 'classification_4':
                    target = target[:, 2:]
            
            # Compute gradients w.r.t the target layer's activations
            target.mean().backward()
        
        # Make sure we have captured the features and gradients
        if self.gradients is None or self.atom_features is None:
            raise RuntimeError("Failed to capture features or gradients. Hooks may not be properly registered.")
            
        # Calculate importance scores based on the selected method
        importance_per_crystal = []
        gradients_per_crystal = []
        guided_gradients_per_crystal = []
        
        # Global average pooling of gradients to get weights
        # Shape: [num_features_in_conv_output]
        alpha_k_c = torch.mean(self.gradients, dim=0)
        
        # Weighted sum of feature maps
        # Shape: [num_atoms_in_batch]
        weighted_activations = self.atom_features * alpha_k_c.unsqueeze(0)
        atom_importance_scores = torch.sum(weighted_activations, dim=1)
        
        # Apply ReLU for standard Grad-CAM (positive contributions only)
        if method in ['grad_cam', 'guided_grad_cam']:
            atom_importance_scores = torch.relu(atom_importance_scores)
            print(f"Atom importance scores (ReLU applied): {atom_importance_scores.min()}, {atom_importance_scores.max()}")
        
        # For Guided Grad-CAM, we need to get the guided gradients
        guided_gradients = None
        if method == 'guided_grad_cam' and self.input_gradients is not None:
            # Apply ReLU to guided gradients
            guided_gradients = torch.relu(self.input_gradients)
            print(f"Guided gradients: {guided_gradients.min()}, {guided_gradients.max()}")
        elif method == 'guided_grad_cam_no_relu' and self.input_gradients is not None:
            guided_gradients = self.input_gradients
            # No ReLU applied to guided gradients
            print(f"Guided gradients (no ReLU): {guided_gradients.min()}, {guided_gradients.max()}")
        
        # Process importance scores for each crystal
        for idx_map in crystal_atom_idx:
            # Get importance scores for atoms in this crystal
            atom_importance = atom_importance_scores[idx_map].detach().cpu().numpy()
            
            # Store raw gradients if requested
            if return_gradients:
                crystal_gradients = self.gradients[idx_map].detach().cpu().numpy()
                gradients_per_crystal.append(crystal_gradients)
            
            # For Guided Grad-CAM, we need to multiply Grad-CAM heatmap with guided gradients
            if method in ['guided_grad_cam', 'guided_grad_cam_no_relu'] and guided_gradients is not None:
                # Get guided gradients for this crystal
                crystal_guided_grads = guided_gradients[idx_map].detach().cpu().numpy()
                guided_gradients_per_crystal.append(crystal_guided_grads)
                
                # Element-wise product of Grad-CAM and guided gradients
                # This is the core of Guided Grad-CAM
                atom_importance = atom_importance.reshape(-1, 1) * crystal_guided_grads
                # Take the sum along feature dimensions
                atom_importance = np.sum(atom_importance, axis=1)
            
            # Normalize importance scores by the maximum absolute value to preserve signs
            max_abs_value = np.max(np.abs(atom_importance)) + 1e-10
            atom_importance = atom_importance / max_abs_value
            
            importance_per_crystal.append(atom_importance)
        
        # Clean up hooks
        for handle in getattr(self, 'handles', []):
            handle.remove()
            
        result = {
            'atom_importance': importance_per_crystal,
            'method': method
        }
        
        if return_gradients:
            result['gradients'] = gradients_per_crystal
            
        if method == 'guided_grad_cam' and guided_gradients is not None:
            result['guided_gradients'] = guided_gradients_per_crystal
            
        return result
    
    # Matplotlib visualization methods removed to simplify codebase
    
    def visualize_atom_importance(self, atoms, atom_importance, atom_elements=None,
                                    title=None, colormap='coolwarm',
                                    highlight_threshold=0.7, show_legend=True, 
                                    show_cell=True,
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
                       atom_elements=None, extra_fea=None, task_idx=0, colormap='coolwarm', 
                       method='grad_cam_no_relu', compare_methods=False, **kwargs):
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
        method : str, default='grad_cam_no_relu'
            Visualization method to use: 'grad_cam', 'grad_cam_no_relu', or 'guided_grad_cam'
        compare_methods : bool, default=False
            If True, generates visualizations for all methods for comparison
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

        # For method comparison, analyze with all methods
        if compare_methods:
            methods = ['grad_cam', 'grad_cam_no_relu', 'guided_grad_cam']
            method_results = {}
            
            for method_name in methods:
                try:
                    result = self.calculate_atom_importance(
                        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                        extra_fea=extra_fea, task_idx=task_idx if not isinstance(task_idx, (list, tuple)) else task_idx[0],
                        method=method_name
                    )
                    
                    method_title = {
                        'grad_cam': 'Grad-CAM (ReLU)',
                        'grad_cam_no_relu': 'Grad-CAM (w/o ReLU)',
                        'guided_grad_cam': 'Guided Grad-CAM'
                    }[method_name]
                    
                    method_results[method_title] = result['atom_importance'][0]
                except Exception as e:
                    print(f"Error calculating {method_name}: {str(e)}")
            
            # Return multi-method comparison
            return self.compare_task_importance(
                atoms=atoms,
                task_importances=method_results,
                atom_elements=atom_elements,
                colormap=colormap,
                **kwargs
            )
            
        # If analyzing multiple tasks, generate task_importances dict
        if isinstance(task_idx, (list, tuple)):
            task_results = {}
            for idx in task_idx:
                task_name = f"Task {idx}"
                if hasattr(self.model, 'task_types') and idx < len(self.model.task_types):
                    task_name = f"Task {idx}: {self.model.task_types[idx]}"
                
                result = self.calculate_atom_importance(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                    extra_fea=extra_fea, task_idx=idx, method=method
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
                extra_fea=extra_fea, task_idx=task_idx, method=method
            )
            
            importance = result['atom_importance'][0]
            
            # Create title with method information
            method_suffix = {
                'grad_cam': '(Grad-CAM with ReLU)',
                'grad_cam_no_relu': '(Grad-CAM w/o ReLU)',
                'guided_grad_cam': '(Guided Grad-CAM)'
            }.get(method, '')
            
            title = f"Task {task_idx} - Atom Importance {method_suffix}"
            if hasattr(self.model, 'task_types') and task_idx < len(self.model.task_types):
                title = f"Task {task_idx} ({self.model.task_types[task_idx]}) - Atom Importance {method_suffix}"
            
            # Return single task visualization
            return self.visualize_atom_importance(
                atoms=atoms,
                atom_importance=importance,
                atom_elements=atom_elements,
                title=title,
                colormap=colormap,
                **kwargs
            )
    
    def compare_visualization_methods(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, atoms,
                                  atom_elements=None, extra_fea=None, task_idx=0, 
                                  colormap='coolwarm', **kwargs):
        """
        Compare different atom importance visualization methods
        
        This is a specialized method that explicitly shows the differences between
        Grad-CAM, Grad-CAM without ReLU, and Guided Grad-CAM visualizations.
        
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
        task_idx : int, default=0
            Index of the task to analyze
        colormap : str, default='coolwarm'
            Matplotlib colormap name
        **kwargs
            Additional visualization parameters
            
        Returns
        -------
        dict
            Dictionary of NGLView widgets for each method
        """
        print("Comparing atom importance visualization methods...")
        print("1. Grad-CAM (with ReLU): Only positive contributions are shown")
        print("2. Grad-CAM (without ReLU): Both positive and negative contributions are preserved")
        print("3. Guided Grad-CAM: Fine-grained visualization combining Grad-CAM with guided backpropagation")
        
        methods = {
            'Grad-CAM (with ReLU)': 'grad_cam',
            'Grad-CAM (without ReLU)': 'grad_cam_no_relu',
            'Guided Grad-CAM': 'guided_grad_cam'
        }
        
        method_results = {}
        
        for title, method_name in methods.items():
            try:
                print(f"\nCalculating {title}...")
                result = self.calculate_atom_importance(
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                    extra_fea=extra_fea, task_idx=task_idx,
                    method=method_name
                )
                method_results[title] = result['atom_importance'][0]
                print(f"✓ {title} calculation complete")
                
                # Print statistics about the importance scores
                importance = result['atom_importance'][0]
                print(f"   Range: [{np.min(importance):.3f}, {np.max(importance):.3f}]")
                print(f"   Mean: {np.mean(importance):.3f}, Std: {np.std(importance):.3f}")
                print(f"   Number of atoms with positive scores: {np.sum(importance > 0)}/{len(importance)}")
                
                if method_name != 'grad_cam':  # Not applicable for standard Grad-CAM which is all positive
                    print(f"   Number of atoms with negative scores: {np.sum(importance < 0)}/{len(importance)}")
                    
            except Exception as e:
                print(f"Error calculating {title}: {str(e)}")
                print("This method will not be included in the comparison")
        
        # Print method comparison summary
        print("\nMethod Comparison Summary:")
        print("---------------------------")
        print("- Grad-CAM (with ReLU): Only positive contributions are shown. Highlights atoms that positively contribute to the prediction.")
        print("- Grad-CAM (without ReLU): Both positive and negative contributions are preserved. Red indicates positive contribution, blue indicates negative.")
        print("- Guided Grad-CAM: Fine-grained visualization that combines class-specificity with high resolution details.")
        
        if not HAS_NGLVIEW or NGLAtomVisualizer is None:
            print("\nNGLView or NGLAtomVisualizer is not available. Cannot create interactive visualization.")
            print("Install with: pip install nglview")
            return method_results
        
        # Return multi-method comparison visualization
        return self.compare_task_importance(
            atoms=atoms,
            task_importances=method_results,
            atom_elements=atom_elements,
            colormap=colormap,
            **kwargs
        )