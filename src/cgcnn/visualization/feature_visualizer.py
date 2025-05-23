'''
Author: zhangshd
Date: 2025-05-20
Description: A module for visualizing feature importance in CGCNN models, focusing on crys_fea and extra_fea inputs
'''

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from typing import Dict, List, Optional, Union, Tuple, Any

class FeatureImportanceVisualizer:
    """
    A class for visualizing feature importance in CGCNN models, specifically for
    analyzing the relative importance of crystal features (crys_fea) and extra features (extra_fea)
    at the conv_to_fc layer using Grad-CAM.
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
        self.features = None
        self.crys_fea_len = None  # Will be determined during analysis
        self.extra_fea_len = None  # Will be determined during analysis
    
    def _save_gradients(self, module, grad_input, grad_output):
        """Hook function to save gradients during backpropagation"""
        # Save gradients flowing into the conv_to_fc layer (grad_input)
        if grad_input and len(grad_input) > 0 and grad_input[0] is not None:
            self.gradients = grad_input[0].detach()
    
    def _save_features(self, module, input_fea, output):
        """Hook function to save input features during forward pass"""
        # Save the input features to the conv_to_fc layer
        if input_fea and len(input_fea) > 0:
            self.features = input_fea[0].detach()
    
    def _register_hooks(self):
        """Register hooks to capture features and gradients from conv_to_fc layer"""
        # Clear any existing hooks
        self.handles = []
        
        # Find the conv_to_fc layer to attach hooks
        if not hasattr(self.model, 'conv_to_fc'):
            raise ValueError("Model does not have a conv_to_fc layer")
        
        target_layer = self.model.conv_to_fc
        
        # Register forward hook to capture input features to conv_to_fc
        self.handles.append(target_layer.register_forward_hook(self._save_features))
        
        # Register backward hook to capture gradients
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradients))
    
    def calculate_feature_importance(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                                    extra_fea=None, task_idx=0, apply_relu=True):
        """
        Calculate feature importance scores for crys_fea and extra_fea using Grad-CAM approach
        
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
        apply_relu : bool, default=True
            Whether to apply ReLU to importance scores (focus on positive contributions only)
            
        Returns
        -------
        dict
            Contains feature importance scores for crys_fea and extra_fea
        """
        # Reset gradients and cached features/gradients
        self.model.zero_grad()
        self.gradients = None
        self.features = None
        
        # Register hooks to capture features and gradients
        self._register_hooks()
        atom_fea_len = self.model.convs[-1].atom_fea_len
        # Determine feature lengths based on model architecture
        if hasattr(self.model, 'embedding_extra'):
            # If model has extra features
            if hasattr(self.model, 'embedding_extra'):
                extra_fea_len = self.model.embedding_extra.out_features
            else:
                extra_fea_len = 0
        else:
            extra_fea_len = 0
        
        self.crys_fea_len = atom_fea_len
        self.extra_fea_len = extra_fea_len
            
        # Forward pass with gradient tracking
        with torch.set_grad_enabled(True):
            outputs, _ = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
            
            target = outputs[task_idx]
            
            # For regression, we compute gradient of the output directly
            # For classification, we compute gradient of the specific class prediction
            if hasattr(self.model, 'task_types') and self.model.task_types[task_idx] == 'classification':
                if target.shape[1] > 1:  # Multi-class
                    # target = torch.max(target, dim=1)[0]
                    ## Use the positive class
                    target = target[:, 1]  # Assuming binary classification
            elif hasattr(self.model, 'task_types') and self.model.task_types[task_idx] == 'classification_4':
                # For classification_4, we take the class with the highest score
                # target = torch.max(target, dim=1)[0]
                target = target[:, 2:]

            # Compute gradients w.r.t the conv_to_fc input features
            target.mean().backward()
        
        # Make sure we have captured the features and gradients
        if self.gradients is None or self.features is None:
            raise RuntimeError("Failed to capture features or gradients. Hooks may not be properly registered.")
        
        # Calculate importance scores (element-wise product of gradients and features)
        feature_importance = self.features * self.gradients
        batch_size = feature_importance.shape[0]
        
        # Keep per-sample importance scores (no batch averaging)
        per_sample_importance = []
        per_sample_crys_importance = []
        per_sample_extra_importance = []
        
        # Process each sample separately
        for i in range(batch_size):
            # Get importance for this sample
            sample_importance = feature_importance[i]
            
            # Apply ReLU if requested (to focus on positive contributions)
            if apply_relu:
                sample_importance = torch.relu(sample_importance)
            
            # Convert to numpy
            sample_importance_np = sample_importance.detach().cpu().numpy()
            
            # Normalize importance 
            # max_abs_value = np.max(np.abs(sample_importance_np)) + 1e-10
            # sample_importance_np = sample_importance_np / max_abs_value
            sample_importance_np = sample_importance_np / np.sum(np.abs(sample_importance_np))  # Normalize to sum to 1
            
            # Split importance scores into crys_fea and extra_fea
            sample_crys_importance = sample_importance_np[:self.crys_fea_len]
            sample_extra_importance = sample_importance_np[self.crys_fea_len:] if self.extra_fea_len > 0 else np.array([])
            
            # Store results for this sample
            per_sample_importance.append(sample_importance_np)
            per_sample_crys_importance.append(sample_crys_importance)
            per_sample_extra_importance.append(sample_extra_importance)
        
        # Also calculate the batch average for backward compatibility and comparison
        avg_importance = torch.mean(feature_importance, dim=0)
        if apply_relu:
            avg_importance = torch.relu(avg_importance)
        avg_importance_np = avg_importance.detach().cpu().numpy()
        # Normalize batch average importance
        # This is optional and can be adjusted based on your needs
        # max_abs_value = np.max(np.abs(avg_importance_np)) + 1e-10
        # avg_importance_np = avg_importance_np / max_abs_value
        avg_importance_np = avg_importance_np / np.sum(np.abs(avg_importance_np))  # Normalize to sum to 1
        
        avg_crys_importance = avg_importance_np[:self.crys_fea_len]
        avg_extra_importance = avg_importance_np[self.crys_fea_len:] if self.extra_fea_len > 0 else np.array([])
        
        # Clean up hooks
        for handle in getattr(self, 'handles', []):
            handle.remove()
        
        return {
            'per_sample_importance': per_sample_importance,  # List of importance arrays for each sample
            'per_sample_crys_importance': per_sample_crys_importance,  # List of crys_fea importance arrays
            'per_sample_extra_importance': per_sample_extra_importance,  # List of extra_fea importance arrays
            'batch_size': batch_size,
            'combined_importance': avg_importance_np,  # For backward compatibility
            'crys_fea_importance': avg_crys_importance,  # For backward compatibility
            'extra_fea_importance': avg_extra_importance,  # For backward compatibility
            'crys_fea_len': self.crys_fea_len,
            'extra_fea_len': self.extra_fea_len
        }
    
    def visualize_feature_importance(self, feature_importance, title=None, figsize=(12, 3), 
                                    colormap='coolwarm', show_labels=True, dpi=100, sample_idx=None):
        """
        Visualize feature importance as a color strip with regions for crys_fea and extra_fea
        
        Parameters
        ----------
        feature_importance : dict
            Dictionary containing feature importance data, as returned by calculate_feature_importance
        title : str, optional
            Plot title
        figsize : tuple, default=(12, 3)
            Figure size (width, height)
        colormap : str, default='coolwarm'
            Matplotlib colormap name
        show_labels : bool, default=True
            Whether to show feature labels
        dpi : int, default=100
            Resolution of the figure
        sample_idx : int, optional
            If provided, visualize a specific sample from the batch. Otherwise, use the batch average.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        # Extract data from importance dict
        crys_fea_len = feature_importance['crys_fea_len']
        extra_fea_len = feature_importance['extra_fea_len']
        
        # Determine if we're visualizing a specific sample or the batch average
        if sample_idx is not None and 'per_sample_importance' in feature_importance:
            # Check if sample_idx is valid
            if sample_idx >= len(feature_importance['per_sample_importance']):
                raise ValueError(f"Sample index {sample_idx} is out of bounds for batch size {feature_importance['batch_size']}")
            
            # Use the specified sample
            combined_importance = feature_importance['per_sample_importance'][sample_idx]
            crys_fea_importance = feature_importance['per_sample_crys_importance'][sample_idx]
            extra_fea_importance = feature_importance['per_sample_extra_importance'][sample_idx]
            
            # Update title to indicate which sample we're visualizing
            if title is None:
                title = f"Feature Importance - Sample {sample_idx}"
            else:
                title = f"{title} - Sample {sample_idx}"
        else:
            # Use batch average (backward compatibility)
            combined_importance = feature_importance['combined_importance']
            crys_fea_importance = feature_importance['crys_fea_importance']
            extra_fea_importance = feature_importance['extra_fea_importance']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Determine the vmin and vmax for colormap
        has_neg_values = np.min(combined_importance) < 0
        if has_neg_values and colormap in ['coolwarm', 'bwr', 'seismic', 'RdBu', 'RdBu_r']:
            # For diverging colormaps with positive and negative values, use symmetric range
            abs_max = max(abs(np.min(combined_importance)), abs(np.max(combined_importance)))
            vmin = -abs_max
            vmax = abs_max
        else:
            # Use actual data range
            vmin = np.min(combined_importance)
            vmax = np.max(combined_importance)
        
        # Create a normalization instance
        norm = Normalize(vmin=0, vmax=0.05)
        
        # Plot importance as a color strip
        total_features = len(combined_importance)
        
        # Create the strip visualization
        # Each feature is represented as a colored vertical line
        for i in range(total_features):
            color_val = combined_importance[i]
            ax.axvspan(i, i+1, color=plt.cm.get_cmap(colormap)(norm(color_val)))
        
        # Add vertical lines to separate crys_fea and extra_fea
        if extra_fea_len > 0:
            ax.axvline(crys_fea_len, color='black', linestyle='-', linewidth=2)
            
            # Add region annotations
            if show_labels:
                ax.text(crys_fea_len/2, 1.05, 'Atom Features', ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax.text(crys_fea_len + extra_fea_len/2, 1.05, 'Lattice Features', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Set x-axis properties
        ax.set_xlim(0, total_features)
        ax.set_xlabel('Feature Index', fontsize=10)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_ylim(0, 1)  # Set fixed height for the strip
        
        # Add colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(colormap)), 
                          ax=ax, orientation='horizontal', pad=0.2, fraction=0.1)
        cbar.set_label('Feature Importance', fontsize=10)
        
        # Mark zero point for diverging data
        if has_neg_values:
            zero_point = -vmin / (vmax - vmin)  # Normalized position of zero
            if 0 < zero_point < 1:  # Only mark if zero is within the range
                cbar.ax.axvline(zero_point, color='black', linestyle='-', linewidth=1)
        
        # Add title
        if title is not None:
            # title = "Feature Importance Visualization"
            plt.title(title, fontsize=14)
        
        # Add statistics as text
        stats_text = (
            f"Crystal Features: {crys_fea_len} features, "
            f"sum importance: {np.sum(crys_fea_importance):.3f}"
        )
        if extra_fea_len > 0:
            stats_text += (
                f"\nExtra Features: {extra_fea_len} features, "
                f"sum importance: {np.sum(extra_fea_importance):.3f}"
            )
        
        # # Add average line markers
        # if crys_fea_len > 0:
        #     crys_avg = np.mean(crys_fea_importance)
        #     ax.hlines(0.5, 0, crys_fea_len, colors='r', linestyles='--', linewidth=1)
        #     ax.text(5, 0.6, f'mean: {crys_avg:.3f}', color='r', ha='left', va='center', fontsize=8, fontweight='bold')
        
        # if extra_fea_len > 0:
        #     extra_avg = np.mean(extra_fea_importance)
        #     ax.hlines(0.5, crys_fea_len, crys_fea_len + extra_fea_len, colors='r', linestyles='--', linewidth=1)
        #     ax.text(crys_fea_len + 5, 0.6, f'mean: {extra_avg:.3f}', color='r', ha='left', va='center', fontsize=8, fontweight='bold')
        
        # Add text with statistics
        plt.figtext(0.5, -0.06, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def analyze_features(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea=None,
                        task_idx=0, apply_relu=True, figsize=(12, 3), colormap='coolwarm', sample_idx=None):
        """
        Complete workflow to analyze and visualize feature importance
        
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
        apply_relu : bool, default=True
            Whether to apply ReLU activation (focus on positive contributions only)
        figsize : tuple, default=(12, 3)
            Figure size for visualization
        colormap : str, default='coolwarm'
            Matplotlib colormap name
        sample_idx : int, optional
            If provided, analyze and visualize a specific sample instead of the batch average
            
        Returns
        -------
        tuple
            (result_dict, fig) containing the importance scores and visualization
        """
        # Calculate feature importance
        result = self.calculate_feature_importance(
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
            extra_fea=extra_fea, task_idx=task_idx, apply_relu=apply_relu
        )
        
        # Create title
        title_prefix = "Feature Importance (ReLU)" if apply_relu else "Feature Importance"
        title = f"{title_prefix} - Task {task_idx}"
        if hasattr(self.model, 'task_types') and task_idx < len(self.model.task_types):
            title = f"{title_prefix} - Task {task_idx} ({self.model.task_types[task_idx]})"
        
        # Visualize the results - either for a specific sample or batch average
        fig = self.visualize_feature_importance(
            result, title=title, figsize=figsize, colormap=colormap, sample_idx=sample_idx
        )
        
        # Determine which data to use for statistics
        if sample_idx is not None and 'per_sample_importance' in result:
            crys_importance = result['per_sample_crys_importance'][sample_idx]
            extra_importance = result['per_sample_extra_importance'][sample_idx] if result['extra_fea_len'] > 0 else np.array([])
            sample_text = f" - Sample {sample_idx}"
        else:
            crys_importance = result['crys_fea_importance']
            extra_importance = result['extra_fea_importance']
            sample_text = ""
        
        # Print some statistics
        print(f"\n{title}{sample_text} Statistics:")
        print(f"Total features: {len(crys_importance) + len(extra_importance)}")
        print(f"Crystal features: {result['crys_fea_len']}")
        if result['extra_fea_len'] > 0:
            print(f"Extra features: {result['extra_fea_len']}")
        
        print(f"\nCrystal feature importance range: [{np.min(crys_importance):.3f}, {np.max(crys_importance):.3f}]")
        print(f"Crystal feature importance mean: {np.mean(crys_importance):.3f}")
        
        if result['extra_fea_len'] > 0:
            print(f"\nExtra feature importance range: [{np.min(extra_importance):.3f}, {np.max(extra_importance):.3f}]")
            print(f"Extra feature importance mean: {np.mean(extra_importance):.3f}")
        
        return result, fig
    
    def compare_feature_importance_across_tasks(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                                             extra_fea=None, task_indices=None, apply_relu=True,
                                             figsize=(12, 10), colormap='coolwarm', sample_idx=None):
        """
        Compare feature importance across multiple tasks
        
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
        task_indices : list, optional
            List of task indices to compare. If None, compare all tasks.
        apply_relu : bool, default=True
            Whether to apply ReLU activation
        figsize : tuple, default=(12, 10)
            Figure size for visualization
        colormap : str, default='coolwarm'
            Matplotlib colormap name
        sample_idx : int, optional
            If provided, compare tasks for a specific sample instead of batch average
            
        Returns
        -------
        tuple
            (task_results, fig) containing importance scores for each task and visualization
        """
        # Determine which tasks to analyze
        if task_indices is None and hasattr(self.model, 'task_types'):
            task_indices = list(range(len(self.model.task_types)))
        elif task_indices is None:
            task_indices = [0]  # Default to task 0
        
        task_results = {}
        all_importances = []
        
        # Calculate importance for each task
        for task_idx in task_indices:
            result = self.calculate_feature_importance(
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                extra_fea=extra_fea, task_idx=task_idx, apply_relu=apply_relu
            )
            
            task_name = f"Task {task_idx}"
            if hasattr(self.model, 'task_types') and task_idx < len(self.model.task_types):
                task_name = f"Task {task_idx} ({self.model.task_types[task_idx]})"
                
            task_results[task_name] = result
            
            # Add appropriate importance data to all_importances list
            if sample_idx is not None and 'per_sample_importance' in result:
                if sample_idx < len(result['per_sample_importance']):
                    all_importances.append(result['per_sample_importance'][sample_idx])
                else:
                    # Fall back to batch average if sample_idx is out of bounds
                    print(f"Warning: Sample index {sample_idx} is out of bounds. Using batch average instead.")
                    all_importances.append(result['combined_importance'])
            else:
                all_importances.append(result['combined_importance'])
        
        # Create a figure with subplots for each task
        n_tasks = len(task_indices)
        fig, axes = plt.subplots(n_tasks, 1, figsize=figsize, sharex=True)
        if n_tasks == 1:
            axes = [axes]  # Make sure axes is a list even for a single task
        
        # Determine global color scale for consistent comparison
        all_importances_flat = np.concatenate(all_importances)
        has_neg_values = np.min(all_importances_flat) < 0
        
        if has_neg_values and colormap in ['coolwarm', 'bwr', 'seismic', 'RdBu', 'RdBu_r']:
            abs_max = max(abs(np.min(all_importances_flat)), abs(np.max(all_importances_flat)))
            vmin = -abs_max
            vmax = abs_max
        else:
            vmin = np.min(all_importances_flat)
            vmax = np.max(all_importances_flat)
            
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot each task's importance
        for i, (task_name, result) in enumerate(task_results.items()):
            ax = axes[i]
            crys_fea_len = result['crys_fea_len']
            extra_fea_len = result['extra_fea_len']
            
            # Determine if we're using per-sample importance or batch average
            if sample_idx is not None and 'per_sample_importance' in result:
                if sample_idx < len(result['per_sample_importance']):
                    combined_importance = result['per_sample_importance'][sample_idx]
                    # Add sample indicator to task name
                    task_name = f"{task_name} - Sample {sample_idx}"
                else:
                    combined_importance = result['combined_importance']
            else:
                combined_importance = result['combined_importance']
                
            total_features = len(combined_importance)
            
            # Create the strip visualization
            for j in range(total_features):
                color_val = combined_importance[j]
                ax.axvspan(j, j+1, color=plt.cm.get_cmap(colormap)(norm(color_val)))
            
            # Add vertical line to separate crys_fea and extra_fea
            if extra_fea_len > 0:
                ax.axvline(crys_fea_len, color='black', linestyle='-', linewidth=2)
            
            # Remove y-axis ticks
            ax.set_yticks([])
            ax.set_ylim(0, 1)
            
            # Add title for each subplot
            ax.set_title(task_name, fontsize=12)
            
            # Add statistics
            crys_avg = np.mean(result['crys_fea_importance'])
            ax.text(5, 0.5, f'mean: {crys_avg:.3f}', color='white', ha='left', va='center', fontsize=8, fontweight='bold')
            
            if extra_fea_len > 0:
                extra_avg = np.mean(result['extra_fea_importance'])
                ax.text(crys_fea_len + 5, 0.5, f'mean: {extra_avg:.3f}', color='white', ha='left', va='center', fontsize=8, fontweight='bold')
        
        # Set common x-axis properties
        axes[-1].set_xlabel('Feature Index', fontsize=12)
        axes[-1].set_xlim(0, total_features)
        
        # Add shared colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(colormap)), 
                          cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Feature Importance', fontsize=12)
        
        # Add overall title
        relu_text = "with ReLU" if apply_relu else "without ReLU"
        sample_text = f", Sample {sample_idx}" if sample_idx is not None else ""
        fig.suptitle(f"Feature Importance Comparison Across Tasks ({relu_text}{sample_text})", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        return task_results, fig
