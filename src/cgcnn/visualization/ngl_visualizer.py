'''
Author: zhangshd
Date: 2025-05-16
Description: Module for visualizing atom importance using NGLView in Jupyter notebooks
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from typing import List, Optional, Union, Dict
import pandas as pd
from IPython.display import display
import traceback

# Initialize Jupyter notebook settings if in Jupyter environment
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')
        # Load widget extension
        try:
            get_ipython().run_line_magic('jupyter', 'nbextension enable --py widgetsnbextension')
            get_ipython().run_line_magic('jupyter', 'nbextension enable --py nglview')
        except:
            pass
except:
    pass

# Check if nglview is available
try:
    import nglview as nv
    HAS_NGLVIEW = True
except ImportError:
    HAS_NGLVIEW = False
    print("NGLView is not installed. To enable interactive 3D visualization, install it with:")
    print("pip install nglview")

class NGLAtomVisualizer:
    """
    A class for visualizing atom importance in MOF structures using NGLView.
    
    This class provides interactive 3D visualization of atom importance scores
    within MOF crystal structures for Jupyter notebooks.
    """
    
    def __init__(self):
        """Initialize the NGLAtomVisualizer."""
        if not HAS_NGLVIEW:
            print("WARNING: NGLView is not available. Install with: pip install nglview")
        self.view = None
        
    def visualize_atom_importance(
        self, 
        atoms, 
        atom_importance: np.ndarray, 
        atom_elements: Optional[List[str]] = None,
        title: str = "Atom Importance Visualization",
        colormap: str = "coolwarm",
        size_scale: float = 0.5,
        min_size: float = 1.0,
        max_size: float = 2.0,
        highlight_threshold: float = 0.5,
        show_legend: bool = True,
        show_labels: bool = True,
        show_cell: bool = True,
        width: str = "100%",
        height: str = "500px",
        min_importance: Optional[float] = None,
        max_importance: Optional[float] = None
    ):
        """
        Visualize atom importance on a crystal structure using NGLView.
        
        This implementation uses a simple and consistent approach:
        1. Atom colors are based on importance values (preserving sign for positive/negative contributions)
        2. Atom sizes are based on absolute importance values (larger atoms = more important)
        3. Colorbar shows the actual range of importance values
        
        Parameters
        ----------
        atoms : ase.Atoms or Structure
            ASE Atoms object or pymatgen Structure representing the crystal structure
        atom_importance : np.ndarray
            Array of importance scores for each atom. Can contain both positive and negative values.
        atom_elements : list, optional
            List of element names for each atom (if not provided, will use atoms.get_chemical_symbols())
        title : str, optional
            Title of the visualization
        colormap : str, optional
            Matplotlib colormap name to use for coloring atoms. Default is 'coolwarm', 
            which is good for diverging data with positive/negative values.
        size_scale : float, optional
            Scaling factor for atom sizes
        min_size : float, optional
            Minimum atom size
        max_size : float, optional
            Maximum atom size
        highlight_threshold : float, optional
            Threshold for highlighting important atoms (based on absolute value, normalized to [0,1])
        show_legend : bool, optional
            Whether to show a color legend
        show_labels : bool, optional
            Whether to show atom labels
        show_cell : bool, optional
            Whether to show the unit cell
        width : str, optional
            Width of the visualization widget
        height : str, optional
            Height of the visualization widget
        min_importance : float, optional
            Minimum value for colormap normalization. If None, computed from data.
        max_importance : float, optional
            Maximum value for colormap normalization. If None, computed from data.
            
        Returns
        -------
        nv.NGLWidget or None
            NGLView widget for interactive visualization or None if NGLView is not available
        """
        if not HAS_NGLVIEW:
            print("NGLView is not installed. Cannot create interactive visualization.")
            return None
        
        # Get atom elements if not provided
        if atom_elements is None:
            try:
                atom_elements = atoms.get_chemical_symbols()
            except AttributeError:
                try:
                    atom_elements = [site.species_string for site in atoms]
                except AttributeError:
                    print("WARNING: Could not determine atom elements. Using default labels.")
                    atom_elements = [f"Atom{i}" for i in range(len(atom_importance))]
        
        # Step 1: Determine color range (preserve positive/negative distinction)
        imp_min = np.min(atom_importance)
        imp_max = np.max(atom_importance)
        has_neg_values = np.any(atom_importance < 0)
        
        # Determine min and max values for color mapping
        if min_importance is not None and max_importance is not None:
            # Use user-provided range
            vmin = min_importance
            vmax = max_importance
        elif has_neg_values and colormap in ['coolwarm', 'bwr', 'seismic', 'RdBu', 'RdBu_r']:
            # For diverging colormaps with positive and negative values, use symmetric range
            abs_max = max(abs(imp_min), abs(imp_max))
            vmin = -abs_max
            vmax = abs_max
        else:
            # Use actual data range
            vmin = imp_min
            vmax = imp_max
        
        # Step 2: Normalize importance values for coloring (preserve sign)
        if vmax != vmin:
            norm_importance = (atom_importance - vmin) / (vmax - vmin)
        else:
            norm_importance = np.ones_like(atom_importance) * 0.5
        
        # Step 3: Calculate atom sizes based on absolute importance
        abs_importance = np.abs(atom_importance)
        max_abs = np.max(abs_importance)
        if max_abs > 0:
            # Normalize absolute values for sizing
            size_importance = abs_importance / max_abs
        else:
            size_importance = np.ones_like(abs_importance) * 0.5
        
        # Step 4: Convert normalized values to colors
        cmap = plt.get_cmap(colormap)
        colors = [to_hex(cmap(val)) for val in norm_importance]
        
        # Create NGLView widget
        try:
            import nglview as nv
            self.view = nv.show_ase(atoms)
            
            # Clear default representation and add base representation
            self.view.clear()
            self.view.add_ball_and_stick(aspectRatio=3.0)
            
            # Apply custom atom colors and sizes
            for i, (color, size_factor) in enumerate(zip(colors, size_importance)):
                atom_size = min_size + (max_size - min_size) * size_factor
                self.view.add_ball_and_stick(selection=f"@{i}", color=color, radius=atom_size*size_scale, aspectRatio=1)
            
            # Initialize GUI components if available
            if hasattr(self.view, '_init_gui') and callable(self.view._init_gui):
                self.view._init_gui()        
            # Highlight important atoms            
            if highlight_threshold is not None:
                # Identify atoms above the importance threshold (based on absolute values)
                important_indices = np.where(size_importance >= highlight_threshold)[0]
                if len(important_indices) > 0:
                    # Create selection of important atoms
                    selection_string = " or ".join([f"@{idx}" for idx in important_indices])
                    
                    # Add special representation for important atoms
                    self.view.add_spacefill(selection=selection_string, opacity=0.4)
                    
                    # Print important atoms to console for reference
                    if atom_elements is not None:
                        print("Important atoms with high contribution values:")
                        for idx in important_indices:
                            if idx < len(atom_elements) and idx < len(atom_importance):
                                element = atom_elements[idx]
                                imp_value = atom_importance[idx]
                                print(f"  {element} (atom {idx}): {imp_value:.3f}")
                        
                        # Add simple atom selection handler for clicked atoms 
                        try:
                            # Create the picked property if it doesn't exist
                            if not hasattr(self.view, 'picked'):
                                self.view.picked = {"atom1": {"serial": -1}}
                            
                            # Define a reliable callback function
                            def on_picked(change):
                                try:
                                    if change and 'new' in change and change['new']:
                                        if 'atom1' in change['new'] and 'index' in change['new']['atom1']:
                                            idx = change['new']['atom1']['index']
                                            if idx < len(atom_elements) and idx < len(atom_importance):
                                                element = atom_elements[idx]
                                                imp_value = atom_importance[idx]
                                                print(f"Selected atom: {element} (atom {idx}), Importance: {imp_value:.3f}")
                                except Exception as e:
                                    # Ignore errors in callback
                                    pass
                            
                            # Register the callback safely
                            if hasattr(self.view, 'observe'):
                                self.view.observe(on_picked, names=["picked"])
                                print("Click on atoms to see detailed importance values in the console")
                        except Exception as e:
                            # If observe fails, don't interrupt the visualization
                            pass
        except Exception as e:
            print(f"Error creating NGLView visualization: {traceback.format_exc()}")
            return None
        
        # Set view properties
        if show_cell:
            self.view.add_unitcell()
        
        self.view.center()
        self.view._remote_call('setSize', target='Widget', args=[width, height])
        
        # Create colorbar legend
        if show_legend:
            self._create_colorbar(
                colormap=colormap,
                title=title,
                importance_values=norm_importance,
                min_val=vmin,
                max_val=vmax,
                has_negative=bool(has_neg_values)
            )
        
        # Print visualization summary
        important_count = np.sum(size_importance >= highlight_threshold)
        print(f"Visualization: {title}")
        print(f"Total atoms: {len(atom_importance)}")
        print(f"Importance range: [{imp_min:.3f}, {imp_max:.3f}]")
        print(f"Color mapping range: [{vmin:.3f}, {vmax:.3f}]")
        print(f"Atoms above importance threshold ({highlight_threshold:.2f}): {important_count}")
        
        return self.view
    
    def _create_colorbar(self, colormap: str, title: str, importance_values: np.ndarray, 
                      min_val: float = 0, max_val: float = 1, has_negative: bool = False):
        """
        Create a colorbar legend for the atom importance visualization.
        
        Parameters
        ----------
        colormap : str
            Matplotlib colormap name
        title : str
            Title for the colorbar
        importance_values : np.ndarray
            Array of normalized importance values
        min_val : float, default=0
            Minimum value represented in the colormap
        max_val : float, default=1
            Maximum value represented in the colormap
        has_negative : bool, default=False
            Whether the data includes negative values
        """
        fig, ax = plt.subplots(figsize=(6, 0.8))
        cmap = plt.get_cmap(colormap)
        
        # Set colorbar label with actual data range and interpretation hint
        if has_negative:
            label = f"Atom Importance [Range: {min_val:.3f} to {max_val:.3f}]"
            subtitle = "Negative (blue) ← Zero → Positive (red)"
        else:
            label = f"Atom Importance [Range: {min_val:.3f} to {max_val:.3f}]"
            subtitle = "Lower ← Importance → Higher"
        
        # Create a colorbar with normalized range (0-1) to match NGLView's coloring
        norm = Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation='horizontal',
            label=label
        )
        
        # Add tick values mapped to original data range
        tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
        cb.set_ticks(tick_positions)
        tick_labels = [f"{min_val + t * (max_val - min_val):.3f}" for t in tick_positions]
        cb.set_ticklabels(tick_labels)
        
        # Mark zero point for diverging data (with both positive and negative values)
        if has_negative:
            zero_point = -min_val / (max_val - min_val)  # Normalized position of zero
            if 0 < zero_point < 1:  # Only mark if zero is within the range
                cb.ax.axvline(zero_point, color='black', linestyle='-', linewidth=1.5)
                # cb.ax.text(zero_point, -0.5, "0", ha='center', va='top', 
                #           color='black', fontsize=9, fontweight='bold')
        
        # Set title for the colorbar
        plt.title(f"{title}\n{subtitle}", fontsize=10)
        
        # Add note about visualization
        # plt.figtext(0.5, -0.3, "Note: Atom colors represent importance values. Atom sizes represent absolute importance.", 
        #            ha='center', fontsize=8, style='italic')
        
        display(fig)
        plt.close(fig)
    
    def visualize_comparison(
        self,
        atoms,
        importance_dict: Dict[str, np.ndarray],
        atom_elements: Optional[List[str]] = None,
        colormap: str = 'coolwarm',
        **kwargs
    ):
        """
        Compare atom importance scores from multiple tasks or models.
        
        Parameters
        ----------
        atoms : ase.Atoms
            ASE Atoms object representing the crystal structure
        importance_dict : Dict[str, np.ndarray]
            Dictionary mapping task/model names to their atom importance arrays.
            These arrays can contain both positive and negative values.
        atom_elements : list, optional
            List of element names for each atom
        colormap : str, default='coolwarm'
            Matplotlib colormap name. 'coolwarm' is good for diverging data with positive/negative values.
        **kwargs : dict
            Additional visualization parameters passed to visualize_atom_importance
            
        Returns
        -------
        Dict[str, nv.NGLWidget]
            Dictionary mapping task/model names to their NGLView widgets
        """
        if not HAS_NGLVIEW:
            print("NGLView is not installed. Cannot create interactive visualization.")
            return None
            
        views = {}
        
        # Find global min and max for consistent color scaling across all tasks
        all_values = np.concatenate(list(importance_dict.values()))
        has_neg = np.any(all_values < 0)
        
        if has_neg and colormap in ['coolwarm', 'bwr', 'seismic', 'RdBu', 'RdBu_r']:
            # For diverging colormaps with positive and negative values, use symmetric range
            abs_max = max(abs(np.min(all_values)), abs(np.max(all_values)))
            global_min = -abs_max
            global_max = abs_max
        else:
            # Use actual data range
            global_min = np.min(all_values)
            global_max = np.max(all_values)
        
        print(f"Comparison visualization using consistent color scale: [{global_min:.3f}, {global_max:.3f}]")
        
        # Create separate visualizations for each task/model with consistent color scaling
        for name, importance in importance_dict.items():
            title = f"{name} Importance (Grad-CAM)"
            view = self.visualize_atom_importance(
                atoms, 
                importance, 
                atom_elements=atom_elements, 
                title=title, 
                colormap=colormap, 
                min_importance=global_min,
                max_importance=global_max,
                **kwargs
            )
            views[name] = view
            
        # Create correlation heatmap for importance scores when comparing multiple tasks
        if len(importance_dict) > 1:
            self._create_correlation_heatmap(importance_dict)
            
        return views
    
    def _create_correlation_heatmap(self, importance_dict: Dict[str, np.ndarray]):
        """
        Create a correlation heatmap for importance scores from multiple tasks/models.
        
        Parameters
        ----------
        importance_dict : Dict[str, np.ndarray]
            Dictionary mapping task/model names to their atom importance arrays
        """
        # Create DataFrame for correlation analysis
        df = pd.DataFrame({name: imp for name, imp in importance_dict.items()})
        
        # Compute correlation matrix
        corr = df.corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black")
        
        # Set labels and title
        task_names = list(importance_dict.keys())
        ax.set_xticks(np.arange(len(task_names)))
        ax.set_yticks(np.arange(len(task_names)))
        ax.set_xticklabels(task_names)
        ax.set_yticklabels(task_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation')
        
        ax.set_title("Correlation of Atom Importance Scores Between Tasks")
        fig.tight_layout()
        
        display(fig)
        plt.close(fig)
    

    

