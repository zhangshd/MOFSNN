# Interactive 3D Visualization for MOF Analysis

## Overview

This guide explains how to use the interactive 3D visualization tools in the MOFSNN project for analyzing atom importance in Metal-Organic Frameworks (MOFs). These tools help researchers better understand which atoms contribute most to specific properties predicted by machine learning models.

## Visualization with NGLView

[NGLView](https://github.com/nglviewer/nglview) is a powerful interactive molecular viewer for Jupyter notebooks that provides:

- **Interactive 3D manipulation**: Rotate, zoom, and pan structures in real-time
- **Multiple representation styles**: Ball-and-stick, spacefill, cartoon, etc.
- **Selection capabilities**: Interactive highlighting of atoms and regions
- **Customizable appearance**: Change colors, styles, and visibility of elements
- **Lightweight web-based rendering**: Works well in Jupyter notebooks and can be exported to HTML

## Installation Instructions

### Installing NGLView

```bash
# Basic installation
pip install nglview

# For Jupyter Notebook
jupyter-nbextension enable nglview --py --sys-prefix

# For JupyterLab
jupyter labextension install nglview-js-widgets
```

## Using the Visualization Tools

### NGLView Visualization API

The `NGLAtomVisualizer` class provides methods for interactive visualization:

```python
# Initialize the visualizer
from src.cgcnn.visualization.ngl_visualizer import NGLAtomVisualizer
visualizer = NGLAtomVisualizer()

# Basic atom importance visualization
view = visualizer.visualize_atom_importance(
    atoms=atoms,                   # ASE Atoms object
    atom_importance=importance,    # Importance scores array
    atom_elements=elements,        # Element symbols (optional)
    title="Atom Importance",       # Plot title
    colormap="coolwarm",           # Colormap for importance
    highlight_threshold=0.7,       # Threshold for highlighting
    show_legend=True,              # Show color legend
    show_labels=True,              # Show atom labels
    show_cell=True,                # Show unit cell
    width="100%",                  # Widget width
    height="500px"                 # Widget height
)

# Multi-task comparison
views = visualizer.visualize_comparison(
    atoms=atoms,                        # ASE Atoms object
    importance_dict={                   # Dictionary of task importance scores
        "Task 1": importance_task1,
        "Task 2": importance_task2
    },
    atom_elements=elements,             # Element symbols (optional)
    show_correlation=True               # Show correlation analysis
)
```

### Understanding the Visualization

Our atom importance visualization uses a simple and intuitive approach:

1. **Color Mapping**: 
   - Atom colors represent importance values (preserving sign)
   - For diverging data, red typically indicates positive contributions, blue negative
   - The colorbar shows the actual range of importance values

2. **Size Mapping**:
   - Atom sizes represent the absolute magnitude of importance
   - Larger atoms = more important (regardless of whether positive or negative)
   - Size scaling can be adjusted with the `size_scale` parameter

3. **Special Features**:
   - Highlighting of important atoms above a threshold
   - Clear marking of zero point in diverging colorbars
   - Consistent color scaling in multi-task comparisons
   - Interactive atom selection with detailed information in console

## Integration with AtomImportanceVisualizer

The main `AtomImportanceVisualizer` class integrates these visualization options:

```python
from src.cgcnn.module.atom_visualizer import AtomImportanceVisualizer

# Create visualizer with a trained model
visualizer = AtomImportanceVisualizer(model)

# NGLView visualization
view = visualizer.visualize_atom_importance_ngl(
    atoms=atoms,
    atom_importance=importance,
    title="Atom Importance",
    colormap="coolwarm"  # Good for data with positive/negative values
)

# Complete analysis workflow with NGLView
view = visualizer.analyze_mof_with_nglview(
    atom_fea=atom_fea,
    nbr_fea=nbr_fea,
    nbr_fea_idx=nbr_fea_idx,
    crystal_atom_idx=crystal_atom_idx,
    atoms=atoms,
    task_idx=0
)
```

## Best Practices

### Interpretation Guidelines

- **Positive Values (Red)**: Atoms that positively contribute to the property
- **Negative Values (Blue)**: Atoms that negatively contribute to the property
- **Absolute Magnitude**: Larger atoms have stronger influence (either positive or negative)
- **Consider Context**: Interpret importance in the context of chemical environment

### Performance Tips

- **For Large Structures**:
  - Reduce the atom representation size
  - Disable labels for better performance
  - Consider using simplified representations (lines instead of balls)

- **Exporting Visualizations**:
  - Use `view._display_image()` and `view._repr_html_()` to capture HTML

## Interactive Features

### Atom Selection

The NGLView visualization offers a streamlined way to interact with the atom importance visualization:

1. **Clicking on Atoms**:
   - Select an atom to see detailed importance information in the console output
   - Console will show element type, atom index, and precise importance value

2. **Important Atom Highlighting**:
   - Atoms above the importance threshold are automatically highlighted
   - These atoms are also listed in the console output for reference
   - You can adjust the threshold using the `highlight_threshold` parameter

### Console Information

The visualization provides several types of information in the console:

- **Summary Statistics**: Range of importance values, threshold information
- **Important Atoms**: List of atoms above the threshold with their importance values
- **Selected Atoms**: Details about atoms you click on in the visualization
- **Error Information**: Any issues with specific visualization features

## Example Use Cases

1. **Identifying Important Functional Groups**:
   - Visualize atom importance to identify which functional groups contribute most to a property
   - Compare importance across different tasks to see how functional roles change

2. **Analyzing Metal Centers**:
   - Focus on metal centers and their coordination environments
   - Highlight atoms above a threshold to identify key interactions

3. **Structure-Property Relationships**:
   - Correlate high-importance atoms with structural features
   - Compare importance patterns across different MOF families
   - Use the interactive selection to probe specific atom contributions

## Jupyter Notebook Examples

For a complete example of interactive visualization in a Jupyter notebook, see the example notebook at `notebooks/16_atom_importance_visualization_interactive.ipynb`.
