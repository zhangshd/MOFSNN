# Atom Importance Visualization in MOFSNN

## Overview

The MOFSNN project now includes advanced tools for visualizing atom-level importance in crystal structures, particularly useful for interpreting CrystalGraphConvNet (CGCNN) models. These visualization tools help identify which atoms most influence the model's predictions, providing insights into structure-property relationships in MOFs.

## Atom Importance Methods

The project implements two approaches to calculate atom importance:

1. **Grad-CAM Implementation** (Current Default)
   - Based on the Gradient-weighted Class Activation Mapping technique
   - Captures both positive and negative contributions of atoms to model predictions
   - Shows how atoms promote or inhibit specific predictions
   - Uses weights derived from gradients flowing into the final convolutional layer
   - Omits the final ReLU activation to preserve positive and negative contributions

2. **Gradient Magnitude** (Legacy Implementation)
   - Based on the magnitude of gradients of the input features
   - Only shows the strength of influence, not the direction (positive/negative)

## Visualization Types

The project supports multiple visualization approaches:

1. **Static 3D Visualization** (Matplotlib-based)
   - Traditional 3D plots with customizable appearance
   - Support for highlighting important atoms
   - Uses diverging colormaps for Grad-CAM (red for positive, blue for negative contributions)

2. **Interactive 3D Visualization** (NGLView-based)
   - Real-time rotation, zooming, and panning
   - Interactive atom selection and inspection
   - Color-coded importance visualization with intuitive colorbar displays
   - Multi-task comparison views with consistent color scaling

## Using the Visualization Tools

### Command Line Usage

You can use the visualization tools from the command line:

```bash
# Basic static visualization
python examples/atom_importance_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --task_idx 0 --save_dir results/atom_importance

# Interactive NGLView visualization
python examples/interactive_atom_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --task_idx 0 --save_html --save_dir results/atom_importance
```

### Jupyter Notebook Usage

For interactive exploration, use the provided Jupyter notebooks:

```bash
# Static visualization notebook
jupyter notebook notebooks/16_atom_importance_visualization.ipynb

# Interactive 3D visualization notebook
jupyter notebook notebooks/16_atom_importance_visualization_interactive.ipynb
```

## Installation Requirements

For interactive visualization, you'll need to install:

```bash
# For NGLView-based visualization
pip install nglview

# For Jupyter Notebook integration
jupyter-nbextension enable nglview --py --sys-prefix
```

## API Usage Example

Here's a simple example of using the visualization API in your code:

```python
from src.cgcnn.module.atom_visualizer import AtomImportanceVisualizer
import ase.io

# Load model and structure
model = load_model_from_dir('/path/to/model')
atoms = ase.io.read('/path/to/structure.cif')

# Create visualizer
visualizer = AtomImportanceVisualizer(model)

# Calculate atom importance
results = visualizer.calculate_atom_importance(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
importance = results['atom_importance'][0]

# Interactive visualization with NGLView
view = visualizer.visualize_atom_importance_ngl(
    atoms=atoms,
    atom_importance=importance,
    title="Atom Importance in MOF Structure",
    highlight_threshold=0.7
)

# Display the view in a Jupyter notebook
display(view)

# Compare importance across multiple tasks
task_views = visualizer.compare_task_importance_ngl(
    atoms=atoms,
    task_importances={
        "Task 0": importance_task0,
        "Task 1": importance_task1
    }
)
```

## Visualization Features

- Grad-CAM based attribution of atom importance
- Both positive (promoting) and negative (inhibiting) contributions are visualized
- Intuitive visualization approach:
  - Colors represent importance values (preserving sign)
  - Atom sizes represent absolute importance (larger = more influential)
  - Colorbar shows actual range of importance values with zero point marked
- Diverging color maps (like 'coolwarm') for easy interpretation of contribution directions
- Customizable highlighting of significant atoms

## Understanding Grad-CAM Visualization

In the atom importance visualization:

- **Red (Positive Values)**: Atoms with positive importance scores contribute positively to the prediction. These atoms help increase the predicted value (for regression) or the predicted class score (for classification).

- **Blue (Negative Values)**: Atoms with negative importance scores contribute negatively to the prediction. These atoms tend to decrease the predicted value or class score.

- **Color Intensity**: The intensity of the color represents the magnitude of importance (brighter colors indicate stronger influence).

- **Atom Size**: Atom size is scaled based on the absolute magnitude of importance, so both strongly positive and strongly negative atoms appear larger.

## Technical Implementation

The Grad-CAM implementation follows these steps:

1. Register hooks to capture feature maps and gradients from the last convolutional layer
2. Perform a forward pass to get model predictions
3. Perform a backward pass to compute gradients of the target output with respect to feature maps
4. Use global average pooling on gradients to compute weights for each feature dimension
5. Calculate importance scores as weighted sums of feature maps
6. Normalize scores while preserving positive and negative values

Unlike standard Grad-CAM, our implementation omits the final ReLU activation to preserve the direction (sign) of contributions, not just their magnitude.

The NGLView visualization implementation uses a simplified, consistent approach:
1. Color values are normalized to [0,1] range while preserving sign information
2. Size values are based on absolute importance (larger = more important)
3. Colorbar displays actual data ranges with the zero point clearly marked
4. Multi-task comparisons use consistent color scaling for easy comparison

## References

1. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618-626).

## Additional Visualization Features

- Support for highlighting atoms above a threshold
- Interactive colorbars and legends
- Customizable atom representation (ball-and-stick, spacefill, etc.)
- Unit cell and crystal structure visualization
- Multi-task comparison for analyzing different property predictions
- Support for different color schemes and scaling options
- Compatible with models using average pooling or attention mechanisms
- Works for both classification and regression tasks
