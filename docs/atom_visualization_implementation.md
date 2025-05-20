# Atom Importance Visualization Methods

## Summary of Implementation

This document summarizes the implementation of different atom importance visualization methods in the MOFSNN project.

## Methods Implemented

1. **Grad-CAM (without ReLU)** - Default Method
   - **File**: `src/cgcnn/visualization/atom_visualizer.py`
   - **API Usage**: `method='grad_cam_no_relu'`
   - **Description**: Modified Grad-CAM that preserves both positive and negative contributions by omitting the final ReLU activation. This allows visualization of atoms that both promote (positive values) and inhibit (negative values) a predicted property.

2. **Grad-CAM (with ReLU)** - Standard Method
   - **File**: `src/cgcnn/visualization/atom_visualizer.py`
   - **API Usage**: `method='grad_cam'`
   - **Description**: Standard Grad-CAM implementation that applies a ReLU activation to the weighted sum of feature maps, highlighting only atoms that positively contribute to predictions.

3. **Guided Grad-CAM** - High-Resolution Method
   - **File**: `src/cgcnn/visualization/atom_visualizer.py`
   - **API Usage**: `method='guided_grad_cam'`
   - **Description**: Combines Guided Backpropagation with Grad-CAM for fine-grained visualization, showing both class-specific focus and structural details.

## Key Features

- All methods use the same API with different `method` parameter values
- Visualization of atom importance can be done using interactive 3D visualization with NGLView
- Methods can be compared side-by-side using the `compare_visualization_methods` function
- Diverging colormaps are used for methods that preserve negative and positive values
- Customizable highlighting threshold for important atoms

## Code Organization

- Core implementation: `src/cgcnn/visualization/atom_visualizer.py`
- 3D visualization utilities: `src/cgcnn/visualization/ngl_visualizer.py`
- Example usage: `examples/atom_importance_visualization.py`
- Method comparison: `examples/compare_atom_visualization_methods.py`
- Unit tests: `tests/test_atom_visualization.py`

## Documentation

- Main documentation: `docs/atom_importance_visualization.md`
- Method comparison guide: `docs/visualization_method_comparison.md`

## Usage

```python
# Basic usage
visualizer = AtomImportanceVisualizer(model)
result = visualizer.calculate_atom_importance(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    method='grad_cam_no_relu'  # or 'grad_cam' or 'guided_grad_cam'
)

# Method comparison
views = visualizer.compare_visualization_methods(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    atoms=structure
)
```

## Command Line

```bash
# Using different methods
python examples/atom_importance_visualization.py --model_path models/checkpoint.pt --cif_path structures/mof.cif --method guided_grad_cam

# Method comparison
python examples/compare_atom_visualization_methods.py --model_path models/checkpoint.pt --cif_path structures/mof.cif --save_html
```

## Future Improvements

1. Add support for more advanced visualization methods like SmoothGrad and Integrated Gradients
2. Implement custom loss functions for specific feature targeting
3. Create batch processing capability for analyzing multiple structures at once
4. Add quantitative metrics for comparing feature importance across methods
