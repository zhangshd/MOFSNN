# CGCNN Visualization Guide

This document provides a comprehensive guide to visualizing both atom importance and feature importance in CGCNN models for MOF stability prediction. All information is based on the actual implementations in `src/cgcnn/visualization/`.

## Overview

The MOFSNN project provides powerful tools for:
1. **Atom-level importance visualization** - Understanding which atoms most influence model predictions
2. **Feature importance visualization** - Analyzing which features (crystal vs extra features) drive model decisions

These visualizations are crucial for interpreting structure-property relationships in MOFs and understanding model behavior.

## Quick Start

### Installation Requirements

```bash
# Core visualization dependencies
pip install matplotlib numpy

# For interactive 3D visualization
pip install nglview

# Enable NGLView in Jupyter
jupyter-nbextension enable nglview --py --sys-prefix
# For JupyterLab
jupyter labextension install nglview-js-widgets
```

### Basic Usage

```python
from src.cgcnn.visualization.atom_visualizer import AtomImportanceVisualizer

# Initialize with trained model
visualizer = AtomImportanceVisualizer(model)

# Calculate atom importance (default: Grad-CAM without ReLU)
result = visualizer.calculate_atom_importance(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    method='grad_cam_no_relu',  # Recommended default
    task_idx=0
)

# Extract importance scores
importance_scores = result['atom_importance']
```

### Interactive 3D Visualization

```python
from src.cgcnn.visualization.ngl_visualizer import NGLAtomVisualizer

# Initialize NGLView visualizer
ngl_viz = NGLAtomVisualizer()

# Create interactive visualization
view = ngl_viz.visualize_atom_importance(
    atoms=atoms,                    # ASE Atoms object
    atom_importance=importance_scores,
    title="MOF Atom Importance",
    colormap="coolwarm",           # Good for diverging data
    highlight_threshold=0.7,       # Highlight top 30% important atoms
    show_legend=True
)
```

## Visualization Methods

The project implements three gradient-based methods for calculating atom importance:

### 1. Grad-CAM without ReLU (Default)

**API**: `method='grad_cam_no_relu'`

**What it shows**: Both positive and negative contributions of atoms to predictions

**Key features**:
- Preserves both promoting and inhibiting effects
- Uses diverging colormaps (red=positive, blue=negative)
- Most informative for understanding complete atom roles

**Best for**: Understanding complete structure-property relationships

### 2. Grad-CAM with ReLU (Standard)

**API**: `method='grad_cam'`

**What it shows**: Only positive contributions to predictions

**Key features**:
- Applies ReLU to focus on promoting atoms only
- Highlights atoms that increase property values
- Simpler interpretation but less complete information

**Best for**: Identifying atoms that promote desired properties

### 3. Guided Grad-CAM (High-Resolution)

**API**: `method='guided_grad_cam'`

**What it shows**: Fine-grained importance with structural details

**Key features**:
- Combines Grad-CAM with Guided Backpropagation
- Provides high-resolution, detailed visualization
- Shows both importance and fine structural features

**Best for**: Detailed analysis requiring fine-grained information

## Comparison of Methods

```python
# Compare all methods side-by-side
views = visualizer.compare_visualization_methods(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
    atoms, 
    task_idx=0
)

# This creates interactive visualizations for all three methods
```

## Understanding the Visualizations

### Color Interpretation

- **Grad-CAM without ReLU**: 
  - Red/warm colors: Atoms promote the property (positive importance)
  - Blue/cool colors: Atoms inhibit the property (negative importance)
  - Color intensity: Strength of effect
  - Zero point clearly marked in colorbar

- **Grad-CAM with ReLU**:
  - Only warm colors shown (positive contributions only)
  - Color intensity: Strength of positive contribution
  - Good for identifying atoms that promote desired properties

- **Guided Grad-CAM**:
  - Similar to Grad-CAM without ReLU but with finer detail
  - High-resolution importance mapping with structural context
  - Shows both class-specific focus and fine structural features

### Size Mapping

In NGLView visualizations:
- **Atom size**: Represents absolute importance magnitude
- **Larger atoms**: More important (regardless of positive/negative direction)
- **Size scaling**: Adjustable with `size_scale` parameter
- **Consistent scaling**: Both strongly positive and negative atoms appear large

### Technical Implementation Notes

The Grad-CAM implementation follows these key steps:

1. **Hook Registration**: Capture feature maps and gradients from the last convolutional layer
2. **Forward Pass**: Get model predictions for the target task
3. **Backward Pass**: Compute gradients of target output with respect to feature maps
4. **Weight Calculation**: Use global average pooling on gradients to compute importance weights
5. **Score Calculation**: Calculate importance as weighted sums of feature maps
6. **Normalization**: Preserve positive and negative values (unlike standard Grad-CAM)

**Key Difference**: Our implementation omits the final ReLU activation to preserve the direction (sign) of contributions, not just their magnitude.

## Advanced Features

### Multi-Task Visualization

```python
# Visualize importance for multiple tasks
task_importance = {}
for task_idx in range(num_tasks):
    result = visualizer.calculate_atom_importance(
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
        task_idx=task_idx
    )
    task_importance[f"Task_{task_idx}"] = result['atom_importance']

# Create comparison view
view = ngl_viz.visualize_comparison(
    atoms=atoms,
    importance_dict=task_importance,
    show_correlation=True
)
```

### Feature Importance Analysis

```python
from src.cgcnn.visualization.feature_visualizer import FeatureImportanceVisualizer

# Analyze crystal vs extra features
feature_viz = FeatureImportanceVisualizer(model)
feature_result = feature_viz.calculate_feature_importance(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea,
    task_idx=0
)

# Complete analysis with visualization
result, fig = feature_viz.analyze_features(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea,
    task_idx=0,
    apply_relu=True,  # Focus on positive contributions
    figsize=(12, 3),
    colormap='coolwarm'
)

# Access importance scores
crys_importance = result['crys_fea_importance'] 
extra_importance = result['extra_fea_importance']

# Compare across multiple tasks
results, fig = feature_viz.compare_feature_importance_across_tasks(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea,
    task_indices=[0, 1, 2],  # Compare first 3 tasks
    figsize=(12, 10)
)
```

## Feature Importance Visualization

Feature importance visualization helps understand which types of features (crystal features vs extra features) are most important for CGCNN model predictions.

### Understanding Feature Types

Modern CGCNN models combine two types of features:
1. **Crystal Features (`crys_fea`)** - Features derived from the crystal graph after graph convolution operations
2. **Extra Features (`extra_fea`)** - Additional features such as cell parameters or other descriptors

### Quick Start - Feature Importance

```python
from src.cgcnn.visualization.feature_visualizer import FeatureImportanceVisualizer

# Initialize with trained model
visualizer = FeatureImportanceVisualizer(model)

# Analyze feature importance
result, fig = visualizer.analyze_features(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea, task_idx=0, apply_relu=True
)

# Display or save
plt.show()
fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
```

### Feature Importance Methods

The visualization uses Grad-CAM at the `conv_to_fc` layer where crystal and extra features are combined:

1. **Forward Pass**: Captures input features to the `conv_to_fc` layer
2. **Backward Pass**: Captures gradients flowing into this layer
3. **Importance Calculation**: Element-wise product of features and gradients
4. **Optional ReLU**: Focus only on positive contributions

### Interpreting Feature Importance

The visualization shows:
- **Colored strip**: All features (crystal left, extra right)
- **Color intensity**: Importance magnitude (bright = high importance)
- **Red/blue colors**: Positive/negative contributions (coolwarm colormap)
- **Statistics**: Mean importance scores for each feature type
- **Clear separation**: Vertical line between crystal and extra features

### Feature Importance API

```python
from src.cgcnn.visualization.feature_visualizer import FeatureImportanceVisualizer

# Initialize
visualizer = FeatureImportanceVisualizer(model)

# Calculate importance scores only
result = visualizer.calculate_feature_importance(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea, task_idx=0, apply_relu=True
)

# Access raw importance data
crys_importance = result['crys_fea_importance']
extra_importance = result['extra_fea_importance']

# Complete analysis with visualization
result, fig = visualizer.analyze_features(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea, task_idx=0, apply_relu=True,
    figsize=(12, 3), colormap='coolwarm'
)

# Compare across multiple tasks
results, fig = visualizer.compare_feature_importance_across_tasks(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea, task_indices=[0, 1, 2], apply_relu=True
)
```

### Advanced Feature Analysis

```python
# Custom colormap and size
result, fig = visualizer.analyze_features(
    # ...parameters...
    figsize=(16, 4),
    colormap='viridis'
)

# Find most important features
crys_importance = result['crys_fea_importance']
most_important_idx = np.argmax(np.abs(crys_importance))
print(f"Most important crystal feature: {most_important_idx}")

# Compare feature type dominance
crys_mean = np.mean(np.abs(result['crys_fea_importance']))
extra_mean = np.mean(np.abs(result['extra_fea_importance']))
print(f"Crystal features mean importance: {crys_mean:.4f}")
print(f"Extra features mean importance: {extra_mean:.4f}")
```

## Troubleshooting

### Common Issues

**No visualization appears:**
- Check if model is in evaluation mode: `model.eval()`
- Verify correct layer names in your model architecture
- Ensure proper tensor dimensions and device placement

**Poor visualization quality:**
- Increase figure DPI: `fig.savefig('output.png', dpi=300)`
- Adjust colormap: try 'viridis', 'plasma', or 'coolwarm'
- Resize figures: `figsize=(16, 12)` for larger displays

**NGLView not working in Jupyter:**
```bash
# Enable extensions
jupyter-nbextension enable nglview --py --sys-prefix
jupyter labextension install nglview-js-widgets

# Restart Jupyter after installation
```

**Memory issues with large structures:**
- Use batch processing for multiple structures
- Clear GPU cache: `torch.cuda.empty_cache()`
- Process subsets of atoms for very large MOFs

**Feature importance issues:**
- Ensure model has both crystal and extra features
- Check if `conv_to_fc` layer exists in your model
- Verify `extra_fea` is passed correctly with `use_cell_params=True`

### Validation

```python
# Verify atom importance calculation
assert result['atom_importance'].shape[0] == len(atoms)
assert not np.any(np.isnan(result['atom_importance']))

# Verify feature importance calculation
if extra_fea is not None:
    expected_crystal_len = model.convs[-1].atom_fea_len
    expected_extra_len = model.embedding_extra.out_features if hasattr(model, 'embedding_extra') else 0
    
    assert len(result['crys_fea_importance']) == expected_crystal_len
    assert len(result['extra_fea_importance']) == expected_extra_len
```

## Code Structure

- **Atom importance**: `src/cgcnn/visualization/atom_visualizer.py`
- **Interactive 3D**: `src/cgcnn/visualization/ngl_visualizer.py` 
- **Feature importance**: `src/cgcnn/visualization/feature_visualizer.py`
- **Example notebook**: `notebooks/16_atom_importance_visualization_interactive.ipynb`

## Best Practices

### For Atom Importance:
- Use `grad_cam_no_relu` method for most interpretable results
- Normalize importance scores across structures for comparison
- Focus on top 10-20% most important atoms for analysis
- Use interactive 3D visualization for detailed exploration

### For Feature Importance:
- Compare crystal vs extra feature importance across different tasks
- Use ReLU for positive contributions, without ReLU for full picture
- Analyze feature patterns across multiple structures
- Consider task-specific importance variations

### Performance Tips:
- Enable GPU acceleration when available
- Use batch processing for multiple structures  
- Clear intermediate tensors to save memory
- Cache results for repeated analysis

## References

1. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.
2. Springenberg, J. T., et al. (2014). Striving for simplicity: The all convolutional net. arXiv preprint.
