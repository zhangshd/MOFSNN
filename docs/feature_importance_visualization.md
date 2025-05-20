# Feature Importance Visualization

This document describes how to use the feature importance visualization module to understand which features (crystal features and extra features) are most important for a CGCNN model's predictions.

## Introduction

Modern CGCNN models often combine two types of features:

1. **Crystal Features (`crys_fea`)** - Features derived from the crystal graph after graph convolution operations
2. **Extra Features (`extra_fea`)** - Additional features such as cell parameters or other descriptors

Understanding the relative importance of these features can provide insights into what information the model uses most when making predictions. The `FeatureImportanceVisualizer` applies a Grad-CAM approach to analyze the importance of these features at the `conv_to_fc` layer where they are combined.

## Technical Approach

The feature importance visualization is based on the Gradient-weighted Class Activation Mapping (Grad-CAM) approach, adapted for feature visualization:

1. **Target Layer**: The visualization targets the `conv_to_fc` layer in the CGCNN model, which is where crystal features and extra features are combined.

2. **Forward Pass**: During forward pass, hooks capture the input features to the `conv_to_fc` layer.

3. **Backward Pass**: During backward pass, hooks capture the gradients flowing into this layer.

4. **Importance Calculation**: Feature importance is calculated as the element-wise product of features and gradients at the `conv_to_fc` layer.

5. **Optional ReLU**: An optional ReLU activation can be applied to focus only on positive contributions.

6. **Visualization**: The importance scores are visualized as a colored strip with clear separation between crystal features and extra features.

## Usage Examples

### Basic Usage

```python
from src.cgcnn.visualization.feature_visualizer import FeatureImportanceVisualizer
from src.cgcnn.module.att_cgcnn import CrystalGraphConvNet

# Load a trained model
model = CrystalGraphConvNet(...)
model.load_state_dict(torch.load('model_checkpoint.pt')['state_dict'])
model.eval()

# Create visualizer
visualizer = FeatureImportanceVisualizer(model)

# Calculate feature importance
result, fig = visualizer.analyze_features(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=extra_fea, task_idx=0, apply_relu=True
)

# Display or save the figure
plt.show()
fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

# Access the importance scores
crys_fea_importance = result['crys_fea_importance']
extra_fea_importance = result['extra_fea_importance']
```

### Command Line Interface

The easiest way to use the visualization is through the provided example script:

```bash
# Visualize feature importance
python examples/feature_importance_visualization.py \
    --model_path /path/to/model/checkpoint.ckpt \
    --cif_path /path/to/structure.cif \
    --task_idx 0 \
    --save_dir results/feature_importance

# Show both positive and negative contributions
python examples/feature_importance_visualization.py \
    --model_path /path/to/model/checkpoint.ckpt \
    --cif_path /path/to/structure.cif \
    --no_relu

# Compare across all tasks
python examples/feature_importance_visualization.py \
    --model_path /path/to/model/checkpoint.ckpt \
    --cif_path /path/to/structure.cif \
    --compare_tasks
    
# Analyze a specific sample in the batch (instead of batch average)
python examples/feature_importance_visualization.py \
    --model_path /path/to/model/checkpoint.ckpt \
    --cif_path /path/to/structure.cif \
    --sample_idx 0
```

## Visualization Interpretation

The feature importance visualization displays:

1. A colored strip representing all features (crystal features on the left, extra features on the right)
2. Color intensity indicating importance (bright colors = high importance)
3. Red/blue colors showing positive/negative contributions (with coolwarm colormap)
4. Statistics about mean importance scores for each feature type
5. Clear separation between crystal and extra features with a vertical line

### Key Insights to Look For

When analyzing the visualization:

1. **Feature Type Dominance**: Compare the average importance of crystal vs. extra features to see which type dominates the model's decisions.

2. **Specific Important Features**: Look for bright spots indicating individual features with high importance.

3. **Task Differences**: When comparing multiple tasks, notice how feature importance patterns differ between tasks.

4. **Sign Patterns**: When using `--no_relu` (without ReLU), notice which features have negative contributions (blue) vs. positive contributions (red).

## API Reference

### FeatureImportanceVisualizer Class

The main class for feature importance visualization.

#### Constructor

```python
visualizer = FeatureImportanceVisualizer(model)
```

- `model`: A trained CrystalGraphConvNet model

#### Methods

##### calculate_feature_importance

```python
result = visualizer.calculate_feature_importance(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=None, task_idx=0, apply_relu=True
)
```

Calculates raw feature importance scores.

- `atom_fea`: Atom features tensor
- `nbr_fea`: Neighbor features tensor
- `nbr_fea_idx`: Neighbor indices tensor
- `crystal_atom_idx`: List of tensors mapping from crystal idx to atom idx
- `extra_fea`: Optional extra features tensor
- `task_idx`: Task index to analyze (default: 0)
- `apply_relu`: Whether to apply ReLU to focus on positive contributions (default: True)

##### visualize_feature_importance

```python
fig = visualizer.visualize_feature_importance(
    feature_importance, title=None, figsize=(12, 3),
    colormap='coolwarm', show_labels=True, dpi=100
)
```

Visualizes feature importance as a colored strip.

- `feature_importance`: Dictionary of importance data (from calculate_feature_importance)
- `title`: Optional plot title
- `figsize`: Figure size as (width, height) tuple
- `colormap`: Matplotlib colormap name
- `show_labels`: Whether to show feature labels
- `dpi`: Resolution of the figure

##### analyze_features

```python
result, fig = visualizer.analyze_features(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=None, task_idx=0, apply_relu=True,
    figsize=(12, 3), colormap='coolwarm'
)
```

Complete workflow to calculate and visualize feature importance.

Parameters are the same as `calculate_feature_importance` plus:
- `figsize`: Figure size as (width, height) tuple
- `colormap`: Matplotlib colormap name

##### compare_feature_importance_across_tasks

```python
results, fig = visualizer.compare_feature_importance_across_tasks(
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
    extra_fea=None, task_indices=None, apply_relu=True,
    figsize=(12, 10), colormap='coolwarm'
)
```

Compares feature importance across multiple tasks.

Parameters are similar to `analyze_features` with:
- `task_indices`: List of task indices to compare (if None, compares all tasks)

## Advanced Usage

### Customizing Visualizations

You can customize the visualizations by modifying the parameters:

```python
# Use a different colormap
result, fig = visualizer.analyze_features(
    # ...other parameters...
    colormap='viridis'
)

# Change the figure size
result, fig = visualizer.analyze_features(
    # ...other parameters...
    figsize=(16, 4)
)
```

### Accessing Raw Importance Data

You can access the raw importance scores for further analysis:

```python
result = visualizer.calculate_feature_importance(
    # ...parameters...
)

# Get importance scores for crystal features
crys_importance = result['crys_fea_importance']

# Get importance scores for extra features
extra_importance = result['extra_fea_importance']

# Find the most important crystal feature
most_important_idx = np.argmax(np.abs(crys_importance))
print(f"Most important crystal feature: {most_important_idx} with score {crys_importance[most_important_idx]}")
```

## Troubleshooting

### No Visualization Appears

If no visualization appears, check if:

1. The model has both crystal and extra features
2. The `conv_to_fc` layer exists in your model
3. The model has been loaded correctly and is in evaluation mode

### Missing Extra Features

If extra features are missing, ensure that:

1. Your model has an `embedding_extra` layer
2. You're passing `extra_fea` correctly to the visualization function
3. Your dataset was created with `use_cell_params=True`

### Poor Visualization Quality

If visualizations look poor, try:

1. Using a different colormap
2. Adjusting the figure size
3. Increasing the DPI for higher resolution
```