# CGCNN Models

This document provides a simplified guide to using CGCNN models for predicting MOF stability properties.

## Overview

The `src/cgcnn` module implements graph neural network approaches that directly process MOF crystal structures from CIF files. These models convert crystal structures into graphs where atoms are nodes and bonds are edges, using this representation to predict various stability properties.

## Quick Start

### Training Models

Use the hyperopt script to train CGCNN models with hyperparameter optimization:

```bash
cd src/cgcnn

# Train attention CGCNN for single stability property
python hyperopt.py --model_cfg att_cgcnn --task_cfg tsd

# Train attention CGCNN for multiple stability properties (multi-task)
python hyperopt.py --model_cfg att_cgcnn --task_cfg tsd_ssd_ws24_water_ws24_water4_ws24_acid_ws24_base_ws24_boiling

# Adjust training parameters
python hyperopt.py --model_cfg att_cgcnn --task_cfg tsd --batch_size 16 --lr 1e-4 --max_epochs 200
```

### Direct Training Without Hyperopt

For faster training with default parameters:

```bash
cd src/cgcnn

# Basic training
python main.py --model_cfg att_cgcnn --task_cfg tsd

# Multi-task training  
python main.py --model_cfg att_cgcnn --task_cfg tsd_ssd_ws24_water_ws24_water4_ws24_acid_ws24_base_ws24_boiling
```

### Inference on New CIF Files

Use the inference script to predict stability for new MOF structures:

```bash
cd src/cgcnn

# Basic inference
python inference.py \
    --input_path data/example_cifs \
    --model_dir results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43 \
    --output_path results/inference_output.csv

# With uncertainty estimation
python inference.py \
    --input_path data/example_cifs \
    --model_dir results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43 \
    --uncertainty_trees_file results/cgcnn_models/version_43/epoch_108/uncertainty_trees.pkl \
    --output_path results/inference_with_uncertainty.csv
```

## Input/Output Formats

### Input Requirements

1. **CIF Files**: Standard crystallographic information files
   - Files should be placed in a directory (e.g., `data/example_cifs/`)
   - Supported formats: `.cif`
   - Files are automatically cleaned and validated

2. **Model Directory**: Trained model checkpoint
   - Contains model weights, configuration, and metadata
   - Example: `results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43`

### Output Format

Predictions are saved as CSV files with columns:
- `MofName`: MOF identifier (from CIF filename)
- `TSD_pred`: Thermal stability prediction (regression)
- `SSD_pred`: Solvent stability prediction (classification probability)
- `WS24_water_pred`: Water stability prediction (classification probability)
- Additional columns for other stability properties
- `*_prob`: Classification probabilities for multi-class labels
- `*_uncertainty`: Uncertainty estimates (if available)

## Model Variants

Three CGCNN variants are available:

1. **Standard CGCNN** (`cgcnn`): Original architecture
2. **Raw CGCNN** (`cgcnn_raw`): Minimal modifications for multi-task learning
3. **Attention CGCNN** (`att_cgcnn`): Enhanced with attention mechanisms (recommended)

## Configuration Options

Common parameters to adjust in training:

```bash
# Training control
--batch_size 16              # Batch size for training
--max_epochs 200            # Maximum training epochs  
--lr 1e-4                   # Learning rate

# Model selection
--model_cfg att_cgcnn       # Model architecture: cgcnn, cgcnn_raw, att_cgcnn
--task_cfg tsd              # Task configuration: tsd, ssd, tsd_ssd, etc.

# Multi-task learning
--loss_aggregation dwa      # Loss aggregation: sum, fixed_weight_sum, dwa
--lr_mult 10                # Learning rate multiplier for task heads

# Model architecture  
--n_conv 3                  # Number of convolution layers
--atom_fea_len 64          # Atom feature length
--h_fea_len 128            # Hidden feature length
--dropout_prob 0.1         # Dropout probability
```

## Evaluation on Test Sets

For models with prepared test datasets:

```python
from src.cgcnn.predict import main as predict_main

# Evaluate model performance
model_dir = "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43"
data_dir = "data/cgcnn_data/TSD"
col2task = {"Label": "TSD"}

# Get predictions and metrics
outputs, metrics = predict_main(model_dir, data_dir, col2task, split="test")
```

## Practical Tips

1. **Model Selection**: Use `att_cgcnn` for best performance
2. **Multi-task Learning**: Combine related stability properties for better generalization
3. **Hyperparameter Tuning**: Use `hyperopt.py` for automatic parameter optimization
4. **GPU Usage**: Ensure CUDA is available for faster training
5. **Batch Size**: Adjust based on GPU memory (typically 8-32)
6. **Uncertainty**: Include uncertainty estimation for more reliable predictions

## Troubleshooting

**Common Issues:**

1. **CIF Processing Errors**: Check CIF file validity and format
2. **Memory Errors**: Reduce batch size or number of workers
3. **Convergence Issues**: Adjust learning rate or use learning rate scheduling

**Performance Optimization:**

- Use attention CGCNN for better accuracy
- Enable multi-task learning when predicting multiple properties
- Utilize uncertainty estimation for screening applications
