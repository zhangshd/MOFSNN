# CGCNN Models

This document details the Crystal Graph Convolutional Neural Network (CGCNN) models implemented in the MOFSNN project for predicting MOF stability properties.

## Overview

The `src/cgcnn` module implements graph neural network approaches, specifically CGCNN and its variants, to predict MOF stability. These models directly use the 3D crystal structure of MOFs from CIF files, learning representations from atomic connectivity patterns.

## Model Architecture

The CGCNN architecture consists of several key components:

1. **Graph Construction**:
   - MOF crystal structures are converted to graphs
   - Atoms become nodes with element-specific features
   - Bonds become edges, weighted by interatomic distances

2. **Convolutional Layers**:
   - Multiple graph convolutional layers process atom and bond features
   - Message passing between atoms captures local chemical environments
   - Pooling operations aggregate information across the crystal

3. **Readout and Prediction**:
   - Global pooling converts variable-sized graphs to fixed-length vectors
   - Fully connected layers produce final predictions
   - Task-specific heads for multi-task learning

## Model Variants

Several CGCNN variants are implemented in the project:

1. **Standard CGCNN** (`cgcnn`):
   - Original CGCNN architecture with minor modifications
   - Suitable for single-task learning

2. **Raw CGCNN** (`cgcnn_raw`):
   - CGCNN with minimal preprocessing of input data
   - Uses raw atomic features without extensive feature engineering

3. **Attention CGCNN** (`att_cgcnn`):
   - CGCNN augmented with attention mechanisms
   - Improved performance for multi-task learning
   - Better captures task-specific atomic contributions

4. **Universal Atom CGCNN** (`cgcnn_uni_atom`):
   - Variant with universal atom type representation
   - Designed to handle diverse element types uniformly

5. **FCNN Models** (`fcnn` and `att_fcnn`):
   - Fully connected neural networks using extracted features
   - Used as baseline comparison for graph-based methods

## Configuration System

The CGCNN implementation uses a configuration system based on the Sacred library (`config.py`):

1. **Base Configuration**:
   - Training parameters (batch size, learning rate, etc.)
   - Optimizer settings
   - Loss function parameters
   - Evaluation metrics

2. **Model-Specific Configurations**:
   - Network architecture details
   - Graph construction parameters 
   - Feature dimensions

3. **Task-Specific Configurations**:
   - Dataset-specific settings
   - Label processing
   - Task weighting for multi-task learning

## Training Process

The training process for CGCNN models includes:

1. **Data Loading**:
   - Load processed CIF files and stability labels
   - Construct crystal graphs on-the-fly or load pre-computed graphs
   - Apply data augmentation if specified

2. **Model Training**:
   - PyTorch Lightning framework for structured training
   - Early stopping based on validation metrics
   - Learning rate scheduling
   - Gradient clipping

3. **Hyperparameter Optimization**:
   - Optuna-based hyperparameter search
   - Trial management and pruning of poor configurations
   - Efficient search space exploration

## Multi-Task Learning

A key feature of the CGCNN implementation is multi-task learning:

1. **Task Formulation**:
   - Joint training on multiple stability prediction tasks
   - Shared backbone network with task-specific heads
   - Weighted loss functions based on task relevance

2. **Loss Aggregation Strategies**:
   - `sum`: Basic summation of individual task losses
   - `fixed_weight_sum`: Weighted sum with predefined task weights
   - `trainable_weight_sum`: Weighted sum with learnable weights
   - `sample_weight_sum`: Weighting based on batch sample counts
   - `dwa`: Dynamic Weight Averaging for automatic weight adjustment

3. **Dynamic Weight Averaging (DWA)**:
   - Automatically adjusts task weights during training
   - Based on relative rate of change in task losses
   - Configurable temperature parameter (`dwa_temp`) controls weight distribution
   - Alpha parameter (`dwa_alpha`) controls adaptation rate

4. **Task Types Support**:
   - `regression`: For continuous values (e.g., thermal decomposition temperature)
   - `classification`: For binary classification tasks
   - `classification_n`: For multi-class classification with n classes (e.g., `classification_4`)

5. **Benefits**:
   - Improved data efficiency
   - Enhanced model generalization
   - Knowledge transfer between related tasks

## Model Training Script

The model training can be performed using the hyperopt.py script:

```python
python src/cgcnn/hyperopt.py with att_cgcnn TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling
```

Configuration options can be adjusted as needed:

```python
python src/cgcnn/hyperopt.py with att_cgcnn TSD batch_size=16 lr=1e-4 random_seed=42
```

## Model Evaluation and Inference

The MOFSNN project provides two distinct scripts for model evaluation and inference:

### Test Set Evaluation (`predict.py`)

The `predict.py` script is designed for evaluating models on prepared test datasets:

1. **Purpose**:
   - Evaluate model performance on test sets with known ground truth
   - Generate detailed performance metrics
   - Create visualizations of model predictions

2. **Functionality**:
   - Loads pre-processed test datasets (not raw CIF files)
   - Generates predictions for all stability tasks
   - Calculates performance metrics (AUROC, accuracy, R², MAE, etc.)
   - Exports results as CSV files and visualizations

3. **Usage**:
   ```python
   from src.cgcnn.predict import main as predict_main
   
   # Evaluate model on test set
   model_dir = "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43"
   data_dir = "data/cgcnn_data/TSD"
   col2task = {"Label": "TSD"}
   split = "test"
   
   # Get predictions and metrics
   outputs, metrics = predict_main(model_dir, data_dir, col2task, split=split)
   ```

### CIF Inference (`inference.py`)

The `inference.py` script handles direct inference on new CIF files:

1. **Purpose**:
   - Process raw CIF files for inference
   - Generate predictions for new MOF structures
   - Support high-throughput screening workflows

2. **Functionality**:
   - Processes raw CIF files into crystal graphs
   - Handles CIF cleaning and standardization
   - Provides uncertainty estimates when available
   - Supports batch processing of large structure libraries

3. **Usage**:
   ```python
   from src.cgcnn.inference import inference
   from pathlib import Path
   
   # Set up paths
   cif_dir = Path("data/raw_data/CoREMOF2019")
   model_dir = Path("results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43")
   saved_dir = Path("results/inference/CoREMOF2019")
   
   # Find CIF files
   cif_list = list(cif_dir.glob("*.cif"))
   
   # Run inference
   results = inference(cif_list, model_dir, saved_dir=saved_dir)
   
   # Access predictions
   predictions_df = pd.DataFrame(results)
   predictions_df.to_csv(saved_dir/"predictions.csv")
   ```

## Key Components

### Data Module

The `datamodule` directory handles data processing for CGCNN:

1. **Dataset Classes**:
   - `LoadGraphData`: Loads pre-processed graph data
   - `LoadGraphDataWithAtomicNumber`: Handles atomic number features
   - `LoadExtraFeatureData`: Processes additional feature types

2. **Data Preparation**:
   - `prepare_data.py`: Converts CIF files to graph data format
   - `clean_cif.py`: Pre-processes CIF files for consistency
   - `data_interface.py`: Provides unified data access interface

### Model Module

The core model implementations are in the `module` directory:

1. **Base Classes**:
   - `MInterface`: Model interface with shared training logic
   - Network layer implementations
   - Loss function definitions

2. **Model Variants**:
   - `CGCNN`: Standard implementation
   - `AttCGCNN`: Attention-enhanced version
   - Other variants with specific architectural modifications

### Utilities

The `utils.py` file provides important utility functions:

1. **Model Management**:
   - Loading and saving models
   - Checkpoint handling
   - Version tracking

2. **Training Support**:
   - Callback definitions
   - Metric calculation
   - Logging utilities

## Example Screening Workflow

The MOF screening workflow typically combines inference with filtering:

```python
from src.cgcnn.inference import inference
from pathlib import Path
import pandas as pd

# Set up paths
cif_dir = Path("data/raw_data/CoREMOF2019")
model_dir = Path("results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43")
saved_dir = Path("results/inference/CoREMOF2019")

# Run inference on all CIF files
cif_list = list(cif_dir.glob("*.cif"))
results = inference(cif_list, model_dir, saved_dir=saved_dir)

# Convert to DataFrame
df_results = pd.DataFrame(results)
df_results.index = df_results["cif_ids"]
df_results.index.name = "MofName"

# Apply stability filters
stable_mofs = df_results[
    (df_results["TSD_pred"] > 300) &  # Thermal stability > 300°C
    (df_results["SSD_pred"] == 1) &   # Solvent stable
    (df_results["WS24_water_pred"] == 1)  # Water stable
]

# Export results
stable_mofs.to_csv(saved_dir/"stable_mofs.csv")
```

## Performance Comparison

The CGCNN models, particularly the attention-enhanced variants, outperform traditional machine learning approaches:

1. **Single-Task Performance**:
   - Better accuracy for classification tasks
   - Improved R² for regression tasks
   - More robust to data variability

2. **Multi-Task Benefits**:
   - Performance improvements on tasks with limited data
   - Better generalization to unseen structures
   - More consistent predictions across related tasks

3. **Model Size and Speed**:
   - Larger model footprint than traditional ML
   - Longer training times but comparable inference speed
   - Scalable to large datasets with appropriate hardware

## Results and Evaluation

Comprehensive evaluation results can be found in the `results/evaluation/` directory:

1. **Task-Specific Performance**:
   - Detailed metrics for each stability task
   - Comparison with baseline methods
   - Ablation studies on model components

2. **External Validation**:
   - Performance on held-out test sets
   - Generalization to new MOF families