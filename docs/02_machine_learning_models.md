# Machine Learning Models

This document provides a simplified guide to using traditional machine learning models in the MOFSNN project for predicting MOF stability properties.

## Overview

The `src/ml` module implements traditional machine learning approaches for predicting MOF stability. The system supports both regression and classification tasks for various stability properties including thermal stability (TSD), solvent stability (SSD), and water stability properties (WS24).

## Quick Start

### Training Models

Use the main training script with basic parameters:

```bash
cd src/ml

# For classification tasks (default)
python main.py \
    --model_type classification \
    --data_dir data/ml_data/WS24 \
    --in_file_name RAC_and_zeo_features_with_id_prop.csv \
    --label_column water_label \
    --name_column MofName \
    --group_column boiling_label \
    --model_list RF SVM \
    --feature_selector_list RFE f_classif \
    --select_des_num_list 50 100 \
    --search_max_evals 50 \
    --search_metric val_AUC

# For regression tasks  
python main.py \
    --model_type regression \
    --data_dir data/ml_data/TSD \
    --in_file_name RAC_and_zeo_features_with_id_prop.csv \
    --label_column Label \
    --name_column MofName \
    --model_list RF SVM \
    --feature_selector_list RFE f_regression \
    --select_des_num_list 50 100 \
    --search_max_evals 50 \
    --search_metric val_R2
```

### Inference on New MOFs

Use the inference script to predict stability for new CIF files:

```bash
cd src/ml

# Single CIF file
python inference.py \
    --input_path /path/to/your/mof.cif \
    --output_path /path/to/predictions.csv

# Directory containing multiple CIF files
python inference.py \
    --input_path /path/to/cif/directory \
    --output_path /path/to/predictions.csv
```

## Input Data Format

### Training Data
The training data should be a CSV file with the following structure:
- `MofName`: MOF identifier
- `Partition`: "train", "test", or "val" 
- `Label`: Target variable (for regression) or class labels (for classification)
- Feature columns: Various molecular descriptors and properties

### CIF Files
For inference, provide standard CIF files containing MOF crystal structures.

## Output

### Training Results
Training produces the following outputs in the results directory:
- Trained model files (`.pkl`)
- Cross-validation metrics
- Feature importance plots
- Chemical space visualizations
- Test set predictions

### Inference Results
Inference produces a CSV file with predictions for all 7 stability properties:
- `TSD`: Thermal stability (regression)
- `SSD`: Solvent stability (regression) 
- `WS24_water`: Water stability (classification)
- `WS24_water4`: Water stability at pH 4 (classification)
- `WS24_acid`: Acid stability (classification)
- `WS24_base`: Base stability (classification)
- `WS24_boiling`: Boiling water stability (classification)

## Model Types

### Available Algorithms
- **RF**: Random Forest
- **SVC/SVR**: Support Vector Machine (Classification/Regression)
- **GP**: Gaussian Process
- **LR**: Linear/Logistic Regression

### Feature Selection Methods
- **RFE**: Recursive Feature Elimination
- **f_classif/f_regression**: F-score based selection
- **mutual_info**: Mutual information based selection

## Configuration

### Key Parameters
- `model_type`: "classification" or "regression"
- `model_list`: List of algorithms to train
- `feature_selector_list`: Feature selection methods
- `select_des_num_list`: Number of features to select
- `search_max_evals`: Hyperparameter optimization iterations
- `k`: Number of cross-validation folds

### Example Configuration
```bash
# Classification example
python main.py \
    --model_type classification \
    --label_column water_label \
    --model_list RF SVC \
    --feature_selector_list RFE \
    --select_des_num_list 50 \
    --search_max_evals 20 \
    --k 5
```

## Training Process

The training pipeline includes these automated steps:

1. **Data Loading**: Load CSV data and split into train/test/validation sets
2. **Feature Processing**: 
   - Scale features using StandardScaler or MinMaxScaler
   - Apply variance filtering to remove low-variance features
   - Select optimal features using specified method
3. **Model Training**:
   - Hyperparameter optimization using Bayesian search
   - Cross-validation for robust evaluation
   - Save trained models and evaluation metrics

## Inference Process

The inference pipeline automatically:

1. **CIF Processing**: Clean and validate CIF files
2. **Feature Generation**: Extract molecular descriptors using automated tools
3. **Prediction**: Apply all 7 trained models to predict stability properties
4. **Output**: Generate CSV file with predictions and confidence estimates

## Performance Metrics

### Regression Tasks
- **R²**: Coefficient of determination
- **RMSE**: Root mean squared error  
- **MAE**: Mean absolute error
- **Pearson/Spearman**: Correlation coefficients

### Classification Tasks
- **ACC**: Accuracy
- **BACC**: Balanced accuracy
- **MCC**: Matthews correlation coefficient
- **AUC**: Area under ROC curve
- **F1**: F1-score

## Advanced Features

### Uncertainty Estimation
Models include distance-based confidence estimation using ball trees to identify when predictions may be unreliable (extrapolating beyond training data).

### Cross-Validation Strategies
- Standard k-fold cross-validation
- Stratified cross-validation for classification
- Group-based cross-validation for related samples

## Tips for Users

1. **Start Simple**: Use default parameters for initial experiments
2. **Feature Selection**: Try different numbers (50, 100, 200) to find optimal feature count
3. **Model Comparison**: Train multiple algorithms (RF, SVC, GP) and compare performance
4. **Data Quality**: Ensure input CIF files are valid and properly formatted
5. **Results Analysis**: Check cross-validation metrics to assess model reliability

## Troubleshooting

### Common Issues
- **Missing dependencies**: Ensure all required packages are installed
- **CIF file errors**: Check CIF file format and crystal structure validity
- **Memory issues**: Reduce batch size or feature count for large datasets
- **Poor performance**: Try different feature selection methods or model types

### Getting Help
Check the log files in `logs/ml_inference/` for detailed error messages and processing information.