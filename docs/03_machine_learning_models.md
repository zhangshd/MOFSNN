# Machine Learning Models

This document details the traditional machine learning models used in the MOFSNN project for predicting MOF stability properties.

## Overview

The `src/ml` module implements various traditional machine learning approaches for predicting MOF stability. These models utilize hand-crafted features derived from MOF structures and are trained on several stability datasets.

## Model Types

The project implements two primary types of machine learning models:

1. **Regression Models**: Used for predicting continuous stability values (e.g., thermal decomposition temperature)
2. **Classification Models**: Used for predicting categorical stability classes (e.g., stable/unstable in certain environments)

## Feature Selection

Feature selection is a critical component of the machine learning pipeline, implemented in several ways:

1. **Filter Methods**:
   - Variance thresholding to remove low-variance features (implemented with `VarianceThreshold`)
   - `f_regression` for regression tasks
   - `f_classif` for classification tasks
   - `mutual_info_regression` and `mutual_info_classif` for non-linear relationships

2. **Wrapper Methods**:
   - Recursive feature elimination (RFE) using Random Forest as base estimator

The feature selection process is managed by the `select_feature` method in the `BaseModel` class, which applies variance filtering before the main feature selection methods.

## Model Implementation

### Regression Models

The regression models are implemented in the `src/ml` module. Key aspects include:

1. **Base Models**:
   - Random Forest Regressor (RF)
   - Support Vector Regression (SVM)
   - Linear Regression (LR)
   - Gaussian Process Regressor (GP)

2. **Hyperparameter Optimization**:
   - Bayesian optimization using Hyperopt for hyperparameter tuning
   - Cross-validation strategies for robust evaluation

3. **Metrics**:
   - R² (coefficient of determination)
   - RMSE (root mean squared error)
   - MAE (mean absolute error)
   - Pearson and Spearman correlation coefficients

### Classification Models

The classification models focus on predicting binary or multi-class stability categories:

1. **Base Models**:
   - Random Forest Classifier (RF)
   - Support Vector Classification (SVM)
   - Logistic Regression (LR)
   - Gaussian Process Classifier (GP)

2. **Performance Metrics**:
   - Accuracy (ACC)
   - Balanced Accuracy (BACC)
   - Matthews Correlation Coefficient (MCC)
   - Area Under ROC Curve (AUC)
   - F1-score (F1)
   - Cross-Entropy (CE)

3. **Class Imbalance Handling**:
   - Class weighting for imbalanced datasets
   - Data balancing for specific label columns

## Training Process

The training process for machine learning models includes:

1. **Data Preprocessing**:
   - Feature scaling (standardization/normalization)
   - Handling missing values
   - Categorical encoding (if applicable)

2. **Model Training**:
   - Cross-validation (typically 5-fold)
   - Hyperparameter optimization
   - Early stopping criteria

3. **Model Selection**:
   - Based on validation performance
   - Consideration of model complexity
   - Ensemble methods for improved performance

## Model Retraining

The notebook `05_ML_retrain.ipynb` implements the retraining process for ML models:

1. **Dataset Loading**:
   - Load processed feature data from `data/ml_data/`
   - Apply consistent preprocessing steps

2. **Feature Selection**:
   - Apply optimal feature selection methods determined from initial experiments
   - Standardize feature sets across stability tasks

3. **Training Procedure**:
   - Train models with optimal hyperparameters
   - Cross-validate performance
   - Save trained models to `results/ml_models/`

## External Test Set Prediction

The notebook `06_ML_prediction_of_external_test_set.ipynb` handles prediction on external test data:

1. **Model Loading**:
   - Load trained models from `results/ml_models/`
   - Apply consistent preprocessing to test data

2. **Prediction**:
   - Generate predictions for external test sets
   - Calculate performance metrics
   - Save results to `results/evaluation/ML_results_external_test.xlsx`

## API and Interface

The ML module provides a consistent interface for working with different model types:

1. **RegressionModel Class** (`src/ml/module.py`):
   - Methods for training, evaluation, and prediction
   - Feature importance calculation
   - Model persistence

2. **ClassificationModel Class** (`src/ml/module.py`):
   - Methods for training, evaluation, and prediction
   - Class probability calibration
   - Model persistence

3. **Interface Functions** (`src/ml/interface.py`):
   - Utility functions for model selection
   - Dataset handling
   - Performance reporting

## Example Usage

```python
# Example of training a regression model for thermal stability
from src.ml.module import RegressionModel
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RegressionModel(random_state=42)

# Load the data
model.load_data(train_X, train_y, test_X=test_X, test_y=test_y)

# Scale features
model.scale_feature(feature_range=(0, 1), saved_dir="results/ml_models")

# Select features
model.select_feature(saved_dir="results/ml_models", feature_selector='f_regression', select_des_num=100)

# Define cross-validation strategy
model.kfold_split(k=5, kfold_type="normal")

# Initialize estimator with parameters
estimator = RandomForestRegressor()
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2
}

# Train the model with cross-validation
model.train(estimator, params, saved_dir="results/ml_models")

# Or train on full dataset
model.fulltrain(estimator, params, saved_dir="results/ml_models")

# Make predictions with optional distance-based confidence estimation
predictions = model.predict(X_test, cal_feature_distance=True, neighbors_num=5)

# Access performance metrics
print(model.all_metrics_df)
```

For classification tasks, the workflow is similar:

```python
# Example of training a classification model for solvent stability
from src.ml.module import ClassificationModel
from sklearn.ensemble import RandomForestClassifier

# Initialize for binary classification
model = ClassificationModel(random_state=42, n_class=2)

# Load and encode categorical labels
model.load_data(train_X, train_y, test_X=test_X, test_y=test_y)
model.label_encode(saved_dir="results/ml_models")

# Scale and select features
model.scale_feature(scaler_name="StandardScaler", saved_dir="results/ml_models")
model.select_feature(feature_selector='f_classif', select_des_num=100, saved_dir="results/ml_models")
model.kfold_split(k=5, kfold_type="stratified")

# Train model
estimator = RandomForestClassifier()
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'class_weight': 'balanced'
}
model.train(estimator, params, saved_dir="results/ml_models")

# Make predictions with probability output
class_predictions = model.predict(X_test, return_prob=False)
probability_predictions = model.predict(X_test, return_prob=True)
```

## Advanced Features

### Confidence Estimation with BallTree

The ML models implement a distance-based confidence estimation using the `sklearn.neighbors.BallTree` class:

1. **Ball Tree Implementation**:
   - The `generate_ball_tree` method builds a BallTree from training data 
   - Distance metrics are calculated using Minkowski distance (configurable parameter p)
   - Feature weights are extracted from trained models to enhance distance calculations

2. **Confidence Calculation**:
   - Reference distance values are calculated from validation sets
   - For new predictions, the distance to training samples is used to estimate prediction confidence
   - Confidence index is calculated as `reference_distance / (actual_distance + epsilon)`

3. **Usage in Predictions**:
   - The `predict` method includes an optional `cal_feature_distance` parameter
   - When enabled, predictions include both the predicted value and a confidence index
   - Higher confidence index values indicate more reliable predictions

This feature is particularly valuable for identifying compounds where the model might be extrapolating beyond its training domain, providing a measure of prediction reliability.

### Cross-Validation Strategies

The framework supports multiple cross-validation strategies implemented in the `kfold_split` method:

1. **KFold**: Standard k-fold cross-validation
2. **StratifiedKFold**: Preserves class distribution in each fold
3. **GroupKFold**: Ensures samples from the same group are not split between folds
4. **LeaveOneOut (LOO)**: Special case where each sample forms a test set
5. **LeaveOneGroupOut (LOGO)**: Each group of samples forms a test set

These strategies can be selected based on dataset characteristics and evaluation requirements.

## Performance Considerations

1. **Computational Efficiency**:
   - Feature selection to reduce dimensionality
   - Efficient model implementations
   - Parallelization for hyperparameter search

2. **Model Interpretability**:
   - Feature importance analysis
   - Partial dependence plots

3. **Ensemble Strategies**:
   - Model averaging
   - Multiple model training with different random seeds

## Results

The machine learning models achieve competitive performance across the different stability prediction tasks:

1. **Thermal Stability**: Regression performance with R² values typically between 0.7-0.8
2. **Solvent Stability**: Classification accuracy typically above 0.85
3. **Water Stability**: Classification performance with F1-scores typically above 0.8

Detailed performance metrics are available in the `results/evaluation/` directory.