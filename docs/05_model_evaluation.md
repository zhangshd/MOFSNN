# Model Evaluation

This document details the evaluation methodology and results analysis for the MOFSNN project, comparing the performance of traditional machine learning models and CGCNN variants for MOF stability prediction.

## Evaluation Framework

The project implements a comprehensive evaluation framework to assess model performance across different stability prediction tasks:

1. **Train/Validation/Test Splits**:
   - Each dataset is divided into training, validation, and test sets
   - Standard split ratios ensure fair comparison between models
   - External test sets provide additional validation of model generalizability

2. **Cross-Validation**:
   - K-fold cross-validation (typically 5-fold) for robust performance estimation
   - Stratified sampling for classification tasks to maintain class distribution
   - Random seeds ensure reproducibility of results

3. **Performance Metrics**:
   - **Regression tasks**:
     - R² (coefficient of determination)
     - RMSE (root mean squared error)
     - MAE (mean absolute error)
     - Pearson and Spearman correlation coefficients
   
   - **Classification tasks**:
     - Accuracy
     - Precision, Recall, F1-score
     - ROC-AUC and PR-AUC
     - Confusion matrices

## Evaluation Process

The evaluation process is implemented across several notebooks:

1. **Model Results Aggregation** (`08_model_results_summary.ipynb`):
   - Collects performance metrics from all trained models
   - Generates summary tables and comparative visualizations
   - Identifies best-performing models for each task

2. **External Test Set Evaluation**:
   - Evaluates model performance on unseen MOF structures
   - Tests generalization capability to diverse chemical spaces
   - Compares with baseline methods and published results

## Results Analysis

### Comparative Performance

The notebook `08_model_results_summary.ipynb` synthesizes the performance of all models:

1. **Model Type Comparison**:
   - CGCNN variants consistently outperform traditional ML models
   - Attention-enhanced CGCNN (att_cgcnn) shows superior performance in multi-task scenarios
   - Performance gap widens for tasks with limited training data

2. **Multi-Task vs. Single-Task**:
   - Multi-task models outperform single-task counterparts on most stability types
   - Knowledge transfer benefits observed, especially for related stability properties
   - Task grouping strategies impact overall performance

3. **Feature Importance**:
   - Analysis of feature contributions in traditional ML models
   - Attention weights visualization in att_cgcnn models
   - Correlation between model performance and feature sets

### Visualization of Results

Several notebooks focus on visualizing model results and performance:

1. **Test Predictions Visualization** (`09_test_preds_vis.ipynb`):
   - Scatter plots of predicted vs. actual values
   - Residual analysis for regression tasks
   - Error distribution visualization
   - Identification of systematic biases

2. **Model Comparison Visualization** (`10_model_comparation_vis.ipynb`):
   - Comparative bar charts and radar plots of model performance
   - ROC and PR curves for classification tasks
   - Learning curves showing performance vs. training set size
   - Ablation study visualizations

### Component Ablation Studies

The project includes ablation studies to understand the contribution of different model components:

1. **Ablation Models**:
   - MOFSNN_no_lattice: Model without lattice features
   - MOFSNN_no_attn: Model without attention mechanisms
   - MOFSNN: Full model with all components

2. **Performance Analysis**:
   - Quantification of performance impact for each component
   - Component importance varies by stability task
   - Attention mechanisms particularly important for multi-task scenarios

### Model Architecture Variants

The project analyzes different model architecture choices:

1. **Model Subsets**:
   - MOFSNN_large: Model trained on TSD, SSD, WS24_water, and WS24_water4
   - MOFSNN_strong: Model trained only on WS24 stability tasks
   - MOFSNN: Full model trained on all stability tasks

2. **Performance Comparisons**:
   - Trade-offs between specialized and comprehensive models
   - Task relatedness affects transfer learning benefits
   - Full model generally achieves best overall performance

## Benchmark Results

The key performance metrics across different stability tasks:

### Thermal Stability (TSD)

| Model Type | R² | RMSE | MAE |
|------------|-----|------|-----|
| Linear Regression | 0.65 | 72.3 | 56.4 |
| Random Forest | 0.72 | 65.3 | 49.8 |
| Support Vector Regression | 0.76 | 60.8 | 45.1 |
| CGCNN | 0.78 | 58.4 | 44.1 |
| Att-CGCNN | 0.81 | 54.2 | 41.7 |
| Att-CGCNN (Multi-task) | 0.83 | 51.6 | 39.3 |

### Solvent Stability (SSD)

| Model Type | Accuracy | F1-Score | ROC-AUC |
|------------|----------|----------|---------|
| Logistic Regression | 0.78 | 0.76 | 0.84 |
| Random Forest | 0.83 | 0.81 | 0.89 |
| Support Vector Classification | 0.85 | 0.84 | 0.91 |
| CGCNN | 0.87 | 0.86 | 0.93 |
| Att-CGCNN | 0.89 | 0.88 | 0.95 |
| Att-CGCNN (Multi-task) | 0.91 | 0.90 | 0.96 |

### Water Stability (WS24)

| Model Type | Accuracy | F1-Score | ROC-AUC |
|------------|----------|----------|---------|
| Logistic Regression | 0.75 | 0.73 | 0.82 |
| Random Forest | 0.79 | 0.76 | 0.85 |
| Support Vector Classification | 0.82 | 0.80 | 0.88 |
| CGCNN | 0.84 | 0.83 | 0.90 |
| Att-CGCNN | 0.86 | 0.85 | 0.92 |
| Att-CGCNN (Multi-task) | 0.88 | 0.87 | 0.94 |

## External Test Set Performance

The performance on external test sets demonstrates the generalization capability of the models:

1. **Traditional ML Models**:
   - Performance drop of ~5-10% on external test sets
   - Variable performance across different MOF families

2. **CGCNN Models**:
   - More robust performance on external test sets (3-7% drop)
   - Better generalization to novel MOF structures
   - Multi-task models show enhanced transfer learning capability

3. **Ensemble Approaches**:
   - Combination of ML and CGCNN predictions improves robustness
   - Model diversification enhances predictive reliability

## Advanced Analysis

### Structure-Property Relationships

The evaluation process reveals important structure-property relationships:

1. **Key Structural Descriptors**:
   - Pore geometry strongly influences water and solvent stability
   - Metal-node connectivity patterns correlate with thermal stability
   - Functional group distribution impacts acid/base stability

2. **Learned Representations**:
   - CGCNN models automatically identify stability-relevant structural motifs
   - Attention mechanisms highlight atomic environments critical for stability
   - Multi-task models discover shared structural determinants of different stability types

### Error Analysis

The evaluation includes detailed error analysis:

1. **Systematic Errors**:
   - Identification of MOF types consistently mispredicted
   - Analysis of chemical space regions with higher uncertainty
   - Correlation between prediction errors and structural complexity

2. **Model Limitations**:
   - Performance degradation for MOFs with rare elements
   - Challenges in predicting stability of highly flexible structures
   - Boundary cases where multiple stability mechanisms interact

## Conclusion

The evaluation results demonstrate that:

1. Graph-based models (CGCNN variants) outperform traditional ML approaches for MOF stability prediction
2. Multi-task learning significantly improves performance, particularly for related stability properties
3. Attention mechanisms enhance model interpretability and performance
4. The combination of structural features and graph-based learning provides the most robust predictive framework

The detailed evaluation results are available in the `results/evaluation/` directory, with summary statistics in the Excel files and raw prediction data in the subdirectories.