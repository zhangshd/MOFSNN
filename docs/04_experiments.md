<!--
 * @Author: zhangshd
 * @Date: 2025-06-03 23:17:55
 * @LastEditors: zhangshd
 * @LastEditTime: 2025-06-04 02:06:38
-->
# Experiments

This document describes the comprehensive experimental pipeline for MOFSNN project, including dataset analysis, model training, performance evaluation, and uncertainty quantification. The experiments are designed to systematically validate the effectiveness of the proposed MOFSNN architecture for MOF stability prediction tasks.

## 1. Dataset Analysis

The dataset analysis provides comprehensive statistical insights into the MOF stability datasets (TSD, SSD, and WS24), including data distribution patterns, label correlations, and cross-dataset intersections.

### Prerequisites

Ensure that all datasets have been preprocessed according to the [Data Preprocessing](./01_data_processing.md#data-processing) guidelines. The preprocessed datasets should be available in their respective directories with proper feature extraction completed.

### Execution

```bash
python src/experiment/dataset_analysis.py
```

**Key Analysis Components:**
- Statistical distribution of stability labels across tasks
- Label correlation matrices
- Cross-dataset MOF overlap analysis using UpSet plots

**Outputs:**
- Statistical summary reports saved to `results/data_analysis/`
- Distribution plots and correlation heatmaps
- Dataset intersection visualizations

## 2. Dataset Reshuffling for Robust Evaluation

To ensure robust model evaluation and statistical significance of results, multiple dataset splits are generated using different random seeds. This procedure is critical for assessing model consistency and avoiding overfitting to specific train-test splits.

### Prerequisites

Complete the initial data preprocessing pipeline as described in the [Data Preprocessing](./01_data_processing.md#dataset-reshuffling) section. The base datasets should be properly formatted and feature-engineered.

### Execution

```bash
python src/data/reshuffle_splits.py
```

**Configuration Options:**
- Multiple random seeds for statistical robustness
- Stratified splitting for classification tasks to maintain label distributions
- Consistent splitting across all tasks for fair comparison

**Outputs:**
- Multiple split versions stored in task-specific directories
- Split metadata and statistics for reproducibility tracking

## 3. Repeated Model Training

This section implements systematic model training across multiple random seeds and dataset splits to ensure statistical significance and reliability of performance metrics. The training pipeline supports both baseline ML models and advanced CGCNN architectures.

### Prerequisites

Ensure completion of:
- Data preprocessing ([Data Preprocessing](./01_data_processing.md))
- Baseline ML model setup ([ML Model Training](./02_machine_learning_models.md))
- CGCNN environment configuration ([CGCNN Model Training](./03_cgcnn_models.md))

### Execution

```bash
## Baseline ML Models Training
# Hyperparameter optimization with 50 evaluations per model
python src/experiment/batch_submit_ml_train.py --search_max_evals 50

## CGCNN Models with Hyperparameter Optimization
# Automated hyperparameter search using Optuna framework
python src/experiment/batch_submit_cgcnn_opt.py 

## CGCNN Models with Fixed Hyperparameters
# Training with pre-optimized hyperparameters for faster execution
python src/experiment/batch_submit_cgcnn_train.py
```

**Monitoring and Outputs:**
- Model checkpoints saved in `results/cgcnn_models/` and `results/ml_models/`
- Hyperparameter search histories for analysis and reproducibility

## 4. Test Set Performance Evaluation

Comprehensive evaluation of trained models on test sets, comparing baseline ML approaches, single-task CGCNN models, multi-task variants, and the full MOFSNN architecture across all stability prediction tasks.

### Prerequisites

Ensure successful completion of model training phases:
- Baseline ML models ([ML Model Training](./02_machine_learning_models.md))
- CGCNN model variants ([CGCNN Model Training](./03_cgcnn_models.md))
- All dataset splits properly configured

### Execution Pipeline

```bash
## Baseline ML Model Evaluation
# Aggregate results across all ML models and tasks
python src/experiment/compare_ml_results.py

## ML Results Visualization
# Generate publication-ready performance plots with custom styling
python src/experiment/compare_ml_results.py --split test --visualize \
    --mae_min 20 --mae_max 72.5 --acc_max 1.05 \
    --annotate_size 10 --bar_width 0.8

## CGCNN Results Aggregation
# Consolidate results from hyperparameter-optimized models
python src/experiment/aggregate_model_results.py --base-dir results/cgcnn_models_opt

## Comprehensive Model Comparison (Averaged Across Seeds)
# Statistical comparison with confidence intervals
python src/experiment/compare_model_performance_repeat.py \
    --config_file results/cgcnn_models_opt/aggregated_model_results.yaml \
    --model_group standard --visualize --mae_max 62.5 --acc_max 1.05

## Best Single Split Comparison
# Performance on the best-performing dataset split
python src/experiment/compare_model_performance.py --split test \
    --config_file configs/model_comparison_config.yaml \
    --model_group standard --visualize \
    --mae_min 20 --mae_max 72.5 --acc_max 1.05 --annotate_size 8
```

**Evaluation Metrics:**
- **Regression Tasks (TSD)**: MAE, R²
- **Classification Tasks (SSD, WS24)**: Accuracy, Balanced accuracy, ROC-AUC
- **Statistical Analysis**: Mean ± standard deviation across multiple seeds
- **Visual Analysis**: Performance plots with error bars and significance testing

**Model Categories Evaluated:**
- **Baseline**: Traditional ML models (Random Forest, Gaussian Process)
- **CGCNN_SG**: Single-task crystal graph neural networks
- **CGCNN_MT**: Multi-task CGCNN without attention
- **MOFSNN**: Full architecture with task-specific attention mechanism

## 5. External Test Set Validation

Independent validation using external test sets that were completely held out during model development. This evaluation includes comparison with published reference models to establish benchmark performance and validate generalization capabilities.

### Prerequisites

Complete the following preparatory steps:
- Model training completion ([ML Model Training](./02_machine_learning_models.md), [CGCNN Model Training](./03_cgcnn_models.md))
- External test set preparation ([Data Preprocessing](./01_data_processing.md#external-test-set))
- Reference model environments setup:
  - `environments/mofsimplify.yaml` for thermal/solvent stability models
  - `environments/reference_ws24.yaml` for water stability models

### Execution Pipeline

```bash
## Reference Model Inference on External Sets
# Thermal stability external test predictions
python src/experiment/batch_reference_stability.py \
    --input_dir data/cgcnn_data/TS_external_test/cifs \
    --output_dir results/reference_pred/TS_external_test

# Water stability external test predictions (multi-worker for efficiency)
python src/experiment/batch_reference_stability.py \
    --input_dir data/cgcnn_data/WS24v2_external_test/cifs \
    --output_dir results/reference_pred/WS24v2_external_test --workers 2

## Reference Results Processing
# Standardize reference model outputs for unified comparison
python src/experiment/reference_external_test_results_process.py

## Our Models' External Test Inference
# Generate predictions from all trained baseline and CGCNN models
python src/experiment/external_test_prediction.py

## Comprehensive External Test Comparison
# Compare all models including reference baselines
python src/experiment/compare_model_performance.py --split external_test \
    --config_file configs/model_comparison_config.yaml \
    --model_group standard --visualize \
    --mae_min 20 --mae_max 72.5 --acc_max 1.05 --annotate_size 8
```

**Reference Model Integration:**
- **MOFSimplify Models**: Published models for thermal and solvent stability
- **WS24 Reference**: Published water stability prediction models
- **Performance Benchmarking**: Direct comparison with state-of-the-art methods

**External Test Validation Features:**
- **Independent Evaluation**: Completely unseen data during model development
- **Reference Comparison**: Performance relative to published benchmarks
- **Generalization Assessment**: Model robustness across different MOF chemistries
- **Real-world Performance**: Practical applicability demonstration

**Key Validation Metrics:**
- Accuracy on completely independent test sets
- Comparison with literature benchmark performance
- Analysis of failure cases and model limitations
- Statistical significance testing across model architectures

## 6. Uncertainty Quantification and Analysis

Comprehensive uncertainty analysis of the MOFSNN model, including uncertainty evolution during training, latent space uncertainty patterns, and identification of high-uncertainty samples for model improvement strategies.

### 6.1 Training Uncertainty Evolution Tracking

Monitor how prediction uncertainty changes throughout the training process to understand model learning dynamics and convergence patterns.

```bash
## Train CGCNN with Epoch-by-Epoch Checkpointing
# Save model state at every epoch for uncertainty tracking
# Note: Modify `model_params` in script to test different hyperparameters
# (e.g., optimizer choice, learning rate decay)
python src/experiment/slurm_train_epoch_save.py
```

### 6.2 Latent Space Uncertainty Analysis

Build k-nearest neighbor trees in the learned latent representation space to analyze uncertainty patterns and model behavior in different regions of the chemical space.

```bash
## Build Latent Vector Trees for All Training Epochs
# Create k-NN trees for uncertainty analysis (k=5 neighbors)
# Process all saved epochs and save to organized output directory
python -u src/experiment/build_latent_vec_tree.py \
    --model_dir results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_52 \
    --k 5 --process_all \
    --output_dir results/uncertainty_evolution/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn_version_52
```

### 6.3 Uncertainty Evolution Visualization

Generate comprehensive visualizations showing how model uncertainty evolves during training and identify optimal stopping points.

```bash
## Visualize Uncertainty Patterns Across Training
# Create plots showing uncertainty metrics vs. training epoch
python src/experiment/track_uncertainty_evolution.py \
    results/uncertainty_evolution/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn_version_45
```

### 6.4 Cross-Model Uncertainty Comparison

Compare uncertainty characteristics across different model architectures and training configurations to identify the most reliable prediction strategies.

```bash
## Compare Uncertainty Across Multiple Models
# Analyze uncertainty patterns between different MOFSNN variants
python src/experiment/compare_model_uncertainty.py
```

**Uncertainty Analysis Features:**
- **Epistemic Uncertainty**: Model uncertainty due to limited training data
- **Aleatoric Uncertainty**: Inherent uncertainty in the data/labels
- **Latent Space Analysis**: Uncertainty patterns in learned chemical representations
- **Temporal Dynamics**: How uncertainty evolves during model training
- **Comparative Analysis**: Uncertainty characteristics across model architectures

**Key Outputs:**
- Uncertainty evolution plots showing training dynamics
- Latent space uncertainty heatmaps and cluster analysis
- High-uncertainty sample identification for targeted data collection
- Model reliability metrics for different prediction confidence levels
- Uncertainty-based model selection criteria

## 7. Data Augmentation Experiments

Systematic evaluation of data augmentation strategies to address class imbalance and improve model performance, including balanced sampling, uncertainty-guided augmentation, and downsampling strategy analysis.

### 7.1 Training with Different Augmentation Strategies

```bash
## Class Balance Augmentation
# Train with balanced class representation without downsampling
python -u src/cgcnn/main.py --task_cfg tsd_ssd_ws24 --model_cfg att_cgcnn \
    --progress_bar --batch_size 32 --max_epochs 500 --max_graph_len 200 \
    --atom_fea_len 144 --extra_fea_len 28 --h_fea_len 144 --n_conv 4 --n_h 8 \
    --dropout_prob 0.55 --use_cell_params --atom_layer_norm \
    --loss_aggregation fixed_weight_sum --dl_sampler random \
    --task_att_type self --aug_noise_std 0.01 --lr 0.001 --lr_mult 1 \
    --group_lr --optim_config fine --patience 50 --task_norm \
    --log_dir results/cgcnn_models_augmented --augment --balance_classes

## No Downsampling Strategy
# Train on full dataset without reducing majority class samples
python -u src/cgcnn/main.py --task_cfg tsd_ssd_ws24 --model_cfg att_cgcnn \
    --progress_bar --batch_size 32 --max_epochs 500 --max_graph_len 200 \
    --atom_fea_len 144 --extra_fea_len 28 --h_fea_len 144 --n_conv 4 --n_h 8 \
    --dropout_prob 0.55 --use_cell_params --atom_layer_norm \
    --loss_aggregation fixed_weight_sum --dl_sampler random \
    --task_att_type self --aug_noise_std 0.01 --lr 0.001 --lr_mult 1 \
    --group_lr --optim_config fine --patience 50 --task_norm \
    --log_dir results/cgcnn_models_augmented

## Uncertainty-Guided Sample Augmentation
# Augment training set with high-uncertainty samples identified from previous models
python -u src/cgcnn/main.py --task_cfg tsd_ssd_ws24 --model_cfg att_cgcnn \
    --progress_bar --batch_size 32 --max_epochs 500 --max_graph_len 200 \
    --atom_fea_len 144 --extra_fea_len 28 --h_fea_len 144 --n_conv 4 --n_h 8 \
    --dropout_prob 0.55 --use_cell_params --atom_layer_norm \
    --loss_aggregation fixed_weight_sum --dl_sampler random \
    --task_att_type self --aug_noise_std 0.01 --lr 0.001 --lr_mult 1 \
    --group_lr --optim_config fine --patience 50 --task_norm \
    --log_dir results/cgcnn_models_augmented --down_sampling --augment \
    --aug_sample_file results/high_uncertainty_samples.xlsx --aug_factor 1
```

### 7.2 Augmentation Strategy Evaluation

```bash
## Standard Augmentation Performance Analysis
# Compare augmented models with baseline performance
python src/experiment/compare_model_performance.py --split test \
    --config_file configs/model_comparison_config.yaml \
    --model_group augmentation --visualize \
    --mae_min 10 --mae_max 62.5 --acc_max 1.05 \
    --annotate_size 10 --bar_width 0.8

## Uncertainty-Based Augmentation Analysis
# Evaluate the effectiveness of uncertainty-guided sample selection
python src/experiment/compare_model_performance.py --split test \
    --config_file configs/model_comparison_config.yaml \
    --model_group augmentation_uncertainty --visualize \
    --mae_min 10 --mae_max 62.5 --acc_max 1.05 \
    --annotate_size 10 --bar_width 0.8
```

**Augmentation Strategies Tested:**
- **Balance Classes**: Oversample minority classes to achieve balanced representation
- **No Downsampling**: Preserve full dataset without reducing majority class size
- **Uncertainty-Guided**: Target high-uncertainty regions identified through model analysis
- **Noise Injection**: Add Gaussian noise to atomic features for regularization

**Key Research Questions:**
- How does class balancing affect model performance on imbalanced datasets?
- Can uncertainty-guided augmentation improve model reliability?

**Performance Metrics:**
- Comparison against baseline models without augmentation

## 8. Inference Efficiency Benchmarking

Comprehensive evaluation of model inference performance and computational efficiency across different architectures, providing practical guidance for deployment scenarios and real-time applications.

### 8.1 Baseline ML Model Inference

```bash
## Single CIF File Inference
# Test inference speed and memory usage on individual MOF structures
python src/ml/inference.py \
    --input_path data/example_cifs/ADAVAE_clean.cif \
    --output_path results/inference/ADAVAE_prediction_ml.csv

## Batch CIF Directory Processing
# Evaluate throughput on multiple MOF structures simultaneously
python src/ml/inference.py \
    --input_path data/example_cifs/ \
    --output_path results/inference/example_mofs_prediction_ml.csv
```

### 8.2 MOFSNN Model Inference

```bash
## Single Structure MOFSNN Prediction
# Compare CGCNN inference performance with baseline models
python src/cgcnn/inference.py \
    --input_path data/example_cifs/ADAVAE_clean.cif \
    --output_path results/inference/ADAVAE_prediction_mofsnn.csv

## Batch MOFSNN Processing
# High-throughput screening capability evaluation
python src/cgcnn/inference.py \
    --input_path data/example_cifs/ \
    --output_path results/inference/example_mofs_prediction_mofsnn.csv
```

### 8.3 High-Performance Computing Integration

```bash
## SLURM Cluster Batch Processing
# Deploy inference jobs with memory constraints (64GB per job)
# Suitable for large-scale MOF database screening
python src/experiment/slurm_inference_batch.py \
    --input_path data/example_cifs/ \
    --output_dir results/inference

## Reference Model Batch Inference
# Benchmark performance against published models
python src/experiment/slurm_reference_infer_batch.py \
    --input_dir data/example_cifs/ \
    --output_dir results/inference
```

**Performance Metrics Evaluated:**
- **Throughput**: Structures processed per second
- **Memory Usage**: Peak RAM consumption during inference
- **Scalability**: Performance degradation with increasing batch sizes
- **Resource Utilization**: CPU/GPU efficiency across model architectures
- **Latency**: Time-to-first-prediction for real-time applications

**Deployment Considerations:**
- **Single Structure Prediction**: Interactive applications and web services
- **Batch Processing**: Large-scale MOF database screening
- **HPC Integration**: Cluster-based high-throughput virtual screening
- **Memory Management**: Optimal batch sizes for different hardware configurations
- **Model Loading**: Initialization overhead and caching strategies

**Expected Outputs:**
- Detailed performance benchmarks comparing all model architectures
- Resource utilization profiles for deployment planning
- Scalability analysis for different hardware configurations
- Recommendations for optimal deployment strategies based on use case requirements

## 9. Experimental Pipeline Summary

This comprehensive experimental framework provides systematic evaluation of MOFSNN across multiple dimensions, ensuring robust and reliable assessment of the proposed methodology.

### Key Experimental Achievements

**Dataset Understanding and Preparation:**
- Comprehensive statistical analysis of MOF stability datasets
- Robust cross-validation through multiple random seed splits
- External test set preparation for independent validation

**Model Development and Training:**
- Systematic comparison across model architectures (Baseline ML, CGCNN variants, MOFSNN)
- Hyperparameter optimization using Bayesian methods
- Multi-seed training for statistical significance

**Performance Validation:**
- Internal validation on held-out test sets
- External validation against independent datasets
- Benchmark comparison with published reference models

**Advanced Analysis:**
- Uncertainty quantification and evolution tracking
- Data augmentation strategy evaluation
- Computational efficiency benchmarking

### Reproducibility Guidelines

All experiments are designed with reproducibility in mind:
- **Configuration Management**: YAML-based parameter specification
- **Random Seed Control**: Fixed seeds for deterministic results
- **Environment Specification**: Conda environment files provided
- **Logging and Monitoring**: Comprehensive experiment tracking
- **Result Archival**: Structured output organization

### Expected Outcomes

The experimental pipeline is designed to demonstrate:
1. **Superior Performance**: MOFSNN outperforms baseline approaches
2. **Statistical Significance**: Results hold across multiple random trials
3. **Generalization Capability**: Strong performance on external test sets
4. **Uncertainty Quantification**: Reliable confidence estimation
5. **Practical Applicability**: Efficient inference for real-world deployment

### Next Steps

Following completion of this experimental pipeline:
- Results can be compiled for publication
- Models can be deployed for practical MOF screening applications
- Uncertainty analysis can guide targeted experimental validation
- Performance benchmarks can inform future model development

For detailed implementation guidance, refer to the individual sections above and the corresponding source code documentation.