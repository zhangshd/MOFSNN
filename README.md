# MOFSNN: Metal-Organic Framework Stability Neural Network

A machine learning framework for predicting the stability of Metal-Organic Frameworks (MOFs).

The data is stored in the `data` folder, with raw data in `data/raw_data`.

## Project Structure

```
MOFSNN/
├── data/               # 数据目录
│   ├── raw_data/       # 原始数据
│   ├── cgcnn_data/     # CGCNN模型数据
│   └── ml_data/        # 传统机器学习模型数据
│
├── notebooks/          # Jupyter笔记本
│   ├── 01_process_TSDandSSD.ipynb
│   ├── 02_process_WS24.ipynb
│   ├── 16_atom_importance_visualization.ipynb
│   └── ...
│
├── examples/           # 示例脚本
│   └── atom_importance_visualization.py
│
├── results/            # 模型输出和结果
│   ├── cgcnn_models/   # CGCNN模型
│   └── ml_models/      # 传统机器学习模型
│
├── src/                # 源代码
│   ├── cgcnn/          # CGCNN模型代码
│   │   ├── module/     # 模型模块
│   │   │   └── ...
│   │   └── visualization/   # 可视化工具
│   │       ├── atom_visualizer.py  # 原子重要性可视化工具(Grad-CAM, Guided Grad-CAM)
│   │       ├── ngl_visualizer.py  # NGLView可视化实现
│   │       └── ...
│   ├── ml/             # 传统机器学习模型代码
│   │   └── mof_inference.py  # 单文件/文件夹推理脚本
│   └── data/           # 数据处理代码
│
└── README.md           # 项目说明
```

## Development Setup

```bash
# 创建虚拟环境
conda create -n mofsnn python=3.9
conda activate mofsnn

# 安装依赖
pip install -r requirements.txt

# 准备数据目录
mkdir -p data/raw_data
```

## ML Model Inference

The project includes a script for making predictions on new MOF structures using pre-trained ML models:

```bash
# Process a single CIF file
python src/ml/mof_inference.py --input_path /path/to/your/mof.cif --output_path results/predictions.csv

# Process a directory containing multiple CIF files
python src/ml/mof_inference.py --input_path /path/to/cif_directory --output_path results/predictions.csv
```

The script predicts all 7 stability properties:
- Thermal Stability (TSD)
- Solvent Stability (SSD)
- Water Stability (WS24_water, binary)
- Water Stability 4-level (WS24_water4)
- Acid Stability (WS24_acid)
- Base Stability (WS24_base)
- Boiling Water Stability (WS24_boiling)

## Model Evaluation Tools

### 模型性能比较脚本

本项目包含两个用于比较模型性能的脚本：

1. `src/experiment/compare_model_performance.py`: 基础版本，处理单路径模型评估
2. `src/experiment/compare_model_performance_repeat.py`: 增强版本，支持将Path配置为列表，可计算多次运行结果的均值和标准差

使用方法示例：
```bash
# 标准版本
python src/experiment/compare_model_performance.py --config_file configs/model_comparison_config.yaml

# 支持重复实验的增强版本
python src/experiment/compare_model_performance_repeat.py --config_file configs/model_comparison_config_repeat.yaml
```

## Raw data
The raw data is stored in the `raw_data` folder, which can get from literature. 
### TSD & SSD 
The thermal stability dataset and solvent stability dataset are get from [Nandy's work](https://pubs.acs.org/doi/10.1021/jacs.1c07217). The data is download from [here](https://zenodo.org/records/5737968/files/SciData.zip?download=1), which is published in [Nandy's another paper](https://www.nature.com/articles/s41597-022-01181-0). The CIF files are matched with MOFs in the CoREMOF2019 databse, which is available from [MOFX-DB](https://mof.tech.northwestern.edu/databases).
### WS24
The water stability, acid stability, base stability, and boiling stability dataset are get from [Terrones's work](https://pubs.acs.org/doi/10.1021/jacs.4c05879).
And the raw data is download from [here](https://zenodo.org/records/12110918).
### Download and extract data
```sh
mkdir -p data/raw_data
cd data/raw_data
wget https://zenodo.org/records/5737968/files/SciData.zip?download=1 -O Nandy_2022_SciData.zip
wget https://zenodo.org/api/records/12110918/files-archive -O WS24v2.zip
wget https://mof.tech.northwestern.edu/Datasets/CoREMOF%202019-mofdb-version:dc8a0295db.zip -O CoREMOF2019.zip
unzip Nandy_2022_SciData.zip
unzip WS24v2.zip
unzip CoREMOF2019.zip
```

## Model Visualization Tools

The project includes advanced tools for visualizing and explaining CGCNN model predictions at different levels:

1. **Atom Importance Visualization** - Understand which atoms influence predictions
2. **Feature Importance Visualization** - Analyze the importance of crystal and extra features
3. **Model Uncertainty Analysis** - Analyze prediction uncertainty using LSE and LSV metrics

### Atom Importance Visualization

This tool helps interpret which atoms most influence the model's predictions, even for models using average pooling rather than attention mechanisms.

#### Using the Atom Visualization Tool

You can visualize atom importance using the provided example script:

```bash
# Run the atom importance visualization script
python examples/atom_importance_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --task_idx 0 --save_dir results/atom_importance
```

Or explore interactively using the Jupyter notebooks:
```bash
# Static visualization notebook
jupyter notebook notebooks/16_atom_importance_visualization.ipynb

# Interactive 3D visualization notebook
jupyter notebook notebooks/16_atom_importance_visualization_interactive.ipynb
```

#### Atom Visualization Features

- Gradient-based attribution of importance to individual atoms
- Multiple visualization methods:
  - Static 3D plots with matplotlib
  - Interactive 3D visualization with NGLView
  - High-quality rendering with OVITO (if installed)
- Support for highlighting important atoms above a threshold
- Multi-task comparison to analyze task-specific importance patterns
- Interactive rotation, zoom, and inspection in 3D
- Compatible with models using average pooling or attention pooling
- Works for both classification and regression tasks

### Feature Importance Visualization

This tool helps analyze the relative importance of crystal features and extra features at the `conv_to_fc` layer where they are combined, using a Grad-CAM approach.

#### Using the Feature Visualization Tool

You can visualize feature importance using the provided example script:

```bash
# Visualize feature importance for a single task
python examples/feature_importance_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --task_idx 0 --save_dir results/feature_importance

# Show both positive and negative contributions
python examples/feature_importance_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --no_relu

# Compare feature importance across all tasks
python examples/feature_importance_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --compare_tasks

# Analyze specific sample in batch (instead of batch average)
python examples/feature_importance_visualization.py --model_path /path/to/model/checkpoint.ckpt --cif_path /path/to/structure.cif --sample_idx 0
```

#### Feature Visualization Capabilities

- Analyze importance of both crystal features (`crys_fea`) and extra features (`extra_fea`)
- Visualize importance as a colored strip with clear separation between feature types
- Apply ReLU to focus on positive contributions or disable it to see both positive and negative influences
- Compare feature importance patterns across multiple tasks
- Analyze individual samples in a batch (rather than batch average)
- Print detailed statistics about the most important features
- Generate high-quality visualizations suitable for publications

### Model Uncertainty Analysis

This tool provides comprehensive uncertainty analysis for CGCNN models using Local Similarity Entropy (LSE) for classification tasks and Local Similarity Variance (LSV) for regression tasks.

#### Using the Uncertainty Analysis Tool

You can perform uncertainty analysis using the command line interface:

```bash
# Run uncertainty analysis on model evaluation results
python src/experiment/model_uncertainty_analysis.py \
    --log_dir /path/to/model/evaluation/results \
    --uncertainty_trees_file /path/to/uncertainty_trees.pkl \
    --output_dir results/uncertainty_analysis \
    --k 5

# Use custom figure size
python src/experiment/model_uncertainty_analysis.py \
    --log_dir /path/to/evaluation/results \
    --uncertainty_trees_file /path/to/uncertainty_trees.pkl \
    --output_dir results/uncertainty_analysis \
    --figsize 24 12
```

Or use the Python API for more flexibility:

```python
from experiment.model_uncertainty_analysis import UncertaintyAnalyzer

# Initialize analyzer
analyzer = UncertaintyAnalyzer(
    log_dir="/path/to/evaluation/results",
    uncertainty_trees_file="/path/to/uncertainty_trees.pkl",
    output_dir="results/uncertainty_analysis"
)

# Run combined analysis for all tasks
analyzer.run_combined_analysis(k=5, figsize=(20, 10))

# Analyze individual tasks
ax, df_summary = analyzer.lse_analysis('SSD', k=5)  # Classification task
ax, df_summary = analyzer.lsv_analysis('TSD', k=5)  # Regression task
```

#### Uncertainty Analysis Features

- **LSE Analysis**: Local Similarity Entropy for classification tasks
  - Measures prediction uncertainty based on neighbor label diversity
  - Analyzes accuracy vs uncertainty cutoff relationships
  - Computes AUROC scores for different uncertainty thresholds
- **LSV Analysis**: Local Similarity Variance for regression tasks  
  - Measures prediction uncertainty based on neighbor label variance
  - Analyzes MAE and R² scores vs uncertainty cutoff relationships
  - Identifies optimal uncertainty thresholds for data filtering
- **Cutoff Analysis**: Performance evaluation at different uncertainty levels
  - Shows how model performance changes when filtering by uncertainty
  - Identifies trade-offs between data retention and prediction accuracy
  - Provides insights for deployment with uncertainty-based filtering
- **Comprehensive Reporting**: 
  - Combined visualization plots for all tasks
  - Numerical results exported to Excel format
  - Summary reports with analysis statistics
  - Individual task analysis with customizable parameters

#### Prerequisites for Uncertainty Analysis

Before running uncertainty analysis, you need:

1. **Model evaluation results**: CSV files with predictions, targets, and probabilities
2. **Latent features**: NPZ files containing last-layer features from the model  
3. **Uncertainty trees**: Pickle file with pre-built ball trees for uncertainty calculation

These can be generated using existing scripts:

```bash
# Build uncertainty trees from model checkpoints
python src/experiment/build_latent_vec_tree.py \
    --model_dir /path/to/model/directory \
    --output_dir /path/to/uncertainty_trees \
    --k 5

# Run model evaluation to get results and features
# (Use existing evaluation scripts in the project)
```

### Interactive 3D Visualization Requirements

For interactive 3D visualization, additional packages are required:

```bash
# For NGLView-based visualization
pip install nglview

# For OVITO-based visualization (optional)
pip install ovito
# or with conda
conda install -c conda-forge ovito
```

### Feature Updates

#### 2025-05-18: Improved Atom Importance Visualization
- Implemented Grad-CAM for atom importance visualization in CGCNN models
- Added support for visualizing both positive and negative atom contributions
- Updated documentation in `docs/atom_importance_visualization.md`
- Modified `AtomImportanceVisualizer` class to use state-of-the-art explainability techniques

#### 2025-05-19: Enhanced Atom Visualization Methods
- Added multiple visualization methods for comprehensive model interpretation:
  - **Grad-CAM with ReLU**: Standard implementation showing positive contributions only
  - **Grad-CAM without ReLU**: Modified version showing both positive and negative contributions
  - **Guided Grad-CAM**: High-resolution visualization combining Grad-CAM with Guided Backpropagation
- Added method comparison functionality in `compare_visualization_methods.py` example
- Created detailed documentation in `docs/visualization_method_comparison.md`
- Updated API to allow method selection via `method` parameter

#### 2025-05-20: Feature Importance Visualization
- Implemented a new feature importance visualization method that uses Grad-CAM approach
- Added visualization for both crystal features (`crys_fea`) and extra features (`extra_fea`)
- Introduced colored strip visualization with clear separation between feature types
- Added support for comparing feature importance across multiple tasks
- Created easy-to-use example script in `examples/feature_importance_visualization.py`
- Added optional ReLU activation for focusing on positive feature contributions
- Supported per-sample analysis for detailed examination of individual samples

#### 2025-05-30: Model Uncertainty Analysis
- Implemented comprehensive uncertainty analysis using LSE (Local Similarity Entropy) and LSV (Local Similarity Variance)
- Created `model_uncertainty_analysis.py` script for automated uncertainty analysis of CGCNN models
- Added support for cutoff analysis to evaluate performance vs uncertainty trade-offs
- Integrated with existing uncertainty tree infrastructure for efficient neighbor searches
- Generated combined visualization plots and numerical results in Excel format
- Added command-line interface and Python API for flexible usage
- Created example script in `examples/uncertainty_analysis_example.py`
- Supports both classification tasks (LSE) and regression tasks (LSV)
- Provides insights for uncertainty-based data filtering and model deployment
