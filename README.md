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
│   └── ...
│
├── results/            # 模型输出和结果
│   ├── cgcnn_models/   # CGCNN模型
│   └── ml_models/      # 传统机器学习模型
│
├── src/                # 源代码
│   ├── cgcnn/          # CGCNN模型代码
│   └── ml/             # 传统机器学习模型代码
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
