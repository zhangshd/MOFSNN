# MOFSNN: Metal-Organic Framework Stability Neural Network

## Overview

MOFSNN (Metal-Organic Framework Stability Neural Network) is a comprehensive framework designed to predict the stability of Metal-Organic Frameworks (MOFs) across different conditions. The project implements both traditional machine learning models and advanced graph neural networks (specifically CGCNN - Crystal Graph Convolutional Neural Networks) to predict various stability properties of MOFs.

## Project Objectives

The primary goal of this project is to develop machine learning models capable of accurately predicting MOF stability under various conditions:

- Thermal stability (TSD dataset)
- Solvent stability (SSD dataset)
- Water stability (WS24 dataset)
- Acid stability (WS24 dataset)
- Base stability (WS24 dataset)
- Boiling stability (WS24 dataset)

## Project Structure

```
MOFSNN/
├── data/               # Data directory
│   ├── raw_data/       # Original MOF data
│   ├── cgcnn_data/     # Processed data for CGCNN models
│   └── ml_data/        # Processed data for traditional ML models
│
├── notebooks/          # Jupyter notebooks for data processing and analysis
│   ├── 01_process_TSDandSSD.ipynb
│   ├── 02_process_WS24.ipynb
│   └── ...
│
├── results/            # Model outputs and results
│   ├── cgcnn_models/   # CGCNN model checkpoints
│   └── ml_models/      # Traditional ML model files
│
├── src/                # Source code
│   ├── cgcnn/          # CGCNN model code
│   └── ml/             # Traditional ML model code
│
└── docs/               # Documentation
```

## Data Sources

The project utilizes several datasets:

1. **TSD & SSD Datasets**: Thermal stability and solvent stability datasets from [Nandy's work](https://pubs.acs.org/doi/10.1021/jacs.1c07217), available from [Zenodo](https://zenodo.org/records/5737968/files/SciData.zip?download=1).

2. **WS24 Dataset**: Water stability, acid stability, base stability, and boiling stability datasets from [Terrones's work](https://pubs.acs.org/doi/10.1021/jacs.4c05879), available from [Zenodo](https://zenodo.org/records/12110918).

3. **CoREMOF2019 Database**: The CIF files are matched with MOFs in the CoREMOF2019 database, available from [MOFX-DB](https://mof.tech.northwestern.edu/databases).

## Getting Started

1. **Clone the repository**

2. **Set up the environment**:
   ```
   conda env create -f environments/mofsnn.yaml
   conda activate mofsnn
   ```

3. **Download and extract the data**:
   ```
   wget https://zenodo.org/records/5737968/files/SciData.zip?download=1 -O Nandy_2022_SciData.zip
   wget https://zenodo.org/api/records/12110918/files-archive -O WS24v2.zip
   wget https://mof.tech.northwestern.edu/Datasets/CoREMOF%202019-mofdb-version:dc8a0295db.zip -O CoREMOF2019.zip
   
   unzip Nandy_2022_SciData.zip -d data/raw_data/Nandy_2022_SciData
   unzip WS24v2.zip -d data/raw_data/WS24v2
   unzip CoREMOF2019.zip -d data/raw_data/CoREMOF2019
   ```

4. **Process the data** using the notebooks in the `notebooks/` directory

5. **Train models** using either the traditional ML approach or the CGCNN approach

6. **Evaluate models** on test sets and analyze results

## Documentation Structure

The documentation is organized into the following sections:

1. [Project Overview](01_overview.md) (this document)
2. [Data Processing](02_data_processing.md)
3. [Machine Learning Models](03_machine_learning_models.md)
4. [CGCNN Models](04_cgcnn_models.md)
5. [Model Evaluation](05_model_evaluation.md)
6. [Inference and Applications](06_inference_applications.md)