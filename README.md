<!--
 * @Author: zhangshd
 * @Date: 2025-06-03 10:17:33
 * @LastEditors: zhangshd
 * @LastEditTime: 2025-06-04 01:48:38
-->
# MOFSNN: Metal-Organic Framework Stability Neural Network

A machine learning framework for predicting the stability of Metal-Organic Frameworks (MOFs).

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
├── data/               # data directory
│   ├── raw_data/       # raw data
│   ├── cgcnn_data/     # processed data for CGCNN models
│   └── ml_data/        # processed data for traditional ML models
│
├── configs/            # information of trained models in yaml format
│
├── results/            # model outputs and results
│   ├── cgcnn_models/   # CGCNN models
│   └── ml_models/      # traditional ML models
│
├── src/                # source code
│   ├── cgcnn/          # CGCNN model code
│   ├── ml/             # traditional machine learning model code
│   ├── data/           # data processing code
│   └── experiment/     # model evaluation, comparison, and visualization code
│
└── docs/               # Documentation
```

## Data Sources

The project utilizes several datasets:

1. **TSD & SSD Datasets**: Thermal stability and solvent stability datasets from [Nandy's work](https://pubs.acs.org/doi/10.1021/jacs.1c07217), available from [Zenodo](https://zenodo.org/records/5737968/files/SciData.zip?download=1).

2. **WS24 Dataset**: Water stability, acid stability, base stability, and boiling stability datasets from [Terrones's work](https://pubs.acs.org/doi/10.1021/jacs.4c05879), available from [Zenodo](https://zenodo.org/records/12110918).

3. **CoREMOF2019 Database**: The CIF files are matched with MOFs in the CoREMOF2019 database, available from [MOFX-DB](https://mof.tech.northwestern.edu/databases).


## Quick Start

To inference the stability of MOFs using the trained MOFSNN, you can use the following command:
```sh
python src/cgcnn/inference.py --input_path <path_to_cif_file_or_directory> --output_path <path_to_save_predictions>
```

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

4. **Process the data** using the scripts in the `src/data/`. This includes:

   - Converting CIF files to crystal graph format
   - Extracting crystal features and extra features
   - Splitting datasets into training, validation, and test sets

5. **Train models** using either the traditional ML approach or the CGCNN approach

6. **Evaluate models** on test sets and analyze results

## Documentation Structure

The documentation is organized into the following sections:

1. [Data Processing](docs/01_data_processing.md)
2. [Machine Learning Models](docs/02_machine_learning_models.md)
3. [CGCNN Models](docs/03_cgcnn_models.md)
4. [Experimentation and Results](docs/04_experiments.md)
