# Data Processing

This document describes the data processing steps in the MOFSNN project, detailing how raw MOF data is prepared for both traditional machine learning models and CGCNN models.

## Raw Data Sources

The project uses several datasets:

1. **Thermal Stability Dataset (TSD)**: Contains thermal decomposition temperatures for MOFs
2. **Solvent Stability Dataset (SSD)**: Contains solvent stability classifications for MOFs
3. **Water Stability Dataset (WS24)**: Contains water stability data for MOFs
4. **Acid/Base/Boiling Stability Datasets**: Additional stability measures from the WS24 dataset
5. **CoREMOF2019**: A database of MOF structures in CIF format that matches with the stability datasets

## Processing Pipeline

### 1. TSD and SSD Data Processing

The processing of TSD and SSD datasets is handled in `notebooks/01_process_TSDandSSD.ipynb`. The key steps include:

1. **Loading raw data**:
   - Load CSV files containing stability data from Nandy's work
   - Match MOF names with CIF files from CoREMOF2019 database

2. **Data cleaning**:
   - Remove rows with missing values
   - Standardize column names for consistency
   - Extract relevant features and labels

3. **CIF file processing**:
   - Copy matching CIF files from CoREMOF2019 to task-specific directories
   - Clean CIF files:
     - Remove overlapping atoms
     - Remove solvent molecules
     - Standardize CIF format

4. **Feature generation**:
   - Generate RACs (Revised Autocorrelation) features
   - Generate Zeo++ features (geometric/topological descriptors)
   - Combine features with stability labels

5. **Graph data preparation**:
   - Convert CIF files to graph data format for CGCNN
   - Define atomic neighbors within specified radius
   - Extract atom features and bond connections

### 2. WS24 Data Processing

The processing of water stability, acid stability, base stability, and boiling stability datasets is handled in `notebooks/02_process_WS24.ipynb`. The steps include:

1. **Loading raw data**:
   - Load stability data from Terrones's work
   - Match MOF names with CIF files from CoREMOF2019 database

2. **Data formatting**:
   - Standardize column names and formats
   - Separate data into different stability types
   - Apply appropriate transformations to labels

3. **CIF file processing**:
   - Same cleaning process as used for TSD/SSD datasets
   - Verify structure integrity after cleaning

4. **Feature extraction**:
   - Generate consistent feature sets across all stability types
   - Ensure feature compatibility for multi-task learning

5. **Partitioning**:
   - Split data into training, validation, and test sets
   - Ensure balanced distribution across stability types

### 3. External Test Set Preparation

The preparation of external test sets is handled in `notebooks/04_prepare_external_test_set.ipynb`, which includes:

1. **Selecting test structures**:
   - Identify MOFs not included in the training sets
   - Ensure diversity of the test set

2. **Processing test structures**:
   - Apply consistent cleaning and feature extraction steps
   - Format data for model evaluation

## Feature Engineering

### Traditional ML Features

For traditional machine learning models, the following features are extracted:

1. **RACs features**:
   - Molecular connectivity-based descriptors
   - Element-specific information
   - Bonding patterns

2. **Zeo++ features**:
   - Surface area calculations
   - Pore volume metrics
   - Channel dimensions
   - Void fraction
   - Geometry descriptors

### CGCNN Features

For CGCNN models, the following features are prepared:

1. **Atom features**:
   - One-hot encoding of element type
   - Additional atomic properties

2. **Bond features**:
   - Distances between connected atoms
   - Graph connectivity information

## Data Format

### ML Model Data

Traditional ML model data is stored as CSV files with:
- MOF identifiers
- Stability labels
- Feature vectors

### CGCNN Model Data

CGCNN model data consists of:
- Processed CIF files
- Graph data files containing atom features and connectivity information
- ID-property mapping files linking MOF identifiers to stability values

## Directory Structure

```
data/
├── raw_data/               # Original data sources
│   ├── CoREMOF2019/        # MOF CIF files database
│   ├── Nandy_2022_SciData/ # TSD and SSD datasets
│   ├── WS24v2/             # Water, acid, base, boiling stability datasets
│   └── popularMOF/         # Additional MOF structures
│
├── cgcnn_data/             # Processed data for CGCNN models
│   ├── CoREMOF2019/        # Processed CoREMOF database
│   ├── SSD/                # Solvent stability data
│   ├── TSD/                # Thermal stability data
│   ├── WS24/               # Water, acid, base, boiling stability data
│   └── TS_external_test/   # External test sets
│
└── ml_data/                # Processed data for ML models
    ├── SSD/                # Solvent stability features
    ├── TSD/                # Thermal stability features
    └── WS24/               # Water, acid, base, boiling features
```

## Scripts and Utilities

The data processing leverages several utility scripts:

1. **CIF cleaning**: `src/cgcnn/datamodule/clean_cif.py`
2. **Feature generation**: `src/ml/featuring/feature_generation.py`
3. **Graph data preparation**: `src/cgcnn/datamodule/prepare_data.py`
4. **RACs calculation**: `src/ml/featuring/RAC_getter.py`
5. **Solvent removal**: `src/ml/featuring/solvent_removal.py`