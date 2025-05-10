# Data Processing

This document describes the data processing workflow for the MOFSNN project. The processing pipeline prepares raw MOF (Metal-Organic Framework) data for both traditional ML models and CGCNN (Crystal Graph Convolutional Neural Networks) models.

## Datasets

The project works with the following datasets:
- **TSD**: Thermal stability dataset
- **SSD**: Solvent removal stability dataset 
- **WS24**: Water stability dataset

## Processing Pipeline

The data processing includes the following steps:
1. Loading raw data from CSV files
2. Cleaning CIF structure files
3. Generating geometric and chemical features
4. Preparing crystal graph data for CGCNN
5. Creating train/validation/test splits
6. Merging features with property data

## Setup and Requirements

The data processing script requires the following dependencies:
- Python 3.8+
- pandas
- numpy
- scikit-learn
- ASE (Atomic Simulation Environment)
- PyMatGen (Python Materials Genomics)
- tqdm

Make sure your project environment has all required dependencies installed.

## Using the Data Processor

The project includes a dedicated data processor script located at `src/data/data_processor.py`. This script replaces the previously used Jupyter notebooks for data processing.

### Command Line Arguments

The data processor supports the following command line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--raw_data_dir` | Directory containing raw data files | `data/raw_data` |
| `--output_dir` | Directory to save processed data for CGCNN models | `data/cgcnn_data_` |
| `--ml_output_dir` | Directory to save processed data for ML models | `data/ml_data_` |
| `--log_dir` | Directory to save logs | `logs/data_processing` |
| `--dataset` | Dataset to process: `all`, `tsd_ssd`, or `ws24` | `all` |
| `--n_cpus` | Number of CPU cores to use for parallel processing | `4` |
| `--radius` | Radius for neighbor finding in crystal graph | `8.0` |
| `--max_num_nbr` | Maximum number of neighbors per atom | `10` |
| `--prob_radius` | Probe radius for geometric feature calculations | `1.86` |
| `--seed` | Random seed for reproducibility | `42` |

### Basic Usage

To process all datasets with default settings:

```bash
python src/data/data_processor.py
```

To process only the TSD and SSD datasets:

```bash
python src/data/data_processor.py --dataset tsd_ssd
```

To process only the WS24 dataset:

```bash
python src/data/data_processor.py --dataset ws24
```

To customize the number of CPU cores for parallel processing:

```bash
python src/data/data_processor.py --n_cpus 8
```

## Data Processing Workflow

### TSD and SSD Datasets

The processing workflow for TSD and SSD datasets follows these steps:

1. **Loading Raw Data**: 
   - Reads raw CSV files from the specified directory
   - Standardizes column names and formats

2. **CIF File Processing**:
   - Copies CIF files from the original dataset
   - Uses refcode as the MOF name identifier
   - Filters out MOFs without corresponding CIF files

3. **CIF Cleaning**:
   - Sanitizes CIF files to ensure they can be properly read by molecular modeling tools
   - Handles errors in CIF files that might prevent further processing

4. **Feature Generation**:
   - Generates RAC (Revised Autocorrelation) and zeolite-inspired features
   - Creates feature files in the specified output directory

5. **Graph Data Preparation**:
   - Prepares graph representation data for CGCNN models
   - Creates neighbor lists and atom type information

6. **Data Analysis**:
   - Computes statistics about atom counts and structure properties
   - Records these statistics in the log files

7. **Data Merging**:
   - Combines features with property data
   - Creates a final CSV file with all necessary information for ML models

The script automatically checks if any step has already been completed to avoid redundant processing, making it efficient for repeated runs.

### WS24 Dataset

The processing workflow for the WS24 dataset follows a similar pattern but includes some dataset-specific steps:

1. **Loading Raw Data**:
   - Reads the features and labels CSV file
   - Processes multi-class water stability labels (water4_label)

2. **CIF File Processing**:
   - Handles two different source directories (WS14s and WS24s)
   - Maps dataset names to appropriate CIF file locations

3. **Train/Val/Test Splitting**:
   - Performs stratified splitting based on multiple label columns
   - Maintains balanced distribution across all stability categories

4. **Label-Specific Processing**:
   - Creates separate datasets for each label type (water, water4, acid, base, boiling)
   - Simplifies model training for specific prediction tasks

The WS24 processing creates multiple output datasets, each optimized for a different prediction task.

## Output Structure

The data processor creates the following directory structure:

```
data/
├── cgcnn_data_/         # CGCNN model data
│   ├── TSD/
│   │   ├── cifs/                     # Original CIF files
│   │   ├── clean_cifs/               # Cleaned CIF files with graph data
│   │   ├── features/                 # Generated features
│   │   ├── id_prop_feat.csv          # Combined ID, property, and features
│   │   └── RAC_and_zeo_features_with_id_prop.csv  # Complete dataset
│   │
│   ├── SSD/             # Same structure as TSD
│   │
│   └── WS24/            # Similar structure with additional label files
│
└── ml_data_/           # Traditional ML model data
    ├── TSD/
    │   └── RAC_and_zeo_features_with_id_prop.csv  # Ready-to-use dataset
    │
    ├── SSD/             # Same structure as TSD
    │
    └── WS24/            # Separate directories for each label type
        ├── water_label/
        ├── water4_label/
        ├── acid_label/
        ├── base_label/
        └── boiling_label/
```

Logs for each processing step are stored in the `logs/data_processing/` directory.

## Performance and Optimization

The data processor includes several optimizations to improve performance:

1. **Caching Mechanism**: Checks for existing processed files to skip completed steps

2. **Parallel Processing**: Uses multiprocessing for computationally intensive tasks

3. **Incremental Updates**: Only updates output files when source files have changed

For large datasets, processing can take significant time. Consider using the `--dataset` flag to process only the datasets you need, and increase `--n_cpus` if more processing power is available.

## Troubleshooting

Common issues and their solutions:

1. **Missing CIF Files**: Check the `mofs_without_cif.txt` file in the output directory to see which MOFs were not found in the original dataset.

2. **Graph Data Preparation Failures**: Review the `mofs_prepare_failed.txt` file and the log files to identify structures that couldn't be processed.

3. **Memory Errors**: Reduce the number of parallel processes (`--n_cpus`) if you encounter memory issues.

For more detailed error information, check the log files in the `logs/data_processing/` directory.

## Dataset Reshuffling

The project includes a dedicated script for checking dataset overlaps and reshuffling train/validation/test splits to ensure consistency across all datasets. This is particularly important for multi-task learning where the same MOF structures appear in multiple datasets.

### Purpose of Reshuffling

The `reshuffle_splits.py` script serves several key functions:
- Creates new consistent partitions ensuring the same MOF structure has the same partition assignment across all datasets
- Generates multiple random seeds for robust model evaluation

### Features

The script provides the following functionalities:
- Maps between CoRE names and refcodes to ensure consistent MOF identification
- Creates stratified splits based on all available label columns
- Maintains proper distribution of different stability class labels across training, validation, and test sets
- Generates multiple dataset versions with different random seeds for model training.

### Using the Reshuffling Script

To reshuffle the datasets and create consistent train/validation/test splits:

```bash
python src/data/reshuffle_splits.py
```

The script automatically:
1. Loads the processed datasets from both ml_data and cgcnn_data directories
2. Creates new stratified splits with proper balance across all label types
3. Generates 5 different random seeds (0-4) for model training
4. Saves the reshuffled datasets with seed suffix (e.g., `RAC_and_zeo_features_with_id_prop_rand0.csv`)

### Output Files

For each random seed (0-4), the script creates the following files:
- In `data/ml_data/TSD/`: `RAC_and_zeo_features_with_id_prop_rand{seed}.csv`
- In `data/ml_data/SSD/`: `RAC_and_zeo_features_with_id_prop_rand{seed}.csv` 
- In `data/ml_data/WS24/`: `RAC_and_zeo_features_with_id_prop_rand{seed}.csv`
- Identical files are also saved in the corresponding `data/cgcnn_data/` directories

When training models that use multiple datasets, always use datasets with the same random seed to ensure proper separation of training, validation, and test data.