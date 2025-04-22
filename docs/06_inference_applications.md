# Inference and Applications

This document details the inference capabilities and practical applications of the MOFSNN framework for predicting MOF stability properties in real-world scenarios.

## Complete Workflow for MOF Screening

For researchers who want to use MOFSNN for MOF screening, here is a complete step-by-step workflow:

### Step 1: Data Preparation

1. **Gather CIF Files**:
   ```bash
   # Create directory for your CIF files
   mkdir -p data/my_screening/cif_files
   
   # Copy your CIF files to this directory
   cp path/to/your/cif/files/*.cif data/my_screening/cif_files/
   ```

2. **Clean CIF Files** (optional but recommended):
   ```python
   from src.cgcnn.datamodule.clean_cif import clean_cif
   from pathlib import Path
   import os
   
   # Set up directories
   cif_dir = Path("data/my_screening/cif_files")
   clean_dir = Path("data/my_screening/clean_cifs")
   os.makedirs(clean_dir, exist_ok=True)
   
   # Clean all CIF files
   for cif_file in cif_dir.glob("*.cif"):
       clean_cif(cif_file, clean_dir / cif_file.name)
   ```

### Step 2: Run Inference

1. **Load Trained Models**:
   ```python
   from src.cgcnn.inference import inference
   from pathlib import Path
   import pandas as pd
   
   # Set paths
   cif_dir = Path("data/my_screening/clean_cifs")
   model_dir = Path("results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43")
   output_dir = Path("results/my_screening")
   os.makedirs(output_dir, exist_ok=True)
   
   # Get all CIF files
   cif_files = list(cif_dir.glob("*.cif"))
   ```

2. **Run Inference**:
   ```python
   # For small datasets
   results = inference(cif_files, model_dir, saved_dir=output_dir)
   
   # For large datasets (batch processing)
   batch_size = 100
   all_results = []
   
   for i in range(0, len(cif_files), batch_size):
       batch_files = cif_files[i:i+batch_size]
       batch_results = inference(batch_files, model_dir, saved_dir=output_dir)
       all_results.append(batch_results)
       print(f"Processed batch {i//batch_size + 1}/{(len(cif_files) + batch_size - 1)//batch_size}")
   
   # Combine results
   results = pd.concat([pd.DataFrame(r) for r in all_results])
   results.to_csv(output_dir / "all_predictions.csv", index=False)
   ```

### Step 3: Filter and Analyze Results

1. **Apply Stability Criteria**:
   ```python
   # Load prediction results
   df = pd.read_csv(output_dir / "all_predictions.csv")
   
   # Apply filters for different stability criteria
   stable_mofs = df[
       (df["TSD_pred"] > 300) &  # Thermal stability > 300°C
       (df["SSD_pred"] == 1) &   # Solvent stable
       (df["WS24_water_pred"] == 1)  # Water stable
   ]
   
   # Export filtered results
   stable_mofs.to_csv(output_dir / "stable_mofs.csv", index=False)
   print(f"Found {len(stable_mofs)} stable MOFs out of {len(df)} total")
   ```

2. **Extract Top Candidates**:
   ```python
   # Sort by thermal stability (for example)
   top_candidates = stable_mofs.sort_values("TSD_pred", ascending=False).head(10)
   
   # Create directory for top candidates
   top_dir = output_dir / "top_candidates"
   os.makedirs(top_dir, exist_ok=True)
   
   # Copy CIF files of top candidates
   for mof_id in top_candidates["cif_ids"]:
       src_file = cif_dir / f"{mof_id}.cif"
       if src_file.exists():
           shutil.copy(src_file, top_dir / f"{mof_id}.cif")
   
   # Export summary of top candidates
   top_candidates.to_csv(top_dir / "top_candidates_summary.csv", index=False)
   ```

### Step 4: Visualize Results

1. **Create Distribution Plots**:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Set up plotting
   plt.figure(figsize=(12, 8))
   
   # Plot thermal stability distribution
   plt.subplot(2, 2, 1)
   sns.histplot(df["TSD_pred"], kde=True)
   plt.axvline(x=300, color='r', linestyle='--')
   plt.title("Thermal Stability Distribution")
   plt.xlabel("Decomposition Temperature (°C)")
   
   # Plot water stability distribution
   plt.subplot(2, 2, 2)
   sns.countplot(x=df["WS24_water_pred"])
   plt.title("Water Stability Distribution")
   plt.xlabel("Stable (1) vs Unstable (0)")
   
   # Save figure
   plt.tight_layout()
   plt.savefig(output_dir / "stability_distributions.png", dpi=300)
   ```

2. **Analysis Report Generation**:
   ```python
   # Create a simple analysis report
   with open(output_dir / "screening_report.txt", "w") as f:
       f.write(f"MOF Screening Analysis Report\n")
       f.write(f"==========================\n\n")
       f.write(f"Total MOFs analyzed: {len(df)}\n")
       f.write(f"Thermally stable MOFs (>300°C): {len(df[df['TSD_pred'] > 300])}\n")
       f.write(f"Solvent stable MOFs: {len(df[df['SSD_pred'] == 1])}\n")
       f.write(f"Water stable MOFs: {len(df[df['WS24_water_pred'] == 1])}\n")
       f.write(f"MOFs stable in all conditions: {len(stable_mofs)}\n\n")
       f.write(f"Top 5 thermally stable MOFs:\n")
       for i, (idx, row) in enumerate(top_candidates.head(5).iterrows()):
           f.write(f"{i+1}. {row['cif_ids']}: {row['TSD_pred']:.1f}°C\n")
   ```

## Running Comparative Analyses with ML Models

To compare CGCNN and traditional ML model predictions:

```python
# Load ML model
from src.ml.module import RegressionModel, ClassificationModel
import joblib

ml_model_path = "results/ml_models/TSD/total_model_f1100_normal5_RandomForestRegressor_42.model"
ml_model = joblib.load(ml_model_path)

# Feature extraction for ML models
from src.ml.featuring import feature_generation as fg

# Generate features for your CIF files
features_df = fg.generate_features_from_cifs(cif_files)

# Make predictions with ML model
ml_predictions = ml_model.predict(features_df)

# Compare with CGCNN predictions
comparison_df = pd.DataFrame({
    "MOF_ID": features_df.index,
    "ML_prediction": ml_predictions,
    "CGCNN_prediction": results["TSD_pred"]
})

# Calculate correlation
correlation = comparison_df[["ML_prediction", "CGCNN_prediction"]].corr().iloc[0, 1]
print(f"Correlation between ML and CGCNN predictions: {correlation:.4f}")
```

## Screening Applications

### MOF Database Screening

The project includes notebooks for large-scale screening of MOF databases:

1. **CoREMOF Screening** (`14-screenning_coremof.ipynb`):
   - Application of trained models to screen the entire CoREMOF database
   - Ranking of MOFs by predicted stability properties
   - Identification of top candidates for specific applications

2. **Filtering Process**:
   - Progressive filtering based on multiple stability criteria (thermal, solvent, water stability)
   - Uncertainty-based filtering to prioritize reliable predictions
   - Exclusion of training samples to focus on novel predictions

3. **Results Organization**:
   - Export of top candidates to CSV files
   - Creation of directories with selected CIF files
   - Systematic tracking of filtering stages

### GCMC Simulations

The integration with Gas Adsorption simulations provides additional insights:

1. **MOF GCMC Analysis** (`15-mof_gcmc.ipynb`):
   - Grand Canonical Monte Carlo simulations for gas adsorption in MOFs
   - Correlation between stability properties and adsorption performance
   - Multi-objective screening considering both stability and functionality

2. **TSA Applications** (`16-gcmc_screen4_tsa.ipynb`):
   - Temperature Swing Adsorption screening
   - Stability requirements for thermal cycling applications
   - Performance-stability trade-off analysis

## Model Deployment

The MOFSNN project provides two distinct approaches for model deployment:

1. **Inference API (`inference.py`)**:
   - Core functionality for processing raw CIF files and making predictions
   - Handles CIF cleaning, graph generation, and model inference
   - Provides uncertainty estimates when available

2. **Example Usage**:
   ```python
   from src.cgcnn.inference import inference
   from pathlib import Path
   
   # Single structure prediction
   cif_path = Path("path/to/structure.cif")
   model_dir = Path("results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn/version_43")
   saved_dir = Path("results/inference/single_cif")
   
   # Optional: uncertainty estimation using pre-built trees
   uncertainty_trees_file = Path("results/evaluation/TSD_SSD_WS24_water_WS24_water4_seed42_att_cgcnn@version_43/uncertainty_trees.pkl")
   
   # Run inference with default CIF cleaning (clean=True)
   results = inference([cif_path], model_dir, saved_dir=saved_dir, 
                      uncertainty_trees_file=uncertainty_trees_file)
   
   # Batch prediction with pre-cleaned CIF files (disable cleaning)
   cif_dir = Path("path/to/clean_cifs")
   cif_list = list(cif_dir.glob("*.cif"))
   batch_results = inference(cif_list, model_dir, saved_dir=saved_dir, clean=False)
   
   # Access results (dictionary with predictions and uncertainties)
   for task in batch_results:
       if task.endswith("_pred"):  # Prediction values
           print(f"{task}: {batch_results[task]}")
       elif task.endswith("_uncertainty"):  # Uncertainty estimates (if provided)
           print(f"{task}: {batch_results[task]}")
   ```

3. **Slurm Integration**:
   - Scripts for running inference on HPC clusters
   - Parallel processing of large datasets
   - Resource allocation management for efficient computation:
   
   ```python
   # Example slurm script from 14-screenning_coremof.ipynb
   slrum_template = """#!/bin/bash
   #SBATCH --job-name={job_name}
   #SBATCH --output={work_dir}/%x_%A.out
   #SBATCH --error={work_dir}/%x_%A.err
   #SBATCH --partition=C9654 
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task={n_cpus}
   #SBATCH --mem-per-gpu=100G
   #SBATCH --gres=gpu:1
   
   export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
   export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH
   
   srun python -u {python_script} --cif_dir {src_cif_dir} --n_cpus {n_cpus}
   """
   python_script = Path("CGCNN_MT/inference.py").absolute()
   ```

## Practical Applications

### Database Screening Process

The MOF database screening process implemented in the project follows these steps:

1. **Data Preprocessing**:
   - Cleaning CIF files from the CoREMOF database
   - Standardizing atom representations
   - Removing solvent molecules

2. **Multi-Criteria Screening**:
   - Setting thermal stability cutoffs (e.g., >300°C)
   - Filtering by predicted solvent and water stability
   - Applying uncertainty thresholds to ensure prediction reliability

3. **Post-Screening Analysis**:
   - Removing known structures from the training set
   - Organizing candidates based on specific application requirements
   - Extracting structure files for further analysis

### Performance-Based Selection

The project enables performance-based selection of MOFs for specific applications:

1. **Stability-Focused Selection**:
   - Identification of thermally stable MOFs (>300°C)
   - Selection of water-stable materials
   - Prioritization of MOFs with multiple favorable stability properties

2. **Application-Specific Filtering**:
   - Targeting specific gas adsorption applications
   - Custom filtering based on multiple property thresholds
   - Integration with adsorption performance metrics

3. **Uncertainty-Aware Selection**:
   - Preference for predictions with low uncertainty scores
   - Confidence-based ranking of candidate materials
   - Balancing performance predictions with prediction reliability

## Industrial Relevance

The screening capabilities have several industrial applications:

1. **Material Discovery**:
   - Accelerating the discovery of stable MOFs for commercial applications
   - Reducing experimental testing through computational pre-screening
   - Targeting specific stability profiles for specialized applications

2. **Application-Specific Selection**:
   - Gas storage and separation
   - Carbon capture
   - Chemical sensing and catalysis

3. **Resource Optimization**:
   - Prioritizing experimental validation efforts
   - Focusing synthesis attempts on promising candidates
   - Reducing time and cost in materials development

## Future Extensions

The inference and screening capabilities can be extended in several ways:

1. **Enhanced Screening Criteria**:
   - Integration of additional stability properties
   - Incorporation of mechanical stability predictions
   - Multi-objective optimization across various property dimensions

2. **Improved Deployment Options**:
   - Web-based interfaces for broader accessibility
   - Integration with materials databases
   - Containerized deployment for reproducible prediction environments

3. **Expanded Applications**:
   - Specialized screening for emerging applications
   - Integration with synthesis prediction models
   - Connection to automated experimental platforms

## Conclusion

The MOFSNN framework provides powerful tools for applying stability predictions to practical MOF screening challenges:

1. The inference pipeline enables efficient prediction of stability properties for large sets of MOF structures
2. Database screening capabilities identify promising stable MOF candidates from large structure libraries
3. Integration with adsorption simulations connects stability predictions to functional performance
4. The systematic filtering approach balances prediction performance with confidence metrics

These capabilities demonstrate the practical utility of the MOFSNN framework for accelerating the discovery and deployment of stable MOFs across various technological applications.