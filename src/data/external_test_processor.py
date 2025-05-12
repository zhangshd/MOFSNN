#!/usr/bin/env python
"""
External test set processing script for MOFSNN project.

This script processes external test sets for both Water Stability 24 (WS24v2) and 
Thermal/Solvent Stability (TS/SS) datasets into formats suitable for both ML and CGCNN models.
It handles data preprocessing, CIF file cleaning, and feature generation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
import logging
import subprocess
import time
from tqdm import tqdm
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.cif")
warnings.filterwarnings("ignore", category=UserWarning, module="moftransformer.utils.prepare_data")

# Get the directory of the script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Get the root directory of the project (two levels up from the script directory)
ROOT_DIR = SCRIPT_DIR.parent.parent
# Add the src directory to the Python path
sys.path.append(str(SCRIPT_DIR.parent))

# Import project-specific modules
try:
    from cgcnn.datamodule.prepare_data import make_prepared_data, get_logger
    from cgcnn.datamodule import clean_cif
    from data_processor import setup_logger
except ImportError:
    raise ImportError("Required project modules not found. Make sure the project structure is correct.")

def process_ws24v2_external_test(args):
    """
    Process the external test set for WS24v2 dataset.
    
    Args:
        args: Command line arguments containing paths and parameters
    
    Returns:
        bool: True if processing was successful
    """
    log_dir = Path(args.log_dir)
    log_file = log_dir / 'external_test_processing.log'
    logger = setup_logger('ws24v2_external_test_processor', log_file)
    logger.info("Starting WS24v2 external test set processing")
    
    # Define paths
    raw_data_dir = Path(args.raw_data_dir)
    ws24v2_ext_set_dir = raw_data_dir / "WS24v2/data_sets/validation_set"
    ws24v2_label_file = ws24v2_ext_set_dir / "sources_labels_v2.csv"
    
    # Output directories
    cgcnn_data_dir = Path(args.output_dir) / "WS24v2_external_test"
    ml_data_dir = Path(args.ml_output_dir) / "WS24v2_external_test"
    
    # Create output directories
    os.makedirs(cgcnn_data_dir, exist_ok=True)
    os.makedirs(ml_data_dir, exist_ok=True)
    cif_dir = cgcnn_data_dir / "cifs"
    clean_cif_dir = cgcnn_data_dir / "clean_cifs"
    os.makedirs(cif_dir, exist_ok=True)
    os.makedirs(clean_cif_dir, exist_ok=True)
    
    # Output files
    id_prop_file = cgcnn_data_dir / "id_prop.csv"
    features_file = cgcnn_data_dir / "RAC_and_zeo_features.csv"
    merged_file = cgcnn_data_dir / "RAC_and_zeo_features_with_id_prop.csv"
    ml_merged_file = ml_data_dir / "RAC_and_zeo_features_with_id_prop.csv"
    
    # Process id_prop data if not already done
    if not id_prop_file.exists():
        logger.info("Loading WS24v2 external test set data")
        df_ws = pd.read_csv(ws24v2_label_file)
        
        # Preprocess data
        df_ws.rename({
            'water_stability_label': 'water4_label',
            'file_name': 'MofName'
        }, axis=1, inplace=True)
        
        # Create binary water label (>2 = stable)
        df_ws["water_label"] = df_ws["water4_label"].apply(lambda x: 1 if x > 2 else 0)
        # Keep the 4-value label as is
        df_ws["water4_label"] = df_ws["water4_label"]
        # Remove extension from MOF names
        df_ws["MofName"] = df_ws["MofName"].apply(lambda x: x.split('.')[0])
        
        # Select and reorder columns
        valid_cols = ['MofName', 'MOF_name', "CCDC_refcode", 'water_label', 'water4_label', 
                       'acid_label', 'base_label', 'boiling_label']
        df_ws = df_ws[valid_cols]
        
        # Add partition column
        df_ws.insert(1, "Partition", "external_test")
        
        logger.info(f"WS24v2 external test set initial shape: {df_ws.shape}")
    else:
        logger.info("Loading existing id_prop file")
        df_ws = pd.read_csv(id_prop_file)
        logger.info(f"Loaded WS24v2 external test set with shape: {df_ws.shape}")
    
    # Copy CIF files if not already done
    all_cifs_exist = True
    for mof_name in df_ws['MofName']:
        if not (cif_dir / f"{mof_name}.cif").exists():
            all_cifs_exist = False
            break
            
    if all_cifs_exist:
        logger.info("All CIF files already exist, skipping copy step")
    else:
        logger.info("Copying CIF files to external test directory")
        failed_mofs = []
        for mof_name in tqdm(df_ws['MofName'], desc="Copying CIF files"):
            src_cif_file = ws24v2_ext_set_dir / f"CIFs/{mof_name}.cif"
            dst_cif_file = cif_dir / f"{mof_name}.cif"
            
            if not src_cif_file.exists():
                logger.warning(f"Source CIF file not found: {src_cif_file}")
                failed_mofs.append(mof_name)
                continue
                
            if not dst_cif_file.exists():
                shutil.copy(src_cif_file, dst_cif_file)
        
        # Remove MOFs without CIF files
        if failed_mofs:
            logger.warning(f"{len(failed_mofs)} MOFs without CIF files")
            df_ws = df_ws[~df_ws['MofName'].isin(failed_mofs)]
            logger.info(f"Updated dataset shape: {df_ws.shape}")
            
            # Save list of failed MOFs
            with open(cgcnn_data_dir / "mofs_without_cif.txt", "w") as f:
                f.write("\n".join(failed_mofs))
    
    # Save id_prop file if needed
    if not id_prop_file.exists():
        logger.info("Saving id_prop file")
        df_ws.to_csv(id_prop_file, index=False)
    
    # Clean CIF files if not already done
    clean_cifs_exist = True
    for cif_file in cif_dir.glob("*.cif"):
        clean_cif_file = clean_cif_dir / cif_file.name
        if not clean_cif_file.exists():
            clean_cifs_exist = False
            break
            
    if clean_cifs_exist:
        logger.info("All clean CIF files already exist, skipping cleaning step")
    else:
        logger.info("Cleaning CIF files")
        clean_cif.main(cif_dir, clean_cif_dir, log_file=cgcnn_data_dir / "clean_cif.log", 
                       santize=args.santize, n_cpus=args.n_cpus)
    
    # Prepare graph data if not already done
    graph_data_complete = True
    for cif_file in clean_cif_dir.glob("*.cif"):
        g_file_name = cif_file.stem + ".graphdata"
        if not (clean_cif_dir / g_file_name).exists():
            graph_data_complete = False
            break
            
    if graph_data_complete:
        logger.info("Graph data already exists, skipping graph data preparation")
    else:
        logger.info("Preparing graph data")
        py_logger = get_logger(filename=str(cgcnn_data_dir / "prepare_graph_data.log"))
        
        # Process structures
        failed_records = []
        for cif_file in tqdm(list(clean_cif_dir.glob("*.cif")), desc="Preparing graph data"):
            g_file_name = cif_file.stem + ".graphdata"
            if not (clean_cif_dir / g_file_name).exists():
                flag = make_prepared_data(cif_file, clean_cif_dir, radius=args.radius, 
                                          max_num_nbr=args.max_num_nbr, logger=py_logger)
                if not flag:
                    logger.warning(f"Failed to generate graph data for {cif_file}")
                    failed_records.append(cif_file.stem)
        
        if failed_records:
            logger.warning(f"Failed to prepare graph data for {len(failed_records)} structures")
            with open(cgcnn_data_dir / "mofs_prepare_failed.txt", "w") as f:
                f.write("\n".join(failed_records))
            
            # Update dataset by removing failed structures
            if failed_records:
                df_ws = df_ws[~df_ws['MofName'].isin(failed_records)]
                df_ws.to_csv(id_prop_file, index=False)
                logger.info(f"Updated dataset shape after removing failed structures: {df_ws.shape}")
    
    # Generate features if not already done
    if features_file.exists():
        logger.info("Feature file already exists, skipping feature generation")
    else:
        logger.info("Generating RACs and Zeo++ features")
        work_dir = ROOT_DIR / "src/ml/featuring"
        process = subprocess.Popen(
            f"python {work_dir/'feature_generation.py'} --cif_dir {clean_cif_dir} --prob_radius {args.prob_radius}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env=os.environ.copy(),
            cwd=str(work_dir)
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Feature generation failed with error: {stderr.decode()}")
        else:
            logger.info("Feature generation completed successfully")
    
    # Merge features with id_prop data if not already done
    if not merged_file.exists():
        logger.info("Merging features with id_prop data")
        if not features_file.exists():
            logger.error("Feature file not found, cannot merge with id_prop data")
            return False
            
        df_feat = pd.read_csv(features_file)
        df_feat.drop(columns=['cif_file'], inplace=True)
        df_feat.rename(columns={'name': 'MofName'}, inplace=True)
        
        # Select columns for merging
        id_prop_cols = ["MofName", "Partition", "water_label", "water4_label", 
                         "acid_label", "base_label", "boiling_label"]
        
        # Merge dataframes
        df_merged = df_ws[id_prop_cols].merge(df_feat, on="MofName", how="left")
        df_merged.to_csv(merged_file, index=False)
        logger.info(f"Merged data shape: {df_merged.shape}")
        
        # Copy to ML data directory
        os.makedirs(os.path.dirname(ml_merged_file), exist_ok=True)
        df_merged.to_csv(ml_merged_file, index=False)
        logger.info(f"Saved merged data to ML directory")
    else:
        logger.info("Merged feature file already exists")
    
    logger.info("WS24v2 external test set processing completed")
    return True

def process_ts_ss_external_test(args):
    """
    Process the external test set for Thermal Stability (TS) and Solvent Stability (SS) datasets.
    
    Args:
        args: Command line arguments containing paths and parameters
    
    Returns:
        bool: True if processing was successful
    """
    log_dir = Path(args.log_dir)
    log_file = log_dir / 'external_test_processing.log'
    logger = setup_logger('ts_ss_external_test_processor', log_file)
    logger.info("Starting TS/SS external test set processing")
    
    # Define paths
    raw_data_dir = Path(args.raw_data_dir)
    ts_ss_dir = raw_data_dir / "Nandy_2022_SciData/blinded_test_set"
    ts_ss_label_file = ts_ss_dir / "blinded_40_elsevier_MOFs.csv"
    core_mof_dir = raw_data_dir / "CoREMOF2019"
    
    # Output directories
    cgcnn_data_dir = Path(args.output_dir) / "TS_external_test"
    ml_data_dir = Path(args.ml_output_dir) / "TS_external_test"
    
    # Create output directories
    os.makedirs(cgcnn_data_dir, exist_ok=True)
    os.makedirs(ml_data_dir, exist_ok=True)
    cif_dir = cgcnn_data_dir / "cifs"
    clean_cif_dir = cgcnn_data_dir / "clean_cifs"
    os.makedirs(cif_dir, exist_ok=True)
    os.makedirs(clean_cif_dir, exist_ok=True)
    
    # Output files
    id_prop_file = cgcnn_data_dir / "id_prop.csv"
    features_file = cgcnn_data_dir / "RAC_and_zeo_features.csv"
    merged_file = cgcnn_data_dir / "RAC_and_zeo_features_with_id_prop.csv"
    ml_merged_file = ml_data_dir / "RAC_and_zeo_features_with_id_prop.csv"
    
    # Process id_prop data if not already done
    if not id_prop_file.exists():
        logger.info("Loading TS/SS external test set data")
        df_ts = pd.read_csv(ts_ss_label_file)
        
        # Calculate metrics from original data if available
        if 'label (solvent removal stability)' in df_ts.columns and 'predicted label (solvent removal stability)' in df_ts.columns:
            from sklearn import metrics
            ss_acc = metrics.accuracy_score(df_ts['label (solvent removal stability)'], 
                                          df_ts['predicted label (solvent removal stability)'])
            ss_auc = metrics.roc_auc_score(df_ts['label (solvent removal stability)'], 
                                        df_ts['predicted probability (solvent removal stability)'])
            ts_r2 = metrics.r2_score(df_ts['label (thermal stability)'], 
                                   df_ts['predicted Td (thermal stability)'])
            ts_mae = metrics.mean_absolute_error(df_ts['label (thermal stability)'], 
                                            df_ts['predicted Td (thermal stability)'])
            
            logger.info('Solvent removal stability accuracy: {:.4f}'.format(ss_acc))
            logger.info('Solvent removal stability AUC: {:.4f}'.format(ss_auc))
            logger.info('Thermal stability R^2: {:.4f}'.format(ts_r2))
            logger.info('Thermal stability MAE: {:.4f}'.format(ts_mae))
        
        # Preprocess data
        df_ts.rename({
            'label (solvent removal stability)': 'ss_label',
            'label (thermal stability)': 'ts_label',
            'CoRE_name': 'MofName'
        }, axis=1, inplace=True)
        
        # Select and reorder columns
        valid_cols = ['MofName', "refcode", 'ts_label', 'ss_label']
        df_ts = df_ts[valid_cols]
        
        # Add partition column
        df_ts.insert(1, "Partition", "external_test")
        
        # Calculate binary thermal stability label (≥359°C = stable)
        df_ts["ts2_label"] = (df_ts["ts_label"] >= 359).astype(int)
        
        logger.info(f"TS/SS external test set shape: {df_ts.shape}")
    else:
        logger.info("Loading existing id_prop file")
        df_ts = pd.read_csv(id_prop_file)
        logger.info(f"Loaded TS/SS external test set with shape: {df_ts.shape}")
    
    # Copy CIF files if not already done
    all_cifs_exist = True
    for mof_name in df_ts['MofName']:
        if not (cif_dir / f"{mof_name}.cif").exists():
            all_cifs_exist = False
            break
            
    if all_cifs_exist:
        logger.info("All CIF files already exist, skipping copy step")
    else:
        logger.info("Copying CIF files to external test directory")
        failed_mofs = []
        for mof_name in tqdm(df_ts['MofName'], desc="Copying CIF files"):
            src_cif_file = core_mof_dir / f"{mof_name}.cif"
            dst_cif_file = cif_dir / f"{mof_name}.cif"
            
            if not src_cif_file.exists():
                logger.warning(f"Source CIF file not found: {src_cif_file}")
                failed_mofs.append(mof_name)
                continue
                
            if not dst_cif_file.exists():
                shutil.copy(src_cif_file, dst_cif_file)
        
        # Remove MOFs without CIF files
        if failed_mofs:
            logger.warning(f"{len(failed_mofs)} MOFs without CIF files")
            df_ts = df_ts[~df_ts['MofName'].isin(failed_mofs)]
            logger.info(f"Updated dataset shape: {df_ts.shape}")
            
            # Save list of failed MOFs
            with open(cgcnn_data_dir / "mofs_without_cif.txt", "w") as f:
                f.write("\n".join(failed_mofs))
    
    # Save id_prop file if needed
    if not id_prop_file.exists():
        logger.info("Saving id_prop file")
        df_ts.to_csv(id_prop_file, index=False)
    
    # Clean CIF files if not already done
    clean_cifs_exist = True
    for cif_file in cif_dir.glob("*.cif"):
        clean_cif_file = clean_cif_dir / cif_file.name
        if not clean_cif_file.exists():
            clean_cifs_exist = False
            break
            
    if clean_cifs_exist:
        logger.info("All clean CIF files already exist, skipping cleaning step")
    else:
        logger.info("Cleaning CIF files")
        clean_cif.main(cif_dir, clean_cif_dir, log_file=cgcnn_data_dir / "clean_cif.log", 
                       santize=args.santize, n_cpus=args.n_cpus)
    
    # Prepare graph data if not already done
    graph_data_complete = True
    for cif_file in clean_cif_dir.glob("*.cif"):
        g_file_name = cif_file.stem + ".graphdata"
        if not (clean_cif_dir / g_file_name).exists():
            graph_data_complete = False
            break
            
    if graph_data_complete:
        logger.info("Graph data already exists, skipping graph data preparation")
    else:
        logger.info("Preparing graph data")
        py_logger = get_logger(filename=str(cgcnn_data_dir / "prepare_graph_data.log"))
        
        # Process structures
        failed_records = []
        for cif_file in tqdm(list(clean_cif_dir.glob("*.cif")), desc="Preparing graph data"):
            g_file_name = cif_file.stem + ".graphdata"
            if not (clean_cif_dir / g_file_name).exists():
                flag = make_prepared_data(cif_file, clean_cif_dir, radius=args.radius, 
                                          max_num_nbr=args.max_num_nbr, logger=py_logger)
                if not flag:
                    logger.warning(f"Failed to generate graph data for {cif_file}")
                    failed_records.append(cif_file.stem)
        
        if failed_records:
            logger.warning(f"Failed to prepare graph data for {len(failed_records)} structures")
            with open(cgcnn_data_dir / "mofs_prepare_failed.txt", "w") as f:
                f.write("\n".join(failed_records))
            
            # Update dataset by removing failed structures
            if failed_records:
                df_ts = df_ts[~df_ts['MofName'].isin(failed_records)]
                df_ts.to_csv(id_prop_file, index=False)
                logger.info(f"Updated dataset shape after removing failed structures: {df_ts.shape}")
    
    # Generate features if not already done
    if features_file.exists():
        logger.info("Feature file already exists, skipping feature generation")
    else:
        logger.info("Generating RACs and Zeo++ features")
        work_dir = ROOT_DIR / "src/ml/featuring"
        process = subprocess.Popen(
            f"python {work_dir/'feature_generation.py'} --cif_dir {clean_cif_dir} --prob_radius {args.prob_radius}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env=os.environ.copy(),
            cwd=str(work_dir)
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Feature generation failed with error: {stderr.decode()}")
        else:
            logger.info("Feature generation completed successfully")
    
    # Merge features with id_prop data if not already done
    if not merged_file.exists():
        logger.info("Merging features with id_prop data")
        if not features_file.exists():
            logger.error("Feature file not found, cannot merge with id_prop data")
            return False
            
        df_feat = pd.read_csv(features_file)
        df_feat.drop(columns=['cif_file'], inplace=True)
        df_feat.rename(columns={'name': 'MofName'}, inplace=True)
        
        # Select columns for merging
        id_prop_cols = ["MofName", "Partition", "ts_label", "ts2_label", "ss_label"]
        
        # Merge dataframes
        df_merged = df_ts[id_prop_cols].merge(df_feat, on="MofName", how="left")
        df_merged.dropna(axis=0, how='any', inplace=True)  # Remove rows with NaN values
        df_merged.to_csv(merged_file, index=False)
        logger.info(f"Merged data shape: {df_merged.shape}")
        
        # Copy to ML data directory
        os.makedirs(os.path.dirname(ml_merged_file), exist_ok=True)
        df_merged.to_csv(ml_merged_file, index=False)
        logger.info(f"Saved merged data to ML directory")
    else:
        logger.info("Merged feature file already exists")
    
    logger.info("TS/SS external test set processing completed")
    return True

def main():
    """Main function to process external test sets."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process external test sets for MOFSNN project.')
    parser.add_argument('--raw_data_dir', type=str, default=str(ROOT_DIR / 'data/raw_data'),
                        help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str, default=str(ROOT_DIR / 'data/cgcnn_data_'),
                        help='Directory to save processed data for CGCNN models')
    parser.add_argument('--ml_output_dir', type=str, default=str(ROOT_DIR / 'data/ml_data_'),
                        help='Directory to save processed data for ML models')
    parser.add_argument('--log_dir', type=str, default=str(ROOT_DIR / 'logs/data_processing'),
                        help='Directory to save logs')
    parser.add_argument('--dataset', type=str, choices=['all', 'ws24v2', 'ts_ss'], default='all',
                        help='Dataset to process: all, ws24v2, or ts_ss')
    parser.add_argument('--n_cpus', type=int, default=4,
                        help='Number of CPU cores to use for parallel processing')
    parser.add_argument('--radius', type=float, default=8.0,
                        help='Radius for neighbor finding in crystal graph')
    parser.add_argument('--max_num_nbr', type=int, default=10,
                        help='Maximum number of neighbors per atom')
    parser.add_argument('--prob_radius', type=float, default=1.86,
                        help='Probe radius for geometric feature calculations')
    parser.add_argument('--santize', action='store_true',
                        help='Whether to sanitize CIF files (remove disorder)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    for dir_path in [args.log_dir, args.output_dir, args.ml_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set up main logger
    log_file = Path(args.log_dir) / 'external_test_processing.log'
    logger = setup_logger('external_test_processor', log_file)
    logger.info("Starting external test set processing with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Process datasets based on arguments
    success = True
    if args.dataset in ['all', 'ws24v2']:
        logger.info("Processing WS24v2 external test set")
        success = process_ws24v2_external_test(args) and success
    
    if args.dataset in ['all', 'ts_ss']:
        logger.info("Processing TS/SS external test set")
        success = process_ts_ss_external_test(args) and success
    
    if success:
        logger.info("External test set processing completed successfully!")
    else:
        logger.error("External test set processing completed with errors")

if __name__ == "__main__":
    main()
