#!/usr/bin/env python
"""
Data processing script for MOFSNN project.

This script processes raw MOF data into formats suitable for both ML and CGCNN models.
It handles TSD, SSD, and WS24 datasets, including cleaning of CIF files, feature 
generation, and train/validation/test splits.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
import logging
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import train_test_split
import pickle
import warnings
import subprocess
from tqdm import tqdm

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
    from ml.featuring.feature_generation import delete_and_remake_folders
except ImportError:
    raise ImportError("Required project modules not found. Make sure the project structure is correct.")

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_atom_num(cif_file):
    """Get the number of atoms in a CIF file."""
    from ase.io import read
    try:
        atoms = read(cif_file)
        return len(atoms)
    except Exception as e:
        return None

def process_tsd_ssd(args):
    """Process TSD and SSD datasets."""
    log_dir = Path(args.log_dir)
    log_file = log_dir / 'tsd_ssd_processing.log'
    logger = setup_logger('tsd_ssd_processor', log_file)
    logger.info("Starting TSD and SSD data processing")
    
    # Define paths
    raw_data_dir = Path(args.raw_data_dir)
    ssd_csv = raw_data_dir / "Nandy_2022_SciData/separate_files/solvent_removal_stability/full_SSD_data.csv"
    tsd_csv = raw_data_dir / "Nandy_2022_SciData/separate_files/thermal_stability/full_TSD_data.csv"
    src_cif_dir = raw_data_dir / "CoREMOF2019"
    
    # Create output directories
    saved_dir = Path(args.output_dir)
    for dataset in ["TSD", "SSD"]:
        for subdir in ["cifs", "clean_cifs", "features"]:
            os.makedirs(saved_dir / dataset / subdir, exist_ok=True)
    
    logger.info("Loading raw data files")
    df_ssd = pd.read_csv(ssd_csv)
    df_tsd = pd.read_csv(tsd_csv)
    df_ssd.dropna(inplace=True)
    df_tsd.dropna(inplace=True)
    
    logger.info(f"SSD dataset shape: {df_ssd.shape}")
    logger.info(f"TSD dataset shape: {df_tsd.shape}")
    
    # Standardize column names
    # Keep both refcode and CoRE_name columns for file mapping
    select_cols1 = ["refcode", "CoRE_name", "assigned_solvent_removal_stability", "partition"]
    select_cols2 = ["refcode", "CoRE_name", "assigned_T_decomp (°C)", "partition"]
    feat_cols = df_ssd.loc[:, "Df (Å)":"D_func-S-3-all"].columns.tolist()
    
    df_ssd = df_ssd[select_cols1 + feat_cols]
    df_tsd = df_tsd[select_cols2 + feat_cols]
    
    # Process each dataset
    for df, task in zip([df_ssd, df_tsd], ["SSD", "TSD"]):
        task_dir = saved_dir / task
        cif_dir = task_dir / "cifs"
        clean_cif_dir = task_dir / "clean_cifs"
        
        # Set refcode as MofName and keep CoRE_name for file lookup
        df.rename(columns={
            "refcode": "MofName", 
            "assigned_solvent_removal_stability" if "assigned_solvent_removal_stability" in df.columns 
            else "assigned_T_decomp (°C)": "Label",
            "partition": "Partition"
        }, inplace=True)
        
        # Check if processing is already completed
        id_prop_feat_file = task_dir / "id_prop_feat.csv"
        rac_zeo_id_prop_file = task_dir / "RAC_and_zeo_features_with_id_prop.csv"
        
        if id_prop_feat_file.exists():
            logger.info(f"{task} id_prop_feat.csv already exists, loading from file")
            df = pd.read_csv(id_prop_feat_file)
        else:
            logger.info(f"Saving {task} dataset metadata")
            df.to_csv(id_prop_feat_file, index=False)
        
        # Copy and clean CIF files if not already done
        all_cifs_exist = True
        for i, row in df.iterrows():
            dst_cif_file = cif_dir / f"{row['MofName']}.cif"
            if not dst_cif_file.exists():
                all_cifs_exist = False
                break
                
        if all_cifs_exist:
            logger.info(f"All CIF files for {task} already exist, skipping copy step")
        else:
            # Copy CIF files
            logger.info(f"Processing CIF files for {task}")
            cif_files_to_process = []
            mofs_without_cif = []
            
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying CIF files for {task}"):
                # Use CoRE_name for source file lookup, but refcode (as MofName) for destination filename
                core_name = row['CoRE_name']
                mof_name = row['MofName']
                
                src_cif_file = src_cif_dir / f"{core_name}.cif"
                dst_cif_file = cif_dir / f"{mof_name}.cif"
                
                if not src_cif_file.exists():
                    logger.warning(f"Source CIF file not found: {src_cif_file}")
                    mofs_without_cif.append(mof_name)
                    continue
                
                if not dst_cif_file.exists():
                    shutil.copy(src_cif_file, dst_cif_file)
                    cif_files_to_process.append(dst_cif_file)
            
            if mofs_without_cif:
                logger.warning(f"{len(mofs_without_cif)} MOFs without CIF files in {task}")
                with open(task_dir / "mofs_without_cif.txt", "w") as f:
                    f.write("\n".join(mofs_without_cif))
                
                # Remove entries without CIF files from the dataframe
                df = df[~df['MofName'].isin(mofs_without_cif)]
                df.to_csv(id_prop_feat_file, index=False)
                logger.info(f"Updated dataset shape after removing MOFs without CIF: {df.shape}")
        
        # Clean CIF files if not already done
        clean_cifs_exist = True
        for cif_file in cif_dir.glob("*.cif"):
            clean_cif_file = clean_cif_dir / cif_file.name
            if not clean_cif_file.exists():
                clean_cifs_exist = False
                break
                
        if clean_cifs_exist:
            logger.info(f"All clean CIF files for {task} already exist, skipping cleaning step")
        else:
            # Clean CIF files
            logger.info(f"Cleaning CIF files for {task}")
            clean_cif.main(cif_dir, clean_cif_dir, log_file=task_dir / "clean_cif.log", santize=True, n_cpus=args.n_cpus)
        
        # Generate features if not already done
        feature_file = task_dir / "RAC_and_zeo_features.csv"
        if feature_file.exists():
            logger.info(f"Feature file for {task} already exists, skipping feature generation")
        else:
            # Generate features using feature_generation.py
            logger.info(f"Generating features for {task}")
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
                logger.error(f"Feature generation for {task} failed with error: {stderr.decode()}")
                continue
            else:
                logger.info(f"Feature generation for {task} completed successfully")
            
        # Prepare graph data if not already done
        graph_data_complete = True
        for cif_file in clean_cif_dir.glob("*.cif"):
            g_file_name = cif_file.stem + ".graphdata"
            if not (clean_cif_dir / g_file_name).exists():
                graph_data_complete = False
                break
                
        if graph_data_complete:
            logger.info(f"Graph data for {task} already exists, skipping graph data preparation")
        else:
            # Prepare graph data
            logger.info(f"Preparing graph data for {task}")
            py_logger = get_logger(filename=str(task_dir / f"prepare_graph_data_{task}.log"))
            
            # Process structures that don't have graph data yet
            for cif_file in tqdm(list(clean_cif_dir.glob("*.cif")), desc=f"Preparing graph data for {task}"):
                g_file_name = cif_file.stem + ".graphdata"
                if not (clean_cif_dir / g_file_name).exists():
                    flag = make_prepared_data(cif_file, clean_cif_dir, radius=args.radius, 
                                            max_num_nbr=args.max_num_nbr, logger=py_logger)
                    if not flag:
                        logger.warning(f"Failed to generate graph data for {cif_file}")
        
        # Analyze atom numbers in structures
        if not (task_dir / "atom_stats.txt").exists():
            atom_nums = []
            for cif_file in tqdm(list(cif_dir.glob("*.cif")), desc=f"Analyzing atoms in {task}"):
                atom_num = get_atom_num(cif_file)
                if atom_num is not None:
                    atom_nums.append(atom_num)
            
            if atom_nums:
                atom_stats = f"{task} atom count statistics: Min={min(atom_nums)}, Max={max(atom_nums)}, " \
                            f"Mean={np.mean(atom_nums):.1f}, Median={np.median(atom_nums)}"
                logger.info(atom_stats)
                
                # Save atom stats for future reference
                with open(task_dir / "atom_stats.txt", "w") as f:
                    f.write(atom_stats)
            else:
                logger.warning(f"No atom statistics could be calculated for {task}")
        else:
            with open(task_dir / "atom_stats.txt", "r") as f:
                logger.info(f.read())
        
        # Merge features with id_prop for this task immediately
        if rac_zeo_id_prop_file.exists():
            logger.info(f"Merged features file for {task} already exists")
            df_new = pd.read_csv(rac_zeo_id_prop_file)
        else:
            logger.info(f"Merging features with ID-property data for {task}")
            df_new = pd.read_csv(task_dir / "RAC_and_zeo_features.csv")
            df = pd.read_csv(id_prop_feat_file)
            
            df_new.drop(columns=['cif_file'], inplace=True)
            df_new.rename(columns={'name': 'MofName'}, inplace=True)
            
            # Merge dataframes
            df_new = df[["MofName", "Label", "Partition"]].merge(df_new, on="MofName", how="right")
            df_new.to_csv(rac_zeo_id_prop_file, index=False)
        
        # Create directory for ML data and save a copy
        ml_data_dir = Path(args.ml_output_dir) / task
        os.makedirs(ml_data_dir, exist_ok=True)
        
        # Save a copy for ML models if not already there
        ml_output_file = ml_data_dir / "RAC_and_zeo_features_with_id_prop.csv"
        if not ml_output_file.exists() or os.path.getmtime(rac_zeo_id_prop_file) > os.path.getmtime(ml_output_file):
            shutil.copy(rac_zeo_id_prop_file, ml_output_file)
            logger.info(f"Copied merged features to ML data directory for {task}")
        
        logger.info(f"Completed processing for {task}, final shape: {df_new.shape}")
    
    logger.info("TSD and SSD data processing completed")
    return True

def process_ws24(args):
    """Process WS24 dataset."""
    log_dir = Path(args.log_dir)
    log_file = log_dir / 'ws24_processing.log'
    logger = setup_logger('ws24_processor', log_file)
    logger.info("Starting WS24 data processing")
    
    # Define paths
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    ml_output_dir = Path(args.ml_output_dir)
    
    src_csv = raw_data_dir / "WS24v2/features/features_and_labels.csv"
    src_cif_dir1 = raw_data_dir / "WS24v2/data_sets/WS14s/CIFs"
    src_cif_dir2 = raw_data_dir / "WS24v2/data_sets/WS24s/CIFs"
    tgt_root_dir = output_dir / "WS24"
    
    # Create output directories
    os.makedirs(tgt_root_dir, exist_ok=True)
    tgt_cif_dir = tgt_root_dir / "cifs"
    clean_cif_dir = tgt_root_dir / "clean_cifs"
    os.makedirs(tgt_cif_dir, exist_ok=True)
    os.makedirs(clean_cif_dir, exist_ok=True)
    
    id_prop_feat_file = tgt_root_dir / "id_prop_feat.csv"
    rac_zeo_features_file = tgt_root_dir / "RAC_and_zeo_features.csv"
    rac_zeo_id_prop_file = tgt_root_dir / "RAC_and_zeo_features_with_id_prop.csv"
    
    # Check if processing is already completed
    if id_prop_feat_file.exists() and rac_zeo_id_prop_file.exists():
        logger.info("WS24 dataset processing already completed, loading from files")
        total_df = pd.read_csv(id_prop_feat_file)
        logger.info(f"Loaded dataset with shape: {total_df.shape}")
    else:
        logger.info("Loading raw data files")
        in_df = pd.read_csv(src_csv)
        logger.info(f"WS24 dataset shape: {in_df.shape}")
        
        # Add water4 class label
        in_df.insert(3, "water4_label", in_df["water_label"].astype(int))
        in_df["water_label"] = in_df["water4_label"].apply(lambda x: 1 if x > 2 else 0)
        in_df.rename(columns={"MOF_name": "MofName"}, inplace=True)
        
        # Check if all CIF files are already copied
        all_cifs_exist = True
        for i in range(len(in_df)):
            n = in_df.iloc[i]["MofName"]
            if not (tgt_cif_dir / f"{n}.cif").exists():
                all_cifs_exist = False
                break
                
        if all_cifs_exist:
            logger.info("All CIF files for WS24 already exist, skipping copy step")
        else:
            # Copy CIF files
            mofs_without_file = []
            logger.info("Copying CIF files for WS24 dataset")
            for i in tqdm(range(len(in_df)), desc="Copying CIF files"):
                n = in_df.iloc[i]["MofName"]
                dataset = in_df.iloc[i]["data_set"]
                
                if dataset == "WS14s":
                    cif_file = src_cif_dir1 / f"{n}.cif"
                elif dataset == "WS24s":
                    cif_file = src_cif_dir2 / f"{n}.cif"
                    
                if not cif_file.exists():
                    logger.warning(f"{cif_file} does not exist")
                    mofs_without_file.append(n)
                    continue
                    
                dst_file = tgt_cif_dir / f"{n}.cif"
                if not dst_file.exists():
                    shutil.copy(cif_file, dst_file)
        
            if len(mofs_without_file) > 0:
                logger.warning(f"Found {len(mofs_without_file)} MOFs without CIF files")
                with open(tgt_root_dir / "mofs_without_cif.txt", "w") as f:
                    f.write("\n".join(mofs_without_file))
        
        # Check if cleaning is already done
        clean_cifs_exist = True
        for cif_file in tgt_cif_dir.glob("*.cif"):
            clean_cif_file = clean_cif_dir / cif_file.name
            if not clean_cif_file.exists():
                clean_cifs_exist = False
                break
                
        if clean_cifs_exist:
            logger.info("All clean CIF files for WS24 already exist, skipping cleaning step")
        else:
            # Clean CIF files
            logger.info("Cleaning CIF files")
            clean_cif.main(tgt_cif_dir, clean_cif_dir, log_file=tgt_root_dir / "clean_cif.log", santize=True, n_cpus=args.n_cpus)
        
        # Check if feature generation is already done
        if rac_zeo_features_file.exists():
            logger.info("Feature file for WS24 already exists, skipping feature generation")
        else:
            # Generate features
            logger.info("Generating features for WS24")
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
                logger.error(f"Feature generation for WS24 failed with error: {stderr.decode()}")
            else:
                logger.info("Feature generation for WS24 completed successfully")
        
        # Check if graph data preparation is already done
        graph_data_complete = True
        for cif_file in clean_cif_dir.glob("*.cif"):
            g_file_name = cif_file.stem + ".graphdata"
            if not (clean_cif_dir / g_file_name).exists():
                graph_data_complete = False
                break
                
        if graph_data_complete:
            logger.info("Graph data for WS24 already exists, skipping graph data preparation")
        else:
            # Prepare graph data
            logger.info("Preparing graph data for WS24")
            py_logger = get_logger(filename=str(tgt_root_dir / "prepare_graph_data.log"))
            
            # Process structures that need graph data preparation
            failed_records = []
            for cif_file in tqdm(list(clean_cif_dir.glob("*.cif")), desc="Preparing graph data"):
                g_file_name = cif_file.stem + ".graphdata"
                if (clean_cif_dir / g_file_name).exists():
                    with open(clean_cif_dir / g_file_name, 'rb') as f:
                        try:
                            cif_id, atom_num, nbr_idx, nbr_dist, *_, cell_params = pickle.load(f)
                            if nbr_idx.shape[0] / atom_num.shape[0] != args.max_num_nbr:
                                logger.warning(f"Number of neighbors is not {args.max_num_nbr} for {g_file_name}")
                                os.remove(str(clean_cif_dir / g_file_name))
                                flag = make_prepared_data(cif_file, clean_cif_dir, 
                                                        logger=py_logger, 
                                                        max_num_nbr=args.max_num_nbr, 
                                                        radius=args.radius)
                                if not flag:
                                    failed_records.append(cif_file.stem)
                        except Exception as e:
                            logger.error(f"Error loading {g_file_name}: {str(e)}")
                            os.remove(str(clean_cif_dir / g_file_name))
                            flag = make_prepared_data(cif_file, clean_cif_dir, 
                                                   logger=py_logger, 
                                                   max_num_nbr=args.max_num_nbr, 
                                                   radius=args.radius)
                            if not flag:
                                failed_records.append(cif_file.stem)
                else:
                    flag = make_prepared_data(cif_file, clean_cif_dir, 
                                           logger=py_logger, 
                                           max_num_nbr=args.max_num_nbr, 
                                           radius=args.radius)
                    if not flag:
                        failed_records.append(cif_file.stem)
        
            if len(failed_records) > 0:
                logger.warning(f"Failed to prepare graph data for {len(failed_records)} structures")
                with open(tgt_root_dir / "mofs_prepare_failed.txt", "w") as f:
                    f.write("\n".join(failed_records))
        
            # Filter out failed structures
            in_df = in_df[~in_df['MofName'].isin(failed_records)]
            logger.info(f"Dataset shape after filtering failed structures: {in_df.shape}")
        
        # Split into train, validation, and test sets if not already done
        if not id_prop_feat_file.exists():
            logger.info("Splitting WS24 dataset into train/val/test sets")
            random_seed = args.seed
            
            label_cols = ['water_label', 'acid_label', 'base_label', 'boiling_label']
            
            train_df, test_df = train_test_split(in_df, test_size=0.2, stratify=in_df[label_cols], random_state=random_seed)
            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df[label_cols], random_state=random_seed)
            
            train_df.insert(2, "Partition", "train")
            val_df.insert(2, "Partition", "val")
            test_df.insert(2, "Partition", "test")
            
            total_df = pd.concat([train_df, val_df, test_df])
            
            # Log split information
            ratio_dict = {}
            for col in label_cols + ["water4_label"]:
                for lb in total_df[col].unique():
                    split_ratios = []
                    for split in ["train", "val", "test"]:
                        sub_df = total_df.loc[(total_df[col] == lb) & (total_df["Partition"] == split)]
                        sub_ratio = len(sub_df) / len(total_df.loc[(total_df[col] == lb)])
                        split_ratios.append([sub_ratio, len(sub_df)])
                    ratio_dict[f"{col}_{lb}(train/val/test)"] = ' : '.join([f"{c}({r:.2f})" for r, c in split_ratios])
            
            for k, v in ratio_dict.items():
                logger.info(f"{k}: {v}")
            
            # Save the processed dataset
            total_df.to_csv(id_prop_feat_file, index=False)
        else:
            total_df = pd.read_csv(id_prop_feat_file)
    
    # Merge features with id_prop if not already done
    if not rac_zeo_id_prop_file.exists():
        logger.info("Merging features with ID-property data for WS24")
        df_new = pd.read_csv(tgt_root_dir / 'RAC_and_zeo_features.csv')
        df = pd.read_csv(id_prop_feat_file)
        
        df_new.drop(columns=['cif_file'], inplace=True)
        df_new.rename(columns={'name': 'MofName'}, inplace=True)
        
        id_prop_cols = ["MofName", "Partition", "water_label", "water4_label", "acid_label", "base_label", "boiling_label"]
        df_new = df[id_prop_cols].merge(df_new, on="MofName", how="left")
        df_new.to_csv(rac_zeo_id_prop_file, index=False)
    else:
        df_new = pd.read_csv(rac_zeo_id_prop_file)
    
    # Create separate datasets for each label
    label_cols = ["water_label", "water4_label", "acid_label", "base_label", "boiling_label"]
    
    ml_data_dir = ml_output_dir / "WS24"
    ml_output_file = ml_data_dir / "RAC_and_zeo_features_with_id_prop.csv"
    os.makedirs(ml_data_dir, exist_ok=True)
    
    # Check if the output file needs to be updated
    if not ml_output_file.exists() or os.path.getmtime(rac_zeo_id_prop_file) > os.path.getmtime(ml_output_file):
        # Create a copy with only the specific label column renamed to "Label"
        task_df = df_new.copy()
        task_df.to_csv(ml_output_file, index=False)
        logger.info(f"Created/updated ML dataset for WS24 at {ml_data_dir}")
    else:
        logger.info(f"ML dataset for WS24 is up to date")
    
    logger.info("WS24 data processing completed")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process MOF datasets for ML and CGCNN models.')
    parser.add_argument('--raw_data_dir', type=str, default=str(ROOT_DIR / 'data/raw_data'),
                      help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str, default=str(ROOT_DIR / 'data/cgcnn_data_'),
                      help='Directory to save processed data for CGCNN models')
    parser.add_argument('--ml_output_dir', type=str, default=str(ROOT_DIR / 'data/ml_data_'),
                      help='Directory to save processed data for ML models')
    parser.add_argument('--log_dir', type=str, default=str(ROOT_DIR / 'logs/data_processing'),
                      help='Directory to save logs')
    parser.add_argument('--dataset', type=str, choices=['all', 'tsd_ssd', 'ws24'], default='all',
                      help='Dataset to process: all, tsd_ssd, or ws24')
    parser.add_argument('--n_cpus', type=int, default=4,
                      help='Number of CPU cores to use for parallel processing')
    parser.add_argument('--radius', type=float, default=8.0,
                      help='Radius for neighbor finding in crystal graph')
    parser.add_argument('--max_num_nbr', type=int, default=10,
                      help='Maximum number of neighbors per atom')
    parser.add_argument('--prob_radius', type=float, default=1.86,
                      help='Probe radius for geometric feature calculations')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    for dir_path in [args.log_dir, args.output_dir, args.ml_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set up main logger
    log_file = Path(args.log_dir) / 'data_processing.log'
    logger = setup_logger('main_processor', log_file)
    logger.info("Starting data processing with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Process datasets based on arguments
    if args.dataset in ['all', 'tsd_ssd']:
        logger.info("Processing TSD and SSD datasets")
        process_tsd_ssd(args)
    
    if args.dataset in ['all', 'ws24']:
        logger.info("Processing WS24 dataset")
        process_ws24(args)
    
    logger.info("Data processing completed successfully!")

if __name__ == "__main__":
    main()
