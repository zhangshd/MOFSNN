#!/usr/bin/env python

# Disable OpenMP verbose output - add at the very beginning before any imports
import os
os.environ["KMP_AFFINITY"] = "disable"  # Disable thread affinity messages
os.environ["KMP_WARNINGS"] = "0"        # Disable KMP warnings
os.environ["OMP_DISPLAY_ENV"] = "FALSE"  # Don't display OpenMP environment variables
os.environ["KMP_SETTINGS"] = "0"        # Don't display OpenMP settings

# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import subprocess
import sys # Import sys for python executable path
import time
import warnings
import tempfile
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
import keras.backend as K
from tensorflow import keras as tf_keras # Use tensorflow's keras
import sklearn
import json
from sklearn.preprocessing import StandardScaler
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

# To run this script, you need to clone the MOFSimplify repository and install Zeo++0.3.
# and install the required dependencies via its environments/environment.yml file.
# Then conda activate the MOFSimplify environment and run this script.

# --- Constants ---
# Set the path to the MOFSimplify directory and the Zeo++ executable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MOFSIMPLIFY_PATH = ROOT_DIR.parent/'MOFSimplify'
ZEO_EXECUTABLE = ROOT_DIR.parent/'zeo++-0.3'/'network'
MODEL_PATH = MOFSIMPLIFY_PATH/'model'
PYTHON_EXECUTABLE = sys.executable # Get the path to the current Python interpreter

# Feature lists
# Base list for RAC features (used for reference)
RAC_FEATURES_BASE = [
    'D_func-I-0-all','D_func-I-1-all','D_func-I-2-all','D_func-I-3-all',
    'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all',
    'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all',
    'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all',
    'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
    'D_func-chi-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
    'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
    'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
    'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
    'D_lc-Z-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
    'D_lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all',
    'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all',
    'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all',
    'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all',
    'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all',
    'D_mc-chi-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all',
    'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all',
    'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all',
    'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-lig-I-0',
    'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2',
    'f-lig-S-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0',
    'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1',
    'f-lig-chi-2', 'f-lig-chi-3', 'func-I-0-all', 'func-I-1-all',
    'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all',
    'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all',
    'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all',
    'func-Z-2-all', 'func-Z-3-all', 'func-chi-0-all', 'func-chi-1-all',
    'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all',
    'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all',
    'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all',
    'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-chi-0-all', 'lc-chi-1-all',
    'lc-chi-2-all', 'lc-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
    'mc-I-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
    'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all',
    'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 'mc-chi-1-all',
    'mc-chi-2-all', 'mc-chi-3-all'
]
# Base list for geometric features (used for reference)
GEO_FEATURES_BASE = [
    'Di', 'Df', 'Dif', 'cell_v', 'VSA', 'GSA', 'VPOV', 'GPOV',
    'POAV_vol_frac', 'PONAV_vol_frac', 'GPOAV', 'GPONAV', 'POAV', 'PONAV'
]

# Features used by the Thermal and Solvent models (148 features)
# Derived by intersecting app.py's RACs+geo lists with columns in model/thermal/ANN/train.csv after removing constant columns
THERMAL_SOLVENT_FEATURES = [
    'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-Z-0-all',
    'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all', 'f-I-0-all', 'f-I-1-all',
    'f-I-2-all', 'f-I-3-all', 'f-T-0-all', 'f-T-1-all', 'f-T-2-all',
    'f-T-3-all', 'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all',
    'mc-chi-0-all', 'mc-chi-1-all', 'mc-chi-2-all', 'mc-chi-3-all',
    'mc-Z-0-all', 'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-I-1-all', # mc-I-0-all removed (constant)
    'mc-I-2-all', 'mc-I-3-all', 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all',
    'mc-T-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
    'D_mc-chi-1-all', 'D_mc-chi-2-all', 'D_mc-chi-3-all', # D_mc-chi-0-all removed (constant)
    'D_mc-Z-1-all', 'D_mc-Z-2-all', 'D_mc-Z-3-all', # D_mc-Z-0-all removed (constant)
    'D_mc-T-1-all', 'D_mc-T-2-all', 'D_mc-T-3-all', # D_mc-T-0-all removed (constant)
    'D_mc-S-1-all', 'D_mc-S-2-all', 'D_mc-S-3-all', # D_mc-S-0-all removed (constant)
    'f-lig-chi-0', 'f-lig-chi-1', 'f-lig-chi-2', 'f-lig-chi-3', 'f-lig-Z-0',
    'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-I-0', 'f-lig-I-1',
    'f-lig-I-2', 'f-lig-I-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2',
    'f-lig-T-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2', 'f-lig-S-3',
    'lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 'lc-chi-3-all',
    'lc-Z-0-all', 'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-I-1-all', # lc-I-0-all removed (constant)
    'lc-I-2-all', 'lc-I-3-all', 'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all',
    'lc-T-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all',
    'D_lc-chi-1-all', 'D_lc-chi-2-all', 'D_lc-chi-3-all', # D_lc-chi-0-all removed (constant)
    'D_lc-Z-1-all', 'D_lc-Z-2-all', 'D_lc-Z-3-all', # D_lc-Z-0-all removed (constant)
    'D_lc-T-1-all', 'D_lc-T-2-all', 'D_lc-T-3-all', # D_lc-T-0-all removed (constant)
    'D_lc-S-1-all', 'D_lc-S-2-all', 'D_lc-S-3-all', # D_lc-S-0-all removed (constant)
    'func-chi-0-all', 'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all',
    'func-Z-0-all', 'func-Z-1-all', 'func-Z-2-all', 'func-Z-3-all',
    'func-I-0-all', 'func-I-1-all', 'func-I-2-all', 'func-I-3-all',
    'func-T-0-all', 'func-T-1-all', 'func-T-2-all', 'func-T-3-all',
    'func-S-0-all', 'func-S-1-all', 'func-S-2-all', 'func-S-3-all',
    'D_func-chi-1-all', 'D_func-chi-2-all', 'D_func-chi-3-all', # D_func-chi-0-all removed (constant)
    'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all', # D_func-Z-0-all removed (constant)
    'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all', # D_func-T-0-all removed (constant)
    'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all', # D_func-S-0-all removed (constant)
    'Df', 'Di', 'Dif', 'GPOAV', 'GPONAV', 'GPOV', 'GSA', 'POAV',
    'POAV_vol_frac', 'PONAV', 'PONAV_vol_frac', 'VPOV', 'VSA', 'cell_v'
]

# Features used by the Water RF model (12 features)
WATER_RF_FEATURES = [
    'mc-Z-3-all', 'D_mc-Z-3-all', 'D_mc-Z-2-all', 'D_mc-Z-1-all',
    'mc-chi-3-all', 'mc-Z-1-all', 'mc-Z-0-all', 'D_mc-chi-2-all',
    'f-lig-Z-2', 'GSA', 'f-lig-I-0', 'func-S-1-all'
]

# Features used by the Acid RF model (8 features)
ACID_RF_FEATURES = [
    'mc-chi-3-all', 'Dif', 'mc-Z-2-all', 'Di', 'f-T-2-all',
    'D_mc-chi-2-all', 'lc-chi-3-all', 'D_mc-S-1-all'
]
# Features used by the Base RF model (13 features)
BASE_RF_FEATURES = [
    'D_mc-chi-1-all', 'f-lig-T-1', 'f-lig-S-2', 'Di', 'Df', 'f-lig-S-0', 
    'GPOV', 'GPOAV', 'f-I-1-all', 'f-lig-chi-0', 'D_lc-Z-3-all', 
    'D_func-alpha-2-all', 'f-lig-Z-0']

# Features used by the Boiling RF model (8 features)
BOILING_FEATURES = [
    'D_mc-chi-3-all', 'f-S-0-all', 'mc-S-0-all', 
    'f-S-2-all', 'D_mc-S-2-all', 
    'Di', 'lc-S-3-all', 'f-Z-3-all'
    ]

# --- Keras Custom Metrics ---
def precision(y_true, y_pred):
    """Keras precision metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val

def recall(y_true, y_pred):
    """Keras recall metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val

def f1(y_true, y_pred):
    """Keras F1 score metric."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def standard_labels(df, key="flag"):
    """
    Standardizes the target column (e.g., 'flag') to be strictly 0 or 1.
    Copied from app.py for consistency.

    Args:
        df (pd.DataFrame): DataFrame to modify.
        key (str): The column name to standardize.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Ensure the key exists before proceeding
    if key not in df.columns:
        print(f"Warning: standard_labels - Key '{key}' not found in DataFrame. Returning original DataFrame.")
        return df
    # Use .loc for safe assignment and handle potential non-numeric values gracefully
    # Convert to numeric first, coercing errors to NaN, then check if equal to 1
    flags_numeric = pd.to_numeric(df[key], errors='coerce')
    flags = np.where(flags_numeric == 1, 1, 0) # 1 if numeric value is 1, else 0 (includes NaN cases)
    df.loc[:, key] = flags # Use .loc for assignment
    return df

# --- Feature Calculation ---
def run_zeopp(probe_radius, cif_path_primitive, output_dir, name_base):
    """
    Runs Zeo++ commands (-ha -res, -sa, -volpo) for a given probe radius.

    Args:
        probe_radius (float): The probe radius (e.g., 1.86 or 1.4).
        cif_path_primitive (str): Path to the primitive CIF file.
        output_dir (str): Directory to save Zeo++ output files.
        name_base (str): Base name for output files (e.g., 'MOFNAME_primitive').

    Returns:
        bool: True if all Zeo++ commands succeed, False otherwise.
    """
    probe_str = str(probe_radius)
    pd_out = os.path.join(output_dir, f'{name_base}_pd.txt')
    sa_out = os.path.join(output_dir, f'{name_base}_sa.txt')
    pov_out = os.path.join(output_dir, f'{name_base}_pov.txt')

    cmd_res = [ZEO_EXECUTABLE, '-ha', '-res', pd_out, cif_path_primitive]
    cmd_sa = [ZEO_EXECUTABLE, '-sa', probe_str, probe_str, '10000', sa_out, cif_path_primitive]
    cmd_volpo = [ZEO_EXECUTABLE, '-volpo', probe_str, probe_str, '10000', pov_out, cif_path_primitive]

    all_success = True
    for cmd, name in zip([cmd_res, cmd_sa, cmd_volpo], ["-res", "-sa", "-volpo"]):
        try:
            print(f"  Running Zeo++ {name} (probe {probe_radius})...")
            # Use capture_output=True for Python 3.7+, otherwise use stdout/stderr=PIPE
            # For Python 3.6 compatibility:
            process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # print(f"  Zeo++ {name} stdout:\n{process.stdout.decode(errors='ignore')}") # Optional: print stdout
            # print(f"  Zeo++ {name} stderr:\n{process.stderr.decode(errors='ignore')}") # Optional: print stderr
        except subprocess.CalledProcessError as e:
            # Decode stderr manually for Python 3.6
            stderr_output = e.stderr.decode(errors='ignore') if e.stderr else 'No stderr output'
            print(f"Error running Zeo++ command (probe {probe_radius}, {name}): {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stderr: {stderr_output}")
            all_success = False
        except FileNotFoundError:
            print(f"Error: Zeo++ executable not found at {ZEO_EXECUTABLE}")
            return False # Cannot continue without Zeo++
        except Exception as e:
            print(f"An unexpected error occurred running Zeo++ {name} (probe {probe_radius}): {e}")
            all_success = False

    return all_success


def parse_zeopp_output(output_dir, name_base, probe_radius):
    """
    Parses the output files from Zeo++ for a specific probe radius.

    Args:
        output_dir (str): Directory containing Zeo++ output files.
        name_base (str): Base name used for output files (e.g., 'MOFNAME_primitive').
        probe_radius (float): The probe radius used (needed for context in case of errors).

    Returns:
        dict: Dictionary containing parsed geometric features. Returns NaNs if parsing fails.
    """
    geo_dict = {key: np.nan for key in GEO_FEATURES_BASE} # Initialize with NaNs
    geo_dict['name'] = name_base # Keep track of the base name

    pd_file = os.path.join(output_dir, f'{name_base}_pd.txt')
    sa_file = os.path.join(output_dir, f'{name_base}_sa.txt')
    pov_file = os.path.join(output_dir, f'{name_base}_pov.txt')

    if not all(os.path.exists(f) for f in [pd_file, sa_file, pov_file]):
        print(f"Warning: Not all Zeo++ output files found for probe {probe_radius}. Geometric features will be NaN.")
        return geo_dict

    try:
        # Parse *_pd.txt
        with open(pd_file) as f:
            line = f.readline().split()
            if len(line) > 3:
                geo_dict['Di'] = float(line[1])
                geo_dict['Df'] = float(line[2])
                geo_dict['Dif'] = float(line[3])
            else:
                print(f"Warning: Unexpected format in {pd_file}. Could not parse Di, Df, Dif.")

        # Parse *_sa.txt
        with open(sa_file) as f:
            content = f.read() # Read whole file for easier parsing
            geo_dict['cell_v'] = float(content.split('Unitcell_volume:')[1].split()[0])
            density = float(content.split('Density:')[1].split()[0])
            geo_dict['VSA'] = float(content.split('ASA_m^2/cm^3:')[1].split()[0])
            geo_dict['GSA'] = float(content.split('ASA_m^2/g:')[1].split()[0])

        # Parse *_pov.txt
        with open(pov_file) as f:
            content = f.read() # Read whole file
            density_pov = float(content.split('Density:')[1].split()[0])
            geo_dict['POAV'] = float(content.split('POAV_A^3:')[1].split()[0])
            geo_dict['PONAV'] = float(content.split('PONAV_A^3:')[1].split()[0])
            geo_dict['GPOAV'] = float(content.split('POAV_cm^3/g:')[1].split()[0])
            geo_dict['GPONAV'] = float(content.split('PONAV_cm^3/g:')[1].split()[0])
            geo_dict['POAV_vol_frac'] = float(content.split('POAV_Volume_fraction:')[1].split()[0])
            geo_dict['PONAV_vol_frac'] = float(content.split('PONAV_Volume_fraction:')[1].split()[0])
            geo_dict['VPOV'] = geo_dict['POAV_vol_frac'] + geo_dict['PONAV_vol_frac']
            if density_pov > 1e-9: # Use density_pov for consistency
                 geo_dict['GPOV'] = geo_dict['VPOV'] / density_pov
            else:
                 print(f"Warning: Near-zero density ({density_pov}) found in {pov_file}. GPOV set to NaN.")
                 geo_dict['GPOV'] = np.nan

    except (IOError, IndexError, ValueError, AttributeError) as e:
        print(f"Error parsing Zeo++ output files for probe {probe_radius}: {e}")
        # Reset to NaNs if parsing fails midway
        geo_dict = {key: np.nan for key in GEO_FEATURES_BASE}
        geo_dict['name'] = name_base
        pass

    return geo_dict

def calculate_features(cif_path, output_dir):
    """
    Calculates RAC and geometric features for a given CIF file.

    Args:
        cif_path (str): Path to the input CIF file.
        output_dir (str): Directory to store intermediate and final feature files.

    Returns:
        tuple: Paths to the two merged descriptor CSV files (one using probe 1.86,
               one using probe 1.4) or (None, None) on failure.
    """
    if not os.path.exists(cif_path):
        print(f"Error: Input CIF file not found at {cif_path}")
        return None, None

    name_base = os.path.splitext(os.path.basename(cif_path))[0]
    cif_dir = os.path.join(output_dir, 'cifs')
    rac_dir = os.path.join(output_dir, 'racs')
    zeo_ts_dir = os.path.join(output_dir, 'zeo_ts') # Thermal/Solvent (1.86)
    zeo_wa_dir = os.path.join(output_dir, 'zeo_wa') # Water/Acid (1.4)
    merged_dir = os.path.join(output_dir, 'merged_descriptors')

    for d in [cif_dir, rac_dir, zeo_ts_dir, zeo_wa_dir, merged_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 1. Prepare CIF and Primitive Cell ---
    temp_cif_path = os.path.join(cif_dir, f"{name_base}.cif")
    shutil.copy(cif_path, temp_cif_path)
    primitive_cif_path = os.path.join(cif_dir, f"{name_base}_primitive.cif")
    primitive_xyz_path = os.path.join(rac_dir, f"{name_base}_primitive.xyz") # Path for RAC xyz

    try:
        print("Generating primitive cell...")
        # Ensure output path exists for get_primitive
        os.makedirs(os.path.dirname(primitive_cif_path), exist_ok=True)
        get_primitive(temp_cif_path, primitive_cif_path)
        if not os.path.exists(primitive_cif_path):
             raise FileNotFoundError("Primitive CIF file not created by get_primitive.")
        print("Primitive cell generated.")
    except Exception as e:
        print(f"Warning: Failed to generate primitive cell using pymatgen: {e}. Using original CIF.")
        # If get_primitive fails, copy the original CIF to be used
        if not os.path.exists(primitive_cif_path):
             shutil.copy(temp_cif_path, primitive_cif_path)

    # --- 2. Calculate RAC Features ---
    print("Calculating RAC features...")
    rac_df_merged = None
    try:
        # Ensure output path exists for get_MOF_descriptors
        os.makedirs(rac_dir, exist_ok=True)

        # --- Run RAC Calculation using subprocess.run ---
        print("  Running RAC calculation script (RAC_getter.py)...")
        rac_script_path = os.path.join(MOFSIMPLIFY_PATH, 'model', 'RAC_getter.py')
        # Pass the original name_base, RAC_getter.py expects to append '_primitive.cif' itself.
        # Ensure cif_dir has a trailing slash for RAC_getter.py's path concatenation.
        cif_dir_with_slash = os.path.join(cif_dir, '')
        cmd_rac = [
            PYTHON_EXECUTABLE,
            rac_script_path,
            cif_dir_with_slash, # Pass directory path with trailing slash
            name_base,
            rac_dir
        ]
        try:
            # Use capture_output=True for Python 3.7+, otherwise use stdout/stderr=PIPE
            # For Python 3.6 compatibility: remove text=True, decode manually
            process_rac = subprocess.run(cmd_rac, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # print(f"  RAC_getter.py stdout:\n{process_rac.stdout.decode(errors='ignore')}") # Optional
            # print(f"  RAC_getter.py stderr:\n{process_rac.stderr.decode(errors='ignore')}") # Optional
            print("  RAC calculation script finished.")
        except subprocess.CalledProcessError as e:
            # Decode stderr manually for Python 3.6
            stderr_output = e.stderr.decode(errors='ignore') if e.stderr else 'No stderr output'
            print(f"Error running RAC calculation script: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stderr: {stderr_output}")
            # Check log file even if subprocess failed
            log_file = os.path.join(rac_dir, 'RAC_getter_log.txt')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f_log:
                    print(f"RAC_getter_log.txt content:\n{f_log.read()}")
            return None, None # Critical failure
        except FileNotFoundError:
             print(f"Error: RAC calculation script not found at {rac_script_path} or Python executable not found at {PYTHON_EXECUTABLE}")
             return None, None
        except Exception as e:
             print(f"An unexpected error occurred running RAC calculation script: {e}")
             return None, None

        # Check if descriptor files were created AFTER running the script
        lc_csv = os.path.join(rac_dir, "lc_descriptors.csv")
        sbu_csv = os.path.join(rac_dir, "sbu_descriptors.csv")
        linker_csv = os.path.join(rac_dir, "linker_descriptors.csv")

        if not all(os.path.exists(f) for f in [lc_csv, sbu_csv, linker_csv]):
             print("Error: RAC descriptor CSV files not generated by RAC_getter.py.")
             log_file = os.path.join(rac_dir, 'RAC_getter_log.txt')
             if os.path.exists(log_file):
                 with open(log_file, 'r') as f_log:
                     print(f"RAC_getter_log.txt content:\n{f_log.read()}")
             # Attempt to run get_MOF_descriptors directly as a fallback
             print("Attempting direct get_MOF_descriptors call as fallback...")
             try:
                 get_MOF_descriptors(primitive_cif_path, 3, path=rac_dir, xyzpath=primitive_xyz_path)
                 if not all(os.path.exists(f) for f in [lc_csv, sbu_csv, linker_csv]):
                     print("Error: Direct get_MOF_descriptors call also failed to generate CSVs.")
                     return None, None # Still failed
                 else:
                     print("Direct get_MOF_descriptors call succeeded.")
             except Exception as direct_call_e:
                 print(f"Error during direct get_MOF_descriptors call: {direct_call_e}")
                 return None, None # Failed fallback

        # Read, average, and merge RAC features
        lc_df = pd.read_csv(lc_csv).mean().to_frame().transpose()
        sbu_df = pd.read_csv(sbu_csv).mean().to_frame().transpose()
        linker_df = pd.read_csv(linker_csv).mean().to_frame().transpose()
        rac_df_merged = pd.concat([lc_df, sbu_df, linker_df], axis=1)
        # Add a name column consistent with geo features for potential future merging needs
        rac_df_merged['name'] = f"{name_base}_primitive"
        print("RAC features calculated and merged.")

    except Exception as e:
        print(f"Error processing RAC features after calculation: {e}")
        return None, None # Critical failure

    # --- 3. Calculate Geometric Features (Two Probe Radii) ---
    primitive_name_base = f"{name_base}_primitive"

    print("Calculating geometric features (probe 1.86 Å)...")
    if not run_zeopp(1.86, primitive_cif_path, zeo_ts_dir, primitive_name_base):
        print("Warning: Failed to run Zeo++ for probe 1.86 Å. Features will be NaN.")
        # Allow continuing, but features will be NaN
    geo_dict_ts = parse_zeopp_output(zeo_ts_dir, primitive_name_base, 1.86)
    geo_df_ts = pd.DataFrame([geo_dict_ts])
    print("Geometric features (probe 1.86 Å) calculated.")

    print("Calculating geometric features (probe 1.4 Å)...")
    if not run_zeopp(1.4, primitive_cif_path, zeo_wa_dir, primitive_name_base):
        print("Warning: Failed to run Zeo++ for probe 1.4 Å. Features will be NaN.")
        # Allow continuing, but features will be NaN
    geo_dict_wa = parse_zeopp_output(zeo_wa_dir, primitive_name_base, 1.4)
    geo_df_wa = pd.DataFrame([geo_dict_wa])
    print("Geometric features (probe 1.4 Å) calculated.")

    # --- 4. Merge and Save Features ---
    print("Merging features...")
    # Ensure RAC DataFrame index matches Geo DataFrame index for concatenation
    # Use the 'name' column for merging if indices don't align naturally
    rac_df_merged.index = [0] # Reset index for simple concat
    geo_df_ts.index = [0]
    geo_df_wa.index = [0]

    # Merge for Thermal/Solvent (Probe 1.86)
    # Concatenate along columns, dropping the duplicate 'name' from rac_df
    merged_df_ts = pd.concat([geo_df_ts, rac_df_merged.drop(columns=['name'])], axis=1)
    output_csv_ts = os.path.join(merged_dir, f"{name_base}_descriptors_ts_1.86.csv")
    merged_df_ts.to_csv(output_csv_ts, index=False)
    print(f"Thermal/Solvent features (probe 1.86) saved to: {output_csv_ts}")

    # Merge for Water/Acid (Probe 1.4)
    merged_df_wa = pd.concat([geo_df_wa, rac_df_merged.drop(columns=['name'])], axis=1)
    output_csv_wa = os.path.join(merged_dir, f"{name_base}_descriptors_wa_1.4.csv")
    merged_df_wa.to_csv(output_csv_wa, index=False)
    print(f"Water/Acid features (probe 1.4) saved to: {output_csv_wa}")

    return output_csv_ts, output_csv_wa

# --- Prediction Functions ---
# Global TF session for Keras models - Initialize lazily if needed
tf_session = None

def get_tf_session():
    """
    Initialize and return a TensorFlow session with proper GPU configuration.
    
    Returns:
        tf.compat.v1.Session: Configured TensorFlow session
    """
    print("Initializing TensorFlow session...")
    try:
        # Configure GPU memory growth to avoid taking all GPU memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # Fix: Use keyword argument for config parameter
        tf_session = tf.compat.v1.Session(config=config)
        
        # Set this session as the default
        tf.compat.v1.keras.backend.set_session(tf_session)
        return tf_session
    except Exception as e:
        print(f"Error initializing TensorFlow session: {e}")
        # Fallback to CPU if GPU initialization fails
        print("Falling back to CPU...")
        tf_session = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(tf_session)
        return tf_session

def predict_thermal(features_csv_path):
    """Predicts thermal stability."""
    print("Predicting thermal stability...")
    try:
        model_path = os.path.join(MODEL_PATH, 'thermal', 'ANN', 'final_model_T_few_epochs.h5')
        train_data_path = os.path.join(MODEL_PATH, 'thermal', 'ANN', 'train.csv')

        if not os.path.exists(model_path):
            print(f"Error: Thermal model not found at {model_path}")
            return "Error: Model file missing"
        if not os.path.exists(train_data_path):
            print(f"Error: Thermal training data not found at {train_data_path}")
            return "Error: Training data missing"
        if not os.path.exists(features_csv_path):
            print(f"Error: Features CSV not found at {features_csv_path}")
            return "Error: Features file missing"

        sess = get_tf_session()
        # Load model within the session context
        with sess.graph.as_default():
             # tf_keras.backend.set_session(sess) # Ensure session is set for this thread/context
             thermal_model = tf_keras.models.load_model(model_path, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})

        # Load data and prepare features
        df_newMOF = pd.read_csv(features_csv_path)
        df_train = pd.read_csv(train_data_path)

        # Check for missing columns in both input and training data
        missing_features_input = [f for f in THERMAL_SOLVENT_FEATURES if f not in df_newMOF.columns]
        if missing_features_input:
            print(f"Error: Missing required features in input {features_csv_path}: {missing_features_input}")
            return f"Error: Missing input features ({', '.join(missing_features_input)})"

        missing_features_train = [f for f in THERMAL_SOLVENT_FEATURES if f not in df_train.columns]
        if missing_features_train:
             print(f"Warning: Missing required features in training data {train_data_path}: {missing_features_train}. Scaling might be inaccurate if these features were important.")
             # Proceed, but be aware of potential issues

        # Select features, handling potential NaNs before scaling
        X_newMOF = df_newMOF[THERMAL_SOLVENT_FEATURES].copy()
        # Simple NaN handling: fill with mean from training data (or 0 if train data also has issues)
        df_train_clean = df_train[THERMAL_SOLVENT_FEATURES].dropna(axis=1, how='all') # Drop cols that are all NaN
        df_train_clean = df_train_clean.dropna() # Drop rows with any NaN

        if df_train_clean.empty:
             print("Error: Training data for scaling is empty after dropping NaNs.")
             return "Error: Training data scaling failed (empty)"

        train_means = df_train_clean.mean()
        X_newMOF.fillna(train_means, inplace=True)
        # Fill any remaining NaNs (if feature was all NaN in train) with 0
        X_newMOF.fillna(0, inplace=True)

        X_newMOF_vals = X_newMOF.values

        # Fit scalers based on cleaned training data
        X_train = df_train_clean[THERMAL_SOLVENT_FEATURES].values
        # Target variable scaling
        y_train_col = "T"
        if y_train_col not in df_train.columns:
             print(f"Error: Target variable '{y_train_col}' not found in {train_data_path}")
             return "Error: Target variable missing in training data"
        df_train_target_clean = df_train.dropna(subset=[y_train_col])
        if df_train_target_clean.empty:
             print(f"Error: No valid target data '{y_train_col}' after dropping NaNs.")
             return "Error: Target data scaling failed (empty)"
        y_train = df_train_target_clean[[y_train_col]].values


        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)

        # Scale input features
        X_newMOF_scaled = x_scaler.transform(X_newMOF_vals)

        # Predict within the session context
        with sess.graph.as_default():
            # tf_keras.backend.set_session(sess)
            prediction_scaled = thermal_model.predict(X_newMOF_scaled)

        # Inverse transform prediction
        prediction = y_scaler.inverse_transform(prediction_scaled)
        prediction_val = float(np.round(prediction[0][0], 2))
        degree_sign = u'\N{DEGREE SIGN}'
        print(f"Thermal stability prediction: {prediction_val}{degree_sign}C")
        return prediction_val

    except Exception as e:
        print(f"Error during thermal prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_solvent(features_csv_path):
    """Predicts solvent removal stability."""
    print("Predicting solvent stability...")
    try:
        model_path = os.path.join(MODEL_PATH, 'solvent', 'ANN', 'final_model_flag_few_epochs.h5')
        # Corrected path for solvent training data (including subdirectory)
        train_data_path = os.path.join(MODEL_PATH, 'solvent', 'ANN', 'dropped_connectivity_dupes', 'train.csv')

        if not os.path.exists(model_path):
            print(f"Error: Solvent model not found at {model_path}")
            return "Error: Model file missing"
        if not os.path.exists(train_data_path):
            print(f"Error: Solvent training data not found at {train_data_path}")
            return "Error: Training data missing"
        if not os.path.exists(features_csv_path):
            print(f"Error: Features CSV not found at {features_csv_path}")
            return "Error: Features file missing"

        sess = get_tf_session()
        
        # --- EXACTLY REPLICATE app.py's run_solvent_ANN function logic ---
        # Define RACs and geo feature lists exactly as in app.py
        RACs = ['D_func-I-0-all','D_func-I-1-all','D_func-I-2-all','D_func-I-3-all',
         'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all',
         'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all',
         'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all',
         'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
         'D_func-chi-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
         'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
         'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
         'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
         'D_lc-Z-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
         'D_lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all',
         'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all',
         'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all',
         'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all',
         'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all',
         'D_mc-chi-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all',
         'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all',
         'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all',
         'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-lig-I-0',
         'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2',
         'f-lig-S-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0',
         'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1',
         'f-lig-chi-2', 'f-lig-chi-3', 'func-I-0-all', 'func-I-1-all',
         'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all',
         'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all',
         'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all',
         'func-Z-2-all', 'func-Z-3-all', 'func-chi-0-all', 'func-chi-1-all',
         'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all',
         'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all',
         'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all',
         'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-chi-0-all', 'lc-chi-1-all',
         'lc-chi-2-all', 'lc-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
         'mc-I-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
         'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all',
         'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 'mc-chi-1-all',
         'mc-chi-2-all', 'mc-chi-3-all']
        geo = ['Df','Di', 'Dif','GPOAV','GPONAV','GPOV','GSA','POAV','POAV_vol_frac',
          'PONAV','PONAV_vol_frac','VPOV','VSA','cell_v']
        
        # Load data exactly as in app.py
        df_train = pd.read_csv(train_data_path)
        df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]
        df_newMOF = pd.read_csv(features_csv_path)
        
        # Select features exactly as in app.py
        features = [val for val in df_train.columns.values if val in RACs+geo]
        
        # Standardize labels as in app.py
        df_train = standard_labels(df_train, key="flag")
        
        # Implement normalize_data_solvent directly
        def normalize_data_solvent_inline(df_train, df_newMOF, fnames, lname):
            _df_train = df_train.copy().dropna(subset=fnames+lname)
            _df_newMOF = df_newMOF.copy().dropna(subset=fnames) 
            X_train = _df_train[fnames].values
            X_newMOF = _df_newMOF[fnames].values  # Same exact order as training set
            y_train = _df_train[lname].values
            
            x_scaler = StandardScaler()
            x_scaler.fit(X_train)
            X_train = x_scaler.transform(X_train)
            X_newMOF = x_scaler.transform(X_newMOF)
            y_train = np.array([1 if x == 1 else 0 for x in y_train.reshape(-1, )])
            return X_train, X_newMOF, y_train, x_scaler
        
        # Process data exactly as in app.py
        X_train, X_newMOF, y_train, x_scaler = normalize_data_solvent_inline(
            df_train, df_newMOF, features, ["flag"])
        
        # Load model and predict exactly as in app.py
        with sess.graph.as_default():
            solvent_model = tf_keras.models.load_model(model_path, 
                                                      custom_objects={'precision': precision, 
                                                                      'recall': recall, 'f1': f1})
            # Use np.round directly on prediction as in app.py (no clip)
            prediction = np.round(solvent_model.predict(X_newMOF), 4)
            prediction_val = float(prediction[0][0])  # Extract the value

        print(f"Solvent stability prediction (probability): {prediction_val}")
        return prediction_val

    except Exception as e:
        print(f"Error during solvent prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_water(features_csv_path):
    """Predicts water stability using Random Forest."""
    print("Predicting water stability...")
    try:
        model_path = os.path.join(MODEL_PATH, 'water_and_acid', 'models', 'water_model.joblib')
        scaler_path = os.path.join(MODEL_PATH, 'water_and_acid', 'models', 'water_scaler.joblib')

        if not os.path.exists(model_path):
            print(f"Error: Water model not found at {model_path}")
            return "Error: Model file missing"
        if not os.path.exists(scaler_path):
            print(f"Error: Water scaler not found at {scaler_path}")
            return "Error: Scaler file missing"
        if not os.path.exists(features_csv_path):
            print(f"Error: Features CSV not found at {features_csv_path}")
            return "Error: Features file missing"

        water_model = joblib.load(model_path)
        water_scaler = joblib.load(scaler_path)

        df_newMOF = pd.read_csv(features_csv_path)

        # Check for missing columns
        missing_features = [f for f in WATER_RF_FEATURES if f not in df_newMOF.columns]
        if missing_features:
            print(f"Error: Missing required features in {features_csv_path}: {missing_features}")
            return f"Error: Missing features ({', '.join(missing_features)})"

        # Select features and handle NaNs (fill with 0 for RF, though scaler should handle)
        X_newMOF = df_newMOF[WATER_RF_FEATURES].copy()
        X_newMOF.fillna(0, inplace=True) # Simple NaN handling for RF features
        X_newMOF_vals = X_newMOF.values

        # Scale features
        X_newMOF_scaled = water_scaler.transform(X_newMOF_vals)

        # Predict probability of the positive class (stable)
        prediction = water_model.predict_proba(X_newMOF_scaled)[:, 1][0]
        prediction_val = float(np.round(prediction, 2))
        print(f"Water stability prediction (probability): {prediction_val}")
        return prediction_val

    except Exception as e:
        print(f"Error during water prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_acid(features_csv_path):
    """Predicts acid stability using Random Forest."""
    print("Predicting acid stability...")
    try:
        model_path = os.path.join(MODEL_PATH, 'water_and_acid', 'models', 'acid_model.joblib')
        scaler_path = os.path.join(MODEL_PATH, 'water_and_acid', 'models', 'acid_scaler.joblib')

        if not os.path.exists(model_path):
            print(f"Error: Acid model not found at {model_path}")
            return "Error: Model file missing"
        if not os.path.exists(scaler_path):
            print(f"Error: Acid scaler not found at {scaler_path}")
            return "Error: Scaler file missing"
        if not os.path.exists(features_csv_path):
            print(f"Error: Features CSV not found at {features_csv_path}")
            return "Error: Features file missing"

        acid_model = joblib.load(model_path)
        acid_scaler = joblib.load(scaler_path)

        df_newMOF = pd.read_csv(features_csv_path)

        # Check for missing columns
        missing_features = [f for f in ACID_RF_FEATURES if f not in df_newMOF.columns]
        if missing_features:
            print(f"Error: Missing required features in {features_csv_path}: {missing_features}")
            return f"Error: Missing features ({', '.join(missing_features)})"

        # Select features and handle NaNs
        X_newMOF = df_newMOF[ACID_RF_FEATURES].copy()
        X_newMOF.fillna(0, inplace=True) # Simple NaN handling
        X_newMOF_vals = X_newMOF.values

        # Scale features
        X_newMOF_scaled = acid_scaler.transform(X_newMOF_vals)

        # Predict probability of the positive class (stable)
        prediction = acid_model.predict_proba(X_newMOF_scaled)[:, 1][0]
        prediction_val = float(np.round(prediction, 2))
        print(f"Acid stability prediction (probability): {prediction_val}")
        return prediction_val

    except Exception as e:
        print(f"Error during acid prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Predict MOF stability (thermal, solvent, water, acid) based on a CIF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("cif_file", help="Path to the input CIF file.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store intermediate files (default: temporary directory)."
        )
    parser.add_argument(
        "--keep_files",
        action="store_true",
        help="Keep intermediate files after execution (default: delete)."
        )
    parser.add_argument(
        "--json_output",
        action="store_true",
        help="Output results in JSON format (useful for batch processing)."
        )

    args = parser.parse_args()

    start_time = time.time()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using specified output directory: {output_dir}")
        cleanup_dir = False # Don't cleanup if user specified directory
    else:
        # Create a temporary directory for calculations
        # Store the path in a variable accessible in the finally block
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="mofsimplify_pred_")
        output_dir = temp_dir_obj.name
        print(f"Using temporary directory: {output_dir}")
        cleanup_dir = not args.keep_files # Cleanup unless --keep_files is specified

    results = {} # Dictionary to store results

    try:
        # Calculate features (using the chosen output directory)
        # features_ts_csv uses probe 1.86, features_wa_csv uses probe 1.4
        features_ts_csv, features_wa_csv = calculate_features(args.cif_file, output_dir)

        if features_ts_csv and features_wa_csv:
            # Run predictions using the correct feature sets
            results['thermal_stability'] = predict_thermal(features_ts_csv)
            results['solvent_stability'] = predict_solvent(features_ts_csv)
            # results['water_stability'] = predict_water(features_wa_csv)
            # results['acid_stability'] = predict_acid(features_wa_csv)

            # Print results clearly
            print("\n--- Stability Predictions ---")
            print(f"Input CIF:          {os.path.abspath(args.cif_file)}")
            print(f"Thermal Stability:  {results.get('thermal_stability', None)}°C")
            print(f"Solvent Stability:  {results.get('solvent_stability', None)} (Probability)")
            # print(f"Water Stability:    {results.get('water_stability', None)} (Probability)")
            # print(f"Acid Stability:     {results.get('acid_stability', None)} (Probability)")
        else:
            print("\n--- Prediction Failed ---")
            print("Feature calculation failed or did not produce necessary files. Cannot proceed with predictions.")
            # Store error indication in results
            results['error'] = "Feature calculation failed"

    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    finally:
        # Clean up temporary directory if created and not told to keep
        if cleanup_dir and 'temp_dir_obj' in locals():
             try:
                 temp_dir_obj.cleanup()
                 print(f"Cleaned up temporary directory: {output_dir}")
             except Exception as e: # Catch potential errors during cleanup
                 print(f"Error removing temporary directory {output_dir}: {e}")
        elif not cleanup_dir:
             print(f"Intermediate files kept in: {output_dir}")

        # Close TensorFlow session if it was initialized
        global tf_session
        if tf_session is not None:
            print("TensorFlow session will be closed automatically on exit.")
            pass # Let garbage collection handle it


    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Output as JSON if requested
    if args.json_output:
        json_results = {
            "cif_file": os.path.abspath(args.cif_file),
            "execution_time": end_time - start_time,
            "predictions": results
        }
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(json_results))
    
    return results

if __name__ == "__main__":
    main()
