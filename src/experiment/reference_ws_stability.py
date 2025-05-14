#!/usr/bin/env python

# Disable OpenMP verbose output - add at the very beginning before any imports
import os
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
import pickle
import json
import traceback # Added to fix previous error
from sklearn.preprocessing import StandardScaler
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
            # Updated for Python 3.7+
            process = subprocess.run(cmd, check=True, capture_output=True, text=True, errors='ignore')
            # Optional: print stdout/stderr if needed for debugging
            # print(f"  Zeo++ {name} stdout:\n{process.stdout}")
            # print(f"  Zeo++ {name} stderr:\n{process.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error running Zeo++ command (probe {probe_radius}, {name}): {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            # stderr is already decoded if text=True was used
            print(f"Stderr: {e.stderr if e.stderr else 'No stderr output'}")
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
               one using probe 1.4) or (None) on failure.
    """
    if not os.path.exists(cif_path):
        print(f"Error: Input CIF file not found at {cif_path}")
        return None

    name_base = os.path.splitext(os.path.basename(cif_path))[0]
    cif_dir = os.path.join(output_dir, 'cifs')
    rac_dir = os.path.join(output_dir, 'racs')
    zeo_wa_dir = os.path.join(output_dir, 'zeo_wa') # Water/Acid (1.4)
    merged_dir = os.path.join(output_dir, 'merged_descriptors')

    for d in [cif_dir, rac_dir, zeo_wa_dir, merged_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 1. Prepare CIF and Primitive Cell ---
    temp_cif_path = os.path.join(cif_dir, f"{name_base}.cif")
    shutil.copy(cif_path, temp_cif_path)
    primitive_cif_path = os.path.join(cif_dir, f"{name_base}_primitive.cif")
    primitive_xyz_path = os.path.join(rac_dir, f"{name_base}_primitive.xyz") # Path for RAC xyz

    try:
        # Ensure output path exists for get_primitive
        os.makedirs(os.path.dirname(primitive_cif_path), exist_ok=True)
        get_primitive(temp_cif_path, primitive_cif_path)
        if not os.path.exists(primitive_cif_path):
             raise FileNotFoundError("Primitive CIF file not created by get_primitive.")
    except Exception as e:
        print(f"Warning: Failed to generate primitive cell using pymatgen: {e}. Using original CIF.")
        # If get_primitive fails, copy the original CIF to be used
        if not os.path.exists(primitive_cif_path):
             shutil.copy(temp_cif_path, primitive_cif_path)

    # --- 2. Calculate RAC Features ---
    rac_df_merged = None
    try:
        # Ensure output path exists for get_MOF_descriptors
        os.makedirs(rac_dir, exist_ok=True)

        # --- Run RAC Calculation using subprocess.run ---
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
            # Updated for Python 3.7+
            print(" ".join(cmd_rac)) # Kept this print for command visibility
            process_rac = subprocess.run(cmd_rac, check=True, capture_output=True, text=True, errors='ignore')
            # Optional: print stdout/stderr if needed for debugging
            # print(f"  RAC_getter.py stdout:\n{process_rac.stdout}")
            # print(f"  RAC_getter.py stderr:\n{process_rac.stderr}")
            print("  RAC calculation script finished.")
        except subprocess.CalledProcessError as e:
            print(f"Error running RAC calculation script: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            # stderr is already decoded if text=True was used
            print(f"Stderr: {e.stderr if e.stderr else 'No stderr output'}")
            # Check log file even if subprocess failed
            log_file = os.path.join(rac_dir, 'RAC_getter_log.txt')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f_log:
                    print(f"RAC_getter_log.txt content:\n{f_log.read()}")
            return None # Critical failure
        except FileNotFoundError:
             print(f"Error: RAC calculation script not found at {rac_script_path} or Python executable not found at {PYTHON_EXECUTABLE}")
             return None
        except Exception as e:
             print(f"An unexpected error occurred running RAC calculation script: {e}")
             return None

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
             try:
                 get_MOF_descriptors(primitive_cif_path, 3, path=rac_dir, xyzpath=primitive_xyz_path)
                 if not all(os.path.exists(f) for f in [lc_csv, sbu_csv, linker_csv]):
                     return None # Still failed
             except Exception as direct_call_e:
                 print(f"Error during direct get_MOF_descriptors call: {direct_call_e}")
                 return None # Failed fallback

        # Read, average, and merge RAC features
        lc_df = pd.read_csv(lc_csv).mean(numeric_only=True).to_frame().transpose()
        sbu_df = pd.read_csv(sbu_csv).mean(numeric_only=True).to_frame().transpose()
        linker_df = pd.read_csv(linker_csv).mean(numeric_only=True).to_frame().transpose()
        rac_df_merged = pd.concat([lc_df, sbu_df, linker_df], axis=1)
        # Add a name column consistent with geo features for potential future merging needs
        rac_df_merged['name'] = f"{name_base}_primitive"

    except Exception as e:
        print(f"Error processing RAC features after calculation: {traceback.format_exc()}")
        return None # Critical failure

    # --- 3. Calculate Geometric Features ---
    primitive_name_base = f"{name_base}_primitive"

    if not run_zeopp(1.4, primitive_cif_path, zeo_wa_dir, primitive_name_base):
        print("Warning: Failed to run Zeo++ for probe 1.4 Å. Features will be NaN.")
        # Allow continuing, but features will be NaN
    geo_dict_wa = parse_zeopp_output(zeo_wa_dir, primitive_name_base, 1.4)
    geo_df_wa = pd.DataFrame([geo_dict_wa])

    # --- 4. Merge and Save Features ---
    # Ensure RAC DataFrame index matches Geo DataFrame index for concatenation
    # Use the 'name' column for merging if indices don't align naturally
    rac_df_merged.index = [0] # Reset index for simple concat
    geo_df_wa.index = [0]

    # Merge for Water/Acid (Probe 1.4)
    merged_df_wa = pd.concat([geo_df_wa, rac_df_merged.drop(columns=['name'])], axis=1)
    output_csv_wa = os.path.join(merged_dir, f"{name_base}_descriptors_wa_1.4.csv")
    merged_df_wa.to_csv(output_csv_wa, index=False)

    return output_csv_wa

def predict_stability(features_csv_path, stability_type):
    """
    Predicts MOF stability using Random Forest models.
    
    Args:
        features_csv_path (str): Path to the CSV file containing MOF features.
        stability_type (str): Type of stability to predict ('water', 'acid', 'base', or 'boiling').
    
    Returns:
        float or str: Prediction probability (0-1) or error message.
    """
    try:
        # Define model configurations
        config = {
            'water': {
                'model_name': 'water_model.pkl',
                'scaler_name': 'water_scaler.pkl',
                'features': WATER_RF_FEATURES
            },
            'acid': {
                'model_name': 'acid_model.pkl',
                'scaler_name': 'acid_scaler.pkl',
                'features': ACID_RF_FEATURES
            },
            'base': {
                'model_name': 'base_model.pkl',
                'scaler_name': 'base_scaler.pkl', 
                'features': BASE_RF_FEATURES
            },
            'boiling': {
                'model_name': 'boiling_model.pkl',
                'scaler_name': 'boiling_scaler.pkl',
                'features': BOILING_FEATURES
            },
            'water4': {
                'model_name': '4_class_water_model.pkl',
                'scaler_name': '4_class_water_scaler.pkl',
                'features': WATER_RF_FEATURES
            }
        }

        if stability_type not in config:
            print(f"Error: Unsupported stability type '{stability_type}'")
            return None
            
        model_path = os.path.join(ROOT_DIR, 'data/raw_data/WS24v2/models', 'models', config[stability_type]['model_name'])
        scaler_path = os.path.join(ROOT_DIR, 'data/raw_data/WS24v2/models', 'models', config[stability_type]['scaler_name'])
        features_list = config[stability_type]['features']

        # Check required files exist
        if not os.path.exists(model_path):
            print(f"Error: {stability_type.capitalize()} model not found at {model_path}")
            return None
        if not os.path.exists(scaler_path):
            print(f"Error: {stability_type.capitalize()} scaler not found at {scaler_path}")
            return None
        if not os.path.exists(features_csv_path):
            print(f"Error: Features CSV not found at {features_csv_path}")
            return None
        # Load model and scaler
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        df_newMOF = pd.read_csv(features_csv_path)

        # Check for missing columns
        missing_features = [f for f in features_list if f not in df_newMOF.columns]
        if missing_features:
            print(f"Error: Missing required features in {features_csv_path}: {missing_features}")
            return None

        # Select features and handle NaNs
        X_newMOF = df_newMOF[features_list].copy()
        X_newMOF.fillna(0, inplace=True)  # Simple NaN handling
        X_newMOF_vals = X_newMOF.values

        # Scale features
        X_newMOF_scaled = scaler.transform(X_newMOF_vals)

        # Predict probability of the positive class (stable)
        if stability_type == 'water4':
            # For 4-class model, predict probabilities for all classes
            prediction_propb = model.predict_proba(X_newMOF_scaled)
            # Get the probability of the first class (stable)
            prediction_propb = np.round(prediction_propb[0], 4).tolist()
            return prediction_propb

        prediction_propb = model.predict_proba(X_newMOF_scaled)[:, 1][0]
        prediction_propb = float(np.round(prediction_propb, 4))
        return prediction_propb

    except Exception as e:
        print(f"Error during {stability_type} prediction: {e}")
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
        cleanup_dir = False # Don't cleanup if user specified directory
    else:
        # Create a temporary directory for calculations
        # Store the path in a variable accessible in the finally block
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="mofsimplify_pred_")
        output_dir = temp_dir_obj.name
        cleanup_dir = not args.keep_files # Cleanup unless --keep_files is specified

    results = {} # Dictionary to store results

    try:
        # Calculate features (using the chosen output directory)
        # features_ts_csv uses probe 1.86, features_wa_csv uses probe 1.4
        features_wa_csv = calculate_features(args.cif_file, output_dir)

        if features_wa_csv:
            # Run predictions using the correct feature sets
            results['water_stability'] = predict_stability(features_wa_csv, 'water')
            results['water4_stability'] = predict_stability(features_wa_csv, 'water4')
            results['acid_stability'] = predict_stability(features_wa_csv, 'acid')
            results['base_stability'] = predict_stability(features_wa_csv, 'base')
            results['boiling_stability'] = predict_stability(features_wa_csv, 'boiling')

            # Print results clearly
            print("\n--- Stability Predictions ---")
            print(f"Input CIF:          {os.path.abspath(args.cif_file)}")
            print(f"Water Stability:    {results.get('water_stability', None)} (Probability)")
            print(f"Water4 Stability:   {results.get('water4_stability', None)} (Probability)")
            print(f"Acid Stability:     {results.get('acid_stability', None)} (Probability)")
            print(f"Base Stability:     {results.get('base_stability', None)} (Probability)")
            print(f"Boiling Stability:  {results.get('boiling_stability', None)} (Probability)")
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
             except Exception as e: # Catch potential errors during cleanup
                 print(f"Error removing temporary directory {output_dir}: {e}")
        elif not cleanup_dir:
             print(f"Intermediate files kept in: {output_dir}")


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
