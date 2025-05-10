#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-05-10
Description: Batch submit CGCNN/MOFSNN model training jobs to SLURM with different CSV files.
This script iterates through different CSV files and model configurations,
submitting SLURM jobs for each combination.
'''

import subprocess
import os
import time
import argparse
import yaml
import glob
from pathlib import Path
import re
import sys

# SLURM job template
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

echo "Starting model training with {csv_file_name}..."
srun python -u {script_exe} {training_args} --down_sampling --csv_file_name {csv_file_name}

echo "Training completed."
""".strip()

# Set up paths
SCRIPT_DIR = Path(__file__).absolute().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR/"src/cgcnn"))

def run_slurm_job(work_dir, job_script_path, executor="sbatch"):
    """
    Submit a SLURM job using the specified executor.
    
    Args:
        work_dir: Working directory
        job_script_path: Path to the job script file
        executor: Command to execute the job (default: sbatch)
        
    Returns:
        Subprocess object
    """
    work_dir = Path(work_dir)
    
    # Create logs directory if it doesn't exist
    logs_dir = work_dir / "slurm_logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Submit the job
    process = subprocess.Popen(
        f"{executor} {job_script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(work_dir)
    )
    
    # Get the output of the job submission
    stdout, stderr = process.communicate()
    
    if stdout:
        print("STDOUT:", stdout.decode().strip())
    if stderr:
        print("STDERR:", stderr.decode().strip())
        
    return process

def get_model_params_from_config(config_path, model_type):
    """
    Extract model parameters from the model comparison config file.
    
    Args:
        config_path: Path to the model comparison config file
        model_type: The model type to extract (CGCNN_SG, CGCNN_MT, MOFSNN)
        
    Returns:
        Dictionary of model paths and their configurations
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_paths = {}
    for model_name, model_info in config['model_dirs_map'].items():
        if model_info['DisplayName'] == model_type:
            # Get the model path
            model_path = model_info['Path']
            
            # Extract version number
            version_match = re.search(r'version_(\d+)$', model_path)
            if version_match:
                version_num = version_match.group(1)
                # Get the model directory (without version)
                model_dir = model_path.rsplit('/', 1)[0]
                
                # Store the model information
                model_paths[model_name] = {
                    'path': model_path,
                    'dir': model_dir,
                    'version': version_num,
                    'model_type': model_info['Model']
                }
    
    return model_paths

def read_hparams_yaml(model_path):
    """
    Read hyperparameters from the model's hparams.yaml file.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary of hyperparameters
    """
    hparams_path = Path(model_path) / "hparams.yaml"
    if not hparams_path.exists():
        print(f"Warning: hparams.yaml not found at {hparams_path}")
        return {}
    
    # Use regular expressions to parse the YAML file manually
    # This avoids issues with Python-specific tags
    hparams = {}
    with open(hparams_path, 'r') as f:
        content = f.read()
        
    # Extract key parameters using regex
    param_patterns = {
        'model_cfg': r'model_cfg: (.+?)$',
        'task_cfg': r'task_cfg: (.+?)$',
        'batch_size': r'batch_size: (\d+)',
        'max_epochs': r'max_epochs: (\d+)',
        'atom_fea_len': r'atom_fea_len: (\d+)',
        'extra_fea_len': r'extra_fea_len: (\d+)',
        'h_fea_len': r'h_fea_len: (\d+)',
        'n_conv': r'n_conv: (\d+)',
        'n_h': r'n_h: (\d+)',
        'dropout_prob': r'dropout_prob: (\d+\.\d+)',
        'lr': r'lr: (\d+\.\d+)',
        'lr_mult': r'lr_mult: (\d+\.\d+)',
        'patience': r'patience: (\d+)',
        'decay_power': r'decay_power: (.+?)$',
        'dl_sampler': r'dl_sampler: (.+?)$',
        'task_att_type': r'task_att_type: (.+?)$',
        'optim': r'optim: (.+?)$',
        'optim_config': r'optim_config: (.+?)$',
    }
    
    # Boolean parameters
    bool_patterns = {
        'use_extra_fea': r'use_extra_fea: (true|false)',
        'use_cell_params': r'use_cell_params: (true|false)',
        'atom_layer_norm': r'atom_layer_norm: (true|false)',
        'group_lr': r'group_lr: (true|false)',
        'task_norm': r'task_norm: (true|false)',
        'att_pooling': r'att_pooling: (true|false)',
        'reconstruct': r'reconstruct: (true|false)',
    }
    
    # Extract parameters
    for param, pattern in param_patterns.items():
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Convert numeric values
            if param in ['batch_size', 'max_epochs', 'atom_fea_len', 'extra_fea_len', 
                        'h_fea_len', 'n_conv', 'n_h', 'patience']:
                hparams[param] = int(value)
            elif param in ['dropout_prob', 'lr', 'lr_mult']:
                hparams[param] = float(value)
            else:
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                hparams[param] = value
    
    # Extract boolean parameters
    for param, pattern in bool_patterns.items():
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            hparams[param] = (value == 'true')
    
    return hparams

def create_training_args(model_params):
    """
    Create a string of training arguments from the model parameters.
    
    Args:
        model_params: Dictionary of model parameters
        
    Returns:
        String of command-line arguments
    """
    args = f"--task_cfg {model_params.get('task_cfg', 'tsd_ssd_ws24')} --model_cfg {model_params.get('model_cfg', 'att_cgcnn')} --progress_bar"
    
    # Add model parameters
    for key, value in model_params.items():
        # Skip task_cfg and model_cfg as they are already added
        if key in ['task_cfg', 'model_cfg']:
            continue
            
        # Handle boolean parameters
        if isinstance(value, bool):
            if value:
                args += f" --{key}"
            continue
            
        # Handle string parameters that contain spaces
        if isinstance(value, str) and ' ' in value:
            args += f" --{key} \"{value}\""
        else:
            args += f" --{key} {value}"
    
    return args

def main():
    """Main function to parse arguments and submit jobs."""
    parser = argparse.ArgumentParser(description="Submit batch CGCNN model training to SLURM with different CSV files")
    parser.add_argument("--model_types", type=str, nargs='+', default=["MOFSNN", "CGCNN_SG", "CGCNN_MT"], 
                        help="Model types to train (CGCNN_SG, CGCNN_MT, MOFSNN)")
    parser.add_argument("--data_dir", type=str, default=str(ROOT_DIR/"data/cgcnn_data"), 
                        help="Base data directory containing CSV files")
    parser.add_argument("--config_file", type=str, default=str(ROOT_DIR/"configs/model_comparison_config.yaml"), 
                        help="Path to model comparison config file")
    parser.add_argument("--csv_pattern", type=str, default="RAC_and_zeo_features_with_id_prop_rand*.csv", 
                        help="Pattern to match CSV files")
    parser.add_argument("--wait_time", type=int, default=5, 
                        help="Wait time in seconds between job submissions")
    parser.add_argument("--dry_run", action="store_true", 
                        help="Create job scripts but don't submit them")
    args = parser.parse_args()

    # Get CSV file names based on pattern
    csv_file_names = [args.csv_pattern.replace("*", str(i)) for i in range(5)]
    for name in csv_file_names:
        print(f"  - {name}")
    
    # Create a directory for job scripts
    job_scripts_dir = SCRIPT_DIR / "job_scripts"
    job_scripts_dir.mkdir(exist_ok=True, parents=True)
    
    # Loop through model types
    for model_type in args.model_types:
        print(f"\nProcessing model type: {model_type}")
        
        # Get model parameters from config
        model_paths = get_model_params_from_config(args.config_file, model_type)
        if not model_paths:
            print(f"No models found for model type '{model_type}' in config file")
            continue
            
        # Loop through models
        for model_name, model_info in model_paths.items():
            print(f"Processing model: {model_name}")
            
            # Get model parameters from hparams.yaml
            model_params = read_hparams_yaml(model_info['path'])
            if not model_params:
                print(f"Could not read parameters for model {model_name}, skipping...")
                continue
                
            # Update model_cfg to match the model type
            model_params['model_cfg'] = model_info['model_type']
            
            # Create training arguments
            training_args = create_training_args(model_params)
            print(f"Training arguments: {training_args}")
            
            # Loop through CSV file names
            for csv_filename in csv_file_names:
                print(f"Processing CSV file: {csv_filename}")
                
                # Create job name (use a simplified format)
                csv_short_name = csv_filename.replace('.csv', '').replace('RAC_and_zeo_features_with_id_prop_', '')
                job_name = f"train_{model_name}_{csv_short_name}"
                
                # Create job script
                job_script = JOB_TEMPLATE.format(
                    job_name=job_name,
                    csv_file_name=csv_filename,
                    training_args=training_args,
                    script_exe=str(ROOT_DIR/"src/cgcnn/main.py")
                )
                
                # Write job script to file
                job_script_path = job_scripts_dir / f"{job_name}.sh"
                with open(job_script_path, "w") as f:
                    f.write(job_script)
                
                print(f"Created job script: {job_script_path}")
                
                # Submit job if not in dry run mode
                if not args.dry_run:
                    print(f"Submitting job: {job_name}")
                    run_slurm_job(SCRIPT_DIR, job_script_path)
                    
                    # Wait to avoid overloading the scheduler
                    time.sleep(args.wait_time)
                else:
                    print(f"DRY RUN: Would submit job: {job_name}")

if __name__ == "__main__":
    main()
