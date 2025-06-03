#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-04-24
Description: Submit CGCNN model training jobs to SLURM with epoch-by-epoch checkpoint saving.
This script creates and submits SLURM job scripts for training CGCNN models,
saving checkpoints for each epoch to enable further analysis.
'''

import subprocess
import os
import time
import argparse
from pathlib import Path

# SLURM job template
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

# Train model with saving each epoch's checkpoint
echo "Starting model training with save_all_epochs..."
srun python -u {script_exe} {training_args} --save_all_epochs

echo "Training completed."
""".strip()

def run_slurm_job(work_dir, executor="sbatch", script_name="run_train.sh"):
    """
    Submit a SLURM job using the specified executor.
    
    Args:
        work_dir: Working directory
        executor: Command to execute the job (default: sbatch)
        script_name: Name of the script file to execute
        
    Returns:
        Subprocess object
    """
    work_dir = Path(work_dir)
    
    # Create logs directory if it doesn't exist
    logs_dir = work_dir / "slurm_logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Submit the job
    process = subprocess.Popen(
        f"{executor} {work_dir/script_name}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(work_dir)
    )
    return process

def create_training_args(task_config, model_config, model_params):
    """
    Create a string of training arguments from the configuration.
    
    Args:
        task_config: Task configuration name
        model_config: Model configuration name
        model_params: Dictionary of model parameters
        
    Returns:
        String of command-line arguments
    """
    args = f"--task_cfg {task_config} --model_cfg {model_config} --progress_bar"
    
    # Add model parameters
    for key, value in model_params.items():
        if isinstance(value, bool):
            if value:
                args += f" --{key}"
            continue
        args += f" --{key} {value}"
    
    return args

def main():
    """Main function to parse arguments and submit jobs."""
    parser = argparse.ArgumentParser(description="Submit CGCNN model training to SLURM with checkpoint saving")
    parser.add_argument("--task_config", type=str, default="tsd_ssd_ws24", 
                        help="Task configuration name")
    parser.add_argument("--model_config", type=str, default="att_cgcnn", 
                        help="Model configuration name")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=500, 
                        help="Maximum number of training epochs")
    # parser.add_argument("--random_seed", type=int, default=42, 
    #                     help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set up paths
    SCRIPT_DIR = Path(__file__).absolute().parent
    ROOT_DIR = SCRIPT_DIR.parent.parent
    
    # Define model configuration with default values
    model_params = {
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'max_graph_len': 200,
        'atom_fea_len': 144,
        'extra_fea_len': 28,
        'h_fea_len': 144,
        'n_conv': 4,
        'n_h': 8,
        'dropout_prob': 0.55,
        'use_extra_fea': False,
        'use_cell_params': True,
        'atom_layer_norm': True,
        'loss_aggregation': "fixed_weight_sum",
        'dl_sampler': 'random',
        'task_att_type': 'self',
        'augment': False,
        'lr': 0.001,
        'lr_mult': 1,
        'group_lr': True,
        'optim_config': "fine",
        'auto_lr_bs_find': False, 
        'patience': 50,
        'att_pooling': False,
        'task_norm': True,
        'reconstruct': False,
        'log_dir': os.path.join(ROOT_DIR, "results/cgcnn_models"),
        'optim': 'adam',
        # 'decay_power': 1.0,
    }
    
    # Create job name
    job_name = f"train_{args.task_config}_{args.model_config}"
    
    # Create training arguments string
    training_args = create_training_args(args.task_config, args.model_config, model_params)
    
    # Create the SLURM job script
    job_script = JOB_TEMPLATE.format(
        job_name=job_name,
        training_args=training_args,
        script_exe=str(ROOT_DIR/"src/cgcnn/main.py")  # Path to the training script
    )
    
    # Write the job script to a file
    slurm_script_path = SCRIPT_DIR / "run_train.sh"
    with open(slurm_script_path, "w") as f:
        f.write(job_script)
    
    print(f"Created SLURM job script: {slurm_script_path}")
    print(f"Job name: {job_name}")
    
    # Submit the job
    process = run_slurm_job(SCRIPT_DIR)
    
    # Get the output of the job submission
    stdout, stderr = process.communicate()
    
    if stdout:
        print("STDOUT:", stdout.decode().strip())
    if stderr:
        print("STDERR:", stderr.decode().strip())
    
    print(f"Submitted job {job_name}")

if __name__ == "__main__":
    main()