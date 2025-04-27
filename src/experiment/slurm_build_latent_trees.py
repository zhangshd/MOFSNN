#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-04-26
Description: Submit batch jobs to SLURM for building latent vector trees from multiple models.
This script creates and submits SLURM job scripts to process multiple model directories
with build_latent_vec_tree.py for uncertainty analysis.
'''

import subprocess
import os
import time
import argparse
import glob
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent.parent

# SLURM job template
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

# Build latent vector trees for model
echo "Starting latent vector tree building for {model_dir}..."

{cmd}

echo "Processing completed."
""".strip()

def run_slurm_job(work_dir, executor="sbatch", script_name="run_build_tree.sh"):
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

def create_build_tree_command(model_dir, output_dir=None, k=5):
    """
    Create a command string for the build_latent_vec_tree.py script.
    
    Args:
        model_dir: Path to the model directory
        output_dir: Directory to save output files
        k: Number of nearest neighbors to use for uncertainty calculation
        
    Returns:
        Command string
    """
    # Get path to the build_latent_vec_tree.py script
    SCRIPT_DIR = Path(__file__).absolute().parent
    script_path = SCRIPT_DIR / "build_latent_vec_tree.py"
    
    # Construct command
    cmd = f"srun python -u {script_path} --model_dir {model_dir} --k {k} --process_all"
    
    if output_dir:
        cmd += f" --output_dir {output_dir}"
    
    return cmd

def find_model_dirs(base_dir, model_pattern="**/version_*", nested=True):
    """
    Find all model directories matching the pattern in the base directory.
    
    Args:
        base_dir: Base directory to search in
        model_pattern: Pattern to match model directories
        nested: Whether to look for nested model directories (e.g., version subdirectories)
        
    Returns:
        List of model directory paths
    """
    base_dir = Path(base_dir)
    
    if nested:
        # Find version_* directories within the model directories
        model_dirs = list(base_dir.glob(model_pattern))
    else:
        # Use the directories directly
        model_dirs = [base_dir / d for d in os.listdir(base_dir) 
                     if (base_dir / d).is_dir()]
    
    return [str(d) for d in model_dirs]

def parse_model_name(model_dir):
    """
    Extract a readable model name from the model directory path.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Model name string
    """
    # Split the path and get the last parts
    path_parts = Path(model_dir).parts
    
    # For version subdirectories, combine parent dir and version
    if "version_" in path_parts[-1]:
        model_name = f"{path_parts[-2]}_{path_parts[-1]}"
    else:
        model_name = path_parts[-1]
    
    # Replace any special characters that might cause issues in job names
    model_name = model_name.replace('-', '_').replace('.', '_')
    
    return model_name

def main():
    """Main function to parse arguments and submit jobs."""
    parser = argparse.ArgumentParser(description="Submit batch jobs to SLURM for building latent vector trees")
    # parser.add_argument("--models_base_dir", type=str, required=True, 
    #                     help="Base directory containing model directories to process")
    parser.add_argument("--output_base_dir", type=str, default=str(ROOT_DIR/"results/uncertainty_evolution"),
                        help="Base directory where outputs will be saved")
    parser.add_argument("--k", type=int, default=5, 
                        help="Number of nearest neighbors to use for uncertainty calculation")
    parser.add_argument("--model_pattern", type=str, default="**/version_*", 
                        help="Pattern to match model directories")
    parser.add_argument("--nested", action="store_true", default=True,
                        help="Whether to look for nested model directories (e.g., version subdirectories)")
    args = parser.parse_args()

    # Set up paths
    SCRIPT_DIR = Path(__file__).absolute().parent
    # models_base_dir = Path(args.models_base_dir)
    
    if args.output_base_dir:
        output_base_dir = Path(args.output_base_dir)
    else:
        output_base_dir = None
    
    # Find all model directories
    # print(f"Searching for model directories in {models_base_dir}")
    # model_dirs = find_model_dirs(models_base_dir, args.model_pattern, args.nested)
    
    
    # if not model_dirs:
    #     print(f"No model directories found in {models_base_dir}")
    #     return

    model_dirs = [
        ROOT_DIR/"results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_45",
        ROOT_DIR/"results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_51",
        ROOT_DIR/"results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_52",
    ]
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Submit jobs for each model directory
    submitted_jobs = []
    
    for i, model_dir in enumerate(model_dirs):
        model_name = parse_model_name(model_dir)
        job_name = f"build_tree_{model_name}"
        
        print(f"\nPreparing job {i+1}/{len(model_dirs)}: {job_name}")
        print(f"Model directory: {model_dir}")
        
        # Create output directory for this model if needed
        if output_base_dir:
            output_dir = output_base_dir / model_name
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = None
        
        # Create command
        cmd = create_build_tree_command(model_dir, output_dir, args.k)
        
        # Create the SLURM job script
        job_script = JOB_TEMPLATE.format(
            job_name=job_name,
            model_dir=model_dir,
            cmd=cmd
        )
        
        # Write the job script to a file
        slurm_script_path = SCRIPT_DIR / "run_build_tree.sh"
        with open(slurm_script_path, "w") as f:
            f.write(job_script)
        
        print(f"Created SLURM job script: {slurm_script_path}")
        
        # Submit the job
        process = run_slurm_job(SCRIPT_DIR)
        
        # Get the output of the job submission
        stdout, stderr = process.communicate()
        
        # Extract job ID from stdout
        if stdout:
            stdout_text = stdout.decode().strip()
            print(f"STDOUT: {stdout_text}")
            
            # Extract job ID
            import re
            job_id_match = re.search(r'Submitted batch job (\d+)', stdout_text)
            if job_id_match:
                job_id = job_id_match.group(1)
                submitted_jobs.append(job_id)
                print(f"Submitted job {job_name} with ID {job_id}")
            else:
                print(f"Could not extract job ID from output: {stdout_text}")
        
        if stderr:
            print(f"STDERR: {stderr.decode().strip()}")
        
        print(f"Submitted job {i+1}/{len(model_dirs)}: {job_name}")
        
        # Avoid overwhelming the scheduler
        time.sleep(1)
    
    print(f"\nSubmitted {len(submitted_jobs)} jobs.")
    print(f"Job IDs: {', '.join(submitted_jobs)}")

if __name__ == "__main__":
    main()