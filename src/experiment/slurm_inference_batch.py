#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-05-16
Description: Submit batch inference jobs to SLURM for both ML and CGCNN models.
This script creates and submits SLURM job scripts to run inference on a directory
of CIF files with both ML and CGCNN models in parallel.
'''

import subprocess
import os
import time
import argparse
import glob
from pathlib import Path
import re

ROOT_DIR = Path(__file__).absolute().parent.parent.parent

# SLURM job template for ML inference
ML_JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
export PATH={conda_path}:$PATH
export LD_LIBRARY_PATH={conda_lib_path}:$LD_LIBRARY_PATH

# Run ML inference
echo "Starting ML inference for {input_path}..."

{ml_cmd}

echo "ML inference completed."
""".strip()

# SLURM job template for CGCNN inference
CGCNN_JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
export PATH={conda_path}:$PATH
export LD_LIBRARY_PATH={conda_lib_path}:$LD_LIBRARY_PATH

# Run CGCNN inference
echo "Starting CGCNN inference for {input_path}..."

{cgcnn_cmd}

echo "CGCNN inference completed."
""".strip()

def run_slurm_job(work_dir, script_path, executor="sbatch"):
    """
    Submit a SLURM job using the specified executor.
    
    Args:
        work_dir: Working directory
        script_path: Path to the script file to execute
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
        f"{executor} {script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(work_dir)
    )
    return process

def create_ml_inference_command(input_path, output_path):
    """
    Create a command string for the ML inference script.
    
    Args:
        input_path: Path to the input CIF file or directory
        output_path: Path to save the output CSV file
        config_file: Path to the ML model configuration file
        
    Returns:
        Command string
    """
    # Get path to the ml inference script
    ml_script_path = ROOT_DIR / "src/ml/inference.py"
    
    # Construct command
    cmd = f"srun python -u {ml_script_path} --input_path {input_path} --output_path {output_path}"

    
    return cmd

def create_cgcnn_inference_command(input_path, output_path,
                                  uncertainty=False
                                  ):
    """
    Create a command string for the CGCNN inference script.
    
    Args:
        input_path: Path to the input CIF file or directory
        output_path: Path to save the output CSV file
        model_dir: Path to the CGCNN model directory
        uncertainty: Whether to enable uncertainty estimation
        temp_dir: Directory to save temporary files
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        no_clean: Whether to skip cleaning CIF files before inference
        
    Returns:
        Command string
    """
    # Get path to the CGCNN inference script
    cgcnn_script_path = ROOT_DIR / "src/cgcnn/inference.py"
    
    # Construct command
    cmd = f"srun python -u {cgcnn_script_path} --input_path {input_path} --output_path {output_path}"
    
    if uncertainty:
        cmd += f" --uncertainty"
    
    return cmd

def extract_job_id(stdout_text):
    """
    Extract job ID from SLURM submission output.
    
    Args:
        stdout_text: Output text from SLURM submission
        
    Returns:
        Job ID string if found, None otherwise
    """
    job_id_match = re.search(r'Submitted batch job (\d+)', stdout_text)
    if job_id_match:
        return job_id_match.group(1)
    return None

def main():
    """Main function to parse arguments and submit jobs."""
    parser = argparse.ArgumentParser(description="Submit batch inference jobs to SLURM for ML and CGCNN models")
    
    # Input and output arguments
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to CIF file or directory containing CIF files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files")
    
    # SLURM configuration arguments
    parser.add_argument("--partition", type=str, default="C9654",
                        help="SLURM partition to use")
    parser.add_argument("--nodelist", type=str, default="c4",
                        help="Node list to use")
    parser.add_argument("--cpus_per_task", type=int, default=1,
                        help="Number of CPUs per task")
    parser.add_argument("--memory", type=str, default="64G",
                        help="Memory allocation (e.g., '64G')")
    parser.add_argument("--conda_path", type=str, 
                        default="/opt/share/miniconda3/envs/mofmthnn/bin",
                        help="Path to conda environment bin directory")
    parser.add_argument("--conda_lib_path", type=str,
                        default="/opt/share/miniconda3/envs/mofmthnn/lib",
                        help="Path to conda environment lib directory")
    
    # CGCNN model arguments
    parser.add_argument("--uncertainty", action="store_true", default=False,
                        help="Whether to enable uncertainty estimation")
    
    args = parser.parse_args()
    
    # Set up paths
    SCRIPT_DIR = Path(__file__).absolute().parent
    input_path = Path(args.input_path).absolute()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    
    # Setup output file paths
    input_name = input_path.stem if input_path.is_file() else input_path.name
    ml_output_path = output_dir / f"ml_inference_{input_name}.csv"
    cgcnn_output_path = output_dir / f"cgcnn_inference_{input_name}.csv"
    
    # Create job names
    ml_job_name = f"ml_inf_{input_name}"
    cgcnn_job_name = f"cgcnn_inf_{input_name}"
    
    # Create commands
    ml_cmd = create_ml_inference_command(
        input_path, 
        ml_output_path, 
    )
    
    cgcnn_cmd = create_cgcnn_inference_command(
        input_path, 
        cgcnn_output_path, 
        uncertainty=args.uncertainty,
    )
    
    # Create the SLURM job scripts
    ml_job_script = ML_JOB_TEMPLATE.format(
        job_name=ml_job_name,
        partition=args.partition,
        nodelist=args.nodelist,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        conda_path=args.conda_path,
        conda_lib_path=args.conda_lib_path,
        input_path=input_path,
        ml_cmd=ml_cmd
    )
    
    cgcnn_job_script = CGCNN_JOB_TEMPLATE.format(
        job_name=cgcnn_job_name,
        partition=args.partition,
        nodelist=args.nodelist,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        conda_path=args.conda_path,
        conda_lib_path=args.conda_lib_path,
        input_path=input_path,
        cgcnn_cmd=cgcnn_cmd
    )
    
    # Write job scripts to files
    ml_script_path = SCRIPT_DIR / "run_ml_inference.sh"
    cgcnn_script_path = SCRIPT_DIR / "run_cgcnn_inference.sh"
    
    with open(ml_script_path, "w") as f:
        f.write(ml_job_script)
    
    with open(cgcnn_script_path, "w") as f:
        f.write(cgcnn_job_script)
    
    print(f"Created SLURM job scripts:")
    print(f"  - ML inference: {ml_script_path}")
    print(f"  - CGCNN inference: {cgcnn_script_path}")
    
    # Submit the jobs
    ml_job_process = run_slurm_job(SCRIPT_DIR, ml_script_path)
    cgcnn_job_process = run_slurm_job(SCRIPT_DIR, cgcnn_script_path)
    
    # Get ML job output
    ml_stdout, ml_stderr = ml_job_process.communicate()
    
    if ml_stdout:
        ml_stdout_text = ml_stdout.decode().strip()
        print(f"ML Inference Job STDOUT: {ml_stdout_text}")
        
        ml_job_id = extract_job_id(ml_stdout_text)
        if ml_job_id:
            print(f"Submitted ML inference job with ID {ml_job_id}")
        else:
            print(f"Could not extract ML job ID from output: {ml_stdout_text}")
    
    if ml_stderr:
        print(f"ML Inference Job STDERR: {ml_stderr.decode().strip()}")
    
    # Get CGCNN job output
    cgcnn_stdout, cgcnn_stderr = cgcnn_job_process.communicate()
    
    if cgcnn_stdout:
        cgcnn_stdout_text = cgcnn_stdout.decode().strip()
        print(f"CGCNN Inference Job STDOUT: {cgcnn_stdout_text}")
        
        cgcnn_job_id = extract_job_id(cgcnn_stdout_text)
        if cgcnn_job_id:
            print(f"Submitted CGCNN inference job with ID {cgcnn_job_id}")
        else:
            print(f"Could not extract CGCNN job ID from output: {cgcnn_stdout_text}")
    
    if cgcnn_stderr:
        print(f"CGCNN Inference Job STDERR: {cgcnn_stderr.decode().strip()}")
    
    print("\nJobs submitted successfully.")
    print(f"Output files will be saved to:")
    print(f"  - ML inference: {ml_output_path}")
    print(f"  - CGCNN inference: {cgcnn_output_path}")

    ## remove the job scripts after submission
    ml_script_path.unlink()
    cgcnn_script_path.unlink()
    print(f"Removed job scripts: {ml_script_path}, {cgcnn_script_path}")

if __name__ == "__main__":
    main()
