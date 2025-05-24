#!/usr/bin/env python
"""
Author: zhangshd
Date: 2025-05-24
Description: Submit batch stability prediction jobs to SLURM for MOF reference stability analysis.
This script creates and submits SLURM job scripts to run batch_reference_stability.py on a directory
of CIF files with specified parameters for thermal/solvent and water/acid/base/boiling stability predictions.
"""

import subprocess
import os
import time
import argparse
import glob
from pathlib import Path
import re

ROOT_DIR = Path(__file__).absolute().parent.parent.parent

# SLURM job template for stability prediction
STABILITY_JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}

# Set up environment paths
export PATH={conda_path}:$PATH
export LD_LIBRARY_PATH={conda_lib_path}:$LD_LIBRARY_PATH

# Run MOF stability prediction
echo "Starting MOF stability prediction for {input_dir}..."
echo "Output directory: {output_dir}"
echo "Workers: {workers}"

{stability_cmd}

echo "MOF stability prediction completed."
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

def create_stability_command(input_dir, output_dir, working_dir=None, limit=None, 
                           workers=1, python_executable_ts=None, python_executable_ws=None):
    """
    Create a command string for the batch stability prediction script.
    
    Args:
        input_dir: Path to the input directory containing CIF files
        output_dir: Path to save the output files
        working_dir: Directory for intermediate files
        limit: Limit the number of CIF files to process
        workers: Number of parallel workers
        python_executable_ts: Python executable for thermal/solvent stability
        python_executable_ws: Python executable for water/acid/base/boiling stability
        
    Returns:
        Command string
    """
    # Get path to the stability prediction script
    stability_script_path = ROOT_DIR / "src/experiment/batch_reference_stability.py"
    
    # Construct base command
    cmd = f"srun python -u {stability_script_path} --input_dir {input_dir} --output_dir {output_dir}"
    
    # Add optional parameters
    if working_dir:
        cmd += f" --working_dir {working_dir}"
    
    if limit:
        cmd += f" --limit {limit}"
    
    if workers and workers > 1:
        cmd += f" --workers {workers}"
    
    if python_executable_ts:
        cmd += f" --python_executable_ts {python_executable_ts}"
    
    if python_executable_ws:
        cmd += f" --python_executable_ws {python_executable_ws}"
    
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
    parser = argparse.ArgumentParser(
        description="Submit batch MOF stability prediction jobs to SLURM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input and output arguments
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing CIF files to process")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--working_dir", type=str, default=None,
                        help="Directory for intermediate files (default: output_dir/working)")
    
    # Processing parameters
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of CIF files to process (for testing)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers to use")
    parser.add_argument("--python_executable_ts", type=str, 
                        default="/opt/share/miniconda3/envs/MOFSimplify/bin/python",
                        help="Python executable for thermal/solvent stability prediction")
    parser.add_argument("--python_executable_ws", type=str,
                        default="/opt/share/miniconda3/envs/test/bin/python", 
                        help="Python executable for water/acid/base/boiling stability prediction")
    
    # SLURM configuration arguments
    parser.add_argument("--partition", type=str, default="C9654",
                        help="SLURM partition to use")
    parser.add_argument("--nodelist", type=str, default="c4",
                        help="Node list to use")
    parser.add_argument("--cpus_per_task", type=int, default=1,
                        help="Number of CPUs per task")
    parser.add_argument("--memory", type=str, default="64G",
                        help="Memory allocation (e.g., '64G')")
    parser.add_argument("--time_limit", type=str, default="24:00:00",
                        help="Time limit for the job (format: HH:MM:SS)")
    parser.add_argument("--conda_path", type=str, 
                        default="/opt/share/miniconda3/envs/mofmthnn/bin",
                        help="Path to conda environment bin directory")
    parser.add_argument("--conda_lib_path", type=str,
                        default="/opt/share/miniconda3/envs/mofmthnn/lib",
                        help="Path to conda environment lib directory")
    
    args = parser.parse_args()
    
    # Set up paths
    SCRIPT_DIR = Path(__file__).absolute().parent
    input_dir = Path(args.input_dir).absolute()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set working directory
    working_dir = args.working_dir
    if working_dir:
        working_dir = Path(working_dir).absolute()
        working_dir.mkdir(exist_ok=True, parents=True)
    
    # Validate input directory
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create job name based on input directory
    input_name = input_dir.name
    job_name = f"reference_infer_{input_name}"
    if args.limit:
        job_name += f"_limit{args.limit}"
    
    # Create command
    stability_cmd = create_stability_command(
        input_dir=input_dir,
        output_dir=output_dir,
        working_dir=working_dir,
        limit=args.limit,
        workers=args.workers,
        python_executable_ts=args.python_executable_ts,
        python_executable_ws=args.python_executable_ws
    )
    
    # Create the SLURM job script
    stability_job_script = STABILITY_JOB_TEMPLATE.format(
        job_name=job_name,
        partition=args.partition,
        nodelist=args.nodelist,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        time_limit=args.time_limit,
        conda_path=args.conda_path,
        conda_lib_path=args.conda_lib_path,
        input_dir=input_dir,
        output_dir=output_dir,
        workers=args.workers,
        stability_cmd=stability_cmd
    )
    
    # Write job script to file
    script_path = SCRIPT_DIR / f"run_stability_{input_name}.sh"
    
    with open(script_path, "w") as f:
        f.write(stability_job_script)
    
    print(f"Created SLURM job script: {script_path}")
    print(f"Job name: {job_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if working_dir:
        print(f"Working directory: {working_dir}")
    if args.limit:
        print(f"Processing limit: {args.limit} files")
    print(f"Workers: {args.workers}")
    print(f"Python executable (TS): {args.python_executable_ts}")
    print(f"Python executable (WS): {args.python_executable_ws}")
    
    # Submit the job
    print(f"\nSubmitting job to SLURM...")
    job_process = run_slurm_job(SCRIPT_DIR, script_path)
    
    # Get job output
    stdout, stderr = job_process.communicate()
    
    if stdout:
        stdout_text = stdout.decode().strip()
        print(f"Job STDOUT: {stdout_text}")
        
        job_id = extract_job_id(stdout_text)
        if job_id:
            print(f"Submitted stability prediction job with ID {job_id}")
        else:
            print(f"Could not extract job ID from output: {stdout_text}")
    
    if stderr:
        print(f"Job STDERR: {stderr.decode().strip()}")
    
    print(f"\nJob submitted successfully.")
    print(f"Monitor job progress with: squeue -j <job_id>")
    print(f"Check job logs in: {SCRIPT_DIR}/slurm_logs/")
    print(f"Results will be saved to: {output_dir}")
    
    # Remove the job script after submission
    script_path.unlink()
    print(f"Removed job script: {script_path}")

if __name__ == "__main__":
    main()
