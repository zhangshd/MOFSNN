#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-05-11
Description: Batch submit ML model training jobs to SLURM with different input files.
This script iterates through different datasets, labels and input files,
submitting SLURM jobs for each combination.
'''

import subprocess
import os
import time
import argparse
from pathlib import Path
import sys
import glob

# Set up paths
SCRIPT_DIR = Path(__file__).absolute().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(ROOT_DIR/"src/ml"))

# SLURM job template for ML training
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

echo "Starting ML model training with {in_file_name}..."
srun python -u {script_exe} --model_type {model_type} --model_list {model_list} \
 --search_max_evals {search_max_evals} --search_metric {search_metric} \
 --label_column {label_column} --group_column {group_column} --name_column {name_column} \
 --feature_selector_list {feature_selector_list} \
 --data_dir {data_dir} --in_file_name {in_file_name} {extra_args}

echo "Training completed."
""".strip()


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


def get_input_file_patterns(data_dir, pattern="RAC_and_zeo_features_with_id_prop_rand*.csv"):
    """
    Get a list of input file patterns based on existing files in the data directory.
    
    Args:
        data_dir: Path to the data directory
        pattern: Glob pattern to match input files
        
    Returns:
        List of input file patterns
    """
    file_pattern = Path(data_dir) / pattern
    files = glob.glob(str(file_pattern))
    
    if not files:
        print(f"No files found matching pattern '{pattern}' in {data_dir}")
        return [pattern]
    
    # Extract patterns from file names
    patterns = []
    for file_path in files:
        file_name = Path(file_path).name
        patterns.append(file_name)
    
    return patterns


def main():
    """Main function to parse arguments and submit jobs."""
    parser = argparse.ArgumentParser(description="Submit batch ML model training to SLURM with different input files")
    parser.add_argument("--tasks", type=str, nargs='+', default=["TSD", "SSD", "WS24"], 
                        help="Tasks/datasets to train on (TSD, SSD, WS24)")
    parser.add_argument("--model_types", type=str, nargs='+', default=["RF", "GP", "SVM"], 
                        help="Model types to train (RF, GP, SVM, etc.)")
    parser.add_argument("--labels", type=str, nargs='+', 
                        default=["Label", "water_label", "water4_label", "acid_label", "base_label", "boiling_label"],
                        help="Labels to train models for (depending on dataset)")
    parser.add_argument("--feature_selectors", type=str, nargs='+', default=["RFE", "f1", "mutual_info"], 
                        help="Feature selection methods to use")
    parser.add_argument("--data_dir_base", type=str, default=str(ROOT_DIR/"data/ml_data"), 
                        help="Base directory containing task data directories")
    parser.add_argument("--in_file_pattern", type=str, default="RAC_and_zeo_features_with_id_prop_rand*.csv", 
                        help="Pattern to match input files")
    parser.add_argument("--search_max_evals", type=int, default=100, 
                        help="Maximum evaluations for hyperparameter search")
    parser.add_argument("--wait_time", type=int, default=2, 
                        help="Wait time in seconds between job submissions")
    parser.add_argument("--dry_run", action="store_true", 
                        help="Create job scripts but don't submit them")
    args = parser.parse_args()

    # Create a directory for job scripts
    job_scripts_dir = SCRIPT_DIR / "job_scripts"
    job_scripts_dir.mkdir(exist_ok=True, parents=True)
    
    # Loop through tasks
    for task in args.tasks:
        print(f"\nProcessing task: {task}")
        
        # Determine model type and search metric based on task
        if task == "TSD":
            model_type = "regression"
            search_metric = "val_R2"
        else:  # SSD and WS24
            model_type = "classification"
            search_metric = "val_AUC"
            
        # Determine data directory for the task
        data_dir = Path(args.data_dir_base) / task
        
        # Get input file patterns for this task
        in_file_patterns = get_input_file_patterns(data_dir, args.in_file_pattern)
        print(f"Found {len(in_file_patterns)} file patterns in {data_dir}")
        
        # Determine which labels to use for this task
        task_labels = []
        if task in ["TSD", "SSD"]:
            task_labels = ["Label"]
        elif task == "WS24":
            task_labels = [label for label in args.labels if label != "Label"]
        
        if not task_labels:
            print(f"No applicable labels found for task {task}, using default 'Label'")
            task_labels = ["Label"]
            
        # Combine model types
        model_list = " ".join(args.model_types)
        feature_selector_list = " ".join(args.feature_selectors)
        
        # Loop through labels
        for label in task_labels:
            print(f"  Processing label: {label}")
            
            # Loop through input files
            for in_file in in_file_patterns:
                print(f"    Processing input file: {in_file}")
                
                # Determine name column based on task
                name_column = "MofName"
                
                # Use label as group column by default
                group_column = label
                
                # Create extra args string
                extra_args = ""
                
                # Create job name
                file_short_name = in_file.replace('.csv', '').replace('RAC_and_zeo_features_with_id_prop', '')
                if file_short_name == "":
                    file_short_name = "base"
                    
                job_name = f"ml_train_{task}_{label}_{file_short_name}"
                
                # Create job script
                job_script = JOB_TEMPLATE.format(
                    job_name=job_name,
                    in_file_name=in_file,
                    script_exe=str(ROOT_DIR/"src/ml/main.py"),
                    model_type=model_type,
                    model_list=model_list,
                    search_max_evals=args.search_max_evals,
                    search_metric=search_metric,
                    label_column=label,
                    group_column=group_column,
                    name_column=name_column,
                    feature_selector_list=feature_selector_list,
                    data_dir=data_dir,
                    extra_args=extra_args
                )
                
                # Write job script to file
                job_script_path = job_scripts_dir / f"{job_name}.sh"
                with open(job_script_path, "w") as f:
                    f.write(job_script)
                
                print(f"    Created job script: {job_script_path}")
                
                # Submit job if not in dry run mode
                if not args.dry_run:
                    print(f"    Submitting job: {job_name}")
                    run_slurm_job(SCRIPT_DIR, job_script_path)
                    
                    # Wait to avoid overloading the scheduler
                    time.sleep(args.wait_time)
                else:
                    print(f"    DRY RUN: Would submit job: {job_name}")

if __name__ == "__main__":
    main()
