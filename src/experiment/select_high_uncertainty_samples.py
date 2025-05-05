'''
Author: zhangshd
Date: 2025-04-28
Description: This script selects samples with high uncertainty for data augmentation.
It reads results from vis_uncertainty_in_latent_space.py, selects the top 20% high uncertainty
samples from the training set for each task, and saves the list to an Excel file.
'''

import os
import sys
# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory of the project (two levels up from the script directory)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Add the src directory to the Python path
sys.path.append(os.path.dirname(SCRIPT_DIR))
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from pathlib import Path

def collect_uncertainty_data(input_dir):
    """
    Collect uncertainty data from CSV files in the input directory.
    
    Args:
        input_dir: Directory containing CSV files with uncertainty data
        
    Returns:
        Dictionary mapping task names to DataFrames with uncertainty data for each split
    """
    input_dir = Path(input_dir)
    
    # Find all CSV files in the input directory
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    
    # Group files by task and split
    task_data = {}
    
    for file_path in csv_files:
        # Parse filename to extract task and split
        # Expected format: {task}_{split}_results.csv
        file_name = file_path.stem
        parts = file_name.split('_')
        
        if len(parts) < 2 or not file_name.endswith('_results'):
            print(f"Skipping file with unexpected name format: {file_path}")
            continue
        
        # Extract task and split
        if len(parts) == 2:  # {task}_results.csv
            task = parts[0]
            split = None
        else:  # {task}_{split}_results.csv
            # Find the position of 'train', 'val', or 'test'
            split_keywords = ['train', 'val', 'test']
            for i, part in enumerate(parts):
                if part in split_keywords:
                    task = '_'.join(parts[:i])
                    split = part
                    break
            else:
                # If no split keyword found, assume the last part is the split
                task = '_'.join(parts[:-2])
                split = parts[-2]
        
        # Initialize task entry if not exists
        if task not in task_data:
            task_data[task] = {}
        
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
            if split:
                task_data[task][split] = df
            else:
                task_data[task]['combined'] = df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return task_data

def select_high_uncertainty_samples(task_data, percentage=20):
    """
    Select samples with high uncertainty from the training set.
    
    Args:
        task_data: Dictionary mapping task names to DataFrames with uncertainty data
        percentage: Percentage of high uncertainty samples to select (default: 20%)
        
    Returns:
        Dictionary mapping task names to DataFrames with selected sample IDs and uncertainty values
    """
    selected_samples = {}
    
    for task, splits in task_data.items():
        if 'train' not in splits:
            print(f"No training data found for task {task}. Skipping.")
            continue
        
        train_df = splits['train']
        
        # Sort by uncertainty in descending order
        sorted_df = train_df.sort_values('uncertainty', ascending=False)
        
        # Select the top percentage of samples
        n_samples = int(len(sorted_df) * percentage / 100)
        selected_df = sorted_df.head(n_samples)
        
        # Store the selected sample IDs and their uncertainty values
        selected_samples[task] = selected_df[['cif_id', 'uncertainty']]
        
        print(f"Task {task}: Selected {n_samples} samples out of {len(sorted_df)} ({percentage}%) with uncertainty ranging from {selected_df['uncertainty'].min():.4f} to {selected_df['uncertainty'].max():.4f}")
    
    return selected_samples

def export_to_excel(selected_samples, output_path):
    """
    Export selected samples to an Excel file, with one sheet per task.
    
    Args:
        selected_samples: Dictionary mapping task names to DataFrames with sample data
        output_path: Path to the output Excel file
        
    Returns:
        Path to the saved Excel file
    """
    with pd.ExcelWriter(output_path) as writer:
        for task, df in selected_samples.items():
            # Add metadata about the selection
            df.attrs['task'] = task
            df.attrs['count'] = len(df)
            
            # Write to a sheet named after the task
            df.to_excel(writer, sheet_name=task, index=False)
    
    print(f"Exported selected samples to {output_path}")
    return output_path

def main():
    parser = ArgumentParser(description="Select high uncertainty samples for data augmentation")
    parser.add_argument("--input_dir", type=str, 
                        default=os.path.join(ROOT_DIR, "results/uncertainty_visualization"),
                        help="Directory containing uncertainty data CSV files")
    parser.add_argument("--output_file", type=str, 
                        default=os.path.join(ROOT_DIR, "results/high_uncertainty_samples_0.5.xlsx"),
                        help="Path to output Excel file")
    parser.add_argument("--percentage", type=float, default=20.0,
                        help="Percentage of high uncertainty samples to select (default: 20%%)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect uncertainty data
    print(f"Collecting uncertainty data from {args.input_dir}")
    task_data = collect_uncertainty_data(args.input_dir)
    
    # Select high uncertainty samples
    print(f"Selecting top {args.percentage}% high uncertainty samples from training set")
    selected_samples = select_high_uncertainty_samples(task_data, args.percentage)
    
    # Export to Excel
    export_to_excel(selected_samples, args.output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
