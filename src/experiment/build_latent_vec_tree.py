#!/usr/bin/env python
'''
Author: zhangshd
Date: 2025-04-24
Description: This script builds ball trees from latent vectors for uncertainty calculation.
It extracts latent vectors from a trained model, then builds and saves ball trees
that can be used for Local Similarity Variance (LSV) or Local Similarity Entropy (LSE) calculation.
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
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import BallTree

from cgcnn.utils import load_model_from_dir
from experiment.vis_uncertainty_in_latent_space import predict_and_collect_features, create_data_module
from experiment.vis_uncertainty_in_latent_space import get_dataloader_from_datamodule

def extract_latent_vectors(model_dir, output_dir=None, split='train', k=5):
    """
    Extract latent vectors from a trained model and save them.
    
    Args:
        model_dir: Directory containing the trained model
        output_dir: Directory to save the extracted vectors
        split: Data split to use ('train', 'val', or 'test')
        k: Number of nearest neighbors to consider for ball tree
        
    Returns:
        Dictionary of task-specific latent vectors, targets, predictions, and ball trees
    """
    if output_dir is None:
        output_dir = Path(model_dir) / "latent_vectors"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {model_dir}")
    model, _ = load_model_from_dir(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create data module
    data_module = create_data_module(model)
    
    # Get dataloader for the specified split
    dataloader = get_dataloader_from_datamodule(model, data_module, split)
    
    # Extract features
    features, preds, targets, cif_ids = predict_and_collect_features(model, dataloader, device)
    
    # Process each task
    results = {}
    
    for task_id, task in enumerate(model.hparams.tasks):
        if task not in features or len(features[task]) == 0:
            print(f"No data for task {task} in {split} set. Skipping.")
            continue
        
        print(f"Processing task: {task}")
        task_type = model.hparams.task_types[task_id]
        
        # Build ball tree for this task
        ball_tree = BallTree(features[task], leaf_size=40)
        
        # Create a dictionary to store task-specific data
        task_data = {
            "features": features[task],
            "targets": targets[task],
            "predictions": preds[task],
            "cif_ids": cif_ids[task],
            "task_type": task_type,
            "ball_tree": ball_tree,
            "k": k
        }
        
        # Save task data as numpy arrays
        np.savez(
            output_dir / f"{task}_{split}_latent_vectors.npz",
            features=features[task],
            targets=targets[task],
            predictions=preds[task],
            cif_ids=np.array(cif_ids[task]),
            task_type=np.array([task_type])
        )
        
        # Store in results
        results[task] = task_data

    # Return the results
    return results

def build_uncertainty_trees(model_dir, output_dir=None, k=5, create_npz=True):
    """
    Build uncertainty trees (ball trees) for each task based on training data latent vectors.
    
    Args:
        model_dir: Directory containing the trained model
        output_dir: Directory to save the output files
        k: Number of nearest neighbors to use for uncertainty calculation
        create_npz: Whether to create npz files with latent vectors
        
    Returns:
        Dictionary of uncertainty trees for each task
    """
    if output_dir is None:
        output_dir = Path(model_dir) / "uncertainty_trees"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract latent vectors from training set
    print("Extracting latent vectors from training set...")
    train_results = extract_latent_vectors(model_dir, output_dir, split='train', k=k)
    
    # Create uncertainty trees
    uncertainty_trees = {}
    
    for task, task_data in train_results.items():
        task_type = task_data["task_type"]
        latent_vectors = task_data["features"]
        
        # Build ball tree for this task using training data
        ball_tree = BallTree(latent_vectors, metric='minkowski', p=2)
        
        # Calculate average distance between training points (needed for normalization)
        # Query the tree for k nearest neighbors of each training point
        dist_train, ind_train = ball_tree.query(latent_vectors, k=k+1, dualtree=True)
        # Skip the first neighbor (the point itself with distance 0)
        dist_train = dist_train[:, 1:]
        # Calculate average distance
        avg_dist_traintrain = dist_train.mean(axis=None)
        print(f"Task {task}: Average distance to {k} nearest neighbors in training data: {avg_dist_traintrain:.4f}")
        
        # Create uncertainty tree with the correct structure for module_utils.py functions
        uncertainty_trees[task] = {
            "tree": ball_tree,  # Key must be "tree" to match module_utils.py
            "labels_train": task_data["targets"],  # Key must be "labels_train"
            "avg_dist_traintrian": avg_dist_traintrain,  # Required for normalization
            "task_type": task_type,
            "k": k
        }
    
    # Save uncertainty trees
    uncertainty_trees_file = output_dir / "uncertainty_trees.pkl"
    with open(uncertainty_trees_file, 'wb') as f:
        pickle.dump(uncertainty_trees, f)
    
    print(f"Saved uncertainty trees to {uncertainty_trees_file}")
    
    return uncertainty_trees

def process_checkpoint(checkpoint_path, output_dir=None, k=5):
    """
    Process a single checkpoint file to extract latent vectors and build uncertainty trees.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save the output files
        k: Number of nearest neighbors to use for uncertainty calculation
        
    Returns:
        Dictionary of uncertainty trees for each task
    """
    # Extract checkpoint info from path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model_dir = os.path.dirname(checkpoint_dir)
    
    if output_dir is None:
        # Create a name based on the checkpoint filename
        checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
        output_dir = Path(model_dir) / f"latent_vectors_{checkpoint_name}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model directly with the specific checkpoint using the updated function
    print(f"Loading model from {model_dir} with checkpoint {checkpoint_path}")
    model, _ = load_model_from_dir(model_dir, custom_checkpoint=checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create data module
    data_module = create_data_module(model)
    
    # Get dataloaders
    train_loader = get_dataloader_from_datamodule(model, data_module, 'train')
    val_loader = get_dataloader_from_datamodule(model, data_module, 'val')
    test_loader = get_dataloader_from_datamodule(model, data_module, 'test')
    
    # Extract features for training, validation and test sets
    print("Extracting training set features...")
    train_features, train_preds, train_targets, train_cif_ids = predict_and_collect_features(model, train_loader, device)
    print("Extracting validation set features...")
    val_features, val_preds, val_targets, val_cif_ids = predict_and_collect_features(model, val_loader, device)
    print("Extracting test set features...")
    test_features, test_preds, test_targets, test_cif_ids = predict_and_collect_features(model, test_loader, device)
    
    # Process each task
    uncertainty_trees = {}
    results = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for task_id, task in enumerate(model.hparams.tasks):
        if task not in train_features or len(train_features[task]) == 0:
            print(f"No data for task {task}. Skipping.")
            continue
        
        print(f"Processing task: {task}")
        task_type = model.hparams.task_types[task_id]
        latent_vectors = train_features[task]
        
        # Build ball tree for this task using training data
        ball_tree = BallTree(latent_vectors, metric='minkowski', p=2)
        
        # Calculate average distance between training points (needed for normalization)
        # Query the tree for k nearest neighbors of each training point
        dist_train, ind_train = ball_tree.query(latent_vectors, k=k+1, dualtree=True)
        # Skip the first neighbor (the point itself with distance 0)
        dist_train = dist_train[:, 1:]
        # Calculate average distance
        avg_dist_traintrain = dist_train.mean(axis=None)
        print(f"Task {task}: Average distance to {k} nearest neighbors in training data: {avg_dist_traintrain:.4f}")
        
        # Create uncertainty tree with the correct structure for module_utils.py functions
        uncertainty_trees[task] = {
            "tree": ball_tree,  # Key must be "tree" to match module_utils.py
            "labels_train": train_targets[task],  # Key must be "labels_train"
            "avg_dist_traintrian": avg_dist_traintrain,  # Required for normalization
            "task_type": task_type,
            "k": k
        }
        
        # For storing in results, we keep the original structure with more data
        results_tree_data = {
            "features": latent_vectors,
            "targets": train_targets[task],
            "predictions": train_preds[task],
            "cif_ids": train_cif_ids[task],
            "task_type": task_type,
            "ball_tree": ball_tree,
            "avg_dist_traintrain": avg_dist_traintrain,
            "k": k
        }
        
        # Save features, targets and predictions for each split
        for split, features, preds, targets, cif_ids in [
            ('train', train_features, train_preds, train_targets, train_cif_ids),
            ('val', val_features, val_preds, val_targets, val_cif_ids),
            ('test', test_features, test_preds, test_targets, test_cif_ids)
        ]:
            if task not in features or len(features[task]) == 0:
                print(f"No data for task {task} in {split} set. Skipping.")
                continue
            
            # Save to npz file
            np.savez(
                output_dir / f"{task}_{split}_latent_vectors.npz",
                features=features[task],
                targets=targets[task],
                predictions=preds[task],
                cif_ids=np.array(cif_ids[task]),
                task_type=np.array([task_type])
            )
            
            # Store in results
            results[split][task] = {
                "features": features[task],
                "targets": targets[task],
                "predictions": preds[task],
                "cif_ids": cif_ids[task],
                "task_type": task_type
            }
    
    # Save uncertainty trees
    uncertainty_trees_file = output_dir / "uncertainty_trees.pkl"
    with open(uncertainty_trees_file, 'wb') as f:
        pickle.dump(uncertainty_trees, f)
    
    print(f"Saved uncertainty trees to {uncertainty_trees_file}")
    
    # Save results dictionary
    results_file = output_dir / "results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved results to {results_file}")
    
    return uncertainty_trees, results

def process_all_checkpoints(model_dir, output_dir=None, k=5):
    """
    Process all checkpoints in a model directory to track uncertainty evolution.
    This optimized version only loads the data module once for all checkpoints.
    
    Args:
        model_dir: Directory containing the trained model and its checkpoints
        output_dir: Directory to save the output files
        k: Number of nearest neighbors to use for uncertainty calculation
        
    Returns:
        List of processed checkpoint paths
    """
    if output_dir is None:
        output_dir = Path(model_dir) / "uncertainty_evolution"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all checkpoint files
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    print(f"Looking for checkpoints in {checkpoint_dir}")
    
    import glob
    import re
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "**/*.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return []
    
    # Extract epoch number from checkpoint file name
    def extract_epoch(filename):
        match = re.search(r'epoch=(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    # Sort checkpoints by epoch
    checkpoint_files.sort(key=extract_epoch)
    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Load base model once - use this to create the data module
    print(f"Loading base model from {model_dir} to initialize data module")
    base_model, _ = load_model_from_dir(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data module once
    print("Creating data module (will be reused for all checkpoints)...")
    data_module = create_data_module(base_model)
    
    # Get dataloaders once
    print("Preparing dataloaders (will be reused for all checkpoints)...")
    train_loader = get_dataloader_from_datamodule(base_model, data_module, 'train')
    val_loader = get_dataloader_from_datamodule(base_model, data_module, 'val')
    test_loader = get_dataloader_from_datamodule(base_model, data_module, 'test')
    
    # Create directory for each checkpoint's results
    processed_checkpoints = []
    
    for checkpoint_file in checkpoint_files:
        epoch = extract_epoch(checkpoint_file)
        print(f"\nProcessing checkpoint for epoch {epoch}: {checkpoint_file}")
        
        # Create output directory for this checkpoint
        checkpoint_output_dir = output_dir / f"epoch_{epoch:03d}"
        checkpoint_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process this checkpoint
        try:
            # Load model with this specific checkpoint using the improved function
            print(f"Loading model from {model_dir} with checkpoint {checkpoint_file}")
            model, _ = load_model_from_dir(model_dir, custom_checkpoint=checkpoint_file)
            model = model.to(device)
            model.eval()  # Set model to evaluation mode
            
            # Extract features for training, validation and test sets using the pre-loaded dataloaders
            print("Extracting training set features...")
            train_features, train_preds, train_targets, train_cif_ids = predict_and_collect_features(model, train_loader, device)
            print("Extracting validation set features...")
            val_features, val_preds, val_targets, val_cif_ids = predict_and_collect_features(model, val_loader, device)
            print("Extracting test set features...")
            test_features, test_preds, test_targets, test_cif_ids = predict_and_collect_features(model, test_loader, device)
            
            # Process each task
            uncertainty_trees = {}
            results = {
                'train': {},
                'val': {},
                'test': {}
            }
            
            for task_id, task in enumerate(model.hparams.tasks):
                if task not in train_features or len(train_features[task]) == 0:
                    print(f"No data for task {task}. Skipping.")
                    continue
                
                print(f"Processing task: {task}")
                task_type = model.hparams.task_types[task_id]
                latent_vectors = train_features[task]
                
                # Build ball tree for this task using training data
                ball_tree = BallTree(latent_vectors, metric='minkowski', p=2)
                
                # Calculate average distance between training points (needed for normalization)
                # Query the tree for k nearest neighbors of each training point
                dist_train, ind_train = ball_tree.query(latent_vectors, k=k+1, dualtree=True)
                # Skip the first neighbor (the point itself with distance 0)
                dist_train = dist_train[:, 1:]
                # Calculate average distance
                avg_dist_traintrain = dist_train.mean(axis=None)
                print(f"Task {task}: Average distance to {k} nearest neighbors in training data: {avg_dist_traintrain:.4f}")
                
                # Create uncertainty tree with the correct structure for module_utils.py functions
                uncertainty_trees[task] = {
                    "tree": ball_tree,  # Key must be "tree" to match module_utils.py
                    "labels_train": train_targets[task],  # Key must be "labels_train"
                    "avg_dist_traintrian": avg_dist_traintrain,  # Required for normalization
                    "task_type": task_type,
                    "k": k
                }
                
                # Save features, targets and predictions for each split
                for split, features, preds, targets, cif_ids in [
                    ('train', train_features, train_preds, train_targets, train_cif_ids),
                    ('val', val_features, val_preds, val_targets, val_cif_ids),
                    ('test', test_features, test_preds, test_targets, test_cif_ids)
                ]:
                    if task not in features or len(features[task]) == 0:
                        print(f"No data for task {task} in {split} set. Skipping.")
                        continue
                    
                    # Save to npz file
                    np.savez(
                        checkpoint_output_dir / f"{task}_{split}_latent_vectors.npz",
                        features=features[task],
                        targets=targets[task],
                        predictions=preds[task],
                        cif_ids=np.array(cif_ids[task]),
                        task_type=np.array([task_type])
                    )
                    
                    # Store in results
                    results[split][task] = {
                        "features": features[task],
                        "targets": targets[task],
                        "predictions": preds[task],
                        "cif_ids": cif_ids[task],
                        "task_type": task_type
                    }
            
            # Save uncertainty trees
            uncertainty_trees_file = checkpoint_output_dir / "uncertainty_trees.pkl"
            with open(uncertainty_trees_file, 'wb') as f:
                pickle.dump(uncertainty_trees, f)
            
            print(f"Saved uncertainty trees to {uncertainty_trees_file}")
            
            # Save results dictionary
            results_file = checkpoint_output_dir / "results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Saved results to {results_file}")
            
            processed_checkpoints.append(checkpoint_file)
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessed {len(processed_checkpoints)} checkpoints")
    
    # Create a summary file with all the checkpoint paths
    summary_file = output_dir / "processed_checkpoints.txt"
    with open(summary_file, 'w') as f:
        for checkpoint in processed_checkpoints:
            f.write(f"{checkpoint}\n")
    
    print(f"Saved processed checkpoint list to {summary_file}")
    
    return processed_checkpoints

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, 
                        default=os.path.join(ROOT_DIR, "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43"),
                        help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, 
                        default=None,
                        help="Directory where uncertainty trees and latent vectors will be saved")
    parser.add_argument("--checkpoint_path", type=str, 
                        default=None,
                        help="Path to a specific checkpoint file to process. If not provided, the best checkpoint is used.")
    parser.add_argument("--process_all", action="store_true",
                        help="Process all checkpoints in the model directory to track uncertainty evolution")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbors to use for uncertainty calculation")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("Starting ball tree construction for uncertainty calculation...")
    print(f"Project root directory: {ROOT_DIR}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"K neighbors: {args.k}")
    
    if args.process_all:
        print("Processing all checkpoints in the model directory")
        process_all_checkpoints(model_dir, output_dir, args.k)
    elif args.checkpoint_path:
        print(f"Processing specific checkpoint: {args.checkpoint_path}")
        process_checkpoint(args.checkpoint_path, output_dir, args.k)
    else:
        print("Building uncertainty trees using the best model checkpoint")
        build_uncertainty_trees(model_dir, output_dir, args.k)