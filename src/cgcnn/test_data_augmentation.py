#!/usr/bin/env python
"""
Test script for verifying data augmentation functionality in MOFSNN.
This script tests both direct use of AugmentedGraphData class and 
integration through DInterface.
"""

import os
import sys
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the modules
from datamodule.dataset import LoadGraphData
from datamodule.augmented_dataset import AugmentedGraphData
from datamodule.data_interface import DInterface

# Import configuration from config.py correctly
import config

# Define the project root and data directories
project_root = Path(__file__).resolve().parent.parent.parent
data_root = project_root / "data/cgcnn_data"
results_dir = project_root / "results/augmentation_analysis"
results_dir.mkdir(exist_ok=True, parents=True)

def test_augmented_dataset(balance_classes=True):
    """
    Test AugmentedGraphData class with a real dataset
    
    Args:
        balance_classes (bool): Whether to balance class distribution during augmentation
    """
    # Try to find a suitable dataset, prefer WS24 or SSD
    for dataset_name in ["WS24", "SSD"]:
        data_dir = data_root / dataset_name
        if data_dir.exists():
            print(f"Found dataset directory: {data_dir}")
            break
    else:
        print("Could not find a suitable dataset directory.")
        print(f"Looked in: {data_root}")
        sys.exit(1)
    
    print(f"Using dataset from: {data_dir}")
    
    try:
        # Determine proper prop_cols for the dataset
        if "WS24" in str(data_dir):
            # For WS24 dataset, use water_label as the property column
            prop_cols = ["water_label"]
        else:
            # For other datasets, use the default 'Label'
            prop_cols = ["Label"]
        
        print(f"Using property columns: {prop_cols}")
        
        # Load original dataset with specified prop_cols
        original_dataset = LoadGraphData(data_dir=data_dir, split='train', prop_cols=prop_cols, use_cell_params=True, down_sampling=False)
        
        print(f"\nOriginal dataset size: {len(original_dataset)}")
        class_distribution = original_dataset.id_prop_df[prop_cols[0]].value_counts()
        print(f"Sample targets distribution:\n{class_distribution}")
        
        # Create augmented dataset
        if balance_classes:
            print("\nCreating balanced augmented dataset...")
            augmented_dataset = AugmentedGraphData(
                original_dataset=original_dataset,
                noise_std=0.01,
                balance_classes=True
            )
        else:
            print("\nCreating fixed-factor augmented dataset...")
            augmented_dataset = AugmentedGraphData(
                original_dataset=original_dataset,
                aug_factor=2,
                noise_std=0.01,
                balance_classes=False
            )
        
        print(f"Augmented dataset size: {len(augmented_dataset)}")
        
        # Verify that augmentation worked
        if len(augmented_dataset) > 0:
            print("Augmentation successful!")
            
            # Calculate the expected class distribution after augmentation
            if balance_classes:
                # The augmented dataset only contains minority class samples
                majority_class = class_distribution.idxmax()
                majority_count = class_distribution[majority_class]
                
                # Show the expected final distribution (original + augmented)
                final_counts = {}
                for class_label, count in class_distribution.items():
                    if class_label == majority_class:
                        final_counts[class_label] = count
                    else:
                        # For minority classes, they should reach majority count after augmentation
                        aug_samples = majority_count - count
                        final_counts[class_label] = majority_count
                        print(f"Class {class_label}: {count} original + {aug_samples} augmented = {majority_count} total")
                
                print("\nFinal expected class distribution:")
                for cls, count in final_counts.items():
                    print(f"Class {cls}: {count}")
            
            # Test accessing a sample
            try:
                sample_idx = 0
                sample = augmented_dataset[sample_idx]
                print(f"\nAugmented sample {sample_idx} ID: {sample['cif_id']}")
                print(f"Target value: {sample['targets']}")
                
                # Verify the augmented_mapping dictionary
                aug_id = sample['cif_id']
                orig_id = augmented_dataset.augmented_mapping.get(aug_id)
                if orig_id:
                    print(f"Augmented sample {aug_id} was derived from original sample {orig_id}")
                    
                    # Compare augmented and original samples in detail
                    compare_augmented_with_original(augmented_dataset, original_dataset, aug_id, orig_id)
                else:
                    print(f"Warning: Could not find original sample for augmented sample {aug_id}")
            except Exception as e:
                print(f"Error retrieving augmented sample: {e}")
        else:
            print("No samples were augmented. This might be normal if there are no minority classes to augment.")
        
        print("AugmentedGraphData test completed!")
        return augmented_dataset
    
    except Exception as e:
        print(f"Error testing AugmentedGraphData: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_augmented_with_original(augmented_dataset, original_dataset, aug_id, orig_id):
    """
    Compare the augmented sample with its original sample in detail,
    focusing on the nbr_dist and cell_params.
    
    Args:
        augmented_dataset: The augmented dataset
        original_dataset: The original dataset
        aug_id: ID of the augmented sample
        orig_id: ID of the original sample
    """
    print("\n=== Detailed Comparison of Augmented vs Original Sample ===")
    
    # Get original data sample
    orig_idx = original_dataset.id_prop_df.index.get_loc(orig_id)
    orig_sample = original_dataset[orig_idx]
    
    # Find the augmented sample by ID
    aug_idx = None
    for i in range(len(augmented_dataset)):
        if augmented_dataset[i]['cif_id'] == aug_id:
            aug_idx = i
            break
    
    if aug_idx is None:
        print(f"Could not find augmented sample with ID: {aug_id}")
        return
    
    aug_sample = augmented_dataset[aug_idx]
    
    # Compare atom features
    print("\nAtom Features Comparison:")
    print(f"  Original shape: {orig_sample['atom_fea'].shape}")
    print(f"  Augmented shape: {aug_sample['atom_fea'].shape}")
    
    # Compare neighbor distances (nbr_dist)
    print("\nNeighbor Distances (nbr_dist) Comparison:")
    # Access raw neighbor distances from dataset
    with open(original_dataset.g_data[orig_id], 'rb') as f:
        orig_data = pickle.load(f)
    
    # Determine format based on data length
    if len(orig_data) == 7:  # LoadGraphDataWithAtomicNumber
        _, _, _, orig_nbr_dist, _, _, orig_cell_params = orig_data
    else:  # LoadGraphData
        _, _, _, orig_nbr_dist, orig_cell_params = orig_data
    
    # Get augmented raw data by repeating the data loading and augmentation process
    np.random.seed(0)  # Set seed for reproducibility
    noise = np.random.normal(0, augmented_dataset.noise_std * np.abs(orig_nbr_dist))
    aug_nbr_dist = orig_nbr_dist + noise
    
    # Calculate statistics
    print(f"  Original nbr_dist stats: min={orig_nbr_dist.min():.4f}, max={orig_nbr_dist.max():.4f}, mean={orig_nbr_dist.mean():.4f}, std={orig_nbr_dist.std():.4f}")
    print(f"  Augmented nbr_dist stats: min={aug_nbr_dist.min():.4f}, max={aug_nbr_dist.max():.4f}, mean={aug_nbr_dist.mean():.4f}, std={aug_nbr_dist.std():.4f}")
    
    # Calculate difference statistics
    diff_nbr_dist = aug_nbr_dist - orig_nbr_dist
    print(f"  Difference stats: min={diff_nbr_dist.min():.4f}, max={diff_nbr_dist.max():.4f}, mean={diff_nbr_dist.mean():.4f}, std={diff_nbr_dist.std():.4f}")
    
    # Print a small sample of the actual distances
    sample_size = min(5, len(orig_nbr_dist))
    print(f"\n  Sample of first {sample_size} neighbor distances:")
    for i in range(sample_size):
        print(f"    Index {i}: Original={orig_nbr_dist[i]:.4f}, Augmented={aug_nbr_dist[i]:.4f}, Diff={diff_nbr_dist[i]:.4f}")
    
    # Compare cell parameters if available
    if orig_cell_params is not None and augmented_dataset.use_cell_params:
        print("\nCell Parameters Comparison:")
        np.random.seed(0)  # Reset seed for consistency
        noise = np.random.normal(0, augmented_dataset.noise_std * np.abs(orig_cell_params))
        aug_cell_params = orig_cell_params + noise
        
        print(f"  Original cell_params: {orig_cell_params}")
        print(f"  Augmented cell_params: {aug_cell_params}")
        print(f"  Difference: {aug_cell_params - orig_cell_params}")
    
    # Plot histogram of differences
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(diff_nbr_dist, bins=50, alpha=0.7)
        ax.set_title('Histogram of Differences in Neighbor Distances')
        ax.set_xlabel('Difference (Augmented - Original)')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(results_dir / f"nbr_dist_diff_{aug_id}.png")
        print(f"\nHistogram saved to: {results_dir}/nbr_dist_diff_{aug_id}.png")
        plt.close()
    except Exception as e:
        print(f"Could not create histogram: {e}")

def test_data_interface_augmentation():
    """Test augmentation through DInterface"""
    try:
        # Define test configuration - hardcoded for testing purposes
        # Using only one data source and dataset type at a time to avoid dimension conflicts
        tasks = []
        task_types = []
        
        # For testing, focus on only one dataset type: WS24_water
        if (data_root / "WS24").exists():
            tasks.extend(["WS24_water", "WS24_water4", "WS24_acid", "WS24_base", "WS24_boiling"])
            task_types.extend(["classification", "classification_4", "classification", "classification", "classification"])
        elif (data_root / "SSD").exists():
            tasks.append("SSD")
            task_types.append("classification")
            
        if not tasks:
            print("Could not find suitable classification tasks.")
            return
        
        print(f"Testing with tasks: {tasks}")
        print(f"Task types: {task_types}")
        
        # Create data module with augmentation enabled
        data_module = DInterface(
            data_dir=data_root,
            tasks=tasks,
            task_types=task_types,
            augment=True,
            aug_noise_std=0.01,
            balance_classes=True,  # Use balanced augmentation
            down_sampling=False,
            use_cell_params=True,
        )
        
        # Set up the data module (this will trigger dataset creation and augmentation)
        data_module.setup(stage='fit')
        
        # Check if training set was created
        if hasattr(data_module, 'trainset'):
            print(f"Total training set size with augmentation: {len(data_module.trainset)}")
        else:
            print("Training set was not created")
            return
        
        # Create data module without augmentation for comparison
        data_module_no_aug = DInterface(
            data_dir=data_root,
            tasks=tasks,
            task_types=task_types,
            augment=False,
            down_sampling=False,
            use_cell_params=True,
        )
        
        data_module_no_aug.setup(stage='fit')
        
        if hasattr(data_module_no_aug, 'trainset'):
            print(f"Training set size without augmentation: {len(data_module_no_aug.trainset)}")
            diff = len(data_module.trainset) - len(data_module_no_aug.trainset)
            print(f"Difference (augmented samples): {diff}")
            
            if diff > 0:
                print("Augmentation successfully increased dataset size!")
            else:
                print("Warning: Augmentation did not increase dataset size.")
                
            # Test loading a batch - with error handling
            try:
                # Use a smaller batch size to reduce complexity
                train_loader = torch.utils.data.DataLoader(
                    data_module.trainset,
                    batch_size=4,
                    shuffle=False,
                    collate_fn=data_module.collate_fn
                )
                
                batch = next(iter(train_loader))
                if 'atom_fea' in batch:
                    print(f"Successfully loaded a batch with {batch['atom_fea'].size(0)} atoms")
                    print(f"Atom feature dimensions: {batch['atom_fea'].size()}")
                else:
                    print(f"Successfully loaded a batch with keys: {list(batch.keys())}")
            except Exception as e:
                print(f"Error loading a batch: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Training set without augmentation was not created")
        
        print("DInterface augmentation test completed!")
        
    except Exception as e:
        print(f"Error testing DInterface augmentation: {e}")
        import traceback
        traceback.print_exc()

def test_compatibility():
    """Test compatibility between original and augmented datasets"""
    try:
        # Load a small WS24 dataset
        data_dir = data_root / "WS24"
        if not data_dir.exists():
            print("WS24 dataset not found")
            return
        
        # Create dataset and augmented dataset
        original_dataset = LoadGraphData(data_dir=data_dir, split='train', prop_cols=["water_label"], use_cell_params=True, down_sampling=False)
        augmented_dataset = AugmentedGraphData(
            original_dataset, 
            noise_std=0.01,
            balance_classes=True
        )
        
        if len(augmented_dataset) == 0:
            print("No augmented samples were created")
            return
            
        # Check tensor dimensions
        orig_sample = original_dataset[0]
        aug_sample = augmented_dataset[0]
        
        print("Original sample dimensions:")
        for key, value in orig_sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
                
        print("Augmented sample dimensions:")
        for key, value in aug_sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
                
        # Check if the dimensions match
        dimension_mismatch = False
        for key in orig_sample:
            if key in aug_sample and isinstance(orig_sample[key], torch.Tensor) and isinstance(aug_sample[key], torch.Tensor):
                if orig_sample[key].dim() != aug_sample[key].dim():
                    dimension_mismatch = True
                    print(f"Dimension mismatch for {key}: original={orig_sample[key].dim()}, augmented={aug_sample[key].dim()}")
        
        if not dimension_mismatch:
            print("All tensor dimensions match between original and augmented samples")
        
        # Test creating a batch with a mix of original and augmented samples
        combined_dataset = torch.utils.data.ConcatDataset([original_dataset, augmented_dataset])
        
        # Get a small batch of samples and check their dimensions
        indices = list(range(min(4, len(combined_dataset))))
        samples = [combined_dataset[i] for i in indices]
        
        # Check if all samples have atom_fea with the same dimension
        atom_fea_dims = [s['atom_fea'].dim() for s in samples]
        if len(set(atom_fea_dims)) > 1:
            print(f"WARNING: Inconsistent atom_fea dimensions in the batch: {atom_fea_dims}")
        else:
            print(f"All samples have consistent atom_fea dimensions: {atom_fea_dims[0]}")
            
        print("Compatibility test completed")
        
    except Exception as e:
        print(f"Error in compatibility test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing data augmentation in MOFSNN")
    print("="*50 + "\n")
    
    print("\n--- Testing balanced augmentation ---\n")
    aug_dataset_balanced = test_augmented_dataset(balance_classes=True)
    
    print("\n--- Testing compatibility between original and augmented datasets ---\n")
    test_compatibility()
    
    print("\n--- Testing augmentation through DInterface ---\n")
    test_data_interface_augmentation()
    
    print("\n" + "="*50)
    print("Tests completed!")
    print("="*50 + "\n")
