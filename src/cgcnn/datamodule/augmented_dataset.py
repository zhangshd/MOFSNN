'''
Author: zhangshd
Date: 2024-08-20 10:00:00
LastEditors: zhangshd
LastEditTime: 2025-04-27 19:53:07
'''
import functools
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datamodule.dataset import LoadGraphData


class AugmentedGraphData(LoadGraphData):
    """
    Data augmentation class for minority classes in classification tasks.
    Performs minor perturbations on nbr_dist and cell_params.
    
    This class identifies minority classes in classification tasks and creates
    augmented versions of these samples by adding small Gaussian noise to 
    nbr_dist and cell_params values.
    
    By default, this class will augment minority class samples to match the majority class count.
    Alternatively, a fixed augmentation factor can be specified.
    """
    
    def __init__(self, original_dataset, aug_factor=None, noise_std=0.01, balance_classes=True):
        """
        Initialize the augmented dataset based on the original dataset.
        
        Args:
            original_dataset (LoadGraphData): The original dataset to augment
            aug_factor (int, optional): How many augmented samples to create for each minority sample.
                                      If None and balance_classes=True, will calculate the factor 
                                      needed to balance class distribution.
            noise_std (float): Standard deviation factor for perturbation (relative to original value)
            balance_classes (bool): If True, augment minority classes to match majority class count.
                                  If False, use fixed aug_factor for all minority samples.
        """
        # Initialize base attributes without calling parent constructor
        self.data_dir = original_dataset.data_dir
        self.split = original_dataset.split
        self.radius = original_dataset.radius
        self.dmin = original_dataset.dmin
        self.step = original_dataset.step
        self.use_cell_params = original_dataset.use_cell_params
        self.use_extra_fea = original_dataset.use_extra_fea
        self.task_id = original_dataset.task_id
        self.prop_cols = original_dataset.prop_cols.copy() if hasattr(original_dataset, 'prop_cols') else None
        self.g_data = original_dataset.g_data
        self.csv_file_name = original_dataset.csv_file_name if hasattr(original_dataset, 'csv_file_name') else None
        self.max_sample_size = original_dataset.max_sample_size if hasattr(original_dataset, 'max_sample_size') else None
        self.down_sampling = original_dataset.down_sampling if hasattr(original_dataset, 'down_sampling') else True
        
        # Store parameters for augmentation
        self.original_dataset = original_dataset
        self.noise_std = noise_std
        self.balance_classes = balance_classes
        
        # Copy required attributes for consistent processing
        self.ari = original_dataset.ari if hasattr(original_dataset, 'ari') else None
        self.gdf = original_dataset.gdf if hasattr(original_dataset, 'gdf') else None
        
        # Identify minority classes for each task
        # Since each dataset typically corresponds to one task, we use the first property column
        task_prop_col = self.prop_cols[0] if self.prop_cols else None
        
        if task_prop_col is None or task_prop_col not in original_dataset.id_prop_df.columns:
            # No valid property column for classification
            self.id_prop_df = pd.DataFrame()
            self.augmented_mapping = {}
            return
            
        # Get class distribution for this task
        class_counts = original_dataset.id_prop_df[task_prop_col].value_counts()
        if len(class_counts) <= 1:
            # Only one class, no minority to augment
            self.id_prop_df = pd.DataFrame()
            self.augmented_mapping = {}
            return
            
        majority_class = class_counts.idxmax()
        majority_count = class_counts[majority_class]
        
        # Collect samples from minority classes
        augmented_rows = []
        self.augmented_mapping = {}  # Maps augmented ID to original ID
        
        # For logging purposes
        class_aug_counts = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        for class_label, count in class_counts.items():
            # Skip majority class
            if class_label == majority_class:
                continue
                
            # Get all samples for this class
            class_samples = original_dataset.id_prop_df[original_dataset.id_prop_df[task_prop_col] == class_label]
            orig_count = len(class_samples)
            
            # Determine how many augmented samples to create for this class
            if balance_classes:
                # Calculate how many total samples needed for this class to match majority class
                total_needed = majority_count
                # Calculate how many augmented samples to create
                aug_samples_needed = total_needed - orig_count
                # Calculate average augmentation factor per original sample (rounded up)
                samples_per_orig = int(np.ceil(aug_samples_needed / orig_count))
                class_aug_counts[class_label] = (orig_count, aug_samples_needed, samples_per_orig)
            else:
                # Use fixed augmentation factor
                samples_per_orig = aug_factor if aug_factor is not None else 1
                aug_samples_needed = orig_count * samples_per_orig
                class_aug_counts[class_label] = (orig_count, aug_samples_needed, samples_per_orig)
            
            # Create augmented samples
            aug_count = 0
            
            # If we need fewer augmented samples than original samples, randomly select which ones to augment
            if aug_samples_needed < orig_count:
                # Randomly select indices to augment
                augment_indices = np.random.choice(
                    len(class_samples), 
                    size=aug_samples_needed, 
                    replace=False
                )
                
                # Get the selected samples for augmentation
                selected_samples = class_samples.iloc[augment_indices]
                
                # Create one augmented version for each selected sample
                for i, (orig_idx, row) in enumerate(selected_samples.iterrows()):
                    aug_id = f"{orig_idx}_aug_0"  # Just one augmentation per sample
                    self.augmented_mapping[aug_id] = orig_idx
                    
                    # Copy the row data for the new augmented sample
                    aug_row = row.copy()
                    augmented_rows.append((aug_id, aug_row))
                    aug_count += 1
                    
                print(f"Class {class_label}: Randomly selected {aug_samples_needed} out of {orig_count} samples for augmentation")
            else:
                # Need to augment each sample multiple times
                for orig_idx, row in class_samples.iterrows():
                    # Determine how many augmented versions to create for this sample
                    for aug_idx in range(samples_per_orig):
                        # Stop if we've created enough augmented samples for this class
                        if balance_classes and aug_count >= aug_samples_needed:
                            break
                            
                        aug_id = f"{orig_idx}_aug_{aug_idx}"
                        self.augmented_mapping[aug_id] = orig_idx
                        
                        # Copy the row data for the new augmented sample
                        aug_row = row.copy()
                        augmented_rows.append((aug_id, aug_row))
                        aug_count += 1
        
        # Create augmented dataframe
        if augmented_rows:
            self.id_prop_df = pd.DataFrame([row for _, row in augmented_rows], 
                                         index=[id for id, _ in augmented_rows])
            
            # Print augmentation statistics
            print("\n=== Data Augmentation Statistics ===")
            print(f"Majority class ({majority_class}): {majority_count} samples")
            for class_label, (orig, aug_needed, per_orig) in class_aug_counts.items():
                print(f"Class {class_label}: {orig} original + {aug_needed} augmented = {orig + aug_needed} total samples")
                if aug_needed < orig:
                    print(f"  Random selection: {aug_needed} samples randomly selected for augmentation")
                else:
                    print(f"  Augmentation ratio: {per_orig:.2f}x per original sample")
            print(f"Total augmented samples: {len(self.id_prop_df)}")
            print("====================================")
        else:
            self.id_prop_df = pd.DataFrame()
    
    def __len__(self):
        """Return the number of augmented samples."""
        return len(self.id_prop_df)
    
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        Get augmented item by adding noise to the original data.
        
        Args:
            idx (int): Index of the augmented sample
            
        Returns:
            Tuple of augmented data in the same format as the original dataset
        """
        # Get augmented sample info
        row = self.id_prop_df.iloc[idx]
        aug_id = row.name
        orig_id = self.augmented_mapping[aug_id]
        
        # Get targets and extra features from the row
        if self.use_extra_fea:
            extra_fea = row.loc["Di":].values.astype(float)
        else:
            extra_fea = []
            
        targets = row[self.prop_cols].values.astype(float)
        
        # Load original graph data
        with open(self.g_data[orig_id], 'rb') as f:
            data = pickle.load(f)
            
        # The structure matches LoadGraphData.__getitem__ return structure
        if len(data) == 7:  # For LoadGraphDataWithAtomicNumber
            cif_id, atom_num, nbr_fea_idx, nbr_dist, uni_idx, uni_count, cell_params = data
            
            # Apply perturbation to nbr_dist
            noise = np.random.normal(0, self.noise_std * np.abs(nbr_dist))
            nbr_dist_augmented = nbr_dist + noise
            
            # Apply perturbation to cell_params if available and used
            if cell_params is not None and self.use_cell_params:
                noise = np.random.normal(0, self.noise_std * np.abs(cell_params))
                cell_params_augmented = cell_params + noise
            else:
                cell_params_augmented = cell_params
                
            # Format return data to match LoadGraphDataWithAtomicNumber.__getitem__
            # Fix: Ensure atom_fea is always 2D by reshaping if needed
            # First convert to atom feature embeddings if we have access to them
            if hasattr(self, 'ari') and self.ari is not None:
                # Use the same method as in LoadGraphData to get 2D atom features
                atom_fea = np.vstack([self.ari.get_atom_fea(i) for i in atom_num])
                atom_fea = torch.Tensor(atom_fea)
            else:
                # If no atom initializer available, make sure it's still 2D
                atom_fea = torch.LongTensor(atom_num)
                if atom_fea.dim() == 1:
                    atom_fea = atom_fea.unsqueeze(-1)  # Add feature dimension
                    
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
            nbr_dist_augmented = torch.FloatTensor(nbr_dist_augmented).view(len(atom_num), -1)
            nbr_fea = self.original_dataset.gdf.expand(nbr_dist_augmented).float()
            targets = torch.FloatTensor(targets)
            extra_fea = torch.FloatTensor(extra_fea)

            if self.use_cell_params:
                cell_params_augmented = torch.FloatTensor(cell_params_augmented)
                extra_fea = torch.cat([extra_fea, cell_params_augmented], dim=-1)

            ret_dict = {
                "atom_fea": atom_fea,
                "nbr_fea": nbr_fea,
                "nbr_fea_idx": nbr_fea_idx,
                "uni_idx": uni_idx,
                "uni_count": uni_count,
                "extra_fea": extra_fea,
                "targets": targets,
                "cif_id": aug_id,
                "task_id": self.task_id
            }

            return ret_dict
                    
        else:  # For LoadGraphData
            cif_id, atom_num, nbr_fea_idx, nbr_dist, cell_params = data
            
            # Apply perturbation to nbr_dist
            noise = np.random.normal(0, self.noise_std * np.abs(nbr_dist))
            nbr_dist_augmented = nbr_dist + noise
            
            # Apply perturbation to cell_params if available and used
            if cell_params is not None and self.use_cell_params:
                noise = np.random.normal(0, self.noise_std * np.abs(cell_params))
                cell_params_augmented = cell_params + noise
            else:
                cell_params_augmented = cell_params
                
            # Format return data to match LoadGraphData.__getitem__
            # Ensure we use the same atom feature initialization as the original dataset
            atom_fea = np.vstack([self.original_dataset.ari.get_atom_fea(i) for i in atom_num])
            atom_fea = torch.Tensor(atom_fea)  # This will be 2D: [num_atoms, feature_dim]
            
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_fea), -1)
            nbr_dist_augmented = torch.FloatTensor(nbr_dist_augmented).view(len(atom_fea), -1)
            nbr_fea = self.original_dataset.gdf.expand(nbr_dist_augmented).float()
            targets = torch.FloatTensor(targets)
            extra_fea = torch.FloatTensor(extra_fea)

            if self.use_cell_params:
                cell_params_augmented = torch.FloatTensor(cell_params_augmented)
                extra_fea = torch.cat([extra_fea, cell_params_augmented], dim=-1)

            ret_dict = {
                "atom_fea": atom_fea,
                "nbr_fea": nbr_fea,
                "nbr_fea_idx": nbr_fea_idx,
                "extra_fea": extra_fea,
                "targets": targets,
                "cif_id": aug_id,
                "task_id": self.task_id
            }

            return ret_dict
