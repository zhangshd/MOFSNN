'''
Author: zhangshd
Date: 2024-08-20 10:00:00
LastEditors: zhangshd
LastEditTime: 2025-05-06 16:35:25
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
    Data augmentation class for minority classes in classification tasks or specific samples.
    Performs minor perturbations on nbr_dist and cell_params.
    
    This class has two modes of operation:
    1. Class-based augmentation: Identifies minority classes in classification tasks and creates
       augmented versions of these samples to balance the class distribution.
    2. Sample-based augmentation: Augments samples based on uncertainty values from an Excel file,
       selecting the top N% of samples with highest uncertainty for each task.
    
    By default, this class will augment minority class samples to match the majority class count.
    Alternatively, a fixed augmentation factor can be specified.
    """
    
    def __init__(self, original_dataset, aug_factor=None, noise_std=0.01, 
                 balance_classes=True, aug_sample_file=None, task=None, 
                 task_type=None, uncertainty_threshold=None):
        """
        Initialize the augmented dataset based on the original dataset.
        
        Args:
            original_dataset (LoadGraphData): The original dataset to augment
            aug_factor (int or dict, optional): How many augmented samples to create for each sample.
                                      Can be an integer for all tasks or a dictionary mapping 
                                      task name to augmentation factor (e.g., {"TSD": 5, "SSD": 2})
            noise_std (float): Standard deviation factor for perturbation (relative to original value)
            balance_classes (bool): If True, augment minority classes to match majority class count.
                                  If False, use fixed aug_factor for all minority samples.
            aug_sample_file (str, optional): Path to Excel file containing uncertainty values for samples.
                                          Each sheet should correspond to a task.
            task (str, optional): The actual task name (e.g. "WS24_water") to match with Excel sheet names.
                                 If None, will try to derive from property column name.
            uncertainty_threshold (float or dict, optional): Threshold for selecting samples with high uncertainty.
                                                         Can be a float (0-1) representing the top percentage
                                                         to select, or a dictionary mapping task name to threshold.
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
        self.aug_sample_file = aug_sample_file
        self.uncertainty_threshold = uncertainty_threshold
        
        # Copy required attributes for consistent processing
        self.ari = original_dataset.ari if hasattr(original_dataset, 'ari') else None
        self.gdf = original_dataset.gdf if hasattr(original_dataset, 'gdf') else None
        
        # If no property columns, create empty dataframe and return
        self.id_prop_df = pd.DataFrame()
        self.augmented_mapping = {}
        if not self.prop_cols:
            return
        
        # Use provided task name if available, otherwise try to derive from property column
        self.task = task
        self.task_type = task_type

        # Check for sample-based augmentation mode (using Excel file)
        if aug_sample_file is not None and os.path.exists(aug_sample_file):
            # If aug_factor is a dictionary, extract the specific factor for this task
            task_aug_factor = None
            if isinstance(aug_factor, dict):
                task_aug_factor = aug_factor.get(self.task, 0)  # Default to 1 if task not found
            else:
                task_aug_factor = aug_factor
            
            # Get the uncertainty threshold for this task
            task_uncertainty_threshold = None
            if isinstance(uncertainty_threshold, dict):
                task_uncertainty_threshold = uncertainty_threshold.get(self.task, 0)  # Default to top 20%
            else:
                task_uncertainty_threshold = uncertainty_threshold if uncertainty_threshold is not None else 0
            
            # Read samples with uncertainty from Excel and filter by threshold
            samples_to_augment = self._read_samples_from_excel(aug_sample_file, task_uncertainty_threshold)
            
            if samples_to_augment and task_aug_factor:
                # Perform sample-based augmentation with task-specific factor
                self._augment_specific_samples(samples_to_augment, task_aug_factor)
                return
        if balance_classes:
            # If we get here, perform regular class-based augmentation
            self._perform_class_based_augmentation(aug_factor)
    
    def _read_samples_from_excel(self, excel_path, threshold=0.2):
        """
        Read sample IDs with uncertainty values from an Excel file and select top N% samples
        with highest uncertainty.
        
        Args:
            excel_path: Path to Excel file
            threshold: Float between 0-1 representing the top percentage of uncertain samples to select
            
        Returns:
            List of sample IDs to augment, or None if the task is not found
        """
        try:
            print(f"Reading samples from {excel_path}")
            
            # Read all sheets from Excel file
            excel_data = pd.read_excel(excel_path, sheet_name=None)
            
            # Find sheet matching current task
            for sheet_name, df in excel_data.items():
                if sheet_name == self.task and 'cif_id' in df.columns and 'uncertainty' in df.columns:
                    # Sort by uncertainty (highest first)
                    df = df.sort_values('uncertainty', ascending=False)
                    
                    # Calculate how many samples to select based on threshold
                    num_samples = int(len(df) * threshold)
                    if num_samples < 1:
                        print(f"Not enough samples to select for task {self.task} (threshold too low)")
                        return None
                    
                    # Select top uncertain samples
                    selected_df = df.head(num_samples)
                    sample_ids = selected_df['cif_id'].astype(str).tolist()
                    
                    # Log uncertainty info
                    min_uncertainty = selected_df['uncertainty'].min()
                    max_uncertainty = selected_df['uncertainty'].max()
                    print(f"Found {len(sample_ids)} samples to augment for task {self.task} (top {threshold*100:.1f}%)")
                    print(f"  Uncertainty range: {min_uncertainty:.4f} to {max_uncertainty:.4f}")
                    print(f"  Selected {len(sample_ids)} out of {len(df)} total samples")
                    
                    return sample_ids
            
            print(f"No matching sheet found for task {self.task} in {excel_path}")
            return None
            
        except Exception as e:
            print(f"Error reading sample file {excel_path}: {e}")
            return None
    
    def _augment_specific_samples(self, sample_ids, aug_factor):
        """
        Augment specific samples identified by their IDs.
        
        Args:
            sample_ids: List of sample IDs to augment
            aug_factor: Augmentation factor (how many copies to create)
        """
        # Create a set of sample IDs for faster lookup
        sample_id_set = set(sample_ids)
        
        # Find matching samples in the original dataset
        matching_samples = []
        for idx in self.original_dataset.id_prop_df.index:
            if str(idx) in sample_id_set:
                matching_samples.append(idx)
        
        # Create augmented versions
        augmented_rows = []
        self.augmented_mapping = {}
        
        # Set augmentation factor - default to 2 if not specified
        samples_per_orig = aug_factor if aug_factor is not None else 2
        
        # Log augmentation info
        print(f"\n=== Sample-Based Augmentation ===")
        print(f"Found {len(matching_samples)} of {len(sample_ids)} requested samples in dataset")
        print(f"Augmentation factor for {self.task}: {samples_per_orig}x")
        
        # Create augmented samples
        for orig_idx in matching_samples:
            # Get original sample data
            row = self.original_dataset.id_prop_df.loc[orig_idx]
            
            # Create multiple augmented versions
            for aug_idx in range(samples_per_orig):
                aug_id = f"{orig_idx}_aug_{aug_idx}"
                self.augmented_mapping[aug_id] = orig_idx
                
                # Create new row for augmented sample
                aug_row = row.copy()
                augmented_rows.append((aug_id, aug_row))
        
        # Create DataFrame with augmented samples
        if augmented_rows:
            self.id_prop_df = pd.DataFrame([row for _, row in augmented_rows], 
                                           index=[id for id, _ in augmented_rows])
            print(f"Created {len(self.id_prop_df)} augmented samples")
        else:
            self.id_prop_df = pd.DataFrame()
            print("No samples were augmented")
        
        print("====================================")
    
    def _perform_class_based_augmentation(self, aug_factor):
        """
        Perform class-based augmentation to balance minority classes.
        This is the original augmentation method from the class.
        
        Args:
            aug_factor: Augmentation factor for fixed augmentation
        """
        # Identify minority classes for each task
        # Since each dataset typically corresponds to one task, we use the first property column
        task_prop_col = self.prop_cols[0] if self.prop_cols else None
        
        if task_prop_col is None or task_prop_col not in self.original_dataset.id_prop_df.columns:
            # No valid property column for classification
            self.id_prop_df = pd.DataFrame()
            self.augmented_mapping = {}
            return
            
        # Get class distribution for this task
        class_counts = self.original_dataset.id_prop_df[task_prop_col].value_counts()
        if len(class_counts) <= 1:
            # Only one class, no minority to augment
            self.id_prop_df = pd.DataFrame()
            self.augmented_mapping = {}
            return
        elif len(class_counts) > 5:
            # Too many classes, cannot handle
            print(f"Too many classes ({len(class_counts)}) for class-based augmentation. Skipping.")
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
            class_samples = self.original_dataset.id_prop_df[self.original_dataset.id_prop_df[task_prop_col] == class_label]
            orig_count = len(class_samples)
            
            # Determine how many augmented samples to create for this class
            if self.balance_classes:
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
                        if self.balance_classes and aug_count >= aug_samples_needed:
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
            print("\n=== Class-Based Augmentation Statistics ===")
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
        if self.task_type == "regression":
            ## add noise to targets if regression
            noise = np.random.normal(0, self.noise_std * np.abs(targets))
            targets = targets + noise

        
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
