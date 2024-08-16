import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodule.dataset import LoadGraphData, LoadExtraFeatureData, LoadGraphDataWithAtomicNumber
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


class DInterface(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=64, num_workers=4, dataset_cls=LoadGraphData,
                 **kwargs):
        
        super().__init__()
        self.root_dir = Path(data_dir)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tasks = kwargs.get('tasks', ['TSD', 'SSD', 'WS24'])
        self.task_types = kwargs.get('task_types', ['regression', 'classification', 'classification'])
        
        self.log_labels = kwargs.get('log_labels', False)
        self.final_train = kwargs.get('final_train', False)
        self.dl_sampler = kwargs.get('dl_sampler', 'random')
        self.kwargs = kwargs
        self.dataset_cls = dataset_cls
        if isinstance(dataset_cls, str):
            self.dataset_cls = eval(dataset_cls)
        self.collate_fn = self.dataset_cls.collate

        print("final_train:", self.final_train)
        print("dl_sampler: ", self.dl_sampler)

    def setup(self, stage=None):
        # This method is called on every GPU

        # Initialize CIFData dataset
        if stage == 'fit' or stage is None:
            if not hasattr(self, 'trainset'):
                self.trainsets = []
                self.trainset_sizes = []
                for i, task in enumerate(self.tasks):
                    dataset_dir = self.root_dir / task
                    trainset = self.dataset_cls(data_dir=dataset_dir, split='train',
                                                task_id=i, **self.kwargs)
                    self.trainset_sizes.append(len(trainset))
                    if self.final_train:
                        trainset.append(
                            self.dataset_cls(data_dir=dataset_dir, split='val',
                                                task_id=i, **self.kwargs))
                        
                    self.trainsets.append(trainset)
                    print(f"Number of {task} training data:", len(trainset))
                self.trainset = ConcatDataset(self.trainsets)
                print("Number of total training data:", len(self.trainset))
                self.task_weights = [len(d) / len(self.trainset) for d in self.trainsets]
            self.train_normalizer()
            if not hasattr(self, 'valset'):
                self.valsets = []
                for i, task in enumerate(self.tasks):
                    dataset_dir = self.root_dir / task
                    valset = self.dataset_cls(data_dir=dataset_dir, split='val',
                                                task_id=i, **self.kwargs)
                                                
                    self.valsets.append(valset)
                    print(f"Number of {task} validation data:", len(valset))
                self.valset = ConcatDataset(self.valsets)
                print("Number of total validation data:", len(self.valset))

        if (stage == 'test' or stage is None) and not hasattr(self, 'testset'):
            self.testsets = []
            for i, task in enumerate(self.tasks):
                dataset_dir = self.root_dir / task
                testset = self.dataset_cls(data_dir=dataset_dir, split='test',
                                                task_id=i, **self.kwargs)
                                                
                self.testsets.append(testset)
                print(f"Number of {task} test data:", len(testset))
            self.testset = ConcatDataset(self.testsets)
            print("Number of total test data:", len(self.testset))


    def train_dataloader(self):
        if 'task' in self.dl_sampler:
            print("Using same_task_in_batch sampler for training data.")
            return DataLoader(self.trainset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=self.num_workers,
                            sampler=SameTaskPriorBatchSchedulerSampler(self.trainset, self.batch_size), 
                            collate_fn=self.collate_fn, pin_memory=True)
            
        elif 'ratio' in self.dl_sampler:
            print("Using ratio sampler for training data.")
            return DataLoader(self.trainset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=self.num_workers,
                            sampler=SameRatioPriorBatchSchedulerSampler(self.trainset, self.batch_size), 
                            collate_fn=self.collate_fn, pin_memory=True)
        else:
            print("Using random sampler for training data.")
            return DataLoader(self.trainset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)
                          

    def val_dataloader(self):
        if 'task' in self.dl_sampler:
            print("Using same_task_in_batch sampler for validation data.")
            return DataLoader(self.valset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=self.num_workers,
                            sampler=SameTaskPriorBatchSchedulerSampler(self.valset, self.batch_size),
                            collate_fn=self.collate_fn, pin_memory=True)
            
        elif 'ratio' in self.dl_sampler:
            print("Using ratio sampler for validation data.")
            return DataLoader(self.valset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=self.num_workers,
                            sampler=SameRatioPriorBatchSchedulerSampler(self.valset, self.batch_size),
                            collate_fn=self.collate_fn, pin_memory=True)
        else:
            print("Using no sampler for validation data.")
            return DataLoader(self.valset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True)
    
    def train_normalizer(self):
        self.normalizers = []
        for i, task_tp in enumerate(self.task_types):
            trainset = self.trainset.datasets[i]
            if 'classification' in task_tp:
                normalizer = Normalizer(torch.Tensor([-1, 0., 1]))
                self.normalizers.append(normalizer)
            else:
                train_targets = torch.Tensor(trainset.id_prop_df.loc[:, trainset.prop_cols].values)
                normalizer = Normalizer(train_targets, log_labels=self.log_labels)
                self.normalizers.append(normalizer)
        return self.normalizers

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, log_labels=False):
        """tensor is taken as a sample to calculate the mean and std"""
        super(Normalizer, self).__init__()
        self.log_labels = log_labels
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-5) # avoid log10(0)
            print("Log10(x+1e-5) transform applied to labels.")
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        self.mean_ = float(self.mean.cpu().numpy())
        self.std_ = float(self.std.cpu().numpy())
        self.device = tensor.device

    def norm(self, tensor):
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-5)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        denormed_tensor = normed_tensor * self.std + self.mean
        if hasattr(self, 'log_labels') and self.log_labels:
            return torch.pow(10, denormed_tensor) - 1e-5
        else:
            return denormed_tensor

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
    def to(self, device):
        """Moves both mean and std to the specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device

        return self  # 返回self以支持链式调用

def split_dataset(data_df, stratify_cols=None, val_size=0.1, test_size=0.1, batch_size=32, random_seed=42):
    np.random.seed(random_seed)
    assert isinstance(data_df, pd.DataFrame)
    if stratify_cols in [None, '', [],  [''], (), ('')]:

        if isinstance(val_size, float) and val_size < 1:
            val_size = max(batch_size, int(len(data_df) * val_size))
        else:
            val_size = max(batch_size, int(val_size))
        if isinstance(test_size, float) and test_size < 1:
                test_size = max(batch_size, int(len(data_df) * test_size))
        elif test_size is None:
                test_size = 0
        else:
            test_size = max(batch_size, int(test_size))

        shuffuled_idxs = np.random.permutation(len(data_df))
        df_val = data_df.iloc[shuffuled_idxs[:val_size]]
        
        if test_size > 0:
            df_test = data_df.iloc[shuffuled_idxs[val_size:val_size+test_size]]
            df_train = data_df.iloc[shuffuled_idxs[val_size+test_size:]]
            
        else:
            df_test = pd.DataFrame()
            df_train = data_df.iloc[shuffuled_idxs[val_size:]]
        
        return df_train, df_val, df_test
    
    else:
        data_df_ = data_df.set_index(stratify_cols, drop=True)
        idxs = data_df_.index.unique()
        print("Number of unique {} tuples: {}".format(stratify_cols, len(idxs)))
        if isinstance(val_size, float) and val_size < 1:
                val_size = max(batch_size, int(len(idxs) * val_size))
        else:
            val_size = max(batch_size, int(val_size))
        if isinstance(test_size, float) and test_size < 1:
                test_size = max(batch_size, int(len(idxs) * test_size))
        elif test_size is None:
                test_size = 0
        else:
            test_size = max(batch_size, int(test_size))

        shuffuled_idxs = idxs[np.random.permutation(len(data_df_.index.unique()))]
        df_val = data_df_.loc[shuffuled_idxs[:val_size]].reset_index()
        
        if test_size > 0:
            df_test = data_df_.loc[shuffuled_idxs[val_size:val_size+test_size]].reset_index()
            df_train = data_df_.loc[shuffuled_idxs[val_size+test_size:]].reset_index()
            
        else:
            df_test = pd.DataFrame()
            df_train = data_df_.loc[shuffuled_idxs[val_size:]].reset_index()
        
        return df_train, df_val, df_test

class SameTaskPriorBatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.dataset_sizes = [len(d) for d in dataset.datasets]
        self.total_size = sum(self.dataset_sizes)

    def __len__(self):
        return self.total_size

    def __iter__(self):
        
        dataset_boundaries = [0] + list(torch.cumsum(torch.tensor(self.dataset_sizes), 0).numpy())
        dataset_indices = [list(range(dataset_boundaries[i], dataset_boundaries[i+1])) for i in range(self.number_of_datasets)]
        dataset_iterators = [iter(RandomSampler(dataset_indices[i])) for i in range(self.number_of_datasets)]

        dataset_idx = 0
        while dataset_idx < self.number_of_datasets:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    sample_idx = next(dataset_iterators[dataset_idx])
                    global_idx = dataset_indices[dataset_idx][sample_idx]
                    batch.append(global_idx)
                except StopIteration:
                    break  # Move to the next dataset if the current is exhausted

            if not batch:  # If the batch is empty, move to the next dataset
                dataset_idx += 1
                continue

            yield from batch

            if len(batch) < self.batch_size:  # Move to the next dataset if the last batch was not full
                dataset_idx += 1

class SameRatioPriorBatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.dataset_sizes = [len(d) for d in dataset.datasets]
        self.total_size = sum(self.dataset_sizes)
        self.dataset_ratios = [len(d) / self.total_size for d in dataset.datasets]

    def __len__(self):
        return self.total_size

    def __iter__(self):
        
        dataset_boundaries = [0] + list(torch.cumsum(torch.tensor(self.dataset_sizes), 0).numpy())
        dataset_indices = [list(range(dataset_boundaries[i], dataset_boundaries[i+1])) for i in range(self.number_of_datasets)]
        dataset_iterators = [iter(RandomSampler(dataset_indices[i])) for i in range(self.number_of_datasets)]
        sampled_num = 0
        batch = []
        while sampled_num < self.total_size:
            for dataset_idx in range(self.number_of_datasets):
                sample_needed = max(4, int(self.batch_size * self.dataset_ratios[dataset_idx]))
                sample_needed = min(sample_needed, len(dataset_indices[dataset_idx]))
                while len(batch) < self.batch_size:
                    if sample_needed == 0:
                        break
                    try:
                        sample_idx = next(dataset_iterators[dataset_idx])
                        global_idx = dataset_indices[dataset_idx][sample_idx]
                        batch.append(global_idx)
                        sample_needed -= 1
                        sampled_num += 1
                    except StopIteration:
                        break  # Move to the next dataset if the current is exhausted

                if len(batch) == self.batch_size:
                    yield from batch
                    batch = []

        if len(batch) > 0:
            yield from batch