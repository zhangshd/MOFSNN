'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:36:30
'''
## This script is adapted from MOFTransformer(https://github.com/hspark1212/MOFTransformer) and CGCNN(https://github.com/txie-93/cgcnn)

from __future__ import print_function, division

import functools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class LoadGraphData(Dataset):
    """ 
    Load CIFDATA dataset from "CIF_NAME.graphdata"
    """
    def __init__(self, data_dir, split, radius=8, dmin=0, step=0.2, 
                 prop_cols=None, use_cell_params=False, use_extra_fea=False,
                 task_id=0, augment=False, **kwargs
                 ):
        
        data_dir = Path(data_dir)
        self.split = split
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.augment = augment
        self.use_cell_params = use_cell_params
        self.use_extra_fea = use_extra_fea
        self.task_id = task_id
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.csv_file_name = kwargs.get("csv_file_name", "RAC_and_zeo_features_with_id_prop.csv")
        self.aug_csv_file_name = kwargs.get("aug_csv_file_name", "RAC_and_zeo_features_with_id_prop_aug.csv")
        self.down_sampling = kwargs.get("down_sampling", True)

        # print("use_cell_params:", self.use_cell_params)
        # print("use_extra_fea:", self.use_extra_fea)

        if  "WS24" in data_dir.name and len(data_dir.name.split("_")) == 2:
            prop_cols = [data_dir.name.split("_")[1] + "_label"]
            # except Exception as e:
            #     prop_cols = ["water_label"]
            if "test" not in data_dir.name:
                data_dir = data_dir.parent / "WS24"
        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        
        self.data_dir = data_dir

        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        print("prop_cols:", self.prop_cols)
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split, self.prop_cols, augment, 
                                      id_prop_file_aug=data_dir/self.aug_csv_file_name,
                                      random_state=42, max_sample_size=self.max_sample_size,
                                      down_sampling=self.down_sampling)
        
        if self.prop_cols and "water4_label" in self.prop_cols:
            self.id_prop_df["water4_label"] -= 1 ## change to 1234 to 0123
        self.id_prop_df.fillna(0, inplace=True)
        file_list = (data_dir / "clean_cifs").glob('*.graphdata')
        self.g_data = {file.stem: file for file in file_list if file.stem in self.id_prop_df.index}
        assert len(self.g_data) == len(self.id_prop_df.index.unique()), f'{len(self.g_data)} != {len(self.id_prop_df.index.unique())}'

        atom_prop_json = Path(__file__).parent/'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_prop_json)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)
        self.g_data.update(new_data.g_data)

    def __len__(self):
        return len(self.id_prop_df)

    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,LCD,PLD,Desity(g/cm^3),VSA(m^2/cm^3),GSA(m^2/g),Vp(cm^3/g),VoidFraction,Label
        cif_id = row.name

        if self.use_extra_fea:
            extra_fea = row.loc["Di":].values.astype(float)
        else:
            extra_fea = []
    
        targets = row[self.prop_cols].values.astype(float)
        
        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, *_, cell_params = data
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == 10.0, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= 10.0 for file: {self.g_data[cif_id]}"

        targets = torch.FloatTensor(targets)

        extra_fea = torch.FloatTensor(extra_fea)

        atom_fea = np.vstack([self.ari.get_atom_fea(i) for i in atom_num])
        atom_fea = torch.Tensor(atom_fea)

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(nbr_dist).view(len(atom_num), -1)
        nbr_fea = self.gdf.expand(nbr_dist).float()
        assert isinstance(nbr_fea, torch.Tensor)

        if self.use_cell_params:
            cell_params = torch.FloatTensor(cell_params)
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        ret_dict = {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "extra_fea": extra_fea,
            "targets": targets,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        return ret_dict
    
    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["targets"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_fea_idx in enumerate(batch_nbr_fea_idx):
            n_i = nbr_fea_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_fea_idx += base_idx
            base_idx += n_i

        dict_batch["atom_fea"] = torch.cat(batch_atom_fea, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_fea_idx, dim=0)
        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["targets"] = torch.stack(batch_targets, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        return dict_batch


class LoadGraphDataWithAtomicNumber(Dataset):

    """ 
    Load CIFDATA dataset from "CIF_NAME.graphdata"
    """
    def __init__(self, data_dir, split, radius=8, dmin=0, step=0.2, 
                 prop_cols=None, use_cell_params=False, use_extra_fea=False,
                 task_id=0, augment=False, **kwargs
                 ):
        data_dir = Path(data_dir)
        self.split = split
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.augment = augment
        self.use_cell_params = use_cell_params
        self.use_extra_fea = use_extra_fea
        self.task_id = task_id
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.csv_file_name = kwargs.get("csv_file_name", "RAC_and_zeo_features_with_id_prop.csv")
        self.aug_csv_file_name = kwargs.get("aug_csv_file_name", "RAC_and_zeo_features_with_id_prop_aug.csv")
        self.down_sampling = kwargs.get("down_sampling", True)

        if  "WS24" in data_dir.name and "test" not in data_dir.name:
            try:
                prop_cols = [data_dir.name.split("_")[1] + "_label"]
            except Exception as e:
                prop_cols = ["water_label"]
            data_dir = data_dir.parent / "WS24"
        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        
        self.data_dir = data_dir
        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        print("prop_cols:", self.prop_cols)
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split, self.prop_cols, augment, 
                                      id_prop_file_aug=data_dir/self.aug_csv_file_name,
                                      random_state=42, max_sample_size=self.max_sample_size,
                                      down_sampling=self.down_sampling)
        
        if self.prop_cols and "water4_label" in self.prop_cols:
            self.id_prop_df["water4_label"] -= 1 ## change to 1234 to 0123
        self.id_prop_df.fillna(0, inplace=True)
        file_list = (data_dir / "clean_cifs").glob('*.graphdata')
        self.g_data = {file.stem: file for file in file_list if file.stem in self.id_prop_df.index}
        assert len(self.g_data) == len(self.id_prop_df.index.unique()), f'{len(self.g_data)} != {len(self.id_prop_df.index.unique())}'

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)
        self.g_data.update(new_data.g_data)

    def __len__(self):
        return len(self.id_prop_df)
    
    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,LCD,PLD,Desity(g/cm^3),VSA(m^2/cm^3),GSA(m^2/g),Vp(cm^3/g),VoidFraction,Label
        cif_id = row.name

        if self.use_extra_fea:
            extra_fea = row.loc["Di":].values.astype(float)
        else:
            extra_fea = []
        
        targets = row[self.prop_cols].values.astype(float)

        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, uni_idx, uni_count, cell_params = data
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == 10.0, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= 10.0 for file: {self.g_data[cif_id]}"

        targets = torch.FloatTensor(targets)

        extra_fea = torch.FloatTensor(extra_fea)

        atom_fea = torch.LongTensor(atom_num) ## use atomic number as feature

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(nbr_dist).view(len(atom_num), -1)
        nbr_fea = self.gdf.expand(nbr_dist).float()
        assert isinstance(nbr_fea, torch.Tensor)

        if self.use_cell_params:
            cell_params = torch.FloatTensor(cell_params)
            extra_fea = torch.cat([extra_fea, cell_params], dim=-1)

        ret_dict = {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
            "extra_fea": extra_fea,
            "targets": targets,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        return ret_dict
    
    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_atom_fea = dict_batch["atom_fea"]
        batch_nbr_fea_idx = dict_batch["nbr_fea_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]
        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["targets"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_fea_idx in enumerate(batch_nbr_fea_idx):
            n_i = nbr_fea_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_fea_idx += base_idx
            base_idx += n_i

        dict_batch["atom_fea"] = torch.cat(batch_atom_fea, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_fea_idx, dim=0)
        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch["targets"] = torch.stack(batch_targets, dim=0)
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        return dict_batch

class LoadExtraFeatureData(Dataset):
    """ 
    Load RACs and ZEOS dataset from csv files"
    """
    def __init__(self, data_dir, split, 
                 prop_cols=None,
                 task_id=0,
                 augment=False, 
                 **kwargs
                 ):
        data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.task_id = task_id
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.csv_file_name = kwargs.get("csv_file_name", "RAC_and_zeo_features_with_id_prop.csv")
        self.aug_csv_file_name = kwargs.get("aug_csv_file_name", "RAC_and_zeo_features_with_id_prop_aug.csv")
        self.down_sampling = kwargs.get("down_sampling", True)

        if  "WS24" in data_dir.name and "test" not in data_dir.name:
            try:
                prop_cols = [data_dir.name.split("_")[1] + "_label"]
            except Exception as e:
                prop_cols = ["water_label"]
            data_dir = data_dir.parent / "WS24"
        assert data_dir.exists(), "Dataset directory not found: {}".format(data_dir)
        self.data_dir = data_dir
        self.prop_cols = prop_cols if prop_cols is not None else ["Label"]
        print("prop_cols:", self.prop_cols)
        self.id_prop_df = sample_data(data_dir/self.csv_file_name, split, self.prop_cols, augment, 
                                      id_prop_file_aug=data_dir/self.aug_csv_file_name,
                                      random_state=42, max_sample_size=self.max_sample_size,
                                      down_sampling=self.down_sampling
                                      )
        
        if self.prop_cols and "water4_label" in self.prop_cols:
            self.id_prop_df["water4_label"] -= 1 ## change to 1234 to 0123
        self.id_prop_df.fillna(0, inplace=True)
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.id_prop_df = pd.concat([self.id_prop_df, new_data.id_prop_df], axis=0)

    def __len__(self):
        return len(self.id_prop_df)

    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        row = self.id_prop_df.iloc[idx]
        ## MofName,Partition,Label,Di,...
        cif_id = row.name

        extra_fea = row.loc["Di":].values.astype(float)
    
        
        targets = row[self.prop_cols].values.astype(float)

        targets = torch.FloatTensor(targets)

        extra_fea = torch.FloatTensor(extra_fea)

        ret_dict = {
            "extra_fea": extra_fea,
            "targets": targets,
            "cif_id": cif_id,
            "task_id": self.task_id
        }

        return ret_dict

    @staticmethod
    def collate(batch):
    
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_extra_fea = dict_batch["extra_fea"]
        batch_targets = dict_batch["targets"]

        dict_batch["extra_fea"] = torch.stack(batch_extra_fea, dim=0)
        dict_batch["targets"] = torch.stack(batch_targets, dim=0)
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])

        return dict_batch
    
    
def sample_data(id_prop_file, split, prop_cols, augment=False, id_prop_file_aug=None, 
                random_state=42, max_sample_size: dict=None, down_sampling=True):
    
    """
    Sample augmented data from dataset
    """
    if max_sample_size is None:
        max_sample_size = {
                "train": 2004,
                "val": 501,
            }
    # if augment and split in ["train", "val"]:
    #     print(f"Sampling {split} data with max_sample_size={max_sample_size[split]}, augment={augment}")
    
    assert os.path.exists(id_prop_file), f'{str(id_prop_file)} not exists'
    id_prop_df = pd.read_csv(id_prop_file, index_col=0)
    if split not in ["train", "val", "test"]:
        return id_prop_df
    id_prop_df = id_prop_df[id_prop_df["Partition"] == split]

    if isinstance(prop_cols, str):
        prop_cols = [prop_cols]
    
    if not augment or id_prop_file_aug is None or not os.path.exists(id_prop_file_aug):
        if len(id_prop_df[prop_cols[0]].unique()) > 5:  ## detect if the task is classification
            return id_prop_df
        elif not prop_cols or prop_cols[0] not in ["acid_label", "base_label", "boiling_label"]:
            return id_prop_df
        elif not down_sampling:
            return id_prop_df

        ## for classification task, we need to sample data from each class
        ## to balance the dataset. We use the minority class as the sample size.
        cls_counts = id_prop_df[prop_cols[0]].value_counts()
        sample_size_cls = cls_counts.min()
        cls_dfs = []
        for cls in cls_counts.index[::-1]:
            cls_df = id_prop_df[id_prop_df[prop_cols[0]] == cls]
            cls_df = cls_df.sample(n=sample_size_cls, 
                                    replace=False, random_state=random_state)
            cls_dfs.append(cls_df)
        id_prop_df = pd.concat(cls_dfs, axis=0)
        
        return id_prop_df

    # if not prop_cols or prop_cols[0] not in ["acid_label", "base_label", "boiling_label"]:
    #     return id_prop_df

    id_prop_df_aug = pd.read_csv(id_prop_file_aug, index_col=0)
    id_prop_df_aug = id_prop_df_aug[id_prop_df_aug["Partition"] == split]
    
    
    if len(id_prop_df[prop_cols[0]].unique()) <= 5:  ## detect if the task is classification
        ## for classification task, we need to sample data from augmented data
        ## to balance the dataset. We use the difference between the max mean sample size
        ## and the current sample size as the sample size for each class.
        cls_counts = id_prop_df[prop_cols[0]].value_counts()
        cls_counts_aug = id_prop_df_aug[prop_cols[0]].value_counts()
        cls_counts_total = cls_counts + cls_counts_aug
        
        sample_size_cls_avg = min(cls_counts_total.min(), max_sample_size[split]//len(cls_counts))
        
        cls_dfs = []
        for cls in cls_counts.index[::-1]:
            cls_df = id_prop_df[id_prop_df[prop_cols[0]] == cls]
            sample_size_cls = sample_size_cls_avg - len(cls_df)
            if sample_size_cls > 0:
                cls_df_aug = id_prop_df_aug[id_prop_df_aug[prop_cols[0]] == cls]
                cls_df_aug = cls_df_aug.sample(n=min(sample_size_cls, len(cls_df_aug)), 
                                               replace=False, random_state=random_state)
                cls_df = pd.concat([cls_df, cls_df_aug], axis=0)
            elif sample_size_cls < 0:
                cls_df = cls_df.sample(n=sample_size_cls_avg, 
                                       replace=True, random_state=random_state)
            cls_dfs.append(cls_df)
        id_prop_df_aug = pd.concat(cls_dfs, axis=0)
    else:
        id_prop_df_aug = id_prop_df_aug.sample(n=max(0, min(max_sample_size[split]-len(id_prop_df), 
                                                     len(id_prop_df_aug))),
                                               replace=False, random_state=random_state)
        id_prop_df_aug = pd.concat([id_prop_df, id_prop_df_aug], axis=0)
    return id_prop_df_aug


def collate_pool(dataset_list):
    
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_extra_fea = [], [], [], []
    crystal_atom_idx, batch_targets, batch_task_ids = [], [], []
    batch_cif_ids = []
    base_idx = 0

    for i, ((atom_fea, nbr_fea, nbr_fea_idx), extra_fea, targets, cif_id, task_id) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_extra_fea.append(extra_fea)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_targets.append(targets)
        batch_cif_ids.append(cif_id)
        batch_task_ids.append(task_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
           torch.stack(batch_extra_fea, dim=0), \
           torch.stack(batch_targets, dim=0), \
           batch_cif_ids, \
           torch.IntTensor(batch_task_ids)

def collate_extra(dataset_list):
    
    batch_extra_fea = []
    batch_targets, batch_task_ids = [], []
    batch_cif_ids = []

    for i, (tup, extra_fea, targets, cif_id, task_id) in enumerate(dataset_list):
        
        batch_extra_fea.append(extra_fea)
        batch_targets.append(targets)
        batch_cif_ids.append(cif_id)
        batch_task_ids.append(task_id)
        
    return tuple(), \
           torch.stack(batch_extra_fea, dim=0), \
           torch.stack(batch_targets, dim=0), \
           batch_cif_ids, \
           torch.IntTensor(batch_task_ids)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

