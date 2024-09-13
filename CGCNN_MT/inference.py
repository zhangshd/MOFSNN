'''
Author: zhangshd
Date: 2024-08-19 15:59:37
LastEditors: zhangshd
LastEditTime: 2024-09-13 22:20:10
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from CGCNN_MT.module.module import MInterface
from CGCNN_MT.module.cgcnn import CrystalGraphConvNet
from CGCNN_MT.datamodule.data_interface import DInterface
from pymatgen.io.cif import CifParser
from ase.io import read
from CGCNN_MT.datamodule.prepare_data import _make_supercell, get_crystal_graph
from CGCNN_MT.datamodule.dataset import AtomCustomJSONInitializer, GaussianDistance
from CGCNN_MT.utils import load_model_from_dir, MODEL_NAME_TO_DATASET_CLS
from CGCNN_MT.datamodule.prepare_data import make_prepared_data
from CGCNN_MT.datamodule.clean_cif import clean_cif
from CGCNN_MT.module.module_utils import calculate_lse_from_tree, calculate_lsv_from_tree
from pytorch_lightning.accelerators import find_usable_cuda_devices
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import Trainer
from pathlib import Path
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os, sys
import functools
import inspect
from tqdm import tqdm
import shutil
matplotlib.use('Agg')



def process_cif(cif, saved_dir, clean=True, **kwargs):

    if isinstance(cif, str):
        cif = Path(cif)
    saved_dir = Path(saved_dir)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(str(saved_dir/__name__))

    graphdata_dir = saved_dir / "graphdata"

    cif_id: str = cif.stem
    graphdata_dir.mkdir(exist_ok=True, parents=True)
    clean_cif_file = graphdata_dir / f"{cif_id}.cif"
    p_graphdata = graphdata_dir / f"{cif_id}.graphdata"
    if not clean_cif_file.exists() and clean:
        flag = clean_cif(cif, clean_cif_file)
        if not flag:
            return None
    else:
        shutil.copy(cif, clean_cif_file)
    if not p_graphdata.exists():
        p_graphdata = make_prepared_data(clean_cif_file, graphdata_dir, logger, **kwargs)
    return p_graphdata
    

class InferenceDataset(Dataset):
    def __init__(self, cif_list, **kwargs):
        """
        Args:
            cif_list (list or str): list of cif file paths or a single cif file path.
        """
        if isinstance(cif_list, (str, Path)):
            self.cif_list = [Path(cif_list)]
        else:
            self.cif_list = [Path(cif) for cif in cif_list]

        self.split = "infer"
        self.radius = kwargs.get("radius", 8)
        self.max_num_nbr = kwargs.get("max_num_nbr", 10)
        self.dmin = kwargs.get("dmin", 0)
        self.step = kwargs.get("step", 0.2)
        self.use_cell_params = kwargs.get("use_cell_params", False)
        self.use_extra_fea = kwargs.get("use_extra_fea", False)
        self.task_id = kwargs.get("task_id", 0)
        self.max_sample_size = kwargs.get("max_sample_size", None)
        self.saved_dir = kwargs.get("saved_dir", Path(os.getcwd())/"inference")
        self.clean = kwargs.get("clean", True)

        self.cif_ids = [cif.stem for cif in self.cif_list]
        self.g_data ={}
        
        atom_prop_json = Path(inspect.getfile(AtomCustomJSONInitializer)).parent/'atom_init.json'
        self.ari = AtomCustomJSONInitializer(atom_prop_json)
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)
    
    def append(self, new_data: Dataset):
        if hasattr(self, 'datasets'):
            self.datasets.append(new_data)
        else:
            self.datasets = [self, new_data]
        self.g_data.update(new_data.g_data)

    def setup(self, stage=None):

        for cif in self.cif_list:
            graphdata_file = process_cif(cif, self.saved_dir, clean=self.clean, 
                                         max_num_nbr=self.max_num_nbr, 
                                         radius=self.radius)
            if graphdata_file:
                self.g_data[cif.stem] = graphdata_file
            else:
                self.cif_ids.remove(cif.stem)
                print(f"Error: {cif} has been removed from the dataset due to errors during data preparation.")
        
    def __len__(self):
        return len(self.g_data)

    @functools.lru_cache(maxsize=None)  # cache load strcutrue
    def __getitem__(self, idx):

        cif_id = self.cif_ids[idx]
        # print(cif_id, self.g_data[cif_id])
        with open(self.g_data[cif_id], 'rb') as f:
            data = pickle.load(f)

        cif_id, atom_num, nbr_fea_idx, nbr_dist, *_, cell_params = data
        assert nbr_fea_idx.shape[0] / atom_num.shape[0] == 10.0, f"nbr_fea_idx.shape[0] / atom_num.shape[0]!= 10.0 for file: {self.g_data[cif_id]}"


        extra_fea = torch.FloatTensor([])

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
        dict_batch["task_id"] = torch.IntTensor(dict_batch["task_id"])
        return dict_batch

def inference(cif_list, model_dir,  saved_dir, uncertainty_trees_file=None, **kwargs):
    """
    Args:    
        cif_list (list or str): list of cif file paths or a single cif file path.
        model (MInterface): trained model.
    """

    # set up model
    clean = kwargs.get("clean", True)
    model, trainer = load_model_from_dir(model_dir)
    if uncertainty_trees_file is not None and os.path.exists(uncertainty_trees_file):
        with open(uncertainty_trees_file, 'rb') as f:
            uncertainty_trees = pickle.load(f)
        print(f"Loaded uncertainty trees from {uncertainty_trees_file}")
    else:
        uncertainty_trees = None

    # set up dataset
    infer_dataset = InferenceDataset(cif_list, saved_dir=saved_dir, clean=clean, **model.hparams)
    infer_dataset.setup()
    infer_loader = DataLoader(infer_dataset, 
                              batch_size=min(len(infer_dataset), model.hparams.get("batch_size", 8)), 
                            #   batch_size = 2,
                              shuffle=False, 
                              num_workers=model.hparams.get("num_workers", 2),
                              collate_fn=infer_dataset.collate
                              )

    outputs = trainer.predict(model, infer_loader)
    all_outputs = {}
    all_outputs[f"cif_ids"] = [d["cif_id"] for d in infer_dataset]
    for task in model.hparams.get("tasks"):
        task_id = model.hparams["tasks"].index(task)
        task_tp = model.hparams["task_types"][task_id]
        all_outputs[f"{task}_pred"] = torch.cat([d[f"{task}_pred"] for d in outputs], dim=0).cpu().numpy().squeeze()
        if "classification" in task_tp:
            all_outputs[f"{task}_prob"] = torch.cat([d[f"{task}_prob"] for d in outputs], dim=0).cpu().numpy()
        if uncertainty_trees is None:
            continue

        all_outputs[f"{task}_uncertainty"] = []
        for d in tqdm(outputs):
            task_fea = d[f'{task}_last_layer_fea'].cpu().numpy().squeeze()
            if "classification" in task_tp:
                all_outputs[f"{task}_uncertainty"].append(calculate_lse_from_tree(uncertainty_trees[task], task_fea, k=uncertainty_trees[task]["k"]))
            else:
                all_outputs[f"{task}_uncertainty"].append(calculate_lsv_from_tree(uncertainty_trees[task], task_fea, k=uncertainty_trees[task]["k"]))
        all_outputs[f"{task}_uncertainty"] = np.concatenate(all_outputs[f"{task}_uncertainty"], axis=0)
    
    return all_outputs

if __name__ == "__main__":

    # cif_list = [
    #     "AVIHIY_clean.cif",
    #     "AVAQIX_clean.cif",
    #     "BIBBUL_clean.cif",
    #     "ELAZEX_clean.cif",
    #     "GOYYEA_clean.cif",
    #     "LIJXEI_clean.cif",
    #     "NOPJUZ_clean.cif",
    # ]
    clean = False
    cif_dir = Path(__file__).parent/"data/CoREMOF2019/clean_cifs"
    # cif_list = [cif_dir/cif for cif in cif_list]
    cif_list = sorted(cif_dir.glob("*.cif"))
    notes = cif_dir.parent.name
    model_dir = Path(__file__).parent/"logs/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43"
    uncertainty_trees_file = Path(__file__).parent/"evaluation/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn@version_43/uncertainty_trees.pkl"
    result_dir = Path(os.getcwd())/f"inference/{notes}"
    model_name = os.path.basename(model_dir)
    results = inference(cif_list, model_dir, saved_dir=result_dir, uncertainty_trees_file=uncertainty_trees_file, clean=clean)
    
    df_res = pd.DataFrame({k:v for k,v in results.items() if k.endswith("_pred") or "uncertainty" in k}, index=results["cif_ids"])
    print(df_res)
    df_res.to_csv(Path(result_dir)/f"infer_results_{model_name}.csv", float_format='%.4f')