
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from CGCNN_MT.module.att_cgcnn import CrystalGraphConvNet as AttCGCNN
from CGCNN_MT.module.att_fcnn import AttFCNN
from CGCNN_MT.module.cgcnn import CrystalGraphConvNet as CGCNN
from CGCNN_MT.module.cgcnn_raw import CrystalGraphConvNet as CGCNNRaw
from CGCNN_MT.module.cgcnn_uni_atom import CrystalGraphConvNet as CGCNNUniAtom
from CGCNN_MT.module.fcnn import FCNN
from CGCNN_MT.datamodule.dataset import LoadGraphData, LoadGraphDataWithAtomicNumber, LoadExtraFeatureData
import pytorch_lightning.callbacks as plc
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from CGCNN_MT.module.module import MInterface


MODEL_NAME_TO_DATASET_CLS = {
        "att_cgcnn": LoadGraphData,
        "cgcnn": LoadGraphData,
        "cgcnn_raw": LoadGraphData,
        "cgcnn_uni_atom": LoadGraphDataWithAtomicNumber,
        "fcnn": LoadExtraFeatureData,
        "att_fcnn": LoadExtraFeatureData,
    }

MODEL_NAME_TO_MODULE_CLS = {
        "att_cgcnn": AttCGCNN,
        "cgcnn": CGCNN,
        "cgcnn_raw": CGCNNRaw,
        "cgcnn_uni_atom": CGCNNUniAtom,
        "fcnn": FCNN,
        "att_fcnn": AttFCNN,
    }

def load_callbacks(patience=10, min_delta=0.0, monitor='val_loss', mode='min', lr_scheduler=None):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor=monitor,
        filename='best-{epoch:02d}-{val_loss:.3f}'.replace("val_loss", monitor),
        save_top_k=1,
        mode=mode,
        save_last=True,
        verbose=True
    ))

    if lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    return callbacks

def load_model_from_dir(model_dir):
    torch.set_float32_matmul_precision("medium")
    model_dir = Path(model_dir)
    with open(model_dir/'hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    
    hparams["model"] = MODEL_NAME_TO_MODULE_CLS[hparams["model_name"]](**hparams)

    trainer = Trainer(default_root_dir=hparams["log_dir"], 
                      accelerator=hparams["accelerator"],
                      devices=find_usable_cuda_devices(1),
                      )
    model_file = [file for file in (model_dir / 'checkpoints').glob('*.ckpt') if 'last' not in file.name][0]
    model = MInterface.load_from_checkpoint(model_file, **hparams)
    return model, trainer

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)