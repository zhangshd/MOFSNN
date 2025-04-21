#!/bin/bash
#SBATCH --job-name=clean_cif_CoREMOF2019
#SBATCH --output=/home/zhangsd/repos/MOFSNN/CGCNN_MT/data/CoREMOF2019/%x_%A.out
#SBATCH --error=/home/zhangsd/repos/MOFSNN/CGCNN_MT/data/CoREMOF2019/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u /home/zhangsd/repos/MOFSNN/CGCNN_MT/datamodule/clean_cif.py --cif_dir /home/zhangsd/repos/MOFSNN/raw_data/CoREMOF2019 --output_dir /home/zhangsd/repos/MOFSNN/CGCNN_MT/data/CoREMOF2019/clean_cifs --sanitize True --log_file /home/zhangsd/repos/MOFSNN/CGCNN_MT/data/CoREMOF2019/clean_cif.log --n_cpus 1
