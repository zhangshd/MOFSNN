#!/bin/bash
#SBATCH --job-name=build_tree_TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn_version_52
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

# Build latent vector trees for model
echo "Starting latent vector tree building for /home/zhangsd/repos/MOFSNN/results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_52..."

srun python -u /home/zhangsd/repos/MOFSNN/src/experiment/build_latent_vec_tree.py --model_dir /home/zhangsd/repos/MOFSNN/results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_52 --k 5 --process_all --output_dir /home/zhangsd/repos/MOFSNN/results/uncertainty_evolution/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn_version_52

echo "Processing completed."