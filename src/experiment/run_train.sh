#!/bin/bash
#SBATCH --job-name=train_tsd_ssd_ws24_att_cgcnn
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --nodelist=c3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:1
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

# Train model with saving each epoch's checkpoint
echo "Starting model training with save_all_epochs..."
srun python -u /home/zhangsd/repos/MOFSNN/src/cgcnn/main.py --task_cfg tsd_ssd_ws24 --model_cfg att_cgcnn --progress_bar --batch_size 32 --max_epochs 500 --max_graph_len 200 --atom_fea_len 144 --extra_fea_len 28 --h_fea_len 144 --n_conv 4 --n_h 8 --dropout_prob 0.55 --use_cell_params --atom_layer_norm --loss_aggregation fixed_weight_sum --dl_sampler random --task_att_type self --lr 0.001 --lr_mult 1 --group_lr --optim_config fine --patience 50 --task_norm --log_dir /home/zhangsd/repos/MOFSNN/results/cgcnn_models --optim adam --decay_power constant_with_warmup --save_all_epochs

echo "Training completed."