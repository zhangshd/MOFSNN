#!/bin/bash
#SBATCH --job-name=ml_train_WS24_boiling_label
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py --model_type classification --model_list RF GP SVM LR  --search_max_evals 100 --search_metric val_AUC  --label_column boiling_label --group_column boiling_label --name_column MofName  --feature_selector_list RFE f1 mutual_info  --data_dir ./data/WS24 --in_file_name RAC_and_zeo_features_with_id_prop.csv
