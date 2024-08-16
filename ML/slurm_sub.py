import subprocess
from pathlib import Path
import os
import time

job_templet = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
export PATH=/opt/share/miniconda3/envs/mofmthnn/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofmthnn/lib/:$LD_LIBRARY_PATH

srun python -u main.py --model_type {model_type} --model_list {model_list} \
 --search_max_evals 100 --search_metric {search_metric} \
 --label_column {label_column} --group_column {label_column} --name_column {name_column} \
 --feature_selector_list RFE f1 mutual_info \
 --data_dir {data_dir} --in_file_name {in_file_name}
"""

def run_slurm_job(work_dir, executor="sbatch", script_name="run"):
    work_dir = Path(work_dir)
    process = subprocess.Popen(
        f"{executor} {work_dir/script_name}",
        # [executor, str(work_dir/'run'), "&"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(work_dir)
    )
    return process

if __name__ == '__main__':
    work_dir = Path("./")
    label_columns = ["water_label","water4_label", "acid_label", "base_label", "boiling_label"]
    script_name = "run_slurm.sh"
    in_file_name="RAC_and_zeo_features_with_id_prop.csv"
    for label_column in label_columns:
        task_name = "WS24"
        name_column="MofName"
        model_type = "classification"
        model_list = ["RF", "GP", "SVM", "LR"]
        model_list = " ".join(model_list)
        search_metric = "val_AUC"
        job_name = f"ml_train_{task_name}_{label_column}"
        data_dir = f"./data/{task_name}"
        
        job_script = job_templet.format(job_name=job_name,
                                        label_column=label_column,
                                        name_column=name_column,
                                        model_type=model_type,
                                        model_list=model_list,
                                        search_metric=search_metric,
                                        data_dir=data_dir,
                                        in_file_name=in_file_name
                                        )
        with open(work_dir/script_name, "w") as f:
            f.write(job_script)
        process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.decode().strip())
        print(f"Submitted job {job_name} with PID {process.pid}")
        time.sleep(1)

    # task_names = ["SSD"]
    # script_name = "run_slurm.sh"
    # model_list = ["RF", "GP", "SVM", "LR"]
    # in_file_name = "RAC_and_zeo_features_with_id_prop.csv"
    # model_list = " ".join(model_list)
    # for task_name in task_names:
    #     job_name = f"ml_train_{task_name}_new_feat"
    #     label_column="Label"
    #     name_column="MofName"
    #     data_dir = f"./data/{task_name}"
    #     if task_name == "TSD":
    #         model_type = "regression"
    #         search_metric = "val_R2"
    #     elif task_name in ["SSD", "WS24"]:
    #         model_type = "classification"
    #         search_metric = "val_AUC"

    #     job_script = job_templet.format(job_name=job_name, 
    #                                     label_column=label_column, 
    #                                     name_column=name_column, 
    #                                     model_type=model_type, 
    #                                     model_list=model_list, 
    #                                     search_metric=search_metric, 
    #                                     data_dir=data_dir,
    #                                      in_file_name=in_file_name
    #                                     )
    #     with open(work_dir/script_name, "w") as f:
    #         f.write(job_script)
    #     process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
    #     while True:
    #         output = process.stdout.readline()
    #         if output == b'' and process.poll() is not None:
    #             break
    #         if output:
    #             print(output.decode().strip())
    #     print(f"Submitted job {job_name} with PID {process.pid}")
        # time.sleep(1)

