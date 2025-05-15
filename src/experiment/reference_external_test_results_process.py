'''
Author: zhangshd
Date: 2025-05-16 01:55:19
LastEditors: zhangshd
LastEditTime: 2025-05-16 01:57:10
'''
import pandas as pd
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR/"data/cgcnn_data"

GROUP_NAMES = [
    "TS_external_test",
    "WS24v2_external_test",
]

LABEL_COLS_MAP = {
    "ts_label": "TSD",
    "ss_label": "SSD",
    "water_label": "WS24_water",
    "water4_label": "WS24_water4",
    "acid_label": "WS24_acid",
    "base_label": "WS24_base",
    "boiling_label": "WS24_boiling",
}

PRED_RESULTS_DIR = ROOT_DIR/"results/reference_pred"

def get_pred_results(group_name):
    """
    Get the prediction results from the reference model.
    """
    json_path = PRED_RESULTS_DIR/group_name/"all_results.json"
    with open(json_path, 'r') as f:
        pred_results = json.load(f)
    pred_df = pd.DataFrame(pred_results)
    return pred_df[["MofName"] + list(LABEL_COLS_MAP.keys())].copy()

def get_ground_truth(group_name):
    """
    Get the ground truth data from the reference model.
    """
    csv_file = DATA_DIR/group_name/"RAC_and_zeo_features_with_id_prop.csv"
    df = pd.read_csv(csv_file)
    return df[["MofName"] + [col for col in LABEL_COLS_MAP.keys() if col in df.columns]].copy()

def process_results():
    """
    Process the prediction results and ground truth data.
    """

    for group_name in GROUP_NAMES:
        pred_df = get_pred_results(group_name)
        gt_df = get_ground_truth(group_name)

        for col, task in LABEL_COLS_MAP.items():
            if col not in gt_df.columns:
                continue
            sub_df_gt = gt_df[["MofName", col]].copy()
            sub_df_gt.rename(columns={col: "GroundTruth"}, inplace=True)
            sub_df_pred = pred_df[["MofName", col]].copy()
            if task == "TSD":
                sub_df_pred.rename(columns={col: "Predicted"}, inplace=True)
            elif task == "WS24_water4":
                ## This is a special case for WS24_water4, which is a 4-class classification task.
                ## The predicted values are in the form of a list of 4 values, which need to be converted to a single value.
                sub_df_pred.insert(1, "Predicted", sub_df_pred[col].apply(lambda x: x.index(max(x)) if isinstance(x, list) else None))
                sub_df_gt["GroundTruth"] = sub_df_gt["GroundTruth"] - 1 # Convert to 0-indexed 
                sub_df_pred.rename(columns={col: "Prob"}, inplace=True)
            else:
                sub_df_pred.insert(1, "Predicted", sub_df_pred[col].apply(lambda x: 1 if x > 0.5 else 0))
                sub_df_pred.rename(columns={col: "Prob"}, inplace=True)
            sub_df = pd.merge(sub_df_gt, sub_df_pred, on="MofName", how="inner")
            print(f"Got {len(sub_df)} comparable samples for {task} in {group_name}.")
            out_csv = PRED_RESULTS_DIR/f"external_test_results_{task}.csv"
            sub_df.to_csv(out_csv, index=False)
            print(f"Saved {task} results to {out_csv}.")
            print("" + "="*50)

if __name__ == "__main__":
    process_results()
    print("Processing completed.")