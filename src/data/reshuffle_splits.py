'''
Author: zhangshd
Date: 2025-05-10 18:25:36
LastEditors: zhangshd
LastEditTime: 2025-05-10 21:46:09

This script checks for overlapping samples between different datasets based on MofName.
'''

import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load datasets
tsd_df = pd.read_csv(ROOT_DIR/'data/ml_data/TSD/RAC_and_zeo_features_with_id_prop.csv')
ssd_df = pd.read_csv(ROOT_DIR/'data/ml_data/SSD/RAC_and_zeo_features_with_id_prop.csv')
ws24_df = pd.read_csv(ROOT_DIR/'data/ml_data/WS24/RAC_and_zeo_features_with_id_prop.csv')
print("TSD dataset:", tsd_df.shape)
print("SSD dataset:", ssd_df.shape)
print("WS24 dataset:", ws24_df.shape)

tsd_raw_df = pd.read_csv(ROOT_DIR/'data/raw_data/Nandy_2022_SciData/separate_files/thermal_stability/full_TSD_data.csv')
ssd_raw_df = pd.read_csv(ROOT_DIR/'data/raw_data/Nandy_2022_SciData/separate_files/solvent_removal_stability/full_SSD_data.csv')
tsd_raw_df.dropna(subset=['CoRE_name', 'refcode'], inplace=True)
ssd_raw_df.dropna(subset=['CoRE_name', 'refcode'], inplace=True)
print("TSD raw data:", tsd_raw_df.shape)
print("SSD raw data:", ssd_raw_df.shape)

name2code_map = {}
code2name_map = {}
for i, row in tsd_raw_df.iterrows():
    if row['refcode'] not in code2name_map:
        name2code_map[row['CoRE_name']] = row['refcode']
        code2name_map[row['refcode']] = row['CoRE_name']
    else:
        assert code2name_map[row['refcode']] == row['CoRE_name'], \
        f"Duplicate refcode {row['refcode']} with different CoRE names: {row['CoRE_name']} and {code2name_map[row['refcode']]}"
for i, row in ssd_raw_df.iterrows():
    if row['refcode'] not in code2name_map:
        name2code_map[row['CoRE_name']] = row['refcode']
        code2name_map[row['refcode']] = row['CoRE_name']
    else:
        assert code2name_map[row['refcode']] == row['CoRE_name'], \
        f"Duplicate refcode {row['refcode']} with different CoRE names: {row['CoRE_name']} and {code2name_map[row['refcode']]}"
tsd_df['MofName'] = tsd_df['MofName'].apply(lambda x: name2code_map[x] if x in name2code_map else x)
ssd_df['MofName'] = ssd_df['MofName'].apply(lambda x: name2code_map[x] if x in name2code_map else x)

# Extract MofNames  
tsd_names = set(tsd_df['MofName'])
ssd_names = set(ssd_df['MofName'])
ws24_names = set(ws24_df['MofName'])

# Check overlaps
tsd_ssd_overlap = tsd_names.intersection(ssd_names)
tsd_ws24_overlap = tsd_names.intersection(ws24_names)
ssd_ws24_overlap = ssd_names.intersection(ws24_names)
all_overlap = tsd_names.intersection(ssd_names, ws24_names)

# Print results
print(f"TSD dataset: {len(tsd_df)} samples")
print(f"SSD dataset: {len(ssd_df)} samples")
print(f"WS24 dataset: {len(ws24_df)} samples")
print(f"Overlap between TSD and SSD: {len(tsd_ssd_overlap)} samples")
print(f"Overlap between TSD and WS24: {len(tsd_ws24_overlap)} samples")
print(f"Overlap between SSD and WS24: {len(ssd_ws24_overlap)} samples")
print(f"Overlap among all three datasets: {len(all_overlap)} samples")
print(f"All unique MOFs: {len(set(tsd_names).union(ssd_names).union(ws24_names))}")

# Check if any samples have different partitions across datasets
print("\nChecking for inconsistent partitions...")

# Create mapping of MofName to partition for each dataset
tsd_partitions = dict(zip(tsd_df['MofName'], tsd_df['Partition']))
ssd_partitions = dict(zip(ssd_df['MofName'], ssd_df['Partition']))
ws24_partitions = dict(zip(ws24_df['MofName'], ws24_df['Partition']))

# Check TSD and SSD overlap
inconsistent = 0
for mof in tsd_ssd_overlap:
    if tsd_partitions[mof] != ssd_partitions[mof]:
        inconsistent += 1
        if inconsistent < 5:  # Limit the output to 5 examples
            print(f"  {mof}: TSD={tsd_partitions[mof]}, SSD={ssd_partitions[mof]}")
        elif inconsistent == 5:  # Limit the output to 5 examples
            print("  ... more inconsistencies found")

print(f"Found {inconsistent} samples with inconsistent partitions between TSD and SSD")

# Check distribution of partitions
print("\nPartition distribution:")
for dataset_name, df in [("TSD", tsd_df), ("SSD", ssd_df), ("WS24", ws24_df)]:
    partition_counts = df['Partition'].value_counts()
    partition_pcts = partition_counts / len(df) * 100
    print(f"{dataset_name}: {', '.join([f'{p}: {c} ({pct:.1f}%)' for p, c, pct in zip(partition_counts.index, partition_counts, partition_pcts)])}")


## reshuffling
tsd_df_ = tsd_df.copy().set_index('MofName')[["Label"]].rename(columns={"Label": "TSD"})
ssd_df_ = ssd_df.copy().set_index('MofName')[["Label"]].rename(columns={"Label": "SSD"})
ws24_df_ = ws24_df.copy().set_index('MofName')[["water_label", "water4_label", "acid_label", "base_label", "boiling_label"]]
all_df = pd.concat([tsd_df_, ssd_df_, ws24_df_], axis=1)
print("Concatenated DataFrame shape:", all_df.shape)
all_df.fillna(100, inplace=True)

cls_label_cols = ["water_label", "acid_label", "base_label", "boiling_label"]
for col in cls_label_cols:
    print(col, "-"*20)
    for v in all_df[col].unique():
        print(f"Number of {v}: {(all_df[col]==v).sum()}")
for seed in range(5):
    train_df, test_df = train_test_split(all_df, test_size=0.2, stratify=all_df[cls_label_cols], random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df[cls_label_cols], random_state=seed)
    train_df.insert(0, "Partition", "train")
    val_df.insert(0, "Partition", "val")
    test_df.insert(0, "Partition", "test")

    total_df = pd.concat([train_df, val_df, test_df])
    name2partition = total_df["Partition"].to_dict()
    
    # Log split information
    ratio_dict = {}
    for col in ["SSD"] + cls_label_cols + ["water4_label"]:
        for lb in total_df[col].unique():
            split_ratios = []
            for split in ["train", "val", "test"]:
                sub_df = total_df.loc[(total_df[col] == lb) & (total_df["Partition"] == split)]
                sub_ratio = len(sub_df) / len(total_df.loc[(total_df[col] == lb)])
                split_ratios.append([sub_ratio, len(sub_df)])
            ratio_dict[f"{col}_{lb}(train/val/test)"] = ' : '.join([f"{c}({r:.2f})" for r, c in split_ratios])

    for k, v in ratio_dict.items():
        print(f"{k}: {v}")

    tsd_df["Partition"] = tsd_df["MofName"].apply(lambda x: name2partition[x] if x in name2partition else None)
    ssd_df['Partition'] = ssd_df["MofName"].apply(lambda x: name2partition[x] if x in name2partition else None)
    ws24_df['Partition'] = ws24_df["MofName"].apply(lambda x: name2partition[x] if x in name2partition else None)

    print("TSD samples without patition: ", tsd_df['Partition'].isna().sum())
    print("SSD samples without patition: ", ssd_df['Partition'].isna().sum())
    print("WS24 samples without patition: ", ws24_df['Partition'].isna().sum())
    print("TSD train/val/test: ", tsd_df['Partition'].value_counts())
    print("SSD train/val/test: ", ssd_df['Partition'].value_counts())
    print("WS24 train/val/test: ", ws24_df['Partition'].value_counts())

    # Map refcode to CoRE_name for TSD and SSD
    tsd_df_ = tsd_df.copy()
    ssd_df_ = ssd_df.copy()
    tsd_df_["MofName"] = tsd_df_["MofName"].apply(lambda x: code2name_map[x] if x in code2name_map else x)
    ssd_df_["MofName"] = ssd_df_["MofName"].apply(lambda x: code2name_map[x] if x in code2name_map else x)

    # Save the reshuffled dataset to CSV files in CGCNN data directory and ML data directory
    ml_data_dir = ROOT_DIR / 'data/ml_data'
    cgcnn_data_dir = ROOT_DIR / 'data/cgcnn_data'

    tsd_df_.to_csv(ml_data_dir / 'TSD' / f'RAC_and_zeo_features_with_id_prop_rand{seed}.csv', index=False)
    ssd_df_.to_csv(ml_data_dir / 'SSD' / f'RAC_and_zeo_features_with_id_prop_rand{seed}.csv', index=False)
    ws24_df.to_csv(ml_data_dir / 'WS24' / f'RAC_and_zeo_features_with_id_prop_rand{seed}.csv', index=False)
    tsd_df_.to_csv(cgcnn_data_dir / 'TSD' / f'RAC_and_zeo_features_with_id_prop_rand{seed}.csv', index=False)
    ssd_df_.to_csv(cgcnn_data_dir / 'SSD' / f'RAC_and_zeo_features_with_id_prop_rand{seed}.csv', index=False)
    ws24_df.to_csv(cgcnn_data_dir / 'WS24' / f'RAC_and_zeo_features_with_id_prop_rand{seed}.csv', index=False)
    print(f"Saved reshuffled datasets for seed {seed} to {ml_data_dir} and {cgcnn_data_dir}")


    
