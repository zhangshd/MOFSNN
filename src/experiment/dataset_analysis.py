"""
Dataset Analysis Script
It analyzes MOF stability datasets including TSD, SSD, and WS24 tasks, creating visualizations
for data distributions, correlations, and dataset intersections.
Author: zhangshd
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cgcnn.datamodule.dataset import LoadGraphData
import matplotlib.gridspec as gridspec
from upsetplot import UpSet
import upsetplot
import random
import re
from itertools import chain, product, combinations
from collections import defaultdict
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'dataset_analysis.log'),
            logging.StreamHandler()
        ]
    )


def load_config() -> Dict[str, Any]:
    """Load default configuration with all parameters"""
    config = {
        # Data paths
        'data_root_dir': str(project_root / "data" / "cgcnn_data"),
        'ssd_raw_csv': str(project_root / "data" / "raw_data" / "Nandy_2022_SciData" / "separate_files" / "solvent_removal_stability" / "full_SSD_data.csv"),
        'tsd_raw_csv': str(project_root / "data" / "raw_data" / "Nandy_2022_SciData" / "separate_files" / "thermal_stability" / "full_TSD_data.csv"),
        
        # Output paths
        'fig_dir': str(project_root / "results" / "data_analysis" / "figures"),
        'numerical_data_dir': str(project_root / "results" / "data_analysis" / "numerical_data"),
        'log_dir': str(project_root / "logs" / "data_processing"),
        
        # Tasks to analyze
        'tasks': ["TSD", "SSD", "WS24_water", "WS24_water4", "WS24_acid", "WS24_base", "WS24_boiling"],
        'task_types': ["regression", "classification", "classification", "classification", "classification", "classification", "classification"],
        
        # Plot settings
        'plot_settings': {
            'figure_dpi': 300,
            'figure_format': ["tif", "svg", "png"],
            'color_palette': "Blues",
            'font_size': 12,
            'title_font_weight': "bold"
        },
        
        # Analysis settings
        'analysis_settings': {
            'upset_plot': {
                'sort_by': "cardinality",
                'sort_categories_by': "-input",
                'facecolor': "#08306b",
                'max_degree_color': "#e3eef9"
            },
            'correlation': {
                'method': "cramers_v",
                'eta_squared': True
            }
        }
    }
    return config


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories"""
    fig_dir = Path(config['fig_dir'])
    numerical_data_dir = Path(config['numerical_data_dir'])
    fig_dir.mkdir(exist_ok=True, parents=True)
    numerical_data_dir.mkdir(exist_ok=True, parents=True)
    
    # Create logs directory if it doesn't exist
    Path("logs/data_processing").mkdir(exist_ok=True, parents=True)


def load_raw_data(config: Dict[str, Any]) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Load raw SSD and TSD data and create MOF name mapping"""
    ssd_raw_csv = config['ssd_raw_csv']
    tsd_raw_csv = config['tsd_raw_csv']
    
    mof_name_map = {}
    
    df_raw_ssd = pd.read_csv(ssd_raw_csv)
    df_raw_tsd = pd.read_csv(tsd_raw_csv)
    df_raw_ssd.dropna(inplace=True)
    df_raw_tsd.dropna(inplace=True)
    
    for df in [df_raw_ssd, df_raw_tsd]:
        for i, row in df.iterrows():
            mof_name = row["CoRE_name"]
            refcode = row["refcode"]
            if mof_name not in mof_name_map:
                mof_name_map[mof_name] = refcode
            else:
                assert mof_name_map[mof_name] == refcode, f"{mof_name} has multiple refcodes: {refcode} and {mof_name_map[mof_name]}"
    
    logging.info(f"Total {len(df_raw_ssd) + len(df_raw_tsd)} samples")
    logging.info(f"Total {len(mof_name_map)} MOFs")
    
    return mof_name_map, df_raw_ssd, df_raw_tsd


def load_task_data(config: Dict[str, Any], mof_name_map: Dict[str, str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data for all tasks"""
    data_root_dir = Path(config['data_root_dir'])
    tasks = config['tasks']
    
    class_map_2 = {0: "unstable", 1: "stable"}
    class_map_4 = {0: "U", 1: "LK", 2: "HK", 3: "TS"}
    
    dfs = {}
    for task in tasks:
        data_dir = data_root_dir / task
        split_dfs = {}
        for split in ["train", "val", "test"]:
            dataset = LoadGraphData(data_dir, split, csv_file_name="id_prop_feat.csv", down_sampling=True)
            split_dfs[split] = dataset.id_prop_df[["Partition"] + dataset.prop_cols].copy()
            split_dfs[split].rename(columns={dataset.prop_cols[0]: "Label"}, inplace=True)
            
            if task in ["TSD", "SSD"]:
                split_dfs[split].index = pd.Series(split_dfs[split].index).apply(lambda x: mof_name_map[x])
            
            if len(split_dfs[split]["Label"].unique()) == 2:
                split_dfs[split]["Label"] = split_dfs[split]["Label"].apply(lambda x: class_map_2[int(x)])
            elif len(split_dfs[split]["Label"].unique()) == 4:
                split_dfs[split]["Label"] = split_dfs[split]["Label"].apply(lambda x: class_map_4[int(x)])
        
        dfs.update({task: split_dfs})
        dfs[task]["total"] = pd.concat(split_dfs.values())
    
    return dfs


def generate_confusion_matrix(true_labels: pd.Series, pred_labels: pd.Series, 
                            true_classes: np.ndarray, pred_classes: List[str]) -> pd.DataFrame:
    """Generate confusion matrix for categorical data"""
    conf_matrix = pd.DataFrame(0, index=true_classes.tolist(), columns=pred_classes)
    for true_label, pred_label in zip(true_labels.tolist(), pred_labels.tolist()):
        conf_matrix.loc[true_label, pred_label] += 1
    return conf_matrix


def plot_dataset_distribution(dfs: Dict[str, Dict[str, pd.DataFrame]], 
                            config: Dict[str, Any]) -> None:
    """Plot dataset distribution visualization"""
    tasks = config['tasks']
    fig_dir = Path(config['fig_dir'])
    numerical_data_dir = Path(config['numerical_data_dir'])
    
    class_map_2 = {0: "unstable", 1: "stable"}
    class_map_4 = {0: "U", 1: "LK", 2: "HK", 3: "TS"}
    
    # Initialize figure
    fig = plt.figure(figsize=(16, 8))
    nrows = 2
    ncols = 4
    gs = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
    
    excel_writer = pd.ExcelWriter(numerical_data_dir / 'Figure2.xlsx')
    
    # Plot regression data swarmplot for TSD
    ax_reg = plt.subplot(gs[:, 0])
    sns.swarmplot(data=dfs["TSD"]["total"], x='Partition', y='Label', ax=ax_reg, 
                  hue='Label', palette='Blues', legend=False)
    ax_reg.set_title('TSD', fontweight='bold')
    ax_reg.set_xlabel('Split')
    ax_reg.set_ylabel('Decomposition Temperature (℃)')
    ax_reg.xaxis.set_ticks_position('top')
    ax_reg.grid(False)
    
    # Add sample count annotations
    y_min, y_max = ax_reg.get_ylim()
    dfs["TSD"]["total"].to_excel(excel_writer, sheet_name='TSD')
    grouped_data = dfs["TSD"]["total"].groupby('Partition').size()
    for i in range(len(grouped_data)):
        count = grouped_data[dfs["TSD"]["total"]['Partition'].unique()[i]]
        ax_reg.text(i, y_min + (y_max - y_min)*0.05, f'N={count}', ha='center', va='bottom')
    
    # Plot classification data confusion matrices
    for i, task in enumerate(tasks[1:]):
        row = i // (ncols - 1)
        col = i % (ncols - 1) + 1
        sub_df = dfs[task]["total"]
        ax = plt.subplot(gs[row, col])
        
        if len(sub_df["Label"].unique()) == 4:
            labels = list(class_map_4.values())
        else:
            labels = list(class_map_2.values())
        
        conf_matrix_df = generate_confusion_matrix(sub_df['Partition'], sub_df['Label'],  
                                                   sub_df['Partition'].unique(), labels)
        conf_matrix_df["sum"] = conf_matrix_df.sum(axis=1)
        conf_matrix_df = pd.concat([conf_matrix_df, pd.DataFrame([conf_matrix_df.sum(axis=0)], index=['sum'])], axis=0)
        conf_matrix_df.to_excel(excel_writer, sheet_name=task)
        
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(task, fontweight='bold')
        ax.set_xlabel('Label')
        ax.set_ylabel('Split')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_label_position('right')
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / 'dataset_distribution.tif', dpi=300)
    plt.savefig(fig_dir / f"dataset_distribution.svg", dpi=300, transparent=True)
    plt.show()
    
    excel_writer.close()
    logging.info("Dataset distribution plot saved")


def calculate_cramers_v(var1: pd.Series, var2: pd.Series) -> float:
    """
    Calculate Cramér's V for categorical variables
    
    Parameters:
    var1: values of categorical variable
    var2: values of categorical variable
    
    Returns:
    Cramér's V value
    """
    contingency_table = pd.crosstab(var1, var2)
    chi2_stat, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2_stat / n
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(phi2 / k)
    return float(cramers_v)


def calculate_eta_squared(continuous_var: pd.Series, categorical_var: pd.Series) -> float:
    """
    Calculate Eta squared for continuous vs categorical variables
    
    Parameters:
    continuous_var: values of continuous variable
    categorical_var: values of categorical variable
    
    Returns:
    Eta squared value
    """
    continuous_values = np.array(continuous_var)
    categorical_values = np.array(categorical_var)
    
    if len(continuous_values) != len(categorical_values):
        raise ValueError("The lengths of the two variables must be the same")

    # Standardize continuous variable
    scaler = StandardScaler()
    standardized_continuous_var = scaler.fit_transform(continuous_values.reshape(-1, 1)).flatten()

    # Encode categorical variable
    encoder = LabelEncoder()
    encoded_cat_var = encoder.fit_transform(categorical_values)

    # Calculate total sum of squares
    mean_overall = np.mean(standardized_continuous_var)
    ss_total = np.sum((standardized_continuous_var - mean_overall) ** 2)

    # Calculate between-group sum of squares
    categories = np.unique(encoded_cat_var)
    ss_between = 0.0
    for cat in categories:
        cat_mask = encoded_cat_var == cat
        cat_data = standardized_continuous_var[cat_mask]
        if len(cat_data) > 0:
            cat_mean = np.mean(cat_data)
            ss_between += len(cat_data) * ((cat_mean - mean_overall) ** 2)
    
    # Calculate Eta squared
    eta_squared = ss_between / ss_total
    return float(eta_squared)


def calculate_dataset_correlations(dfs: Dict[str, Dict[str, pd.DataFrame]], 
                                 config: Dict[str, Any]) -> pd.DataFrame:
    """Calculate correlations between datasets"""
    tasks = config['tasks']
    task_types = config['task_types']
    
    corr = []
    for task_i, task_tp_i in zip(tasks, task_types):
        corr_row = []
        for task_j, task_tp_j in zip(tasks, task_types):
            if task_i == task_j:
                corr_row.append(1)
                continue
                
            df_i = dfs[task_i]["total"].reset_index()
            df_j = dfs[task_j]["total"].reset_index()
            
            # Drop Partition column if it exists
            if "Partition" in df_i.columns:
                df_i = df_i.drop(columns=["Partition"])
            if "Partition" in df_j.columns:
                df_j = df_j.drop(columns=["Partition"])
            
            # Use the index column as merge key
            merge_key = df_i.columns[0]  # First column should be the index (MOF ID)
            
            df_i = df_i.rename(columns={"Label": f"{task_i}Label"}).copy()
            df_j = df_j.rename(columns={"Label": f"{task_j}Label"}).copy()
            intersection_df = df_i.merge(df_j, on=merge_key, how="inner")
            
            logging.info(f"{task_i} vs {task_j}: {len(intersection_df)}")
            
            if len(intersection_df) == 0:
                corr_row.append(0)
                continue
                
            all_task_tps = [task_tp_i, task_tp_j]
            
            if "regression" in all_task_tps:
                all_labels = [intersection_df[f"{task_i}Label"], intersection_df[f"{task_j}Label"]]
                reg_labels = all_labels[all_task_tps.index("regression")]
                clf_labels = all_labels[all_task_tps.index("classification")]
                corr_coef = calculate_eta_squared(reg_labels, clf_labels)
            else:
                corr_coef = calculate_cramers_v(intersection_df[f"{task_i}Label"], intersection_df[f"{task_j}Label"])
            
            corr_row.append(corr_coef)
        corr.append(corr_row)
    
    corr_df = pd.DataFrame(corr, index=tasks, columns=tasks)
    return corr_df


def plot_correlation_heatmap(corr_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='Blues', ax=ax, cbar=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    fig_dir = Path(config['fig_dir'])
    plt.savefig(fig_dir / 'dataset_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info("Correlation heatmap saved")


def create_upset_plot(dfs: Dict[str, Dict[str, pd.DataFrame]], 
                     corr_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Create UpSet plot for dataset intersections"""
    tasks = config['tasks']
    fig_dir = Path(config['fig_dir'])
    numerical_data_dir = Path(config['numerical_data_dir'])
    
    all_cif_ids = {task: dfs[task]["total"].index for task in tasks}
    all_samples = set().union(*all_cif_ids.values())
    
    data = []
    for sample in all_samples:
        row = []
        for task in tasks:
            if sample in all_cif_ids[task]:
                row.append(True)
            else:
                row.append(False)
        
        if sum(row) >= 2:
            # Get indices of True values
            true_indices = [i for i, value in enumerate(row) if value]
            
            # Generate combinations
            one_true_combinations = list(combinations(true_indices, 1))
            two_true_combinations = list(combinations(true_indices, 2))
            
            for combo in two_true_combinations:
                new_list = [False] * len(row)
                for index in combo:
                    new_list[index] = True
                data.append(new_list + [sum(new_list)])
                
            for combo in one_true_combinations:
                new_list = [False] * len(row)
                for index in combo:
                    new_list[index] = True
                data.append(new_list + [sum(new_list)])
            continue
            
        data.append(row + [sum(row)])
    
    upset_data = pd.DataFrame(data, columns=tasks + ['count'])
    upset_data.set_index(tasks, inplace=True)
    
    # Create UpSet Plot
    upset = UpSet(upset_data, subset_size='count', sort_by='cardinality', sort_categories_by="-input",
                  facecolor="#08306b", show_counts='%d', totals_plot_elements=0)
    upset.style_subsets(max_degree=1, facecolor="#e3eef9")
    plot_dict = upset.plot()
    intersections_ax = plot_dict['intersections']
    intersections_ax.set_ylabel(intersections_ax.get_ylabel(), fontsize=14, fontweight='bold')
    
    # Add correlation heatmap as inset
    ax_inset = inset_axes(intersections_ax, width="50%", height="60%", loc='upper right', 
                          bbox_to_anchor=(0, 0., 0.95, 0.9), bbox_transform=intersections_ax.transAxes, borderpad=0)
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='Blues', ax=ax_inset, cbar=False)
    
    ax_inset.set_xticklabels(ax_inset.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax_inset.set_title('Dataset Correlations', fontsize=14, fontweight='bold')
    
    # Add colorbar for the inset plot
    inset_colorbar = inset_axes(ax_inset, width="5%", height="100%", loc='right', 
                                bbox_to_anchor=(0.55, 0, 0.5, 1), bbox_transform=ax_inset.transAxes, borderpad=0.1)
    plt.colorbar(ax_inset.collections[0], cax=inset_colorbar)
    
    fig = intersections_ax.figure
    fig.set_size_inches(16, 10)
    fig.figure.set_dpi(300)
    fig.savefig(fig_dir / 'dataset_intersections.tif', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'dataset_intersections.svg', dpi=300, transparent=True)
    plt.show()
    
    # Save numerical data
    excel_writer = pd.ExcelWriter(numerical_data_dir / 'Figure3.xlsx')
    upset_data.to_excel(excel_writer, sheet_name='Upset Plot')
    corr_df.to_excel(excel_writer, sheet_name='Dataset Correlation')
    excel_writer.close()
    
    logging.info("UpSet plot and correlation data saved")


def main():
    """Main function to run the complete dataset analysis"""
    parser = argparse.ArgumentParser(description="Dataset Analysis for MOF Stability Prediction")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    
    # Create directories
    create_directories(config)
    
    logging.info("Starting dataset analysis...")
    
    # Load raw data and create MOF name mapping
    mof_name_map, df_raw_ssd, df_raw_tsd = load_raw_data(config)
    
    # Load task data
    dfs = load_task_data(config, mof_name_map)
    
    # Get all unique MOFs
    all_mofs = []
    for task in config['tasks']:
        df = dfs[task]["total"]
        all_mofs.extend(df.index.tolist())
    all_mofs = list(set(all_mofs))
    logging.info(f"Total unique MOFs across all tasks: {len(all_mofs)}")
    
    # Plot dataset distribution
    plot_dataset_distribution(dfs, config)
    
    # Calculate correlations
    corr_df = calculate_dataset_correlations(dfs, config)
    logging.info("Dataset correlations calculated")
    logging.info(f"Correlation matrix:\n{corr_df}")
    
    # Plot correlation heatmap
    plot_correlation_heatmap(corr_df, config)
    
    # Create UpSet plot
    create_upset_plot(dfs, corr_df, config)
    
    logging.info("Dataset analysis completed successfully!")


if __name__ == "__main__":
    main()
