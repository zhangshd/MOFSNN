#!/usr/bin/env python
"""
Author: zhangshd
Date: 2025-05-30
Description: Model uncertainty analysis script.
This script performs uncertainty analysis on CGCNN models using LSE and LSV metrics.
It loads model results, uncertainty trees, and generates comprehensive uncertainty
analysis plots including cutoff analysis for different tasks.

Before running this script, ensure you have the necessary directories and files set up:
- `log_dir`: Directory containing model evaluation results (CSV and NPZ files), 
    which can be prepared using the `src/cgcnn/predict.py` script.
- `uncertainty_trees_file`: Path to the uncertainty trees pickle file,
    which can be generated using the `src/experiment/uncertainty_cutoff_analysis.py` script.
- `output_dir`: Directory where analysis outputs will be saved.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pickle
import shutil
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional, Any
from matplotlib.axes import Axes

# Project path setup
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(SCRIPT_DIR.parent))  # Add src to path

from cgcnn.module.module_utils import calculate_lse_from_tree, calculate_lsv_from_tree


class UncertaintyAnalyzer:
    """
    Main class for performing uncertainty analysis on CGCNN models.
    """
    
    def __init__(self, log_dir: str, uncertainty_trees_file: str, output_dir: str):
        """
        Initialize the uncertainty analyzer.
        
        Args:
            log_dir: Directory containing model evaluation results
            uncertainty_trees_file: Path to uncertainty trees pickle file
            output_dir: Directory to save analysis outputs
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different outputs
        (self.output_dir / "figures").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "numerical_data").mkdir(exist_ok=True, parents=True)
        
        # Task definitions and mappings
        self.tasks = ["TSD", "SSD", "WS24_water", "WS24_water4", "WS24_acid", "WS24_base", "WS24_boiling"]
        self.targets_map = {
            "SSD": {0: "unstable", 1: "stable"},
            "WS24_water": {0: "unstable", 1: "stable"},
            "WS24_water4": {0: "U", 1: "LK", 2: "HK", 3: "TS"},
            "WS24_acid": {0: "unstable", 1: "stable"},
            "WS24_base": {0: "unstable", 1: "stable"},
            "WS24_boiling": {0: "unstable", 1: "stable"},
        }
        
        # Load data
        self.latent_feas = {}
        self.targets = {}
        self.predictions = {}
        self.probs = {}
        self.uncertainty_trees = {}
        
        self._load_model_data()
        self._load_uncertainty_trees(uncertainty_trees_file)
    
    def _load_model_data(self):
        """Load latent features, targets, predictions, and probabilities."""
        print("Loading model evaluation data...")
        
        for task in self.tasks:
            for split in ['val', 'test']:
                try:
                    # Load latent features
                    latent_fea_file = self.log_dir / f"{task}_{split}_latent_vectors.npz"
                    if latent_fea_file.exists():
                        latent_fea = np.load(latent_fea_file)
                        self.latent_feas[f"{task}_{split}"] = latent_fea[list(latent_fea.keys())[0]]
                    else:
                        print(f"Warning: Latent features file {latent_fea_file} does not exist for {task}_{split}.")
                        continue
                    
                    # Load results
                    results_file = self.log_dir / f"{split}_results_{task}.csv"
                    if results_file.exists():
                        res_df = pd.read_csv(results_file)
                        
                        # Copy test files to output directory for reference
                        if split == "test":
                            # shutil.copy(latent_fea_file, self.output_dir / "numerical_data" / f"{split}_last_layer_fea_{task}.npz")
                            shutil.copy(results_file, self.output_dir / "numerical_data" / f"{split}_results_{task}.csv")
                        
                        self.targets[f"{task}_{split}"] = res_df['GroundTruth'].values
                        self.predictions[f"{task}_{split}"] = res_df['Predicted'].values
                        
                        # Load probabilities if available
                        if "Prob" in res_df.columns:
                            self.probs[f"{task}_{split}"] = np.array([
                                eval(p) if isinstance(p, str) else p for p in res_df['Prob'].values
                            ])
                            print(f"Loaded probabilities for {task}_{split}:\n{self.probs[f'{task}_{split}'][:5]} ...")

                    else:
                        print(f"Warning: Results file {results_file} does not exist for {task}_{split}.")
                        continue
                        
                except Exception as e:
                    print(f"Warning: Could not load data for {task}_{split}: {e}")
    
    def _load_uncertainty_trees(self, uncertainty_trees_file: str):
        """Load uncertainty trees from pickle file."""
        print(f"Loading uncertainty trees from {uncertainty_trees_file}...")
        
        with open(uncertainty_trees_file, "rb") as f:
            self.uncertainty_trees = pickle.load(f)
        
        print(f"Loaded uncertainty trees for tasks: {list(self.uncertainty_trees.keys())}")
    
    def lse_analysis(self, task: str, k: int = 10, **kwargs) -> Tuple[Axes, pd.DataFrame]:
        """
        Perform Local Similarity Entropy (LSE) analysis for classification tasks.
        
        Args:
            task: Task name
            k: Number of nearest neighbors
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of matplotlib axis and summary dataframe
        """
        print(f"Performing LSE analysis for task: {task}")
        
        # Calculate LSE for test set
        scaled_avg_knn_dist = calculate_lse_from_tree(
            self.uncertainty_trees[task],
            self.latent_feas[f"{task}_test"], 
            k=k,
            scale=False
        )
        
        # Extract plotting parameters
        alpha = kwargs.get('alpha', 0.8)
        xmax = kwargs.get('xmax', None)
        ax = kwargs.get('ax', None)
        frac_cutoff = kwargs.get('frac_cutoff', 0.8)
        x_label = kwargs.get('x_label', True)
        y_label_left = kwargs.get('y_label_left', True)
        y_label_right = kwargs.get('y_label_right', True)
        legend = kwargs.get('legend', False)
        tick_font_size = kwargs.get('tick_font_size', 12)
        label_font_size = kwargs.get('label_font_size', 12)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Initialize result lists
        cutoffs = []
        performances_in = []
        performances_out = []
        aucs_in = []
        fracs = []
        
        # Analyze performance at different cutoff values
        for i in range(1, 101, 1):
            cutoff = 0.01 * i
            
            # Split data by cutoff
            in_mask = scaled_avg_knn_dist <= cutoff
            out_mask = scaled_avg_knn_dist > cutoff
            
            sub_targets = self.targets[f"{task}_test"][in_mask]
            sub_preds = self.predictions[f"{task}_test"][in_mask]
            sub_targets_out = self.targets[f"{task}_test"][out_mask]
            sub_preds_out = self.predictions[f"{task}_test"][out_mask]
            sub_probs_in = self.probs[f"{task}_test"][in_mask]
            
            cutoffs.append(cutoff)
            performances_in.append(metrics.accuracy_score(sub_targets, sub_preds))
            performances_out.append(metrics.accuracy_score(sub_targets_out, sub_preds_out))
            
            # Calculate AUC if possible
            try:
                if len(np.unique(sub_targets)) > 2:  # Multi-class case
                    aucs_in.append(metrics.roc_auc_score(sub_targets, sub_probs_in, multi_class="ovr"))
                elif  len(np.unique(sub_targets)) == 2:  # Binary case
                    aucs_in.append(metrics.roc_auc_score(sub_targets, sub_probs_in[:, 1]))
                else:
                    aucs_in.append(metrics.roc_auc_score(sub_targets, sub_probs_in))
            except ValueError:
                aucs_in.append(0.5)
            
            frac = len(sub_targets) / len(self.targets[f"{task}_test"])
            fracs.append(frac)
            
            if frac > frac_cutoff:
                print(f"LSE cutoff={cutoff:.2f}, Accuracy={metrics.accuracy_score(sub_targets, sub_preds):.3f}")
                break
        
        # Plot results
        metric_full = metrics.accuracy_score(self.targets[f"{task}_test"], self.predictions[f"{task}_test"])
        
        sns.scatterplot(x=cutoffs, y=performances_in, ax=ax, marker='D', 
                       label='ACC of Data inside Cutoff', alpha=alpha)
        sns.scatterplot(x=cutoffs, y=performances_out, ax=ax, marker='^', 
                       label='ACC of Data outside Cutoff', alpha=alpha)
        
        if x_label:
            ax.set_xlabel('Uncertainty Cutoff')
        if y_label_left:
            ax.set_ylabel('Accuracy')
        ax.set_ylim(-0.01, 1.05)
        
        # Add secondary y-axis for fraction
        ax2 = ax.twinx()
        sns.scatterplot(x=cutoffs, y=fracs, ax=ax2, color='g', marker='o', 
                       label='Retained Data Fraction', alpha=alpha)
        
        if y_label_right:
            ax2.set_ylabel('Fraction')
        ax2.set_ylim(-0.01, 1.05)
        
        if xmax is None:
            xmax = cutoffs[-1]
        ax.set_xlim(0, xmax)
        
        # Add reference line
        ax.hlines(y=metric_full, xmin=0, xmax=xmax, colors='blue', linestyles='dashed', 
                  label=f'MAE/ACC of full test set', alpha=alpha)
        
        # Add text annotations
        ax2.text(xmax*0.65, 0.3, "\n".join([
            f'ACC={performances_in[-1]:.2f}',
            f'AUROC={aucs_in[-1]:.2f}',
            'when:',
            'LSE$_{cutoff}$=' + f'{cutoffs[-1]:.2f}',
        ]), fontsize=tick_font_size, ha='left', va='center', color='r')
        
        ax2.text(xmax*0.78, metric_full*0.96, f"ACC={metric_full:.2f}", 
                fontsize=tick_font_size, ha='left', va='center', color='b')
        
        # Handle legends
        if legend:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2, loc='lower right')
            ax2.legend().set_visible(False)
        else:
            ax.legend().set_visible(False)
            ax2.legend().set_visible(False)
        
        ax.set_title(f'Accuracy & Fraction vs LSE Cutoff ({task})', 
                    weight='bold', fontsize=label_font_size)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        ax.xaxis.label.set_size(label_font_size)
        ax.yaxis.label.set_size(label_font_size)
        ax2.tick_params(axis='both', labelsize=tick_font_size)
        ax2.yaxis.label.set_size(label_font_size)
        
        # Create summary dataframe
        df_sum = pd.DataFrame({
            'LSE Cutoff': cutoffs,
            'Accuracy inside Cutoff': performances_in,
            'Accuracy outside Cutoff': performances_out,
            'Retained Data Fraction': fracs,
            'AUROC inside Cutoff': aucs_in
        })
        
        return ax, df_sum
    
    def lsv_analysis(self, task: str, k: int = 10, **kwargs) -> Tuple[Axes, pd.DataFrame]:
        """
        Perform Local Similarity Variance (LSV) analysis for regression tasks.
        
        Args:
            task: Task name
            k: Number of nearest neighbors
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of matplotlib axis and summary dataframe
        """
        print(f"Performing LSV analysis for task: {task}")
        print(self.latent_feas.keys())
        # Calculate LSV for test set
        avg_distances = calculate_lsv_from_tree(
            self.uncertainty_trees[task], 
            self.latent_feas[f"{task}_test"], 
            k=k,
            scale=False
        )
        
        # Extract plotting parameters
        alpha = kwargs.get('alpha', 0.8)
        xmax = kwargs.get('xmax', None)
        ax = kwargs.get('ax', None)
        frac_cutoff = kwargs.get('frac_cutoff', 0.8)
        x_label = kwargs.get('x_label', True)
        y_label_left = kwargs.get('y_label_left', True)
        y_label_right = kwargs.get('y_label_right', True)
        legend = kwargs.get('legend', False)
        tick_font_size = kwargs.get('tick_font_size', 12)
        label_font_size = kwargs.get('label_font_size', 12)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Scale the distances
        scaler = MinMaxScaler()
        scaled_avg_knn_dist = scaler.fit_transform(avg_distances.reshape(-1, 1)).reshape(-1)
        
        # Initialize result lists
        cutoffs = []
        performances_in = []
        performances_out = []
        fracs = []
        r2_scores_in = []
        
        # Analyze performance at different cutoff values
        for i in range(1, 96, 1):
            cutoff = 0.001 * i
            
            # Split data by cutoff
            in_mask = scaled_avg_knn_dist <= cutoff
            out_mask = scaled_avg_knn_dist > cutoff
            
            sub_targets = self.targets[f"{task}_test"][in_mask]
            sub_preds = self.predictions[f"{task}_test"][in_mask]
            sub_targets_out = self.targets[f"{task}_test"][out_mask]
            sub_preds_out = self.predictions[f"{task}_test"][out_mask]
            
            cutoffs.append(cutoff)
            performances_in.append(metrics.mean_absolute_error(sub_targets, sub_preds))
            performances_out.append(metrics.mean_absolute_error(sub_targets_out, sub_preds_out))
            r2_scores_in.append(metrics.r2_score(sub_targets, sub_preds))
            
            frac = len(sub_targets) / len(self.targets[f"{task}_test"])
            fracs.append(frac)
            
            if frac > frac_cutoff:
                print(f"LSV cutoff={cutoff:.2f}, MAE={metrics.mean_absolute_error(sub_targets, sub_preds):.1f}, "
                      f"R2={metrics.r2_score(sub_targets, sub_preds):.2f}")
                break
        
        # Plot results
        metric_full = metrics.mean_absolute_error(self.targets[f"{task}_test"], self.predictions[f"{task}_test"])
        
        sns.scatterplot(x=cutoffs, y=performances_in, ax=ax, marker='D', 
                       label='MAE/ACC of Data inside Cutoff', alpha=alpha)
        sns.scatterplot(x=cutoffs, y=performances_out, ax=ax, marker='^', 
                       label='MAE/ACC of Data outside Cutoff', alpha=alpha)
        
        if x_label:
            ax.set_xlabel('Uncertainty Cutoff')
        if y_label_left:
            ax.set_ylabel('Mean Absolute Error (℃)')
        ax.set_ylim(10, 60)
        
        # Add secondary y-axis for fraction
        ax2 = ax.twinx()
        sns.scatterplot(x=cutoffs, y=fracs, ax=ax2, color='g', marker='o', 
                       label='Retained Data Fraction', alpha=alpha)
        
        if y_label_right:
            ax2.set_ylabel('Fraction')
        ax2.set_ylim(-0.01, 1.05)
        
        if xmax is None:
            xmax = cutoffs[-1]
        ax.set_xlim(0, xmax)
        
        # Add reference line
        ax.hlines(y=metric_full, xmin=0, xmax=xmax, colors='blue', linestyles='dashed',
                  label=f'MAE/ACC of full test set', alpha=alpha)
        
        # Add text annotations
        ax2.text(xmax*0.65, 0.3, "\n".join([
            f'MAE={performances_in[-1]:.1f}',
            f'R2={r2_scores_in[-1]:.2f}',
            'when:',
            'LSV$_{cutoff}$=' + f'{cutoffs[-1]:.2f}',
        ]), fontsize=tick_font_size, ha='left', va='center', color='r')
        
        ax.text(xmax*0.05, metric_full*0.95, f"MAE={metric_full:.1f}", 
               fontsize=tick_font_size, ha='left', va='center', color='b')
        
        # Handle legends
        if legend:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
            ax2.legend().set_visible(False)
        else:
            ax.legend().set_visible(False)
            ax2.legend().set_visible(False)
        
        ax.set_title(f'MAE & Fraction vs LSV Cutoff ({task})', 
                    weight='bold', fontsize=label_font_size)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        ax.xaxis.label.set_size(label_font_size)
        ax.yaxis.label.set_size(label_font_size)
        ax2.tick_params(axis='both', labelsize=tick_font_size)
        ax2.yaxis.label.set_size(label_font_size)
        
        # Create summary dataframe
        df_sum = pd.DataFrame({
            'LSV Cutoff': cutoffs,
            'MAE/ACC inside Cutoff': performances_in,
            'MAE/ACC outside Cutoff': performances_out,
            'Retained Data Fraction': fracs,
            'R2 inside Cutoff': r2_scores_in
        })
        
        return ax, df_sum
    
    def run_combined_analysis(self, k: int = 5, figsize: Tuple[int, int] = (20, 10)):
        """
        Run combined uncertainty analysis for all tasks and create comprehensive plots.
        
        Args:
            k: Number of nearest neighbors for uncertainty calculation
            figsize: Figure size for the combined plot
        """
        print("Running combined uncertainty analysis...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 4, figure=fig)
        
        ncols = 4
        nrows = 2
        
        # Create Excel writer for numerical data
        excel_file = self.output_dir / "numerical_data" / "uncertainty_analysis_results.xlsx"
        excel_writer = pd.ExcelWriter(excel_file)
        
        # TSD (regression) analysis - takes first column
        tsd_ax = gs[0, 0]
        ax, df_sum = self.lsv_analysis(
            'TSD', k=k, ax=fig.add_subplot(tsd_ax), 
            frac_cutoff=0.8, x_label=False, y_label_left=True, 
            y_label_right=False, legend=True
        )
        df_sum.to_excel(excel_writer, sheet_name="TSD", index=False)
        
        # Classification tasks analysis - remaining positions
        clf_tasks = ["SSD", "WS24_water", "WS24_water4", "WS24_acid", "WS24_base", "WS24_boiling"]
        for i, task in enumerate(clf_tasks):
            n = i + 1
            row = n // ncols
            col = n % ncols
            lse_ax = gs[row, col]
            
            ax, df_sum = self.lse_analysis(
                task, k=k, ax=fig.add_subplot(lse_ax), 
                frac_cutoff=0.8,
                x_label=(row == 1) | (col == 3 and row == 0),
                y_label_left=(col == 0 and row == 1) | (col == 1 and row == 0),
                y_label_right=(col == 2 and row == 1) | (col == 3 and row == 0),
                legend=False
            )
            df_sum.to_excel(excel_writer, sheet_name=task, index=False)
        
        # Save the combined plot
        plt.tight_layout()
        fig_file = self.output_dir / "figures" / "uncertainty_cutoff_analysis.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        
        svg_file = self.output_dir / "figures" / "uncertainty_cutoff_analysis.svg"
        plt.savefig(svg_file, dpi=200, transparent=True, bbox_inches='tight')
        
        plt.show()
        excel_writer.close()
        
        print(f"Saved combined analysis plot to: {fig_file}")
        print(f"Saved numerical results to: {excel_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of the uncertainty analysis."""
        print("Generating summary report...")
        
        report_file = self.output_dir / "uncertainty_analysis_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("Uncertainty Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log Directory: {self.log_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("Tasks Analyzed:\n")
            for task in self.tasks:
                if f"{task}_test" in self.targets:
                    n_samples = len(self.targets[f"{task}_test"])
                    f.write(f"  - {task}: {n_samples} test samples\n")
            
            f.write(f"\nUncertainty Trees Available: {list(self.uncertainty_trees.keys())}\n")
            
            f.write("\nOutput Files Generated:\n")
            f.write("  - uncertainty_cutoff_analysis.png/svg: Combined uncertainty analysis plots\n")
            f.write("  - uncertainty_analysis_results.xlsx: Numerical results for all tasks\n")
            f.write("  - uncertainty_analysis_summary.txt: This summary report\n")
            
            f.write("\nAnalysis Methods:\n")
            f.write("  - LSE (Local Similarity Entropy): Used for classification tasks\n")
            f.write("  - LSV (Local Similarity Variance): Used for regression tasks (TSD)\n")
            f.write("  - Cutoff Analysis: Performance vs uncertainty threshold analysis\n")
        
        print(f"Summary report saved to: {report_file}")


def main():
    """Main function to run uncertainty analysis."""
    parser = ArgumentParser(description="Perform uncertainty analysis on CGCNN model results")
    
    parser.add_argument("--log_dir", type=str, default=ROOT_DIR / "results/evaluation/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn@version_43",
                       help="Directory containing model evaluation results")
    parser.add_argument("--uncertainty_trees_file", type=str, default=ROOT_DIR / "results/cgcnn_models/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_43/epoch_108/uncertainty_trees.pkl",
                       help="Path to uncertainty trees pickle file")
    parser.add_argument("--output_dir", type=str, default=ROOT_DIR / "results/uncertainty_analysis",
                       help="Directory to save analysis outputs")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of nearest neighbors for uncertainty calculation")
    parser.add_argument("--figsize", type=int, nargs=2, default=[20, 10],
                       help="Figure size as width height")
    
    args = parser.parse_args()
    
    print("Starting uncertainty analysis...")
    print(f"Log directory: {args.log_dir}")
    print(f"Uncertainty trees file: {args.uncertainty_trees_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"K neighbors: {args.k}")
    
    # Initialize analyzer
    analyzer = UncertaintyAnalyzer(
        log_dir=args.log_dir,
        uncertainty_trees_file=args.uncertainty_trees_file,
        output_dir=args.output_dir
    )
    
    # Run combined analysis
    analyzer.run_combined_analysis(k=args.k, figsize=tuple(args.figsize))
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("Uncertainty analysis completed successfully!")


if __name__ == "__main__":
    main()
