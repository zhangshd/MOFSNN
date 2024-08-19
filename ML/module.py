'''
Author: zhangshd
Date: 2024-08-15 15:51:31
LastEditors: zhangshd
LastEditTime: 2024-08-17 20:02:53
'''
import os
import time
import copy
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
from matplotlib import colors
plt.switch_backend('agg')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, LeaveOneOut, LeaveOneGroupOut, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, matthews_corrcoef, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import BallTree
import copy


def sec_to_time(seconds):
    """Convert seconds to time format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


def weighted_cross_entropy(y_true, y_pred_proba, n_class):
    """Compute the weighted cross-entropy loss for multi-class classification."""
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    class_counts = np.bincount(y_true, minlength=n_class)
    total_samples = len(y_true)
    class_weights = total_samples / (n_class * class_counts)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    # print(y_true)
    y_true_one_hot = np.eye(n_class)[y_true]
    weighted_ce = - (y_true_one_hot * np.log(y_pred_proba)) * class_weights
    return weighted_ce.sum(axis=1).mean()


def split_train_test(df, test_size=0.1, group_column="", random_state=0):
    """Split the dataset into train and test sets."""
    if group_column:
        train_index, test_index = train_test_split(df.index, test_size=test_size, stratify=df[group_column], random_state=random_state)
    else:
        train_index, test_index = train_test_split(df.index, test_size=test_size, random_state=random_state)
    print(f'Train test split successfully: train/test = {len(train_index)}/{len(test_index)}')
    return train_index, test_index


def plot_scatter(targets, predictions, title: str=None, metrics: dict=None, outfile: str=None):

    targets = np.array(targets)
    predictions = np.array(predictions)
    max_value = max(targets.max(), predictions.max())
    min_value = min(targets.min(), predictions.min())
    offset = (max_value-min_value)*0.06
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.5)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f"Groud Truth")
    ax.set_ylabel(f"Predictions")

    ax.set_xlim(min_value - offset, max_value + offset)
    ax.set_ylim(min_value - offset, max_value + offset)

    ax.plot([min_value, max_value], [min_value, max_value], 'r--')  # 'r--'表示红色虚线

    if metrics:
        text_content = ""
        for k, v in metrics.items():
            text_content += f"{k}: {v:.4f}\n"
        ax.text(max_value - offset*6, min_value + offset, 
            text_content, 
            fontsize=12, color='red')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax

def plot_confusion_matrix(targets, predictions, title=None, outfile=None):
    
    cm = confusion_matrix(targets, predictions)
    if title is None:
        title = f"Confusion Matrix"
    num_classes = len(cm)
    acc = (cm.diagonal().sum()/cm.sum())*100
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel('Groud Truth')
    ax.set_ylabel('Predictions')
    ax.set_title(title+f'(ACC={acc:.2f}%)')
    ax.set_aspect('equal')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight', format='png')
    return fig, ax


class BaseModel:
    """Base class for Regression and Classification models."""

    def __init__(self, random_state=0):
        self.random_state = random_state
        self.color_list = list(colors.XKCD_COLORS.values())
        self.n_jobs = min(int(0.8 * os.cpu_count()), 64)
        self.feature_selector_name = "null"
        self.feature_select_num = 0
        self.model_type = None
        self.full_trained = False

    def load_data(self, train_X, train_y, test_X=None, test_y=None, valid_X=None, valid_y=None, train_groups=None):
        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)
        if test_X is not None and test_y is not None:
            self.test_X = np.array(test_X)
            self.test_y = np.array(test_y)
        else:
            self.test_X = None
            self.test_y = None
        if valid_X is not None and valid_y is not None:
            self.valid_X = np.array(valid_X)
            self.valid_y = np.array(valid_y)
        else:
            self.valid_X = None
            self.valid_y = None
        self.train_groups = np.array(train_groups)
        print("=" * 50)
        print(f"Train_X shape: {self.train_X.shape}")
        print(f"Train_y shape: {self.train_y.shape}")
        if self.test_X is not None:
            print(f"Test_X shape: {self.test_X.shape}")
            print(f"Test_y shape: {self.test_y.shape}")
        if self.valid_X is not None:
            print(f"Valid_X shape: {self.valid_X.shape}")
            print(f"Valid_y shape: {self.valid_y.shape}")
        print("=" * 50)

    def scale_feature(self, feature_range=(0, 1), saved_dir="", saved_file_note="", scaler_name="MinMaxScaler"):
        print(f"Scaled feature range: {feature_range}")
        self.scaler = MinMaxScaler(feature_range=feature_range) if scaler_name == "MinMaxScaler" else StandardScaler()
        self.train_X_init = self.train_X.copy()
        self.train_X = self.scaler.fit_transform(self.train_X)
        if self.test_X is not None and self.test_y is not None:
            self.test_X_init = self.test_X
            self.test_X = self.scaler.transform(self.test_X)
        if self.valid_X is not None and self.valid_y is not None:
            self.valid_X_init = self.valid_X
            self.valid_X = self.scaler.transform(self.valid_X)
        if saved_dir:
            scaler_x_file = os.path.join(saved_dir, f"scaler_{saved_file_note}.pkl")
            with open(scaler_x_file, 'wb') as f:
                joblib.dump(self.scaler, f)

    def select_feature(self, saved_dir="", feature_selector='f1', select_des_num=100, saved_file_note=""):
        t1 = time.time()
        self.variance_filter = VarianceThreshold(threshold=0)
        self.train_X_filtered = self.variance_filter.fit_transform(self.train_X)
        print(f"Features shape after variance filter: {self.train_X_filtered.shape}")
        print(f'Executing feature selection on features by {feature_selector}.')
        select_des_num = min(select_des_num, self.train_X_filtered.shape[1])
        if self.model_type == "regression":
            if feature_selector == 'RFE':
                base_model = RandomForestRegressor(n_estimators=20, random_state=self.random_state, n_jobs=self.n_jobs)
                self.selector = RFE(base_model, n_features_to_select=select_des_num, step=0.01)
            elif feature_selector in ['f1', 'f_regression']:
                self.selector = SelectKBest(score_func=f_regression, k=select_des_num)
            elif feature_selector in ['mutual_info','mutual_info_regression']:
                self.selector = SelectKBest(score_func=mutual_info_regression, k=select_des_num)
            else:
                raise NotImplementedError(f"Feature selector choice: REF/f1/mutual_info, {feature_selector} is not implemented.")
        elif self.model_type == "classification":
            if feature_selector == 'RFE':
                base_model = RandomForestClassifier(n_estimators=20, random_state=self.random_state, n_jobs=self.n_jobs)
                self.selector = RFE(base_model, n_features_to_select=select_des_num, step=0.01)
            elif feature_selector in ['f1', 'f_classif']:
                self.selector = SelectKBest(score_func=f_classif, k=select_des_num)
            elif feature_selector in ['mutual_info','mutual_info_classif']:
                self.selector = SelectKBest(score_func=mutual_info_classif, k=select_des_num)
            else:
                raise NotImplementedError(f"Feature selector choice: REF/f1/mutual_info, {feature_selector} is not implemented.")
        else:
            raise NotImplementedError(f"model_type choice: regression/classification, {self.model_type} is not implemented.")
        self.selector.fit(self.train_X_filtered, self.train_y)
        self.train_X_selected = self.selector.transform(self.train_X_filtered)
        if self.test_X is not None and self.test_y is not None:
            self.test_X_filtered = self.variance_filter.transform(self.test_X)
            self.test_X_selected = self.selector.transform(self.test_X_filtered)
        if self.valid_X is not None and self.valid_y is not None:
            self.valid_X_filtered = self.variance_filter.transform(self.valid_X)
            self.valid_X_selected = self.selector.transform(self.valid_X_filtered)
        self.feature_selector_name = feature_selector
        self.feature_select_num = select_des_num
        if saved_dir:
            variance_file = os.path.join(saved_dir, f"variance_{saved_file_note}.pkl")
            with open(variance_file, 'wb') as f:
                joblib.dump(self.variance_filter, f)
            selector_file = os.path.join(saved_dir, f'selector_{saved_file_note}.pkl')
            with open(selector_file, 'wb') as f:
                joblib.dump(self.selector, f)
        print(f"Selected feature num: {self.train_X_selected.shape[1]}")
        print(f'Time cost for selection: {sec_to_time(time.time() - t1)}')
        print('')

    def kfold_split(self, k=5, kfold_type="normal"):
        np.random.seed(self.random_state)
        train_val_idxs = []
        self.k = k
        if kfold_type in [None, "none"] or k == 1:
            print("Using none KFold, all training.")
            train_idx = np.random.permutation(len(self.train_X))
            val_idx = np.random.permutation(len(self.train_X))
            train_val_idxs.append([train_idx, val_idx])
        elif kfold_type == 'group' and k != 1:
            print("Using GroupKFold.")
            kf = GroupKFold(k)
            for train_idx, val_idx in kf.split(self.train_X, groups=self.train_groups):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        elif kfold_type == 'stratified' and k != 1:
            print("Using StratifiedKFold.")
            kf = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in kf.split(self.train_X, y=self.train_groups):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        elif kfold_type == 'loo':
            print("Using LeaveOneOut.")
            kf = LeaveOneOut()
            for train_idx, val_idx in kf.split(self.train_X):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        elif kfold_type == 'logo':
            print("Using LeaveOneGroupOut.")
            kf = LeaveOneGroupOut()
            for train_idx, val_idx in kf.split(self.train_X, groups=self.train_groups):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        else:
            print("Using normal KFold.")
            kf = KFold(k, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in kf.split(self.train_X):
                train_val_idxs.append([np.random.permutation(train_idx), np.random.permutation(val_idx)])
        self.kfold_type = kfold_type
        self.train_val_idxs = train_val_idxs

    def save_total_model(self, saved_dir, saved_file_note=""):
        total_model_file = os.path.join(saved_dir, f"total_model_{self.feature_selector_name}{self.feature_select_num}_{self.kfold_type}{self.k}_{self.model_name}_{saved_file_note}.model")
        with open(total_model_file, "wb+") as f:
            joblib.dump(self, f)
        print(f"The total model file has been saved: {total_model_file}")

    def load_total_model(self, model_file):
        with open(model_file, 'rb+') as f:
            new_model = joblib.load(f)
        for key, value in new_model.__dict__.items():
            self.__setattr__(key, value)

    def draw_predictions(self, y_true, y_pred, saved_dir, saved_file_note="", data_group="test"):
        
        fig_file = os.path.join(saved_dir, f"{data_group}_predicted_{saved_file_note}.png")
        if self.model_type == "regression":
            metrics = {"R2": r2_score(y_true, y_pred), 
                       "MSE": mean_squared_error(y_true, y_pred), 
                       "MAE": mean_absolute_error(y_true, y_pred)}
            plot_scatter(y_true, y_pred, title=f"{data_group} predicted", metrics=metrics, outfile=fig_file)
        elif self.model_type == "classification":
            plot_confusion_matrix(y_true, y_pred, title=f"{data_group} predicted", outfile=fig_file)
        else:
            raise NotImplementedError(f"model_type choice: regression/classification, {self.model_type} is not implemented.")

    def generate_ball_tree(self, p=1, neighbors_num=1, saved_dir=""):
        self.balltrees = []
        self.ref_dist_values = []
        self.feature_weights_list = []
        if self.full_trained:
            model = self.model
            if hasattr(model, "coef_"):
                feature_weights = model.coef_
            elif hasattr(model, "feature_importances_"):
                feature_weights = model.feature_importances_
            else:
                warnings.warn("The trained models have no attribute like 'coef_' or 'feature_importances_'")
                feature_weights = np.array([1] * self.train_X_selected.shape[1])
            scaler_w = MinMaxScaler(feature_range=(0.1, 0.9))
            feature_weights = scaler_w.fit_transform(feature_weights.reshape(-1, 1)).squeeze()
            self.feature_weights_list.append(feature_weights)
            balltree = BallTree(self.train_X_selected, metric='minkowski', p=p)
            if saved_dir:
                balltree_file = os.path.join(saved_dir, f"balltree_{self.feature_selector_name}{self.feature_select_num}_full_{self.model_name}.pkl")
                with open(balltree_file, 'wb') as f:
                    joblib.dump(balltree, f)
            self.balltrees.append(balltree)
            dist_mean = balltree.query(self.train_X_selected, k=neighbors_num, dualtree=True)[0][:, -1].mean()
            ref_dist_value = 2.5 * np.quantile(dist_mean, 0.75) - 1.5 * np.quantile(dist_mean, 0.25)  # Q3+1.5IQR
            self.ref_dist_values.append(ref_dist_value)
        else:
            for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
                model = self.models[i]
                if hasattr(model, "coef_"):
                    feature_weights = model.coef_
                elif hasattr(model, "feature_importances_"):
                    feature_weights = model.feature_importances_
                else:
                    warnings.warn("The trained models have no attribute like 'coef_' or 'feature_importances_'")
                    feature_weights = np.array([1] * self.train_X_selected.shape[1])
                scaler_w = MinMaxScaler(feature_range=(0.1, 0.9))
                feature_weights = scaler_w.fit_transform(feature_weights.reshape(-1, 1)).squeeze()
                self.feature_weights_list.append(feature_weights)
                kf_train_X = self.train_X_selected[train_idx]
                kf_val_X = self.train_X_selected[val_idx]
                balltree = BallTree(kf_train_X, metric='minkowski', p=p)
                if saved_dir:
                    with open(os.path.join(saved_dir, f"balltree_p{p}_fold_{i + 1}.pkl"), "wb+") as f:
                        joblib.dump(balltree, f)
                dist, ind = balltree.query(kf_val_X, k=neighbors_num, dualtree=True)
                dist_mean = dist.mean(axis=1)
                ref_dist_value = 2.5 * np.quantile(dist_mean, 0.75) - 1.5 * np.quantile(dist_mean, 0.25)  # Q3+1.5IQR
                self.ref_dist_values.append(ref_dist_value)
                print("*" * 50)
                print(f"Get reference distance value from validation set of fold {i + 1}: \033[36;1m{ref_dist_value}\033[0m")
                self.balltrees.append(balltree)

    def visualize_chem_space(self, train_X, test_X, saved_dir, method="tSNE", notes=""):
        """Visualize chemical space using tSNE or PCA."""
        all_X = np.concatenate((train_X, test_X), axis=0)
        if method == "tSNE":
            dimension_model = TSNE(n_components=2, perplexity=30, random_state=self.random_state)
            reduction_X = dimension_model.fit_transform(all_X)
        elif method == "PCA":
            dimension_model = PCA(n_components=2)
            reduction_X = dimension_model.fit_transform(all_X)
        else:
            raise NotImplementedError
        dimension_model_name = dimension_model.__class__.__name__
        reduction_X_tr = reduction_X[:len(train_X)]
        reduction_X_te = reduction_X[len(train_X):]
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.plot(reduction_X_tr[:, 0], reduction_X_tr[:, 1], linestyle='', marker='+',
                 color=self.color_list[-1], markerfacecolor='w', markersize=8, label="training set")
        plt.plot(reduction_X_te[:, 0], reduction_X_te[:, 1], linestyle='', marker='o',
                 color=self.color_list[-2], markerfacecolor='w', markersize=6, label="test set")
        plt.xlabel(f"{dimension_model_name}1", fontdict={'fontsize': 15})
        plt.ylabel(f"{dimension_model_name}2", fontdict={'fontsize': 15})
        plt.legend(loc="best")
        plt.savefig(os.path.join(saved_dir, f"train_test_distribution_{dimension_model_name}_{notes}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


class RegressionModel(BaseModel):
    """Regression Model class with methods for training and evaluation."""

    def __init__(self, random_state=0):
        super().__init__(random_state)
        self.metrics_list = ["R2", "RMSE", "MAE", "Pearson", "Spearman"]
        self.model_type = "regression"
        

    def cal_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        df_y = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        pearson_corr = df_y.corr().iloc[0, 1]
        spearman_corr = df_y.corr("spearman").iloc[0, 1]
        return r2, rmse, mae, pearson_corr, spearman_corr

    def train(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = copy.deepcopy(params)
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessRegressor) and "kernel" not in self.params:
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        train_metrics_all, val_metrics_all, test_metrics_all = [], [], []
        test_pred_all, val_pred_all, val_y_all = [], [], []
        self.models = []
        if self.valid_X_selected is not None:
            ## if validation set is provided, train model without cross-validation
            print("=" * 50)
            print(f"Train/validation num: {len(self.train_y)}/{len(self.valid_y)}")
            self.model.fit(self.train_X_selected, self.train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(self.train_X_selected, self.valid_X_selected, saved_dir=saved_dir, method="tSNE", notes="valid")
            train_pred = self.model.predict(self.train_X_selected)
            
            valid_pred = self.model.predict(self.valid_X_selected)
            
            train_metrics = self.cal_metrics(self.train_y, train_pred)
            valid_metrics = self.cal_metrics(self.valid_y, valid_pred)
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(valid_metrics)
            val_pred_all.extend(valid_pred)
            
            val_y_all.extend(self.valid_y)
            if hasattr(self, "test_X_selected") and self.test_y is not None:
                
                test_pred = self.model.predict(self.test_X_selected)
                test_pred_all.append(test_pred)
                
                test_metrics = self.cal_metrics(self.test_y, test_pred)
                test_metrics_all.append(test_metrics)
                if saved_dir:
                    self.visualize_chem_space(self.train_X_selected, self.test_X_selected, saved_dir=saved_dir, method="tSNE", notes="test")
            self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_y_all, test_pred_all, saved_dir)
            print('Total run time:', sec_to_time(time.time() - tick))
            return
        for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
            print("=" * 50)
            print(f"Train/validation num: {len(train_idx)}/{len(val_idx)}")
            kf_train_X = self.train_X_selected[train_idx]
            kf_train_y = self.train_y[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            kf_val_y = self.train_y[val_idx]
            self.model.fit(kf_train_X, kf_train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}_{i + 1}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(kf_train_X, kf_val_X, saved_dir=saved_dir, method="tSNE", notes=f"fold{i + 1}")
            kf_train_pred = self.model.predict(kf_train_X)
            kf_val_pred = self.model.predict(kf_val_X)
            train_metrics_all.append(self.cal_metrics(kf_train_y, kf_train_pred))
            val_metrics_all.append(self.cal_metrics(kf_val_y, kf_val_pred) if self.kfold_type != "loo" else [None] * len(self.metrics_list))
            val_pred_all.extend(kf_val_pred)
            val_y_all.extend(kf_val_y)
            if hasattr(self, "test_X_selected") and self.test_y is not None:
                test_pred = self.model.predict(self.test_X_selected)
                test_pred_all.append(test_pred)
                test_metrics_all.append(self.cal_metrics(y_true=self.test_y, y_pred=test_pred))
        self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_y_all, test_pred_all, saved_dir)
        print('Total run time:', sec_to_time(time.time() - tick))

    def _aggregate_metrics(self, train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_y_all, test_pred_all, saved_dir):
        metrics_list = self.metrics_list
        self.train_metrics_df = pd.DataFrame(train_metrics_all, columns=["tr_" + s for s in metrics_list],
                                             index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        self.val_metrics_df = pd.DataFrame(val_metrics_all, columns=["val_" + s for s in metrics_list],
                                           index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        metrics_dfs = [self.train_metrics_df, self.val_metrics_df]

        self.val_pred_all = np.array(val_pred_all, dtype=np.float32)
        self.val_y_all = np.array(val_y_all, dtype=np.float32)

        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_metrics_df = pd.DataFrame(test_metrics_all, columns=["te_" + s for s in metrics_list],
                                                index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
            metrics_dfs.append(self.test_metrics_df)
            self.test_pred = np.mean(test_pred_all, axis=0)
            df_te_pred = pd.DataFrame([self.test_y.squeeze(), self.test_pred.squeeze()],
                                      index=['true_value', "pred_value"]).T
            if saved_dir:
                test_pred_file = os.path.join(saved_dir, f"test_predicted_{self.model_name}.csv")
                df_te_pred.to_csv(test_pred_file, index=False)

        all_metrics_df = pd.concat(metrics_dfs, axis=1).T
        all_metrics_df['mean'] = all_metrics_df.mean(axis=1)
        if self.kfold_type == "loo":
            val_mean_metric = self.cal_metrics(self.val_y_all, self.val_pred_all)
            for c_idx, col in enumerate(["val_" + s for s in metrics_list]):
                all_metrics_df.loc[col, 'mean'] = val_mean_metric[c_idx]
        self.all_metrics_df = all_metrics_df.T
        print('*' * 50)
        print("All results for k-fold cross validation: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if (("R2" in col) or ("RMSE" in col))]])

    def predict(self, X, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self, "scaler"):
            X = self.scaler.transform(X)
        if hasattr(self, "variance_filter"):
            X = self.variance_filter.transform(X)
        if hasattr(self, "selector"):
            X = self.selector.transform(X)
        all_y_pred = []
        for model in self.models:
            y_pred = model.predict(X)
            all_y_pred.append(y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0).reshape((-1, 1))
        if cal_feature_distance and hasattr(self, "balltrees"):
            dist_means = []
            confident_indexes = []
            for i, balltree in enumerate(self.balltrees):
                dist, ind = balltree.query(X, k=neighbors_num, dualtree=True)
                dist_mean = dist.mean(axis=1).reshape((-1, 1))
                dist_means.append(dist_mean)
                confident_index = self.ref_dist_values[i] / (dist_mean + 0.00001)
                confident_indexes.append(confident_index)
            dist_mean = np.mean(dist_means, axis=0)
            confident_index = np.mean(confident_indexes, axis=0)
            y_pred_mean = np.hstack([y_pred_mean, dist_mean, confident_index])
        return y_pred_mean

    def fulltrain(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = params
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessRegressor):
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        if self.valid_X_selected is not None:
            ## concatenate train and valid data for training
            train_X_selected = np.vstack([self.train_X_selected, self.valid_X_selected])
            train_y = np.hstack([self.train_y, self.valid_y])
        self.model.fit(train_X_selected, train_y)
        if saved_dir:
            with open(os.path.join(saved_dir, f'{self.model_name}_full.pkl'), 'wb+') as f:
                joblib.dump(self.model, f)
        train_pred = self.model.predict(train_X_selected)
        train_metrics = self.cal_metrics(train_y, train_pred)
        self.all_metrics_df.loc['full', ["tr_" + s for s in self.metrics_list]] = train_metrics
        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_pred = self.model.predict(self.test_X_selected)
            test_metrics = self.cal_metrics(self.test_y, self.test_pred)
            self.all_metrics_df.loc['full', ["te_" + s for s in self.metrics_list]] = test_metrics
        # print(self.all_metrics_df.columns)
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if (("R2" in col) or ("RMSE" in col))]])
        print('Total run time:', sec_to_time(time.time() - tick))
        self.full_trained = True


class ClassificationModel(BaseModel):
    """Classification Model class with methods for training and evaluation."""

    def __init__(self, random_state=0, n_class=2):
        super().__init__(random_state)
        self.prob_thd = 0.5
        self.metrics_list = ["ACC", "BACC", "MCC", "AUC", "F1", "CE"]
        self.n_class = n_class
        self.model_type = "classification"
        self.full_trained = False

    def cal_metrics(self, y_true, y_pred, y_pred_prob=None, n_class=2):
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred)
        acc = round(accuracy_score(y_true, y_pred), 4)
        mcc = round(matthews_corrcoef(y_true, y_pred), 4)
        bacc = round(balanced_accuracy_score(y_true, y_pred), 4)
        f1 = round(f1_score(y_true, y_pred, average='macro'), 4)
        if y_pred_prob is not None:
            y_pred_prob = np.array(y_pred_prob, dtype=np.float32)
            ce = weighted_cross_entropy(y_true, y_pred_prob, n_class)
            if n_class == 2 and len(y_pred_prob.shape) == 2:
                y_pred_prob = y_pred_prob[:, 1]
            if n_class > 2:
                auc = round(roc_auc_score(y_true, y_pred_prob, multi_class='ovo', average='macro'), 4)
            else:
                auc = round(roc_auc_score(y_true, y_pred_prob), 4)
        else:
            ce = 0
            auc = 0
        return acc, bacc, mcc, auc, f1, ce
    def label_encode(self, saved_dir=None, saved_file_note=""):
        self.label_encoder = LabelEncoder()
        self.train_y_init = copy.deepcopy(self.train_y)
        self.train_y = self.label_encoder.fit_transform(self.train_y)
        if hasattr(self, "test_y") and self.test_y is not None:
            self.test_y_init = copy.deepcopy(self.test_y)
            self.test_y = self.label_encoder.transform(self.test_y)
        if hasattr(self, "valid_y") and self.valid_y is not None:
            self.valid_y_init = copy.deepcopy(self.valid_y)
            self.valid_y = self.label_encoder.transform(self.valid_y)
        if saved_dir:
            with open(os.path.join(saved_dir, f"label_encoder_{saved_file_note}.pkl"), 'wb+') as f:
                joblib.dump(self.label_encoder, f)

    def train(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = copy.deepcopy(params)
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessClassifier) and "kernel" not in self.params:
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        train_metrics_all, val_metrics_all, test_metrics_all = [], [], []
        test_pred_all, test_pred_prob_all, val_pred_all, val_pred_prob_all, val_y_all = [], [], [], [], []
        self.models = []

        if self.valid_X_selected is not None:
            ## if validation set is provided, train model without cross-validation
            print("=" * 50)
            print(f"Train/validation num: {len(self.train_y)}/{len(self.valid_y)}")
            self.model.fit(self.train_X_selected, self.train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(self.train_X_selected, self.valid_X_selected, saved_dir=saved_dir, method="tSNE", notes="valid")
            train_pred = self.model.predict(self.train_X_selected)
            train_pred_prob = self.model.predict_proba(self.train_X_selected)
            valid_pred = self.model.predict(self.valid_X_selected)
            valid_pred_prob = self.model.predict_proba(self.valid_X_selected)
            train_metrics = self.cal_metrics(self.train_y, train_pred, train_pred_prob, n_class=self.n_class)
            valid_metrics = self.cal_metrics(self.valid_y, valid_pred, valid_pred_prob, n_class=self.n_class)
            train_metrics_all.append(train_metrics)
            val_metrics_all.append(valid_metrics)
            val_pred_all.extend(valid_pred)
            val_pred_prob_all.extend(valid_pred_prob)
            val_y_all.extend(self.valid_y)
            if hasattr(self, "test_X_selected") and self.test_y is not None:
                test_pred_prob = self.model.predict_proba(self.test_X_selected)
                test_pred = self.model.predict(self.test_X_selected)
                test_pred_all.append(test_pred)
                test_pred_prob_all.append(test_pred_prob)
                test_metrics = self.cal_metrics(self.test_y, test_pred, test_pred_prob, n_class=self.n_class)
                test_metrics_all.append(test_metrics)
                if saved_dir:
                    self.visualize_chem_space(self.train_X_selected, self.test_X_selected, saved_dir=saved_dir, method="tSNE", notes="test")
            self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_pred_prob_all, val_y_all, test_pred_all, saved_dir)
            print('Total run time:', sec_to_time(time.time() - tick))
            return
        
        for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
            print("=" * 50)
            print(f"Train/validation num: {len(train_idx)}/{len(val_idx)}")
            kf_train_X = self.train_X_selected[train_idx]
            kf_train_y = self.train_y[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            kf_val_y = self.train_y[val_idx]
            self.model.fit(kf_train_X, kf_train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir:
                with open(os.path.join(saved_dir, f'{self.model_name}_{i + 1}.pkl'), 'wb+') as f:
                    joblib.dump(self.model, f)
                self.visualize_chem_space(kf_train_X, kf_val_X, saved_dir=saved_dir, method="tSNE", notes=f"fold{i + 1}")
            if self.model_name == "SVC":
                kf_train_pred_prob = self.model.decision_function(kf_train_X)
                kf_train_pred = self.model.predict(kf_train_X)
                kf_val_pred_prob = self.model.decision_function(kf_val_X)
                if len(kf_train_pred_prob.shape) == 1:
                    kf_train_pred_prob = np.stack((-kf_train_pred_prob, kf_train_pred_prob), axis=1)
                    kf_val_pred_prob = np.stack((-kf_val_pred_prob, kf_val_pred_prob), axis=1)
                kf_val_pred = self.model.predict(kf_val_X)
            else:
                kf_train_pred_prob = self.model.predict_proba(kf_train_X)
                kf_train_pred = self.model.predict(kf_train_X)
                kf_val_pred_prob = self.model.predict_proba(kf_val_X)
                kf_val_pred = self.model.predict(kf_val_X)

            train_metrics_all.append(self.cal_metrics(kf_train_y, kf_train_pred, kf_train_pred_prob, n_class=self.n_class))
            val_metrics_all.append(self.cal_metrics(kf_val_y, kf_val_pred, kf_val_pred_prob, n_class=self.n_class) if self.kfold_type != "loo" else [None] * len(self.metrics_list))
            val_pred_all.extend(kf_val_pred)
            val_pred_prob_all.extend(kf_val_pred_prob)
            val_y_all.extend(kf_val_y)

            if hasattr(self, "test_X_selected") and self.test_y is not None:
                if self.model_name == "SVC":
                    test_pred_prob = self.model.decision_function(self.test_X_selected)
                    test_pred = self.model.predict(self.test_X_selected)
                    if len(test_pred_prob.shape) == 1:
                        test_pred_prob = np.stack((-test_pred_prob, test_pred_prob), axis=1)
                else:
                    test_pred_prob = self.model.predict_proba(self.test_X_selected)
                    test_pred = self.model.predict(self.test_X_selected)
                test_pred_all.append(test_pred)
                test_pred_prob_all.append(test_pred_prob)
                test_metrics_all.append(self.cal_metrics(y_true=self.test_y, y_pred=test_pred, y_pred_prob=test_pred_prob, n_class=self.n_class))
        self._aggregate_metrics(train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_pred_prob_all, val_y_all, test_pred_all, saved_dir)
        print('Total run time:', sec_to_time(time.time() - tick))

    def _aggregate_metrics(self, train_metrics_all, val_metrics_all, test_metrics_all, val_pred_all, val_pred_prob_all, val_y_all, test_pred_all, saved_dir, test_pred_prob_all=None):
        metrics_list = self.metrics_list
        self.train_metrics_df = pd.DataFrame(train_metrics_all, columns=["tr_" + s for s in metrics_list],
                                             index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        self.val_metrics_df = pd.DataFrame(val_metrics_all, columns=["val_" + s for s in metrics_list],
                                           index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
        metrics_dfs = [self.train_metrics_df, self.val_metrics_df]

        self.val_pred_all = np.array(val_pred_all, dtype=np.float32)
        self.val_y_all = np.array(val_y_all, dtype=np.float32)
        self.val_pred_prob_all = np.array(val_pred_prob_all, dtype=np.float32)

        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_metrics_df = pd.DataFrame(test_metrics_all, columns=["te_" + s for s in metrics_list],
                                                index=[f'fold_{i + 1}' for i in range(len(self.train_val_idxs))])
            metrics_dfs.append(self.test_metrics_df)
            ## for classification, test_pred is the predicted class label which appears most frequently in all predictions from different models
            self.test_pred_all = np.array(test_pred_all, dtype=np.int8)
            self.test_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=self.test_pred_all)
            df_te_pred = pd.DataFrame([self.test_y.squeeze(), self.test_pred.squeeze()],
                                      index=['true_value', "pred_value"]).T
            if saved_dir:
                test_pred_file = os.path.join(saved_dir, f"test_predicted_{self.model_name}.csv")
                df_te_pred.to_csv(test_pred_file, index=False)

        all_metrics_df = pd.concat(metrics_dfs, axis=1).T
        all_metrics_df['mean'] = all_metrics_df.mean(axis=1)
        if self.kfold_type == "loo":
            val_mean_metric = self.cal_metrics(self.val_y_all, self.val_pred_all, self.val_pred_prob_all, n_class=self.n_class)
            for c_idx, col in enumerate(["val_" + s for s in metrics_list]):
                all_metrics_df.loc[col, 'mean'] = val_mean_metric[c_idx]
        self.all_metrics_df = all_metrics_df.T
        print('*' * 50)
        print("All results for k-fold cross validation: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if ("tr" not in col and col.split("_")[-1] in ["MCC", "ACC", "BACC", "AUC", "F1"])]])
    
    def predict(self, X, return_prob=False, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self, "scaler"):
            X = self.scaler.transform(X)
            print("Scaled X shape:", X.shape)
        if hasattr(self, "variance_filter"):
            X = self.variance_filter.transform(X)
            print("Variance filtered X shape:", X.shape)
        if hasattr(self, "selector"):
            X = self.selector.transform(X)
            print("Selected X shape:", X.shape)
        all_y_pred = []
        for model in self.models:
            if self.model_name == "SVC":
                y_pred = model.decision_function(X)
                y_pred = np.stack((-y_pred, y_pred), axis=1)
            else:
                y_pred = model.predict_proba(X)
            all_y_pred.append(y_pred)
        # print(all_y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0)
        if not return_prob:
            y_pred_mean = y_pred_mean.argmax(axis=1)  # prob -> class_label
            y_pred_mean = y_pred_mean.reshape((-1, 1))
        if cal_feature_distance and hasattr(self, "balltrees"):
            dist, ind = self.balltrees[0].query(X, k=neighbors_num, dualtree=True)
            dist_mean = dist.mean(axis=1).reshape((-1, 1))
            y_pred_mean = y_pred_mean.reshape((-1, 1))
            y_pred_mean = np.hstack([y_pred_mean, dist_mean])
            if hasattr(self, "ref_dist_values"):
                confident_index = self.ref_dist_values[0] / (dist_mean + 0.00001)
                y_pred_mean = np.hstack([y_pred_mean, confident_index])
        return y_pred_mean

    def fulltrain(self, estimator, params, saved_dir=""):
        self.model = estimator
        self.params = copy.deepcopy(params)
        self.model_name = self.model.__class__.__name__
        tick = time.time()
        if isinstance(self.model, GaussianProcessClassifier):
            kernel = ConstantKernel(constant_value=self.params.pop("constant_value")) \
                     * RBF(length_scale=self.params.pop("length_scale")) \
                     + WhiteKernel(noise_level=self.params.pop("noise_level"))
            self.params.update(kernel=kernel)
        self.model.set_params(**self.params)
        if self.valid_X_selected is not None:
            ## concatenate train and valid data for training
            train_X_selected = np.vstack([self.train_X_selected, self.valid_X_selected])
            train_y = np.hstack([self.train_y, self.valid_y])
        print(f"Full training with {len(train_X_selected)} samples")
        self.model.fit(train_X_selected, train_y)
        # self.models = [copy.copy(self.model)]
        if saved_dir:
            with open(os.path.join(saved_dir, f'{self.model_name}_full.pkl'), 'wb+') as f:
                joblib.dump(self.model, f)
        train_pred = self.model.predict(train_X_selected)
        if self.model_name == "SVC":
            train_pred_prob = self.model.decision_function(train_X_selected)
            if len(train_pred_prob.shape) == 1:
                train_pred_prob = np.stack((-train_pred_prob, train_pred_prob), axis=1)
        else:
            train_pred_prob = self.model.predict_proba(train_X_selected)
        train_metrics = self.cal_metrics(train_y, train_pred, train_pred_prob, n_class=self.n_class)
        self.all_metrics_df.loc['full', ["tr_" + s for s in self.metrics_list]] = train_metrics
        if hasattr(self, "test_X_selected") and self.test_y is not None:
            self.test_pred = self.model.predict(self.test_X_selected)
            if self.model_name == "SVC":
                test_pred_prob = self.model.decision_function(self.test_X_selected)
                if len(test_pred_prob.shape) == 1:
                    test_pred_prob = np.stack((-test_pred_prob, test_pred_prob), axis=1)
            else:
                test_pred_prob = self.model.predict_proba(self.test_X_selected)
            self.test_pred_prob = test_pred_prob
            test_metrics = self.cal_metrics(self.test_y, self.test_pred, self.test_pred_prob, n_class=self.n_class)
            self.all_metrics_df.loc['full', ["te_" + s for s in self.metrics_list]] = test_metrics
        print('*' * 50)
        print("All results for full training: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if ("tr" not in col and col.split("_")[-1] in ["MCC", "ACC", "BACC", "AUC", "F1"])]])
        print('Total run time:', sec_to_time(time.time() - tick))
        self.full_trained = True

if __name__ == "__main__":
    pass