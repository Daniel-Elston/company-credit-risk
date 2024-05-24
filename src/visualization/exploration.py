from __future__ import annotations

import logging
import warnings
from pathlib import Path

import dcor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import delayed
from joblib import Parallel

from utils.file_handler import save_to_parquet
from utils.setup_env import setup_project_env

warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid")
project_dir, config, setup_logs = setup_project_env()


class Visualiser:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fig_path = Path(config["path"]["exploration"])
        self.corr_path = Path(config["path"]["correlation"])
        self.var_path = Path(config["path"]["variance"])

    def generate_pair_plot(self, df, title):
        sns.pairplot(
            df, diag_kind='kde',
            plot_kws={'alpha': 0.8, 's': 2, 'edgecolor': 'k'})
        plt.savefig(Path(f'{self.fig_path}/{title}.png'))
        plt.close()

    def generate_corr_plot(self, df, title, method='pearson'):
        correlation_matrix = df.corr(method=method)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f",
            cbar=True, square=True, vmax=1, vmin=-1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            annot_kws={"size": 8})
        plt.title(f'Correlation Matrix Heatmap for {title} (Method: {method})')
        plt.savefig(Path(f'{self.fig_path}/{title}.png'))
        plt.close()
        return correlation_matrix

    def generate_distance_corr_plot(self, df, title, run_number):
        columns = df.columns
        dist_corr_matrix = pd.DataFrame(index=columns, columns=columns)
        for col1 in columns:
            for col2 in columns:
                if col1 == col2:
                    dist_corr_matrix.loc[col1, col2] = 1.0
                else:
                    dist_corr = dcor.distance_correlation(df[col1], df[col2])
                    dist_corr_matrix.loc[col1, col2] = dist_corr

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            dist_corr_matrix.astype(float), annot=True, fmt=".2f",
            cbar=True, square=True, vmax=1, vmin=-1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            annot_kws={"size": 8})
        plt.title(f'Distance Correlation Heatmap for {title}')
        plt.savefig(Path(f'{self.fig_path}/{title}.png'))
        plt.close()
        save_to_parquet(dist_corr_matrix, Path(f"{self.corr_path}/exploration_{run_number}.parquet"))
        return dist_corr_matrix

    def compute_and_save_variance(self, df, run_number):
        variance_matrix = round(df.var(), 2)
        variance_matrix_df = pd.DataFrame(variance_matrix, columns=['Variance'])
        save_to_parquet(variance_matrix_df, Path(f"{self.var_path}/exploration_{run_number}.parquet"))
        return variance_matrix_df

    def exploration_filing(self, run_number):
        dir_path = Path(f'{self.path_exp}/exploration_{run_number}')
        dir_path.mkdir(parents=True, exist_ok=True)

    def pipeline(self, df, run_number, **kwargs):
        groups = kwargs.get('groups')

        self.logger.info(
            f'Running Visualisation Pipeline. Exploration Run Number {run_number}...')

        dist_store, dist_names = list(groups.values())[:-2], list(groups.keys())[:-2]
        methods = ['pearson', 'spearman', 'kendall']
        cols = groups['all']

        Parallel(n_jobs=4)(
            delayed(self.generate_pair_plot)(
                df[i], f'exploration_{run_number}/pair_plot_{j}') for i, j in zip(dist_store, dist_names))

        for method in methods:
            self.generate_corr_plot(df[cols], f'exploration_{run_number}/corr_map_all_{method}', method=method)

        self.generate_distance_corr_plot(df[cols], f'exploration_{run_number}/corr_map_all_dist', run_number)
        self.compute_and_save_variance(df[cols], run_number)

        self.logger.info(
            f'Variance and Correlation Matrix saved to: reports/analysis/.../exploration_{run_number}.parquet')
        self.logger.info(
            f'Visualisation Pipeline Completed. Figures saved to: ``{self.fig_path}/exploration_{run_number}/*.png``')
