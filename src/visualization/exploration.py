from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils.file_handler import save_to_parquet
from utils.setup_env import setup_project_env
sns.set_theme(style="darkgrid")
project_dir, config, setup_logs = setup_project_env()


class Visualiser:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dir_path = Path(config["path"]["exploration"])

    def generate_pair_plot(self, df, title):
        sns.pairplot(
            df, diag_kind='kde',
            plot_kws={'alpha': 0.8, 's': 2, 'edgecolor': 'k'})
        plt.savefig(Path(f'{self.dir_path}/{title}.png'))

    def generate_heat_plot(self, df, title):
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f",
            cbar=True, square=True, vmax=1, vmin=-1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            annot_kws={"size": 8})
        plt.title(f'Correlation Matrix Heatmap for {title} Financial Metrics {config["year"]}')
        plt.savefig(Path(f'{self.dir_path}/{title}.png'))
        return correlation_matrix

    def exploration_filing(self, run_number):
        dir_path = Path(f'{self.path_exp}/exploration_{run_number}')
        dir_path.mkdir(parents=True, exist_ok=True)

    def pipeline(self, df, run_number, **kwargs):
        groups = kwargs.get('groups')

        self.logger.info(
            f'Running Visualisation Pipeline. Exploration Run Number {run_number}...')

        dist_store, dist_names = list(groups.values())[:-2], list(groups.keys())[:-2]
        corr_store, corr_names = list(groups.values())[:-1], list(groups.keys())[:-1]

        for i, j in zip(dist_store, dist_names):
            self.generate_pair_plot(df[i], f'exploration_{run_number}/pair_plot_{j}')

        for i, j in zip(corr_store, corr_names):
            self.generate_heat_plot(df[i], f'exploration_{run_number}/corr_map_{j}')

        cols = groups['all']
        corr_mat = self.generate_heat_plot(df[cols], f'exploration_{run_number}/corr_map_all')
        save_to_parquet(corr_mat, Path(f'{config['path']['correlation']}/exploration_{run_number}.csv'))

        self.logger.info(
            f'Visualisation Pipeline Completed. Figures saved to: ``{self.dir_path}/exploration_{run_number}/*.png``')
