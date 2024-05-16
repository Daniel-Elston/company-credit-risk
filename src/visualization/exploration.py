from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils.setup_env import setup_project_env
sns.set_theme(style="darkgrid")
project_dir, config, setup_logs = setup_project_env()


class Visualiser:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def select_by_metric(self, df, company_name, metric: str):
        company = df[df['Company name'] == company_name]
        company_id = company.index[0]
        metric_selected = df[df.columns.str.contains(metric)]
        company_metric = metric_selected[metric_selected.index == company_id]
        return company_metric

    def generate_pair_plot(self, df, cols, title):
        select_metrics = df[cols]
        sns.pairplot(
            select_metrics, diag_kind='kde',
            plot_kws={'alpha': 0.8, 's': 2, 'edgecolor': 'k'})
        plt.savefig(Path(f'{config['path']['exploration']}/{title}.png'))

    def generate_heat_plot(self, df, cols, title):
        correlation_data = df[cols]
        correlation_matrix = correlation_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f",
            cbar=True, square=True, vmax=1, vmin=-1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            annot_kws={"size": 8})
        plt.title(f'Correlation Matrix Heatmap for {title} Financial Metrics {config['year']}')
        plt.savefig(Path(f'{config['path']['exploration']}/{title}.png'))
        return correlation_matrix

    def generate_trends(self, df, metric: str, date_cols):
        cols = [f'{metric}.{year}' for year in date_cols]
        mean_values = df[cols].mean(axis=0)
        plt.figure(figsize=(14, 7))
        plt.plot(date_cols, mean_values)
        plt.title(f'{metric} Trends Over Years')
        plt.xlabel('Year')
        plt.ylabel(f'{metric}')
        plt.savefig(Path(f'{config['path']['exploration']}/{metric}_trends.png'))

    def exploration_filing(self, run_number):
        dir_path = f'{config['path']['exploration']}/exploration_{run_number}'
        if os.path.isdir(dir_path):
            pass
        else:
            self.logger.info(f'Creating ``{dir_path}`` directory.')
            os.mkdir(dir_path)

    def pipeline(self, df, groups, run_number):
        self.logger.info(
            f'Running Visualiser Pipeline. Run Number {run_number}...')

        dist_store, dist_names = list(groups.values())[:-2], list(groups.keys())[:-2]
        # corr_store, corr_names = list(groups.values())[:-1], list(groups.keys())[:-1]

        for i, j in zip(dist_store, dist_names):
            self.generate_pair_plot(df, i, f'exploration_{run_number}/pair_plot_{j}')

        # for i, j in zip(corr_store, corr_names):
        #     self.generate_heat_plot(df, i, f'exploration_{run_number}/corr_map_{j}')

        corr_mat = self.generate_heat_plot(df, groups['all'], f'exploration_{run_number}/corr_map_all')
        corr_mat.to_csv(Path(f'{config['path']['correlation']}/exploration_{run_number}.csv'))

        self.logger.info(
            f'Visualiser Pipeline Completed. Figures saved to: ``{config['path']['exploration']}/exploration_{run_number}/*.png``')
