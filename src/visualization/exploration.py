from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import seaborn as sns

from utils.config_ops import amend_features
from utils.setup_env import setup_project_env
sns.set_theme(style="darkgrid")
project_dir, config, setup_logs = setup_project_env()
# import statistics


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
            plot_kws={'alpha': 0.8, 's': 10})
        plt.title(f'Pair Plot of {title} Metrics {config['year']}')
        plt.savefig(f'reports/analysis/figures/{title}.png')

    def generate_heat_plot(self, df, cols, title):
        correlation_data = df[cols]
        correlation_matrix = correlation_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f",
            cbar=True, square=True, vmax=1, vmin=-1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            annot_kws={"size": 6})
        plt.title(f'Correlation Matrix Heatmap for {
                  title} Financial Metrics {config['year']}')
        plt.savefig(f'reports/analysis/figures/corr_map_{title}.png')

    def generate_trends(self, df, metric: str, date_cols):
        cols = [f'{metric}.{year}' for year in date_cols]
        mean_values = df[cols].mean(axis=0)
        plt.figure(figsize=(14, 7))
        plt.plot(date_cols, mean_values)
        plt.title(f'{metric} Trends Over Years')
        plt.xlabel('Year')
        plt.ylabel(f'{metric}')
        plt.savefig(f'reports/analysis/figures/{metric}_trends.png')

    def pipeline(self, df):
        self.logger.info(
            'Running Visualiser Pipeline.')
        raw, grow, vol, further = amend_features(config)

        for i, j in zip([raw, grow, vol, further], ['raw', 'grow', 'vol', 'further']):
            self.generate_pair_plot(df, i, f'pair_plot_{j}')

        # self.generate_heat_plot(df, raw_features, 'Raw')
        # self.generate_heat_plot(df, engineered_features, 'Engineered')
        # self.generate_heat_plot(df, df_single_date.columns, 'All')
        # self.generate_trends(df.sample(1000), 'EBIT', date_cols)
        self.logger.info(
            'Visualiser Pipeline Completed. Figures saved to: ``reports/analysis/figures & transforms/*.png``')
