from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from database.db_ops import DataBaseOps
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
sns.set_theme(style="darkgrid")
project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class Analysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def stratified_random_sample(self, df):
        df_strat = df.groupby('Sector 1', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(0.1 * len(x)))))
        return df_strat

    def select_by_metric(self, df, company_name, metric: str):
        company = df[df['Company name'] == company_name]
        company_id = company.index[0]
        metric_selected = df[df.columns.str.contains(metric)]
        company_metric = metric_selected[metric_selected.index == company_id]
        return company_metric

    def identify_outliers(self, df):
        outlier_store = []
        meta_store = []
        for column in df.select_dtypes(include=[np.number]).columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            lower_counts = (df[column] < lower_bound).sum()
            upper_counts = (df[column] > upper_bound).sum()

            perc_lower = round(lower_counts / len(df) * 100, 2)
            perc_upper = round(upper_counts / len(df) * 100, 2)

            if perc_lower > 10 or perc_upper > 10:
                meta_store.append((perc_lower+perc_upper)/2)
                outlier_dict = {
                    'Column': column,
                    'Lower Outliers': perc_lower,
                    'Upper Outliers': perc_upper}
                outlier_store.append(outlier_dict)

        df_outlier_perc = round(sum(meta_store)/len(meta_store), 2)
        self.logger.debug('Dataframe outlier percentage: %s', df_outlier_perc)

        filepath = Path('reports/analysis/outliers.json')
        save_json(outlier_store, filepath)

    def pipeline(self, df):
        self.logger.info('Running Analysis Pipeline.')
        df_strat = self.stratified_random_sample(df)
        self.identify_outliers(df)
        self.logger.info(
            'Analysis Pipeline Completed. Identified Outliers saved to: ``reports/analysis/outliers.json``')
        return df_strat


class Visualiser:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_pair_plot(self, df, cols, title):
        select_metrics = df[cols]
        sns.pairplot(
            select_metrics, diag_kind='kde',
            plot_kws={'alpha': 0.8, 's': 10})
        plt.title(f'Pair Plot of {title} Metrics 2020')
        plt.savefig(f'reports/figures/pair_plot_{title}.png')

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
                  title} Financial Metrics 2020')
        plt.savefig(f'reports/figures/corr_map_{title}.png')

    def generate_trends(self, df, metric: str, date_cols):
        cols = [f'{metric}.{year}' for year in date_cols]
        mean_values = df[cols].mean(axis=0)
        plt.figure(figsize=(14, 7))
        plt.plot(date_cols, mean_values)
        plt.title(f'{metric} Trends Over Years')
        plt.xlabel('Year')
        plt.ylabel(f'{metric}')
        plt.savefig(f'reports/figures/{metric}_trends.png')

    def get_plotting_cols(self, df):
        df_numeric = df.select_dtypes(include=[np.number])
        remove_dates = df.columns[df.columns.str.contains(
            '2019|2018|2017|2016|2015')]
        df_single_date = df_numeric.drop(remove_dates, axis=1)

        raw_features = [
            'Turnover.2020', 'EBIT.2020', 'PLTax.2020', 'Leverage.2020', 'ROE.2020',
            'TAsset.2020', 'debt_to_eq2020', 'op_marg2020', 'asset_turnover2020', 'roa2020']
        engineered_features = [
            'growth_Turnover.2020', 'growth_MScore.2020', 'growth_EBIT.2020', 'growth_PLTax.2020',
            'growth_ROE.2020', 'growth_TAsset.2020', 'growth_Leverage.2020', 'debt_to_eq2020',
            'op_marg2020', 'asset_turnover2020', 'roa2020']

        return df_single_date, raw_features, engineered_features

    def pipeline(self, df, date_cols):
        self.logger.info('Running Visualiser Pipeline.')
        df_single_date, raw_features, engineered_features = self.get_plotting_cols(
            df)

        self.generate_pair_plot(df, raw_features, 'Raw')
        self.generate_pair_plot(df, engineered_features, 'Engineered')
        self.generate_heat_plot(df, raw_features, 'Raw')
        self.generate_heat_plot(df, engineered_features, 'Engineered')
        self.generate_heat_plot(df, df_single_date.columns, 'All')
        self.generate_trends(df.sample(1000), 'EBIT', date_cols)
        self.logger.info(
            'Visualiser Pipeline Completed. Figures saved to: ``reports/figures/*.png``')
