from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import boxcox
from scipy.stats import kurtosis
from scipy.stats import skew

from database.db_ops import DataBaseOps
from utils.config_ops import amend_features
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
                    'Upper Outliers': perc_upper
                }
                outlier_store.append(outlier_dict)

        df_outlier_perc = round(sum(meta_store)/len(meta_store), 2)
        self.logger.debug('Dataframe outlier percentage: %s', df_outlier_perc)

        filepath = Path('reports/analysis/outliers.json')
        save_json(outlier_store, filepath)

    def evaluate_skew(self, df):
        df = df.select_dtypes(include=[np.number])
        skew_store = []
        for column in df.columns:
            original_skew = skew(df[column].dropna())
            transformed_skew = skew(
                np.log1p(df[column].clip(lower=0)).dropna())

            if abs(original_skew) > 1:
                skew_dict = {
                    'Column': column,
                    'Original Skew': round(original_skew, 2),
                    'Transformed Skew': round(transformed_skew, 2)
                }
                skew_store.append(skew_dict)

        filepath = Path('reports/analysis/skew.json')
        save_json(skew_store, filepath)

    def apply_box_cox_1p(self, df, cols):
        shifts = df[cols].min().apply(lambda x: 1 - x if x <= 0 else 0)
        df[cols] += shifts
        df[cols] = df[cols].apply(lambda x: boxcox1p(x, 0))
        return df

    def apply_box_cox(self, df, cols):
        shifts = df[cols].min().apply(lambda x: 1 - x if x <= 0 else 0)
        df[cols] += shifts
        df[cols] = df[cols].apply(lambda x: boxcox(x, 0))
        return df

    def apply_log(self, df, cols):
        df[cols] = df[cols].clip(lower=0).apply(np.log1p)
        return df

    def apply_sqrt(self, df, cols):
        df[cols] = np.sqrt(df[cols].clip(lower=0))
        return df

    def apply_inverse_sqrt(self, df, cols):
        df[cols] = 1 / np.sqrt(df[cols].clip(lower=0))
        return df

    def apply_inv(self, df, cols):
        df[cols] = 1 / df[cols].clip(lower=0)
        return df

    def pipeline(self, df):
        self.logger.info(
            'Running Analysis Pipeline.')
        df_strat = self.stratified_random_sample(df)
        # self.identify_outliers(df)
        self.evaluate_skew(df)

        num_cols = df.select_dtypes(include=[np.number])
        # df_strat = self.apply_log(df_strat, num_cols.columns)
        # df_strat = self.apply_box_cox_1p(df_strat, num_cols.columns)
        df_strat = self.apply_box_cox(df_strat, num_cols.columns)
        # df_strat = self.apply_sqrt(df_strat, num_cols.columns)
        # df_strat = self.apply_inverse_sqrt(df_strat, num_cols.columns)
        # df_strat = self.apply_inv(df_strat, num_cols.columns)

        self.logger.info(
            'Analysis Pipeline Completed. Identified Outliers saved to: ``reports/analysis/outliers.json``')
        return df_strat


class Visualiser:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_pair_plot(self, df, cols, title):
        # df[cols] = np.log1p(df[cols].clip(lower=0))
        # df[cols] = boxcox(df[cols], 0)
        # df[cols] = boxcox1p(df[cols], 0)
        select_metrics = df[cols]
        sns.pairplot(
            select_metrics, diag_kind='kde',
            plot_kws={'alpha': 0.8, 's': 10})
        plt.title(f'Pair Plot of {title} Metrics {config['year']}')
        plt.savefig(f'reports/analysis/transforms/{title}.png')

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

        # for i,j in zip([raw, grow, vol, further], ['raw', 'grow', 'vol', 'further']):
        #     self.generate_pair_plot(df, i, f'2_t1_pp_{j}')

        # self.generate_heat_plot(df, raw_features, 'Raw')
        # self.generate_heat_plot(df, engineered_features, 'Engineered')
        # self.generate_heat_plot(df, df_single_date.columns, 'All')
        # self.generate_trends(df.sample(1000), 'EBIT', date_cols)
        self.logger.info(
            'Visualiser Pipeline Completed. Figures saved to: ``reports/analysis/figures & transforms/*.png``')


class SkewDetector:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_log(self, df, cols):
        df[cols] = df[cols].clip(lower=0).apply(np.log1p)
        return df

    def apply_box_cox_1p(self, df, cols):
        shifts = df[cols].min().apply(lambda x: 1 - x if x <= 0 else 0)
        df[cols] += shifts
        df[cols] = df[cols].apply(lambda x: boxcox1p(x, 0))
        return df

    def apply_box_cox(self, df, cols):
        shifts = df[cols].min().apply(lambda x: 1 - x if x <= 0 else 0)
        df[cols] += shifts
        df[cols] = df[cols].apply(lambda x: boxcox(x, 0))
        return df

    def apply_sqrt(self, df, cols):
        df[cols] = np.sqrt(df[cols].clip(lower=0))
        return df

    def apply_inverse_sqrt(self, df, cols):
        df[cols] = 1 / np.sqrt(df[cols].clip(lower=0))
        return df

    def apply_inv(self, df, cols):
        df[cols] = 1 / df[cols].clip(lower=0)
        return df

    def apply_transformation(self, df, cols, transform_func):
        return transform_func(df, cols)

    def analyze_skew_and_kurtosis(self, df, cols, transform_func, file_idx):
        skew_store = []
        for column in cols:
            original_skew = skew(df[column].dropna())
            original_kurtosis = kurtosis(df[column].dropna())

            # Apply transformation
            transformed_df = self.apply_transformation(
                df.copy(), [column], transform_func)
            transformed_skew = skew(transformed_df[column].dropna())
            transformed_kurtosis = kurtosis(transformed_df[column].dropna())

            skew_dict = {
                'Column': column,
                'Original Skew': round(original_skew, 2),
                'Transformed Skew': round(transformed_skew, 2),
                'Original Kurtosis': round(original_kurtosis, 2),
                'Transformed Kurtosis': round(transformed_kurtosis, 2)
            }
            skew_store.append(skew_dict)

        filepath = Path(f'reports/analysis/skew_kurt/{file_idx}.json')
        save_json(skew_store, filepath)

    def pipeline(self, df):
        raw, grow, vol, further = amend_features(config)
        cols = raw+grow+vol+further

        trans_funcs = [self.apply_log, self.apply_box_cox_1p, self.apply_box_cox,
                       self.apply_sqrt, self.apply_inverse_sqrt, self.apply_inv]
        trans_indx = ['log', 'cox1p', 'cox', 'sqrt', 'inv_sqrt', 'inv']
        for transform, idx in zip(trans_funcs, trans_indx):
            self.analyze_skew_and_kurtosis(df, cols, transform, idx)
        # self.analyze_skew_and_kurtosis(df, cols,self.apply_box_cox_1p)
