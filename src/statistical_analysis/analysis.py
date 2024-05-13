from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

from utils.file_handler import load_json
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class Sampling:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def stratified_random_sample(self, df, seed=42):
        df_strat = df.groupby('Sector 1', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(0.1 * len(x))), random_state=seed))
        return df_strat


class OutlierDetection:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

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

    def pipeline(self, df):
        self.logger.info(
            'Running Analysis Pipeline.')

        self.identify_outliers(df)

        self.logger.info(
            'Analysis Pipeline Completed. Identified Outliers saved to: ``reports/analysis/outliers.json``')


class GenerateSkewAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_transformation(self, df, cols, transform_func):
        return transform_func(df, cols)

    def analyze_skew_and_kurtosis(self, df, cols, transform_func, file_idx):
        skew_store = []
        for column in cols:
            original_skew = skew(df[column].dropna())
            original_kurtosis = kurtosis(df[column].dropna())

            transformed_df = transform_func(df.copy(), column)

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

        filepath = Path(f'reports/analysis/skew_analysis/{file_idx}.json')
        save_json(skew_store, filepath)

    def pipeline(self, df, cols, transform_funcs):
        self.logger.info(
            'Running Skew Generation and Analysis Pipeline.')

        # raw, grow, vol, further = amend_features(config)
        # cols = raw+grow+vol+further
        for transform, idx in zip(transform_funcs, config['trans_idx']):
            self.analyze_skew_and_kurtosis(df, cols, transform, idx)

        self.logger.info(
            'Skew Generation and Analysis Pipeline Completed. Data saved to: ``reports/analysis/skew_analysis/*.json``')


class EvaluateSkewAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_results(self):
        results = {}
        for transform in config['trans_idx']:
            filepath = Path(f'reports/analysis/skew_analysis/{transform}.json')
            data = load_json(filepath)
            results[transform] = data
        return results

    def compile_transform_data(self, data):
        """Retrieve transformed skew values for each column and transform"""
        column_transforms = {}
        for transform, records in data.items():
            for record in records:
                column = record['Column']
                skew_value = record['Transformed Skew']
                if column not in column_transforms:
                    column_transforms[column] = {}
                column_transforms[column][transform] = skew_value
        return column_transforms

    def get_optimal_transform(self, column_transforms):
        """Retrieve the transform with the lowest absolute skew for each column"""
        optimal_transforms = {}
        for column, transforms in column_transforms.items():
            min_skew = float('inf')
            optimal_transform = None
            for transform, skew_value in transforms.items():
                if abs(skew_value) < min_skew:
                    min_skew = abs(skew_value)
                    optimal_transform = transform
            optimal_transforms[column] = (
                optimal_transform, transforms[optimal_transform])

            filepath = Path('reports/analysis/transform_map.json')
            save_json(optimal_transforms, filepath)
        return optimal_transforms

    def pipeline(self):
        self.logger.info(
            'Running Skew Evaluation and Analysis Pipeline. Analysing files: ``reports/analysis/skew_analysis/*.json`')

        results = self.load_results()
        compiled_data = self.compile_transform_data(results)
        self.get_optimal_transform(compiled_data)

        self.logger.info(
            'Skew Evaluation and Analysis Pipeline Completed. Results saved to: ``reports/analysis/transform_map.json``')
