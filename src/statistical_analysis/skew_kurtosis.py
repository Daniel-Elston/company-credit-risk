from __future__ import annotations

import logging
from pathlib import Path

from scipy.stats import kurtosis
from scipy.stats import skew

from utils.file_handler import load_json
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


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

        filepath = Path(f'{config['path']['skew']}/{file_idx}.json')
        save_json(skew_store, filepath)

    def pipeline(self, df, cols, transform_funcs):
        self.logger.info(
            'Running Skew Generation and Analysis Pipeline.')

        for transform, idx in zip(transform_funcs, config['trans_idx']):
            self.analyze_skew_and_kurtosis(df, cols, transform, idx)

        self.logger.info(
            f'Skew Generation and Analysis Pipeline Completed. Data saved to: ``{config['path']['skew']}/*.json``')


class EvaluateSkewAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_results(self):
        results = {}
        for transform in config['trans_idx']:
            filepath = Path(f'{config['path']['skew']}/{transform}.json')
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

            filepath = Path(f'{config['path']['skew']}/transform_map.json')
            save_json(optimal_transforms, filepath)
        return optimal_transforms

    def pipeline(self):
        self.logger.info(
            f'Running Skew Evaluation and Analysis Pipeline. Analysing files: ``{config['path']['skew']}/*.json`')

        results = self.load_results()
        compiled_data = self.compile_transform_data(results)
        self.get_optimal_transform(compiled_data)

        self.logger.info(
            f'Skew Evaluation and Analysis Pipeline Completed. Results saved to: ``{config['path']['maps']}/transform_map.json``')
