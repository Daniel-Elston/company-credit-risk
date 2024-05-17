from __future__ import annotations

import logging
from pathlib import Path

from scipy.stats import kurtosis
from scipy.stats import skew

from utils.file_handler import load_json
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class GenerateDistAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_skew_and_kurtosis(self, df, cols, transform_func, transform_name):
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

        filepath = Path(f'{config['path']['skew']}/{transform_name}.json')
        save_json(skew_store, filepath)

    def pipeline(self, df, trans_map, **kwargs):
        cols = kwargs.get('cont')
        self.logger.info(
            'Generating Distribution Analysis Pipeline.')

        for transform_name, transform_func in trans_map.items():
            self.analyze_skew_and_kurtosis(df, cols, transform_func, transform_name)

        self.logger.info(
            f'Distribution Analysis Generation Completed. Data saved to: ``{config['path']['skew']}/*.json``')


class EvaluateDistAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_results(self, trans_map):
        results = {}
        for transform in trans_map.keys():
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
                kurtosis_value = record['Transformed Kurtosis']
                if column not in column_transforms:
                    column_transforms[column] = {}
                column_transforms[column][transform] = (
                    skew_value, kurtosis_value)
        return column_transforms

    def get_optimal_transform(self, column_transforms, skew_weight, kurt_weight):
        """Retrieve the transform with the lowest combined metric for each column"""
        optimal_transforms = {}
        for column, transforms in column_transforms.items():
            min_metric = float('inf')
            optimal_transform = None
            for transform, (skew_value, kurtosis_value) in transforms.items():
                combined_metric = abs(skew_value)*skew_weight + \
                    abs(kurtosis_value)*kurt_weight
                if combined_metric < min_metric:
                    min_metric = combined_metric
                    optimal_transform = transform
            optimal_transforms[column] = (
                optimal_transform, transforms[optimal_transform])

        filepath = Path(f'{config["path"]["skew"]}/transform_map.json')
        save_json(optimal_transforms, filepath)
        return optimal_transforms

    def pipeline(self, trans_map, skew_weight, kurt_weight):
        self.logger.info(
            f'Running Evaluation of Distribution Analysis. Analysing files: ``{config['path']['skew']}/*.json`')

        results = self.load_results(trans_map)
        compiled_data = self.compile_transform_data(results)
        self.get_optimal_transform(compiled_data, skew_weight, kurt_weight)

        self.logger.info(
            f'Distribution Analysis Evaluation Completed. Results saved to: ``{config['path']['maps']}/transform_map.json``')
