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
        self.save_path = Path(config['path']['skew'])

    def analyze_column(self, df, col, transformed_df):
        """Get pre/post transform measures for skew and kurtosis"""
        original_skew = skew(df)
        original_kurtosis = kurtosis(df)

        transformed_skew = skew(transformed_df)
        transformed_kurtosis = kurtosis(transformed_df)

        return {
            'Column': col,
            'Original Skew': round(original_skew, 2),
            'Transformed Skew': round(transformed_skew, 2),
            'Original Kurtosis': round(original_kurtosis, 2),
            'Transformed Kurtosis': round(transformed_kurtosis, 2)
        }

    def analyze_skew_and_kurtosis(self, df, cols, transform_func, transform_name):
        """Passes each column to transform_func and returns skew and kurtosis analysis"""
        skew_store = []
        for col in cols:
            transformed_df = transform_func(df, col)
            result = self.analyze_column(df[col], col, transformed_df[col])
            skew_store.append(result)

        filepath = Path(f'{self.save_path}/{transform_name}.json')
        save_json(skew_store, filepath)

    def pipeline(self, df, trans_map, **feature_groups):
        cols = feature_groups['cont']
        self.logger.info(
            'Generating Distribution Analysis Pipeline.')

        for transform_name, transform_func in trans_map.items():
            self.analyze_skew_and_kurtosis(df, cols, transform_func, transform_name)

        self.logger.info(
            f'Distribution Analysis Generation Completed. Data saved to: ``{self.save_path}/*.json``')


class EvaluateDistAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_path = Path(config['path']['skew'])
        self.save_path = Path(config['path']['maps'])

    def load_results(self, trans_map):
        results = {}
        for transform_name in trans_map.keys():
            filepath = Path(f'{self.load_path}/{transform_name}.json')
            data = load_json(filepath)
            results[transform_name] = data
        return results

    def compile_transform_data(self, data):
        """Compile data into column, transform_name: (skew, kurtosis) dict"""
        column_transforms = {}
        for transform_name, records in data.items():
            for r in records:
                column = r['Column']
                column_transforms.setdefault(column, {})[transform_name] = (
                    r['Transformed Skew'], r['Transformed Kurtosis'])
        return column_transforms

    def get_combined_metric(self, skew_value, kurtosis_value, skew_weight, kurt_weight):
        return abs(skew_value)*skew_weight + abs(kurtosis_value)*kurt_weight

    def identify_transform(self, transforms, skew_weight, kurt_weight):
        """Find the optimal transform with the lowest combined metric."""
        min_metric = float('inf')
        optimal_transform = None
        for transform, (skew_value, kurtosis_value) in transforms.items():
            combined_metric = self.get_combined_metric(skew_value, kurtosis_value, skew_weight, kurt_weight)
            if combined_metric < min_metric:
                min_metric = combined_metric
                optimal_transform = transform
        return optimal_transform

    def retrieve_transform(self, compiled_data, skew_weight, kurt_weight):
        """Retrieve the transform with the lowest combined metric for each column."""
        optimal_transforms = {}
        for column, transforms in compiled_data.items():
            optimal_transform = self.identify_transform(transforms, skew_weight, kurt_weight)
            optimal_transforms[column] = (optimal_transform, transforms[optimal_transform])

        filepath = Path(f'{self.save_path}/transform_map.json')
        save_json(optimal_transforms, filepath)
        return optimal_transforms

    def pipeline(self, trans_map, skew_weight, kurt_weight):
        self.logger.info(
            f'Running Evaluation of Distribution Analysis. Analysing files: ``{self.load_path}/*.json`')

        results = self.load_results(trans_map)
        compiled_data = self.compile_transform_data(results)
        self.retrieve_transform(compiled_data, skew_weight, kurt_weight)

        self.logger.info(
            f'Distribution Analysis Evaluation Completed. Results saved to: ``{self.save_path}/transform_map.json``')
