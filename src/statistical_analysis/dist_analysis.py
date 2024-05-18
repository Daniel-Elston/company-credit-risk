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
        self.dir_skew = Path(config['path']['skew'])

    def analyze_column(self, df, column, transform_func):
        original_skew = skew(df[column])
        original_kurtosis = kurtosis(df[column])

        transformed_df = transform_func(df, column)
        transformed_skew = skew(transformed_df[column])
        transformed_kurtosis = kurtosis(transformed_df[column])

        return {
            'Column': column,
            'Original Skew': round(original_skew, 2),
            'Transformed Skew': round(transformed_skew, 2),
            'Original Kurtosis': round(original_kurtosis, 2),
            'Transformed Kurtosis': round(transformed_kurtosis, 2)
        }

    def analyze_skew_and_kurtosis(self, df, cols, transform_func, transform_name):
        skew_store = [self.analyze_column(df, column, transform_func) for column in cols]
        filepath = Path(f'{self.dir_skew}/{transform_name}.json')
        save_json(skew_store, filepath)

    def pipeline(self, df, trans_map, **kwargs):
        cols = kwargs['cont']
        self.logger.info(
            'Generating Distribution Analysis Pipeline.')

        for transform_name, transform_func in trans_map.items():
            self.analyze_skew_and_kurtosis(df, cols, transform_func, transform_name)

        self.logger.info(
            f'Distribution Analysis Generation Completed. Data saved to: ``{self.dir_skew}/*.json``')


class EvaluateDistAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dir_skew = Path(config['path']['skew'])
        self.dir_maps = Path(config['path']['maps'])

    def load_results(self, trans_map):
        results = {}
        for transform_name in trans_map.keys():
            filepath = Path(f'{self.dir_skew}/{transform_name}.json')
            data = load_json(filepath)
            results[transform_name] = data
        return results

    def compile_transform_data(self, data):
        """Retrieve transformed skew values for each column and transform"""
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

    def retrieve_transform(self, column_transforms, skew_weight, kurt_weight):
        """Retrieve the transform with the lowest combined metric for each column."""
        optimal_transforms = {}

        for column, transforms in column_transforms.items():
            optimal_transform = self.identify_transform(transforms, skew_weight, kurt_weight)
            optimal_transforms[column] = (optimal_transform, transforms[optimal_transform])

        filepath = Path(f'{config["path"]["maps"]}/transform_map.json')
        save_json(optimal_transforms, filepath)
        return optimal_transforms

    def pipeline(self, trans_map, skew_weight, kurt_weight):
        self.logger.info(
            f'Running Evaluation of Distribution Analysis. Analysing files: ``{self.dir_skew}/*.json`')

        results = self.load_results(trans_map)
        compiled_data = self.compile_transform_data(results)
        self.retrieve_transform(compiled_data, skew_weight, kurt_weight)

        self.logger.info(
            f'Distribution Analysis Evaluation Completed. Results saved to: ``{self.dir_maps}/transform_map.json``')

    # def analyze_skew_and_kurtosis(self, df: dd.DataFrame, cols: list[str], transform_func, transform_name: str):
    #     skew_store = []
    #     for column in cols:
    #         result = delayed(self.analyze_column)(df, column, transform_func)
    #         skew_store.append(result)

    #     skew_store = delayed(skew_store).compute()
    #     table = pa.Table.from_pylist(skew_store)
    #     pq.write_table(table, f'{config["path"]["skew"]}/{transform_name}.parquet')
