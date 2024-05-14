from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils.file_handler import save_json
from utils.setup_env import setup_project_env
# from utils.file_handler import load_json
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
