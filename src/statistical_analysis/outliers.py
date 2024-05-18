from __future__ import annotations

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import StatisticState
from utils.setup_env import setup_project_env

project_dir, config, setup_logs = setup_project_env()


class HandleOutliers:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ss = StatisticState()

    def _compute_zscore_mask(self, df, cols) -> pd.Series:
        z_scores = np.abs(zscore(df[cols]))
        return (z_scores < self.ss.zscore_threshold).all(axis=1)

    def _compute_iqr_mask(self, df, cols) -> pd.Series:
        Q1 = df[cols].quantile(self.ss.iqr_quantiles[0])
        Q3 = df[cols].quantile(self.ss.iqr_quantiles[1])
        IQR = Q3 - Q1
        Q1_thresh = Q1 - self.ss.iqr_factor * IQR
        Q3_thresh = Q3 + self.ss.iqr_factor * IQR
        return ~((df[cols] < Q1_thresh) | (df[cols] > Q3_thresh)).any(axis=1)

    def _compute_isolation_forest_mask(self, df, cols) -> pd.Series:
        iso_forest = IsolationForest(contamination=self.ss.iso_contamination, random_state=self.ss.random_state)
        outliers = iso_forest.fit_predict(df[cols])
        return outliers != -1

    def _compute_lof_mask(self, df, cols) -> pd.Series:
        lof = LocalOutlierFactor(n_neighbors=self.ss.lof_n_neighbors, contamination=self.ss.lof_contamination)
        outliers = lof.fit_predict(df[cols])
        return outliers != -1

    def _compute_dbscan_mask(self, df, cols) -> pd.Series:
        clustering = DBSCAN(eps=self.ss.dbscan_eps, min_samples=self.ss.dbscan_min_samples).fit(df[cols])
        return clustering.labels_ != -1

    def partition_mask(self, df, cols, mask_func):
        mask = mask_func(df, cols)
        return df[mask]

    def replace_outliers(self, ddf: dd.DataFrame, cols, method) -> dd.DataFrame:
        method_function_mapping = {
            'zscore': self._compute_zscore_mask,
            'iqr': self._compute_iqr_mask,
            'iso': self._compute_isolation_forest_mask,
            'lof': self._compute_lof_mask,
            'dbscan': self._compute_dbscan_mask,
        }
        mask_func = method_function_mapping[method]
        def partition_mask(df): return self.partition_mask(df, cols, mask_func)

        return ddf.map_partitions(partition_mask, meta=ddf)

    def pipeline(self, ddf: dd.DataFrame, **kwargs) -> dd.DataFrame:
        cont = kwargs.get('cont')
        self.logger.info('Running Outlier Handling pipeline.')

        for method in ['lof', 'iqr', 'zscore']:
            ddf = self.replace_outliers(ddf, cont, method=method)
            self.logger.debug('Data shape after %s: %s', method, ddf.shape)

        self.logger.info('Outlier Handling pipeline complete.')
        return ddf


#     def pipeline(self, df, **kwargs):
#         cont = kwargs.get('cont')
#         self.logger.info(
#             'Running Outlier Handling pipeline.')
#         initial_shape = df.shape

#         # df = self.replace_outliers(df, cont, method='lof')
#         # self.logger.debug(
#         #     'Data shape after lof: %s', df.shape)

#         # df = self.replace_outliers(df, cont, method='iqr')
#         # self.logger.debug(
#         #     'Data shape after iqr: %s', df.shape)

#         # df = self.replace_outliers(df, cont, method='zscore')
#         # self.logger.debug(
#         #     'Data shape after zscore: %s', df.shape)
#         for method in ['lof', 'iqr', 'zscore']:
#             df = self.replace_outliers(df, cont, method=method)
#             self.logger.debug('Data shape after %s: %s', method, df.shape)

#         processed_shape = df.shape
#         shape_diff = (initial_shape[0] - processed_shape[0], initial_shape[1] - processed_shape[1])
#         self.logger.debug(
#             'Initial Shape: %s, Processed Shape: %s, Shape Difference: %s (Rows Removed: %s, Columns Changed: %s)',
#             initial_shape, processed_shape, shape_diff, shape_diff[0], shape_diff[1])
#         self.logger.info(
#             'Outlier Handling pipeline complete.')
#         return df
