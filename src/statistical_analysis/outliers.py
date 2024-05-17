from __future__ import annotations

import logging

import numpy as np
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

    def remove_outliers_zscore(self, df, cols):
        z_scores = np.abs(zscore(df[cols]))
        return df[(z_scores < self.ss.zscore_threshold).all(axis=1)]

    def remove_outliers_iqr(self, df, cols):
        Q1 = df[cols].quantile(self.ss.iqr_quantiles[0])
        Q3 = df[cols].quantile(self.ss.iqr_quantiles[1])
        IQR = Q3 - Q1
        Q1_thresh = Q1 - self.ss.iqr_factor * IQR
        Q3_thresh = Q3 + self.ss.iqr_factor * IQR
        return df[~((df[cols] < Q1_thresh) | (df[cols] > Q3_thresh)).any(axis=1)]

    # def winsorize(self, df, cols, win_quantiles=(0.01, 0.99)):
    #     for col in cols:
    #         lower_bound = df[col].quantile(win_quantiles[0])
    #         upper_bound = df[col].quantile(win_quantiles[1])
    #         df[col] = np.clip(df[col], lower_bound, upper_bound)
    #     return df

    def remove_outliers_isolation_forest(self, df, cols):
        iso_forest = IsolationForest(
            contamination=self.ss.iso_contamination, random_state=self.ss.random_state)
        outliers = iso_forest.fit_predict(df[cols])
        return df[outliers != -1]

    def remove_outliers_lof(self, df, cols):
        lof = LocalOutlierFactor(
            n_neighbors=self.ss.lof_n_neighbors, contamination=self.ss.lof_contamination)
        outliers = lof.fit_predict(df[cols])
        return df[outliers != -1]

    def remove_outliers_dbscan(self, df, cols):
        clustering = DBSCAN(
            eps=self.ss.dbscan_eps, min_samples=self.ss.dbscan_min_samples).fit(df[cols])
        return df[clustering.labels_ != -1]

    def replace_outliers(self, df, cols, method='zscore'):
        if method == 'zscore':
            return self.remove_outliers_zscore(df, cols)
        elif method == 'iqr':
            return self.remove_outliers_iqr(df, cols)
        # elif method == 'winsorize':
        #     return self.winsorize(df, cols, **kwargs)
        elif method == 'iso':
            return self.remove_outliers_isolation_forest(df, cols)
        elif method == 'lof':
            return self.remove_outliers_lof(df, cols)
        elif method == 'dbscan':
            return self.remove_outliers_dbscan(df, cols)
        else:
            raise ValueError("Unsupported outlier detection method.")

    def pipeline(self, df, continuous, discrete):
        self.logger.debug(
            'Running HandleOutliers pipeline. Data shape: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='lof')
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after lof: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='iqr')
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after iqr: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='zscore')
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after zscore: %s', df.shape)

        # df = self.replace_outliers(df, long_tails, method='zscore', threshold=4.0)
        # self.logger.debug('HandleOutliers pipeline complete. Data shape after SECOND zscore: %s', df.shape)

        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape: %s', df.shape)

        return df
