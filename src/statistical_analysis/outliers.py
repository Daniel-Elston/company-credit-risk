from __future__ import annotations

import logging

import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class HandleOutliers:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def remove_outliers_zscore(self, df, cols, threshold=3):
        z_scores = np.abs(zscore(df[cols]))
        return df[(z_scores < threshold).all(axis=1)]

    def remove_outliers_iqr(self, df, cols, factor=1.5):
        Q1 = df[cols].quantile(0.05)
        Q3 = df[cols].quantile(0.95)
        IQR = Q3 - Q1
        return df[~((df[cols] < (Q1 - factor * IQR)) | (df[cols] > (Q3 + factor * IQR))).any(axis=1)]

    def winsorize(self, df, cols, quantiles=(0.01, 0.99)):
        for col in cols:
            lower_bound = df[col].quantile(quantiles[0])
            upper_bound = df[col].quantile(quantiles[1])
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def remove_outliers_isolation_forest(self, df, cols, contamination=0.01):
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(df[cols])
        return df[outliers != -1]

    def remove_outliers_lof(self, df, cols):
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        outliers = lof.fit_predict(df[cols])
        return df[outliers != -1]

    def remove_outliers_dbscan(self, df, cols, eps=0.5, min_samples=5):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df[cols])
        return df[clustering.labels_ != -1]

    def replace_outliers(self, df, cols, method='zscore', **kwargs):
        if method == 'zscore':
            return self.remove_outliers_zscore(df, cols, **kwargs)
        elif method == 'iqr':
            return self.remove_outliers_iqr(df, cols, **kwargs)
        elif method == 'winsorize':
            return self.winsorize(df, cols, **kwargs)
        elif method == 'iso':
            return self.remove_outliers_isolation_forest(df, cols, **kwargs)
        elif method == 'lof':
            return self.remove_outliers_lof(df, cols)
        elif method == 'dbscan':
            return self.remove_outliers_dbscan(df, cols, **kwargs)
        else:
            raise ValueError("Unsupported outlier detection method.")

    def pipeline(self, df, continuous, discrete):
        self.logger.debug(
            'Running HandleOutliers pipeline. Data shape: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='lof', contamination=0.5)
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after lof: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='iqr', factor=6.0)
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after iqr: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='zscore', threshold=12.0)
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after zscore: %s', df.shape)

        # df = self.replace_outliers(df, long_tails, method='zscore', threshold=4.0)
        # self.logger.debug('HandleOutliers pipeline complete. Data shape after SECOND zscore: %s', df.shape)

        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape: %s', df.shape)

        return df
