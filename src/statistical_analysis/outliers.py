from __future__ import annotations

import logging

import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from utils.config_ops import continuous_discrete
from utils.setup_env import setup_project_env
# from utils.config_ops import get_feature_columns
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

    # def get_cols(self, df):
    #     continuous, discrete = continuous_discrete(config, df)
    #     growth_cols = get_feature_columns(continuous, 'growth_', False)
    #     cont_no_growth = [col for col in continuous if col not in growth_cols]

    #     # volatility = get_feature_columns(continuous, 'volatility', False)

    #     growth_ebits = get_feature_columns(continuous, 'growth_EBIT', False)
    #     growth_pltax = get_feature_columns(continuous, 'growth_PLTax', False)
    #     growth_roe = get_feature_columns(continuous, 'growth_ROE', False)
    #     ebits = get_feature_columns(continuous, 'EBIT.', True)
    #     pltax = get_feature_columns(continuous, 'PLTax.', True)
    #     leverage = get_feature_columns(continuous, 'Leverage.', True)
    #     long_tails = growth_ebits + growth_pltax + growth_roe + ebits + pltax + leverage

    #     debt_to_eq = get_feature_columns(continuous, 'debt_to_eq', False)
    #     op_marg = get_feature_columns(continuous, 'op_marg', False)
    #     roa = get_feature_columns(continuous, 'roa', False)
    #     growth_cols = get_feature_columns(continuous, 'growth_', False)
    #     ebits = get_feature_columns(continuous, 'EBIT.', True)
    #     cols = debt_to_eq + op_marg + roa + growth_cols + ebits

    #     return continuous, cont_no_growth, growth_cols, long_tails, cols

    def pipeline(self, df):
        self.logger.debug(
            'Running HandleOutliers pipeline. Data shape: %s', df.shape)

        # continuous, cont_no_growth, growth_cols, long_tails, cols = self.get_cols(
        #     df)
        continuous, discrete = continuous_discrete(config, df)

        df = self.replace_outliers(
            df, continuous, method='lof', contamination=0.5)
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after lof: %s', df.shape)

        df = self.replace_outliers(df, continuous, method='iqr', factor=6.0)
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after iqr: %s', df.shape)

        # df = self.replace_outliers(df, long_tails, method='iqr', factor=4.0)
        # self.logger.debug(
        #     'HandleOutliers pipeline complete. Data shape after SECOND iqr: %s', df.shape)

        df = self.replace_outliers(
            df, continuous, method='zscore', threshold=12.0)
        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape after zscore: %s', df.shape)

        # df = self.replace_outliers(df, long_tails, method='zscore', threshold=4.0)
        # self.logger.debug('HandleOutliers pipeline complete. Data shape after SECOND zscore: %s', df.shape)

        self.logger.debug(
            'HandleOutliers pipeline complete. Data shape: %s', df.shape)

        return df
