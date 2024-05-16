from __future__ import annotations

import logging

import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer

from utils.file_handler import load_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class StoreTransforms:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_log(self, df, cols):
        df[cols] = df[cols].clip(lower=0).apply(np.log1p)
        return df

    def apply_sqrt(self, df, cols):
        df[cols] = np.sqrt(df[cols].clip(lower=0))
        return df

    def apply_inv_sqrt(self, df, cols):
        eps = 1e-8
        df[cols] = 1 / np.sqrt(df[cols] + eps)
        return df

    def apply_inv(self, df, cols):
        eps = 1e-8
        df[cols] = 1 / (df[cols] + eps)
        return df

    def apply_power(self, df, cols, method='yeo-johnson'):
        pt = PowerTransformer(method=method)
        df[cols] = pt.fit_transform(df[[cols]])
        return df

    def get_transform_lists(self):
        trans_funcs = [
            self.apply_log, self.apply_sqrt,
            self.apply_inv_sqrt, self.apply_inv,
            self.apply_power]
        return trans_funcs

    def get_transform_map(self):
        trans_map = {
            'log': self.apply_log,
            'sqrt': self.apply_sqrt,
            'inv_sqrt': self.apply_inv_sqrt,
            'inv': self.apply_inv,
            'power': self.apply_power
        }
        return trans_map

    def get_transform_info(self):
        trans_map = self.get_transform_map()
        trans_funcs = self.get_transform_lists()
        return trans_map, trans_funcs


class ApplyTransforms:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calc_skew(self, df):
        return round((skew(df)).mean(), 2)

    def calc_kurtosis(self, df):
        return round((df.kurtosis()).mean(), 2)

    def cols_to_transform(self, optimal_transforms, shape_threshold=0.1):
        """Get columns that fall below Skew/Kurt threshold"""
        high_vals = {
            col: np.mean([abs(skew), abs(kurt)])
            for col, (transform, [skew, kurt]) in optimal_transforms.items()
            if np.mean([abs(skew), abs(kurt)]) > shape_threshold}
        return list(high_vals.keys())

    def apply_transforms(self, df_transform, optimal_transforms, trans_map):
        """Apply the optimal transform to each column in the dataframe"""
        for col in df_transform.columns:
            transform, _ = optimal_transforms[col]
            if transform in trans_map:
                df_transform = trans_map[transform](df_transform, col)
            else:
                self.logger.error('Transform not found: %s', transform)
                raise ValueError(f'Transform not found: {transform}')
        return df_transform

    def pipeline(self, df, trans_map, shape_threshold):
        self.logger.info(
            'Applying Distribution Transformations.')
        optimal_transforms = load_json(
            f'{config['path']['skew']}/transform_map.json')
        cols = self.cols_to_transform(optimal_transforms, shape_threshold)

        df_transform = df[cols].copy()
        pre_transform_skew = self.calc_skew(df_transform)
        pre_transform_kurtosis = self.calc_kurtosis(df_transform)

        df_transform = self.apply_transforms(
            df_transform, optimal_transforms, trans_map)

        post_transform_skew = self.calc_skew(df_transform)
        post_transform_kurtosis = self.calc_kurtosis(df_transform)
        df[cols] = df_transform[cols]

        self.logger.info(
            'Transforms applied. Pre-transform skew: %s. Post-transform skew: %s',
            pre_transform_skew, post_transform_skew)
        self.logger.info(
            'Transforms applied. Pre-transform kurtosis: %s. Post-transform kurtosis: %s',
            pre_transform_kurtosis, post_transform_kurtosis)
        return df
