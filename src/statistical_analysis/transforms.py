from __future__ import annotations

import logging

import numpy as np
from scipy.special import boxcox1p
from scipy.stats import boxcox
from scipy.stats import skew

from utils.config_ops import amend_features
from utils.file_handler import load_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class StoreTransforms:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_log(self, df, cols):
        df[cols] = df[cols].clip(lower=0).apply(np.log1p)
        return df

    def apply_box_cox_1p(self, df, col):
        shift = df[col].min()
        if shift <= 0:
            df[col] += (1 - shift)
        df[col] = boxcox1p(df[col], 0)
        return df

    def apply_box_cox(self, df, col):
        shift = df[col].min()
        if shift <= 0:
            df[col] += (1 - shift)
        df[col], _ = boxcox(df[col])
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

    def get_transform_lists(self):
        trans_funcs = [
            self.apply_log, self.apply_box_cox_1p, self.apply_box_cox,
            self.apply_sqrt, self.apply_inv_sqrt, self.apply_inv]
        return trans_funcs

    def get_transform_map(self):
        trans_map = {
            'log': self.apply_log,
            'cox1p': self.apply_box_cox_1p,
            'cox': self.apply_box_cox,
            'sqrt': self.apply_sqrt,
            'inv_sqrt': self.apply_inv_sqrt,
            'inv': self.apply_inv
        }
        return trans_map

    def get_transform_info(self):
        trans_map = self.get_transform_map()
        trans_funcs = self.get_transform_lists()
        return trans_map, trans_funcs


class ApplyTransforms:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_transformation(self, df, cols, transform_func):
        return transform_func(df, cols)

    def pipeline(self, df, trans_map):
        self.logger.info(
            'Applying Distribution Transformations.')
        optimal_transforms = load_json('reports/analysis/transform_map.json')

        raw, grow, vol, further = amend_features(config)
        cols = raw+grow+vol+further
        df_transform = df[cols]

        pre_transform_skew = round((skew(df_transform)).mean(), 2)
        for col, (transform, _) in optimal_transforms.items():
            if transform in trans_map:
                df_transform = trans_map[transform](df_transform, col)
        post_transform_skew = round((skew(df_transform)).mean(), 2)

        df[cols] = df_transform[cols]
        self.logger.info(
            'Transforms applied. Pre-transform skew: %s. Post-transform skew: %s', pre_transform_skew, post_transform_skew)
        return df
