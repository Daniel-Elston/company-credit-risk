from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer

from utils.file_handler import load_json
from utils.setup_env import setup_project_env

project_dir, config, setup_logs = setup_project_env()


@dataclass
class StoreTransforms:
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    trans_map: dict = field(init=False)

    def __post_init__(self):
        self.trans_map = self.get_transform_map()
        self.logger.info("Transforms initialized from file: ``src/statistical_analysis/transforms.py``.")

    # def apply_log(self, df, cols: list):
    #     df[cols] = df[cols].clip(lower=0).apply(np.log1p)
    #     return df

    def apply_sqrt(self, df, cols: list):
        df[cols] = np.sqrt(df[cols].clip(lower=0))
        return df

    def apply_inv_sqrt(self, df, cols: list):
        eps = 1e-8
        df[cols] = 1 / np.sqrt(df[cols] + eps)
        return df

    def apply_inv(self, df, cols: list):
        eps = 1e-8
        df[cols] = 1 / (df[cols] + eps)
        return df

    def apply_power(self, df, cols: list, method: str = 'yeo-johnson'):
        pt = PowerTransformer(method=method)
        df[cols] = pt.fit_transform(df[[cols]])
        return df

    def get_transform_map(self):
        return {
            # 'log': self.apply_log,
            'sqrt': self.apply_sqrt,
            'inv_sqrt': self.apply_inv_sqrt,
            'inv': self.apply_inv,
            'power': self.apply_power
        }


class ApplyTransforms:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_path = Path(config['path']['maps'])

    def _calc_skew(self, df):
        return round((skew(df.dropna())).mean(), 2)

    def _calc_kurtosis(self, df):
        return round((kurtosis(df.dropna())).mean(), 2)

    def cols_to_transform(self, optimal_transforms, shape_threshold):
        """Get columns that fall below Skew/Kurt threshold"""
        high_vals = {
            col: np.mean([abs(skew), abs(kurt)])
            for col, (transform, [skew, kurt]) in optimal_transforms.items()
            if np.mean([abs(skew), abs(kurt)]) >= shape_threshold}
        return list(high_vals.keys())

    def apply_transforms(self, df_transform, optimal_transforms, trans_map):
        """Apply the optimal transform to each column in the dataframe"""
        for col in df_transform.columns:
            transform, _ = optimal_transforms.get(col, (None, None))
            if transform in trans_map:
                df_transform = trans_map[transform](df_transform, col)
            else:
                self.logger.warning('No transform found for column: %s', col)
        return df_transform

    def pipeline(self, df, trans_map, shape_threshold):
        self.logger.info('Applying Distribution Transformations.')
        optimal_transforms = load_json(Path(f'{self.load_path}/transform_map.json'))
        cols = self.cols_to_transform(optimal_transforms, shape_threshold)

        if not cols:
            self.logger.warning("No columns meet the shape threshold criteria.")
            return df

        pre_transform_skew = self._calc_skew(df[cols])
        pre_transform_kurtosis = self._calc_kurtosis(df[cols])

        df_transform = self.apply_transforms(df[cols].copy(), optimal_transforms, trans_map)

        post_transform_skew = self._calc_skew(df_transform)
        post_transform_kurtosis = self._calc_kurtosis(df_transform)

        df[cols] = df_transform[cols]

        self.logger.debug('Pre-transform skew: %s. Post-transform skew: %s', pre_transform_skew, post_transform_skew)
        self.logger.debug('Pre-transform kurtosis: %s. Post-transform kurtosis: %s', pre_transform_kurtosis, post_transform_kurtosis)

        self.logger.info('Distribution Transformations Applied')
        return df
