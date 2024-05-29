from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.setup_env import setup_project_env

project_dir, config, setup_logs = setup_project_env()


class BuildFeatures:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_mean_years(self, df, metric_cols):
        for metric in metric_cols:
            cols = df.columns[df.columns.str.contains(f'{metric}.')]
            df[f'{metric}_mean'] = df[cols].mean(axis=1)
        return df

    def build_growth(self, df, metric_cols, date_cols):
        growth_data = {}

        for idx in range(1, len(date_cols)):
            current_year = date_cols[idx - 1]
            previous_year = date_cols[idx]

            for metric in metric_cols:
                current_metric = f'{metric}.{current_year}'
                previous_metric = f'{metric}.{previous_year}'
                growth_col = f'growth_{metric}_{current_year}'

                growth_data[growth_col] = (
                    df[current_metric] - df[previous_metric]) / (
                    df[previous_metric]).replace(0, np.nan)

                mean = growth_data[growth_col].mean()
                growth_data[growth_col] = (
                    growth_data[growth_col].fillna(mean))

        growth_df = pd.DataFrame(growth_data)
        for metric in metric_cols:
            growth_cols = [col for col in growth_df.columns if f'growth_{metric}_' in col]
            df[f'growth_{metric}_mean'] = growth_df[growth_cols].mean(axis=1)
        return df

    def build_volatility(self, df, metric_cols, date_cols):
        for metric in metric_cols:
            cols = [f'{metric}.{year}' for year in date_cols if f'{metric}.{year}' in df.columns]
            df[f'volatility_{metric}'] = df[cols].std(axis=1)
        return df

    def build_metrics(self, df):
        eps = 1e-6
        df['fur_debt_to_eq'] = (
            df['TAsset_mean'] / (
                df['TAsset_mean'] - df['Leverage_mean'] + eps)
        )
        df['fur_op_marg'] = (
            df['EBIT_mean'] / (
                df['Turnover_mean'] + eps)
        )
        df['fur_asset_turnover'] = (
            df['Turnover_mean'] / (
                df['TAsset_mean'] + eps)
        )
        df['fur_roa'] = (
            df['EBIT_mean'] / (
                df['TAsset_mean'] + eps)
        )
        return df

    def pipeline(self, df):
        self.logger.info(
            'Running Feature Building pipeline.')
        initial_shape = df.shape

        metric_cols = ['MScore', 'TAsset', 'Leverage', 'EBIT', 'Turnover', 'ROE', 'PLTax']
        date_cols = ['2020', '2019', '2018', '2017', '2016', '2015']

        df = self.get_mean_years(df, metric_cols)

        df = self.build_growth(df, metric_cols, date_cols)
        df = self.build_volatility(df, metric_cols, date_cols)
        df = self.build_metrics(df)

        processed_shape = df.shape
        shape_diff = (processed_shape[0] - initial_shape[0], processed_shape[1] - initial_shape[1])
        self.logger.debug(
            'Initial Shape: %s, Processed Shape: %s, Shape Difference: %s (Rows Removed: %s, Columns Changed: %s)',
            initial_shape, processed_shape, shape_diff, shape_diff[0], shape_diff[1])
        self.logger.info(
            'Feature Building pipeline complete.')
        return df
