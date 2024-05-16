from __future__ import annotations

import logging

import numpy as np

from utils.setup_env import setup_project_env
# from database.db_ops import DataBaseOps

project_dir, config, setup_logs = setup_project_env()
# creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class BuildFeatures:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_growth(self, df, metric_cols, date_cols):
        for idx in range(1, len(date_cols)):
            current_year = date_cols[idx - 1]
            previous_year = date_cols[idx]

            for metric in metric_cols:
                current_metric = f'{metric}.{current_year}'
                previous_metric = f'{metric}.{previous_year}'

                df[f'growth_{metric}{current_year}'] = (
                    df[current_metric] - df[previous_metric]) / (df[previous_metric]).replace(0, np.nan)

                mean = df[f'growth_{metric}{current_year}'].mean()
                df[f'growth_{metric}{current_year}'] = df[f'growth_{
                    metric}{current_year}'].fillna(mean)
        return df

    def build_volatility(self, df, metric_cols, date_cols):
        for metric in metric_cols:
            cols = [f'{metric}.{year}' for year in date_cols if f'{
                metric}.{year}' in df.columns]
            df[f'volatility_{metric}'] = df[cols].std(axis=1)
        return df

    def build_metrics(self, df, date_cols):
        eps = 1e-6
        for year in date_cols:
            df[f'fur_debt_to_eq{year}'] = df[f'TAsset.{year}'] / \
                (df[f'TAsset.{year}'] - df[f'Leverage.{year}'] + eps)
            df[f'fur_op_marg{year}'] = df[f'EBIT.{year}'] / \
                (df[f'Turnover.{year}'] + eps)
            df[f'fur_asset_turnover{year}'] = df[f'Turnover.{
                year}'] / (df[f'TAsset.{year}'] + eps)
            df[f'fur_roa{year}'] = df[f'EBIT.{year}'] / \
                (df[f'TAsset.{year}'] + eps)
        return df

    def pipeline(self, df, metric_cols, date_cols):
        self.logger.debug(
            'Running BuildFeatures pipeline. Dataframe shape: %s', df.shape)
        df = self.build_growth(df, metric_cols, date_cols)
        df = self.build_volatility(df, metric_cols, date_cols)
        df = self.build_metrics(df, date_cols)
        self.logger.debug(
            'BuildFeatures pipeline complete. New dataframe shape: %s', df.shape)
        return df
