from __future__ import annotations

import logging

from database.db_ops import DataBaseOps
from utils.setup_env import setup_project_env

project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class BuildFeatures:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_growth(self, df):
        # dates = range(2015, 2021)
        dates = ['2020', '2019', '2018', '2017', '2016', '2015']
        metric_cols = [col for col in df.columns if any(
            x in col for x in dates)]

        for metric in metric_cols:
            df[f'growth_{metric}'] = (
                df[f'{metric}'] - df[f'{metric}']) / df[f'{metric}']
        return df

    def build_volatility(self, df):
        dates = range(2015, 2021)
        metric_cols = [col for col in df.columns if any(
            x in col for x in dates)]

        for year, metric in zip(dates, metric_cols):
            df[f'volatility_{metric}.{year}'] = df[f'{
                metric}.{year}'].std(axis=1)
        return df

    def build_metrics(self, df):
        for year in range(2015, 2021):
            df[f'debt_to_eq{year}'] = df[f'TAsset.{
                year}'] / (df[f'TAsset.{year}'] - df[f'Leverage.{year}'])
            df[f'op_marg{year}'] = df[f'EBIT.{
                year}'] / df[f'Turnover.{year}']
            df[f'asset_turnover{year}'] = df[f'Turnover.{
                year}'] / df[f'TAsset.{year}']
            df[f'roa{year}'] = df[f'EBIT.{
                year}'] / df[f'TAsset.{year}']
        return df

    def pipeline(self, df):
        df = self.build_growth(df)
        # df = self.build_volatility(df)
        # df = self.build_metrics(df)
        # print(df.iloc[:,-8:])
        # print(df.shape)
        return df
