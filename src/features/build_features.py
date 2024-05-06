from __future__ import annotations

import logging

from database.db_ops import DataBaseOps
from utils.setup_env import setup_project_env

project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class BuildFeatures:
    def __init__(self, df):
        self.df = df
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_growth(self):
        # dates = range(2015, 2021)
        dates = ['2020', '2019', '2018', '2017', '2016', '2015']
        metric_cols = [col for col in self.df.columns if any(
            x in col for x in dates)]

        for metric in metric_cols:
            self.df[f'growth_{metric}'] = (
                self.df[f'{metric}'] - self.df[f'{metric}']) / self.df[f'{metric}']

    def build_volatility(self):
        dates = range(2015, 2021)
        metric_cols = [col for col in self.df.columns if any(
            x in col for x in dates)]

        for year, metric in zip(dates, metric_cols):
            self.df[f'volatility_{metric}.{year}'] = self.df[f'{
                metric}.{year}'].std(axis=1)

    def build_metrics(self):
        for year in range(2015, 2021):
            self.df[f'debt_to_eq{year}'] = self.df[f'TAsset.{
                year}'] / (self.df[f'TAsset.{year}'] - self.df[f'Leverage.{year}'])
            self.df[f'op_marg{year}'] = self.df[f'EBIT.{
                year}'] / self.df[f'Turnover.{year}']
            self.df[f'asset_turnover{year}'] = self.df[f'Turnover.{
                year}'] / self.df[f'TAsset.{year}']
            self.df[f'roa{year}'] = self.df[f'EBIT.{
                year}'] / self.df[f'TAsset.{year}']

    def pipeline(self):
        self.build_growth()
        # self.build_volatility(df)
        # self.build_metrics(df)
        # print(self.df.iloc[:,-8:])
        # print(self.df.shape)
        return self.df
