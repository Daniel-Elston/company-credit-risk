from __future__ import annotations

import logging

import pandas as pd
import pyarrow.parquet as pq

from database.db_ops import DataBaseOps
from utils.setup_env import setup_project_env
# import matplotlib.pyplot as plt

project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class Exploration:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.table, self.df, self.report = self.load_data()
        self.df_strat = self.stratified_random_sample()
        self.df_by_metric = self.combine_dt()

    def load_data(self):
        table = pq.read_table('data/interim/combined.parquet')
        report = pd.read_excel('reports/analysis/report.xlsx')
        df = table.to_pandas()
        return table, df, report

    def stratified_random_sample(self):
        df_strat = self.df.groupby('Sector 2').apply(
            lambda x: x.sample(frac=0.1))
        return df_strat

    def combine_dt(self):
        turnover = self.df[['Company name', 'Turnover.2020', 'Turnover.2019',
                            'Turnover.2018', 'Turnover.2017', 'Turnover.2016', 'Turnover.2015']]
        earnings_bit = self.df[['Company name', 'EBIT.2020', 'EBIT.2019',
                                'EBIT.2018', 'EBIT.2017', 'EBIT.2016', 'EBIT.2015']]
        pl_tax = self.df[['Company name', 'PLTax.2020', 'PLTax.2019',
                          'PLTax.2018', 'PLTax.2017', 'PLTax.2016', 'PLTax.2015']]
        mscore = self.df[['Company name', 'MScore.2020', 'MScore.2019',
                          'MScore.2018', 'MScore.2017', 'MScore.2016', 'MScore.2015']]
        leverage = self.df[['Company name', 'Leverage.2020', 'Leverage.2019',
                            'Leverage.2018', 'Leverage.2017', 'Leverage.2016', 'Leverage.2015']]
        roe = self.df[['Company name', 'ROE.2020', 'ROE.2019',
                       'ROE.2018', 'ROE.2017', 'ROE.2016', 'ROE.2015']]
        tasset = self.df[['Company name', 'TAsset.2020', 'TAsset.2019',
                          'TAsset.2018', 'TAsset.2017', 'TAsset.2016', 'TAsset.2015']]
        df_by_metric = [turnover, earnings_bit,
                        pl_tax, mscore, leverage, roe, tasset]
        return df_by_metric

    def sorting_table(self):
        sorted_table = self.table.sort_by([("ROE.2020", "descending")])

        top_10 = sorted_table.slice(0, 50)
        bot_10 = sorted_table.slice(len(sorted_table) - 50, 50)

        top_10_companies = top_10.select(['Company name', 'ROE.2020'])
        bottom_10_companies = bot_10.select(['Company name', 'ROE.2020'])

        print(top_10_companies.to_pandas())
        print(bottom_10_companies.to_pandas())

    def select_by_metric(self, company_name):
        company = self.df[self.df['Company name'] == company_name]
        company_id = company.index[0]
        turnover = self.df_by_metric[0]
        company_turnover = turnover[turnover.index == company_id]
        print(company_turnover)
        # plt.plot(company_turnover.iloc[:, 1:].columns, company_turnover.iloc[:, 1:].values[0])
        # plt.show()

    def main(self):
        # table, df, report = self.load_data()
        # df_by_metric = self.combine_dt(df)
        # self.stratified_random_sample(df)
        # self.query_report(report)
        # self.sorting_table(table)
        self.select_by_metric('SOLARPARK DELPHINUS GMBH & CO. KG')


if __name__ == '__main__':
    Exploration().main()
