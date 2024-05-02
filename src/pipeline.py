from __future__ import annotations

import logging

from src.data.make_dataset import LoadData
from src.data.quality_assessment import QualityAssessment
from utils.setup_env import setup_project_env
# from database.db_ops import DataBaseOps
project_dir, config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_make_dataset(self):
        load = LoadData()
        load.table_to_parquet(config['export_tables'])
        table = load.create_pa_table()
        df = load.create_pd_df(table)
        return table, df

    def run_quality_assessment(self, df):
        qa = QualityAssessment()
        qa.generate_exploratory_report(df, 'report.xlsx', 'info.xlsx')

    def main(self):
        pa_table, df = self.run_make_dataset()
        self.run_quality_assessment(df)


if __name__ == '__main__':
    DataPipeline().main()
