from __future__ import annotations

import logging

from src.data.make_dataset import LoadData
from src.data.processing import Processor
from src.data.quality_assessment import QualityAssessment
from src.visualization.exploration import Exploration
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.table, self.df = self.run_make_dataset()

    def run_make_dataset(self):
        load = LoadData()
        load.table_to_parquet(config['export_tables'])
        table = load.create_pa_table()
        df = load.create_pd_df(table)
        # create_pq_file = load.create_pq_file()
        return table, df

    def run_quality_assessment(self):
        qa = QualityAssessment()
        qa.generate_exploratory_report(self.df, 'report.xlsx', 'info.xlsx')

    def run_initial_processing(self):
        process = Processor(self.df)
        process.main()

    def run_exploration(self, df):
        exp = Exploration()
        exp.select_by_metric()

    def main(self):
        pa_table, df = self.run_make_dataset()
        self.run_quality_assessment()
        self.run_initial_processing()

        # self.run_exploration(df)


if __name__ == '__main__':
    DataPipeline().main()
