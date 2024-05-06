from __future__ import annotations

import logging

from src.data.make_dataset import LoadData
from src.data.processing import InitialProcessor
from src.data.quality_assessment import QualityAssessment
from src.features.build_features import BuildFeatures
from src.visualization.exploration import Exploration
from utils.setup_env import setup_project_env
# from src.data.processing import FurtherProcessor
project_dir, config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_make_dataset(self):
        load = LoadData()
        self.table, self.df, self.metric_cols, self.date_cols = load.pipeline()

    def run_quality_assessment(self):
        qa = QualityAssessment()
        qa.generate_exploratory_report(self.df, 'report.xlsx', 'info.xlsx')

    def run_initial_processing(self):
        process = InitialProcessor(self.df)
        self.df = process.pipeline()

    def run_feature_engineering(self):
        build = BuildFeatures(self.df)
        self.df = build.pipeline()

    def run_exploration(self):
        exp = Exploration()
        print(exp.stratified_random_sample())

    def main(self):
        # pa_table, df, metric_cols, date_cols =
        self.run_make_dataset()
        # self.run_quality_assessment()
        self.run_initial_processing()
        # self.run_exploration()
        self.run_feature_engineering()


if __name__ == '__main__':
    DataPipeline().main()
