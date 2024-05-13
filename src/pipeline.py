from __future__ import annotations

import logging

from src.data.make_dataset import LoadData
from src.data.processing import FurtherProcessor
from src.data.processing import InitialProcessor
from src.data.processing import StoreTransforms
from src.data.quality_assessment import QualityAssessment
from src.features.build_features import BuildFeatures
from src.statistics.statistical_analysis import EvaluateSkewAnalysis
from src.statistics.statistical_analysis import GenerateSkewAnalysis
from src.statistics.statistical_analysis import Sampling
from src.visualization.exploration import Visualiser
from utils.setup_env import setup_project_env
# from utils.file_handler import load_json
# from utils.config_ops import amend_features
# from utils.file_handler import save_json
project_dir, config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # @staticmethod
    def run_make_dataset(self):
        """Loads PGSQL tables -> .parquet -> pd.DataFrame"""
        load = LoadData()
        self.table, self.df, self.metric_cols, self.date_cols = load.pipeline()

    def run_quality_assessment(self):
        qa = QualityAssessment()
        qa.generate_exploratory_report(self.df, 'report.xlsx', 'info.xlsx')

    def run_initial_processing(self):
        """Removes NaNs, duplicates and encodes categorical features"""
        process = InitialProcessor()
        self.df = process.pipeline(self.df)

    def run_feature_engineering(self):
        """Builds features"""
        build = BuildFeatures()
        self.df = build.pipeline(self.df, self.metric_cols, self.date_cols)

    def generate_stratified_sample(self):
        """Generate stratified sample"""
        sample = Sampling()
        self.df_stratified = sample.stratified_random_sample(self.df)

    def run_exploration(self):
        """Visualise Stratified Data"""
        visual = Visualiser()
        visual.pipeline(self.df_stratified)

    def run_further_processing(self):
        """Remove outliers"""
        process = FurtherProcessor()
        self.df = process.pipeline(self.df)

    def run_statistical_analysis(self):
        """Run statistical analysis"""
        transform_funcs = StoreTransforms().get_transform_lists()
        GenerateSkewAnalysis().pipeline(self.df, transform_funcs)
        EvaluateSkewAnalysis().pipeline()

    def apply_transforms(self):
        pass

    def main(self):
        self.run_make_dataset()
        self.run_quality_assessment()
        self.run_initial_processing()
        self.run_feature_engineering()
        self.generate_stratified_sample()
        self.run_exploration()
        self.run_further_processing()
        # self.run_exploration()
        self.run_statistical_analysis()
        # self.run_exploration()


if __name__ == '__main__':
    DataPipeline().main()
