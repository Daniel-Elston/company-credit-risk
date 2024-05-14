from __future__ import annotations

import logging

from src.data.make_dataset import LoadData
from src.data.processing import FurtherProcessor
from src.data.processing import InitialProcessor
from src.data.quality_assessment import QualityAssessment
from src.features.build_features import BuildFeatures
from src.statistical_analysis.analysis_utils import Sampling
from src.statistical_analysis.correlations import EvaluateCorrAnalysis
from src.statistical_analysis.correlations import GenerateCorrAnalysis
from src.statistical_analysis.skew_kurtosis import EvaluateSkewAnalysis
from src.statistical_analysis.skew_kurtosis import GenerateSkewAnalysis
from src.statistical_analysis.transforms import ApplyTransforms
from src.statistical_analysis.transforms import StoreTransforms
from src.visualization.exploration import Visualiser
from utils.config_ops import continuous_discrete
from utils.setup_env import setup_project_env
# from utils.file_handler import load_json
# from utils.file_handler import save_json
project_dir, config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.df = pd.read_parquet(Path('data/interim/processed.parquet'))

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
        initial_process = InitialProcessor()
        self.df = initial_process.pipeline(self.df)

    def run_feature_engineering(self):
        """Builds features"""
        build = BuildFeatures()
        self.df = build.pipeline(self.df, self.metric_cols, self.date_cols)
        self.cont, self.disc = continuous_discrete(config, self.df)

    def run_exploration(self, run_number):
        """Visualise Stratified Data"""
        df_stratified = Sampling().stratified_random_sample(self.df)
        visual = Visualiser()
        visual.pipeline(df_stratified, self.cont, run_number)
        GenerateCorrAnalysis().pipeline(run_number)

    def run_further_processing(self):
        """Remove outliers"""
        process = FurtherProcessor()
        self.df = process.pipeline(self.df)

    def run_statistical_analysis(self):
        """Run statistical analysis"""
        self.trans_map, self.trans_funcs = StoreTransforms().get_transform_info()
        GenerateSkewAnalysis().pipeline(self.df, self.cont, self.trans_funcs)
        EvaluateSkewAnalysis().pipeline()

    def apply_transforms(self):
        self.df = ApplyTransforms().pipeline(self.df, self.cont, self.trans_map)

    def run_statistical_evaluations(self):
        diff12, diff23, diff13 = EvaluateCorrAnalysis().pipeline()

    def main(self):
        self.run_make_dataset()
        self.run_quality_assessment()
        self.run_initial_processing()
        self.run_feature_engineering()

        self.run_exploration(run_number=1)
        self.run_further_processing()
        self.run_exploration(run_number=2)
        self.run_statistical_analysis()
        self.apply_transforms()
        self.run_exploration(run_number=3)

        self.run_statistical_evaluations()


if __name__ == '__main__':
    DataPipeline().main()
