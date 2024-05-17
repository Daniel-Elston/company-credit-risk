from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pprint import pprint

import pandas as pd

from src.data.make_dataset import LoadData
from src.data.processing import InitialProcessor
from src.data.quality_assessment import QualityAssessment
from src.features.build_features import BuildFeatures
from src.statistical_analysis.correlations import EvaluateCorrAnalysis
from src.statistical_analysis.correlations import GenerateCorrAnalysis
from src.statistical_analysis.eiganvalues import AnalyseEigenValues
from src.statistical_analysis.outliers import HandleOutliers
from src.statistical_analysis.skew_kurtosis import EvaluateDistAnalysis
from src.statistical_analysis.skew_kurtosis import GenerateDistAnalysis
from src.statistical_analysis.transforms import ApplyTransforms
from src.statistical_analysis.transforms import StoreTransforms
from src.visualization.exploration import Visualiser
from utils.my_utils import continuous_discrete
from utils.my_utils import group_features
from utils.my_utils import stratified_random_sample
from utils.setup_env import setup_project_env
# from pathlib import Path
# from src.data.processing import FurtherProcessor
# from utils.file_handler import load_json
# from utils.file_handler import save_json
project_dir, config, setup_logs = setup_project_env()


@dataclass
class DataState:
    df: pd.DataFrame = None
    cont: list = field(default_factory=list)
    disc: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)
    trans_map: dict = field(default_factory=dict)
    trans_funcs: dict = field(default_factory=dict)

    def __post_init__(self):
        print(f"Initialized DataState: {self}")

    def print_state(self):
        for field_info in fields(self):
            attr_name = field_info.name
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, dict):
                print(f"{attr_name}:------------------------------------------------------------------------------")
                pprint(attr_value)
            else:
                print(f"{attr_name}:------------------------------------------------------------------------------")
                print(attr_value)


class DataPipeline:
    def __init__(self):
        self.ds = DataState()
        self.logger = logging.getLogger(self.ds.__class__.__name__)
        # self.ds.df = pd.read_parquet(Path('data/interim/df_outliers_rem.parquet'))
        # self.ds.cont, self.ds.disc = continuous_discrete(config, self.ds.df)
        # self.ds.groups = group_features(self.ds.cont, self.ds.disc)
        self.ds.trans_map, self.ds.trans_funcs = StoreTransforms().get_transform_info()

    def run_make_dataset(self):
        """Loads PGSQL tables -> .parquet -> pd.DataFrame"""
        load = LoadData()
        _, self.ds.df = load.pipeline(config['export_tables'])

    def run_quality_assessment(self):
        """Generates quality assessment report"""
        qa = QualityAssessment()
        qa.generate_exploratory_report(self.ds.df, 'report.xlsx', 'info.xlsx')

    def run_initial_processing(self):
        """Removes NaNs, duplicates and encodes categorical features"""
        initial_process = InitialProcessor()
        self.ds.df = initial_process.pipeline(self.ds.df)

    def run_feature_engineering(self):
        """Builds features"""
        build = BuildFeatures()
        self.ds.df = build.pipeline(self.ds.df)
        self.ds.cont, self.ds.disc = continuous_discrete(config, self.ds.df)
        self.ds.groups = group_features(self.ds.cont, self.ds.disc)
        self.ds.print_state()

    def run_handle_outliers(self):
        """Removes Outliers"""
        outliers = HandleOutliers()
        self.ds.df = outliers.pipeline(self.ds.df, self.ds.cont, self.ds.disc)

    def run_exploration(self, run_number):
        """Visualise Stratified Data"""
        df_stratified = stratified_random_sample(self.ds.df)
        Visualiser().pipeline(df_stratified, self.ds.groups, run_number)
        GenerateCorrAnalysis().pipeline(run_number)

    def run_distribution_analysis(self, skew_weight=1, kurt_weight=1):
        """Run statistical analysis"""
        GenerateDistAnalysis().pipeline(self.ds.df, self.ds.cont, self.ds.trans_funcs)
        EvaluateDistAnalysis().pipeline(skew_weight, kurt_weight)

    def apply_transforms(self):
        """Apply transformations"""
        transform = ApplyTransforms()
        self.ds.df = transform.pipeline(self.ds.df, self.ds.trans_map, shape_threshold=0)
        self.ds.print_state()

    def run_correlation_analysis(self):
        """Run correlation analysis"""
        EvaluateCorrAnalysis().pipeline()
        AnalyseEigenValues().pipeline()

    def main(self):
        try:
            self.run_make_dataset()
            self.run_quality_assessment()
            self.run_initial_processing()
            self.run_feature_engineering()
            self.run_exploration(run_number='0')

            self.run_handle_outliers()
            # self.df.to_parquet(Path('data/interim/df_outliers_rem.parquet'))
            self.run_exploration(run_number='1')

            self.run_distribution_analysis()
            self.apply_transforms()
            self.run_exploration(run_number='2')
            self.run_correlation_analysis()

            self.run_distribution_analysis()
            self.apply_transforms()
            self.run_exploration(run_number='3')
            self.run_correlation_analysis()

        except Exception as e:
            self.logger.exception(f'Error: {e}', exc_info=e)
            raise


if __name__ == '__main__':
    DataPipeline().main()
