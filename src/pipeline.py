from __future__ import annotations

import gc
import logging
from time import time

import dask.dataframe as dd

from config import DataState
from config import StatisticState
from src.data.make_dataset import LoadData
from src.data.processing import InitialProcessor
from src.data.quality_assessment import QualityAssessment
from src.features.build_features import BuildFeatures
from src.statistical_analysis.correlations import EvaluateCorrAnalysis
from src.statistical_analysis.correlations import GenerateCorrAnalysis
from src.statistical_analysis.dist_analysis import EvaluateDistAnalysis
from src.statistical_analysis.dist_analysis import GenerateDistAnalysis
from src.statistical_analysis.eiganvalues import EvaluateEigenValues
from src.statistical_analysis.eiganvalues import GenerateEigenValues
from src.statistical_analysis.outliers import HandleOutliers
from src.statistical_analysis.transforms import ApplyTransforms
from src.visualization.exploration import Visualiser
from utils.my_utils import stratified_random_sample
from utils.setup_env import setup_project_env
# from src.data.processing import FurtherProcessor
# from utils.file_handler import load_json
# from utils.file_handler import save_json
project_dir, config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self):
        self.ds = DataState()
        self.ss = StatisticState()
        self.logger = logging.getLogger(self.ds.__class__.__name__)
        # self.ds.df = pd.read_parquet('data/interim/df_out.parquet')

    def run_make_dataset(self):
        """Loads PGSQL tables -> .parquet -> pd.DataFrame"""
        load = LoadData()
        _, self.ds.df = load.pipeline(config['export_tables'])
        # self.ds.df = stratified_random_sample(self.ds.df)

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

    def run_handle_outliers(self):
        """Removes Outliers"""
        outliers = HandleOutliers()
        self.ds.df = dd.from_pandas(self.ds.df, npartitions=10)
        self.ds.df = outliers.pipeline(self.ds.df, **self.ds.feature_groups)
        self.ds.df = self.ds.df.compute()
        gc.collect()

    def run_exploration(self, run_n):
        """Visualise Stratified Data"""
        df_stratified = stratified_random_sample(self.ds.df)
        visualiser = Visualiser()
        visualiser.pipeline(df_stratified, run_n, **self.ds.feature_groups)
        gc.collect()

    def run_distribution_analysis(self):
        """Runs distribution analysis"""
        gen_dist_analysis = GenerateDistAnalysis()
        gen_dist_analysis.pipeline(self.ds.df, self.ds.trans_map, **self.ds.feature_groups)
        eval_dist_analysis = EvaluateDistAnalysis()
        eval_dist_analysis.pipeline(self.ds.trans_map)
        gc.collect()

    def apply_transforms(self):
        """Applies the optimal transform to each continuous feature"""
        transform = ApplyTransforms()
        self.ds.df = transform.pipeline(self.ds.df, self.ds.trans_map, self.ss.shape_threshold)
        gc.collect()

    def run_correlation_analysis(self):
        """Runs correlation analysis"""
        gen_corr_analysis = GenerateCorrAnalysis()
        gen_corr_analysis.pipeline()
        eval_corr_analysis = EvaluateCorrAnalysis()
        eval_corr_analysis.pipeline()

    def run_eigen_analysis(self):
        """Runs eigenvalue analysis"""
        gen_eigen_values = GenerateEigenValues()
        gen_eigen_values.pipeline()
        eval_eigen_values = EvaluateEigenValues()
        eval_eigen_values.pipeline()

    def main(self):
        t1 = time()
        try:
            self.run_make_dataset()
            self.run_quality_assessment()
            self.run_initial_processing()
            self.run_feature_engineering()
            self.ds.update_grouped_features()
            self.run_exploration(run_n=0), gc.collect()
            self.run_handle_outliers(), gc.collect()
            self.run_exploration(run_n=1), gc.collect()
            self.run_distribution_analysis(), gc.collect()
            self.apply_transforms(), gc.collect()
            self.run_exploration(run_n=2), gc.collect()
            self.run_distribution_analysis(), gc.collect()
            self.apply_transforms(), gc.collect()
            self.run_exploration(run_n=3), gc.collect()
            self.run_correlation_analysis()
            self.run_eigen_analysis()
        except Exception as e:
            self.logger.exception(f'Error: {e}', exc_info=e)
            raise

        t2 = time()
        print(f'Pipeline Elapsed Time: {round(t2 - t1, 2)} seconds')


if __name__ == '__main__':
    DataPipeline().main()

# total pipeline run1 time = 250s
# total pipeline run2 time = 83s
