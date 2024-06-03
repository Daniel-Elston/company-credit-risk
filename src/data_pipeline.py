from __future__ import annotations

import logging
from pathlib import Path
from time import time

from config import DataState
from src.data.make_dataset import LoadData
from src.data.processing import InitialProcessor
from src.data.quality_assessment import QualityAssessment
from src.features.build_features import BuildFeatures
from utils.file_handler import save_to_parquet
from utils.my_utils import stratified_random_sample
from utils.setup_env import setup_project_env
project_dir, project_config, setup_logs = setup_project_env()


class DataPipeline:
    def __init__(self, data_state: DataState):
        self.ds = data_state
        self.logger = logging.getLogger(self.ds.__class__.__name__)
        self.save_path = Path(project_config['path']['interim'])

    def run_make_dataset(self):
        """Loads PGSQL tables -> .parquet -> pd.DataFrame"""
        load = LoadData()
        _, self.ds.df = load.pipeline(project_config['export_tables'])
        self.ds.df = stratified_random_sample(self.ds.df)

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

    def main(self):
        t1 = time()
        try:
            self.run_make_dataset()
            self.run_quality_assessment()
            self.run_initial_processing()
            self.run_feature_engineering()
            self.ds.update_feature_groups()
            save_to_parquet(self.ds.df, f'{self.save_path}/{self.ds.checkpoints[0]}.parquet')

        except Exception as e:
            self.logger.exception(f'Error: {e}', exc_info=e)
            raise
        self.logger.info(f'Pipeline Elapsed Time: {round(time()-t1, 2)} seconds')


if __name__ == '__main__':
    DataPipeline().main()
