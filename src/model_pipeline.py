from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from time import time

from config import DataState
from config import ModelConfig
from src.data.processing import FurtherProcessor
from src.features.pca import PrincipleComponentsAnalysis
from src.models.train_model import ClusteringTool
from utils.file_handler import load_from_parquet
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, project_config, setup_logs = setup_project_env()


class ModelPipeline:
    def __init__(self, data_state: DataState, config: ModelConfig):
        self.ds = data_state
        self.config = config
        self.logger = logging.getLogger(self.ds.__class__.__name__)
        self.save_path = Path(project_config['path']['interim'])
        self.checkpoints = [
            'raw',
            'outliers',
            'transform1',
            'transform2',]

    def apply_scaling(self):
        scale = FurtherProcessor()
        self.ds.df = scale.pipeline(self.ds.df, **self.ds.feature_groups)

    def select_model_features(self, feature_groups):
        omit_cols = set(feature_groups['groups']['vol'])
        continuous = set(feature_groups['continuous'])
        training_cols = list(continuous - omit_cols)
        target_cols = ['MScore_mean']
        return training_cols, target_cols

    def run_pca(self, training_features):
        pca = PrincipleComponentsAnalysis(self.config)
        self.ds.df_pca = pca.pipeline(self.ds.df, training_features)

    def run_clustering(self):
        clustering = ClusteringTool(self.config)
        clustering.pipeline(self.ds.df_pca, run_number=1, **self.config.clustering_params)

    def main(self):
        t1 = time()
        try:
            df_raw = load_from_parquet(f'{self.save_path}/{self.checkpoints[1]}.parquet')
            self.ds.df = load_from_parquet(f'{self.save_path}/{self.checkpoints[3]}.parquet')
            self.ds.update_feature_groups()

            round_msc = round(df_raw[['MScore_mean']], 0)
            group_msc = round_msc.groupby('MScore_mean').value_counts()
            print(group_msc)

            self.apply_scaling()

            training_features, target_features = self.select_model_features(self.ds.feature_groups)
            self.run_pca(training_features)

            self.run_clustering()

        except Exception as e:
            self.logger.exception(f'Error: {e}', exc_info=e)
            raise
        finally:
            save_json(asdict(self.config), 'reports/analysis/model_state.json')
        self.logger.info(f'Pipeline Elapsed Time: {round(time()-t1, 2)} seconds')


if __name__ == '__main__':
    ModelPipeline().main()
