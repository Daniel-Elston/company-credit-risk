from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from time import time

import numpy as np

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

    def apply_scaling(self):
        """Scale data using Standard Scaler"""
        scale = FurtherProcessor()
        self.ds.df = scale.pipeline(self.ds.df, **self.ds.feature_groups)

    def select_model_features(self, feature_groups):
        """Select features for model training"""
        omit_cols = set(feature_groups['groups']['vol'])
        continuous = set(feature_groups['continuous'])
        training_cols = list(continuous - omit_cols)
        return training_cols

    def run_pca(self, training_features):
        """Perform PCA on the scaled data."""
        pca = PrincipleComponentsAnalysis(self.config)
        self.ds.df_pca = pca.pipeline(self.ds.df, training_features)

    def run_clustering(self):
        """Perform clustering on the PCA data."""
        clustering = ClusteringTool(self.config)
        df = clustering.pipeline(self.ds.df_pca, run_number=1, **self.config.clustering_params)
        return df

    def review_matches(self, df):
        df_raw = load_from_parquet(f'{self.save_path}/{self.ds.checkpoints[1]}.parquet')

        mapping = {
            0: 1.0,
            2: 0.0,
            1: 2.0,
            3: 3.0
        }
        df['Cluster'] = df['Cluster'].replace(mapping)
        df['target'] = round(df_raw['MScore_mean'], 0)
        mask = np.where(df['target'] == df['Cluster'])
        res = (len(mask[0])/len(df))*100

        self.logger.info(f'Matches: {round(res, 2)}%')

    def main(self):
        t1 = time()
        try:
            self.ds.df = load_from_parquet(f'{self.save_path}/{self.ds.checkpoints[3]}.parquet')
            self.ds.update_feature_groups()

            self.apply_scaling()

            training_features = self.select_model_features(self.ds.feature_groups)
            self.run_pca(training_features)

            df = self.run_clustering()
            self.review_matches(df)

        except Exception as e:
            self.logger.exception(f'Error: {e}', exc_info=e)
            raise
        finally:
            save_json(asdict(self.config), 'reports/analysis/model_state.json')
        self.logger.info(f'Pipeline Elapsed Time: {round(time()-t1, 2)} seconds')


if __name__ == '__main__':
    ModelPipeline().main()
