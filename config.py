from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from pprint import pformat

import pandas as pd

from src.statistical_analysis.transforms import StoreTransforms
from utils.my_utils import group_features
from utils.setup_env import setup_project_env
project_dir, project_config, setup_logs = setup_project_env()


@dataclass
class DataState:
    df: pd.DataFrame = None
    trans: StoreTransforms = field(default_factory=StoreTransforms)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    trans_map: dict = field(init=False, default_factory=dict)
    feature_groups: dict = field(init=False, default_factory=dict)
    df_pca: pd.DataFrame = None

    def __post_init__(self):
        self.trans_map = self.trans.get_transform_map()
        post_init_dict = {
            'df': self.df,
            'df_pca': self.df_pca,
            'logger': self.logger
        }
        self.logger.debug(f"Initialized DataState: {pformat(post_init_dict)}")

    def update_feature_groups(self):
        """Update continuous, discrete, and grouped features."""
        # self.feature_groups = group_features(config, self.df)
        self.feature_groups = group_features(self.df)

    def __repr__(self):
        return pformat(self.__dict__)


@dataclass
class StatisticConfig:
    random_state: int = 42
    shape_threshold: float = 0
    zscore_threshold: float = 12.0
    iqr_factor: float = 6.0
    iqr_quantiles: tuple = (0.05, 0.95)
    iso_contamination: float = 0.01
    lof_contamination: float = 0.05
    lof_n_neighbors: int = 20
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5


@dataclass
class ModelConfig:
    random_state: int = 42
    var_threshold: float = 0.1
    n_components: int = 3
    n_clusters: int = 4
    eps: float = 0.5
    min_samples: int = 5
    clustering_methods: list = field(default_factory=lambda: ['kmeans', 'dbscan', 'agglomerative'])
    clustering_params: dict = field(init=False)

    def __post_init__(self):
        self.clustering_params = {
            'kmeans': {
                'n_clusters': self.n_clusters, 'random_state': self.random_state},
            'dbscan': {
                'eps': self.eps, 'min_samples': self.min_samples},
            'agglomerative': {
                'n_clusters': self.n_clusters}
        }
