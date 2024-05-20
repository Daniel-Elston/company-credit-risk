from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from pprint import pformat

import pandas as pd

from src.statistical_analysis.transforms import StoreTransforms
from utils.my_utils import grouped_features
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


@dataclass
class DataState:
    df: pd.DataFrame = None
    trans: StoreTransforms = field(default_factory=StoreTransforms)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    trans_map: dict = field(init=False, default_factory=dict)
    feature_groups: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.trans_map = self.trans.get_transform_map()
        post_init_dict = {
            'df': self.df,
            'logger': self.logger
        }
        self.logger.debug(f"Initialized DataState: {pformat(post_init_dict)}")

    def update_grouped_features(self):
        """Update continuous, discrete, and grouped features."""
        self.feature_groups = grouped_features(config, self.df)


@dataclass
class StatisticState:
    random_state: int = 42
    skew_weight: float = 1
    kurt_weight: float = 1
    shape_threshold: float = 0
    zscore_threshold: float = 12.0
    iqr_factor: float = 6.0
    iqr_quantiles: tuple = (0.05, 0.95)
    iso_contamination: float = 0.01
    lof_contamination: float = 0.05
    lof_n_neighbors: int = 20
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
