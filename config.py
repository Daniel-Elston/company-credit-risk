from __future__ import annotations

from dataclasses import dataclass

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


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
