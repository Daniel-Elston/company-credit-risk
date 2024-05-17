from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pprint import pprint

import pandas as pd


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


@dataclass
class StatisticState:
    random_state: int = 42
    skew_weight: float = 1
    kurt_weight: float = 1
    shape_threshold: float = 0.1
    zscore_threshold: float = 12.0
    iqr_factor: float = 6.0
    iqr_quantiles: tuple = (0.05, 0.95)
    iso_contamination: float = 0.01
    lof_contamination: float = 0.05
    lof_n_neighbors: int = 20
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
