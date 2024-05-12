from __future__ import annotations

import logging
from pathlib import Path
from pprint import pprint

import pandas as pd

from src.data.processing import FurtherProcessor
from src.visualization.exploration import Analysis
from src.visualization.exploration import SkewDetector
from src.visualization.exploration import Visualiser
from utils.file_handler import load_json
from utils.setup_env import setup_project_env
# from utils.config_ops import amend_features
# from utils.file_handler import save_json
project_dir, config, setup_logs = setup_project_env()


class AnalysisPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df = pd.read_parquet(Path('data/interim/processed.parquet'))

    def run_exploration(self):
        analyse = Analysis()
        visual = Visualiser()
        skew = SkewDetector()
        df_stratified = analyse.pipeline(self.df)
        visual.pipeline(df_stratified)
        skew.pipeline(df_stratified)

    def run_further_processing(self):
        process = FurtherProcessor()
        self.df = process.pipeline(self.df)
        self.df.to_parquet(Path('data/interim/processed.parquet'))

    def access_skew_json(self):
        # transform_idx = ['log', 'cox1p', 'cox', 'sqrt', 'inv_sqrt', 'inv']
        filepath = Path('reports/analysis/skew_kurt/cox1p.json')
        data = load_json(filepath)

        skew_store = {}
        kurt_store = {}
        for entry in data:
            if abs(entry['Transformed Skew']) > 1:
                skew_store[entry['Column']] = entry['Transformed Skew']
            if abs(entry['Transformed Kurtosis']) > 1:
                kurt_store[entry['Column']] = entry['Transformed Kurtosis']

        pprint(skew_store)
        pprint(kurt_store)

    def main(self):
        self.run_exploration()
        self.run_further_processing()
        # df = pd.read_parquet(Path('data/interim/processed.parquet'))
        self.run_exploration()
        self.access_skew_json()


if __name__ == '__main__':
    AnalysisPipeline().main()
