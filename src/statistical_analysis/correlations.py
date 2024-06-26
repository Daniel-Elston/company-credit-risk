from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils.file_handler import load_from_parquet
from utils.file_handler import load_json
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class GenerateCorrAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_path = Path(config['path']['correlation'])
        self.save_path = Path(config['path']['correlation'])

    def load_corr_store(self, run_number):
        filepath = Path(f'{self.load_path}/exploration_{run_number}.parquet')
        data = load_from_parquet(filepath)
        return data

    def corr_frobenius_norm(self, data, run_number):
        sum_corr = round(
            data.abs().sum().sum() - np.trace(data.abs().to_numpy()), 4)

        total_elements = data.size - len(data)
        avg_corr = round(sum_corr / total_elements, 4)

        frobenius_norm = round(np.linalg.norm(data, 'fro'), 4)

        store = {
            'sum': sum_corr,
            'avg': avg_corr,
            'fro': frobenius_norm,
        }
        filepath = Path(f'{self.save_path}/corr_fro_results_{run_number}.json')
        save_json(store, filepath)

    def pipeline(self):
        self.logger.info(
            f'Generating Correlation Analysis. Analysing files: ``{self.load_path}/exploration_n.csv``')

        for run_number in range(1, 4):
            data = self.load_corr_store(run_number)
            self.corr_frobenius_norm(data, run_number)

        self.logger.info(
            f'Correlation Analysis Completed. Results saved to: ``{self.save_path}/corr_fro_results_n.json``')


class EvaluateCorrAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_path = Path(config['path']['correlation'])

    def get_fro_diff(self):
        fro_store = []
        for run_number in range(1, 4):
            filepath = Path(f'{self.load_path}/corr_fro_results_{run_number}.json')
            data = load_json(filepath)
            frobenius_norm = data['fro']
            fro_store.append(frobenius_norm)

        diff12 = round(fro_store[1] - fro_store[0], 4)
        diff23 = round(fro_store[2] - fro_store[1], 4)
        diff13 = round(fro_store[2] - fro_store[0], 4)
        return diff12, diff23, diff13

    def pipeline(self):
        self.logger.info(
            f'Evaluating Correlation Analysis. Analysing files: ``{self.load_path}/corr_fro_results_n.json``')

        *_, diff13 = self.get_fro_diff()

        self.logger.debug(
            'Frobenius Norm Differences: %s', diff13)
        self.logger.info(
            'Correlation Analysis Completed. Frobenius Norm Differences Generated')
