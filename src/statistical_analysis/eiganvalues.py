from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.file_handler import load_from_parquet
from utils.file_handler import load_json
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()

sns.set_theme(style="darkgrid")


class GenerateEigenValues:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_path = Path(config['path']['correlation'])
        self.save_path = Path(config['path']['eigen'])

    def load_corr_store(self, run_number):
        filepath = Path(f'{self.load_path}/exploration_{run_number}.parquet')
        data = load_from_parquet(filepath)
        return data

    def corr_eigenvalues(self, data, run_number):
        corr_eig = np.linalg.eigvals(data)
        store = {
            'eigen_values': corr_eig.tolist()
        }
        filepath = Path(f'{self.save_path}/eigen_values_{run_number}.json')
        save_json(store, filepath)

    def pipeline(self):
        self.logger.info(
            f'Generating Eigenvalues. Analysing files: ``{self.load_path}/*.parquet``')
        for run_number in range(1, 4):
            data = self.load_corr_store(run_number)
            self.corr_eigenvalues(data, run_number)

        self.logger.info(
            f'Eigenvalues Analysis results saved to: ``{self.save_path}/*``')


class EvaluateEigenValues:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_path = Path(config['path']['eigen'])
        self.save_path = Path(config['path']['eigen'])

    def load_corr_store(self, run_number):
        filepath = Path(f'{self.load_path}/eigen_values_{run_number}.json')
        data = load_json(filepath)
        return data

    def proportion_variance_explained(self, eigenvalues):
        total_variance = sum(eigenvalues)
        return [eig / total_variance for eig in eigenvalues]

    def plot_variance_explained(self, eigenvalues, run_number):
        variance_explained = self.proportion_variance_explained(eigenvalues)
        length = len(variance_explained)
        cumsum = np.cumsum(variance_explained)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, length + 1), cumsum, marker='o', linestyle='--')
        plt.title(f"Proportion of Variance Explained - Run {run_number}")
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Proportion of Variance Explained')
        plt.savefig(Path(f'{self.save_path}/scree_{run_number}.png'))

    def pipeline(self):
        self.logger.info(
            f'Generating Eigenvalues. Analysing files: ``{self.load_path}/*.json``')

        for run_number in range(1, 4):
            data = self.load_corr_store(run_number)
            eigenvalues = data['eigen_values']
            self.plot_variance_explained(eigenvalues, run_number)

        self.logger.info(
            f'Eigenvalues Analysis results saved to: ``{self.save_path}/*``')
