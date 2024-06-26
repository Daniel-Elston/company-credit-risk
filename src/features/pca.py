from __future__ import annotations

import logging
import warnings

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from config import ModelConfig
from utils.setup_env import setup_project_env
project_dir, project_config, setup_logs = setup_project_env()

warnings.filterwarnings("ignore")


class PrincipleComponentsAnalysis:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def feature_selection_variance(self, df):
        """Select features based on variance threshold."""
        selector = VarianceThreshold(self.config.var_threshold)
        selected_data = selector.fit_transform(df)
        return pd.DataFrame(selected_data, columns=df.columns[selector.get_support()])

    def perform_pca(self, df):
        """Perform PCA on the scaled data."""
        index = df.index
        pca = PCA(n_components=self.config.n_components)
        pca_data = pca.fit_transform(df)
        explained_variance = pca.explained_variance_ratio_
        pca_columns = [f'PC{i+1}' for i in range(pca_data.shape[1])]
        return pd.DataFrame(pca_data, columns=pca_columns, index=index), explained_variance

    def pipeline(self, df, training_feautres):
        self.logger.info(
            'Running PCA pipeline. Data shape: %s', df.shape)

        training_feautres = ['Leverage_mean', 'ROE_mean', 'PLTax_mean', 'EBIT_mean', 'fur_debt_to_eq']  # ,'fur_roa','fur_op_marg']
        df = df[training_feautres]

        # df = self.feature_selection_variance(df)
        selected_cols = df.columns

        df, explained_variance = self.perform_pca(df)

        self.logger.info(
            'PCA explained variance: %s', explained_variance, 'With selected features: %s', selected_cols)
        self.logger.info(
            'PCA pipeline complete. Data shape: %s', df.shape)
        return df
