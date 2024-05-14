from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from utils.config_ops import amend_features
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


warnings.filterwarnings("ignore")


class InitialProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def remove_data(self, df):
        len1 = len(df)
        df = df.drop(columns=['No'])
        df = df.dropna()
        df = df[~df[['Company name']].duplicated(keep='first')]
        len2 = len(df)
        total_removed = len1 - len2
        self.logger.debug('Removing data: Rows removed: %s', total_removed)
        return df

    def map_categorical(self, df):
        self.logger.debug('Mapping categorical')
        to_map = [
            'Country', 'MScore.2020', 'MScore.2019',
            'MScore.2018', 'MScore.2017', 'MScore.2016', 'MScore.2015']
        mapping = {
            'Italy': 0, 'France': 1, 'Spain': 2, 'Germany': 3,
            'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6, 'CC': 7, 'C': 8, 'D': 9
        }
        df[to_map] = df[to_map].replace(mapping)
        return df

    def encode_categorical(self, df):
        self.logger.debug('Encoding categories')
        label_encoder = LabelEncoder()
        df['Combined_Sector'] = df['Sector 2'] + \
            "_" + df['Sector 1']
        df[['Combined_Sector']] = df[['Combined_Sector']].apply(
            label_encoder.fit_transform)
        sector_map = {category: idx for idx,
                      category in enumerate(label_encoder.classes_)}

        filepath = Path('reports/analysis/maps/sector_map.json')
        if os.path.isfile(filepath):
            pass
        else:
            save_json(sector_map, filepath)
        return df

    def pipeline(self, df):
        self.logger.info(
            'Running InitialProcessor pipeline. Data shape: %s', df.shape)
        df = self.remove_data(df)
        df = self.map_categorical(df)
        df = self.encode_categorical(df)
        self.logger.info(
            'InitialProcessor pipeline complete. Data shape: %s', df.shape)
        return df


class FurtherProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def replace_outliers(self, df, method, quantiles):
        for column in df.select_dtypes(include=[np.number]).columns:
            if method == "cap":
                lower_bound, upper_bound = df[column].quantile(
                    quantiles[0]), df[column].quantile(quantiles[1])
                df[column] = np.where(
                    df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(
                    df[column] > upper_bound, upper_bound, df[column])
            elif method == "median":
                median = df[column].median()
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[column] = np.where((df[column] < lower_bound) | (
                    df[column] > upper_bound), median, df[column])
            elif method == "winsorize":
                lower_bound, upper_bound = df[column].quantile(
                    quantiles[0]), df[column].quantile(quantiles[1])
                df[column] = np.where(
                    df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(
                    df[column] > upper_bound, upper_bound, df[column])
            elif method == "zscore":
                mean = df[column].mean()
                std = df[column].std()
                df[column] = np.where(
                    (df[column] < mean - 2 * std) | (df[column] > mean + 2 * std), mean, df[column])
        return df

    def scale_data(self, df):
        scaler = StandardScaler()
        df_numeric = df.select_dtypes(include=[np.number])
        scale_cols = df_numeric.columns
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        return df

    def pipeline(self, df):
        self.logger.debug(
            'Running FurtherProcessor pipeline. Data shape: %s', df.shape)

        _, grow, *_ = amend_features(config)

        self.replace_outliers(df, "winsorize", (0.05, 0.95))
        self.replace_outliers(df[grow], 'zscore', (0.05, 0.95))

        self.logger.debug(
            'FurtherProcessor pipeline complete. Data shape: %s', df.shape)
        return df


# investment_grade = [0,1,2,3] # low to moderate credit risk
# speculative_grade = [4,5,6,7,8,9] # high credit risk, potentially defaulted
#
# AAA = Highest credit quality
# AA = Very high credit quality
# A = High credit quality
# BBB = Good credit quality
# BB = Speculative
# B = Highly speculative
# CCC Substantial credit risk
# CC = Very high levels of credit risk
# C = Near default
# D = Default
