from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


warnings.filterwarnings("ignore")


class InitialProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.save_path = Path(config['path']['maps'])

    def remove_data(self, df):
        try:
            df = df.drop(columns=['No'])
        except KeyError:
            pass
        df = df.dropna()
        df = df[~df[['Company name']].duplicated(keep='first')]
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
        df['Combined_Sector'] = df['Sector 2'] + "_" + df['Sector 1']
        df[['Combined_Sector']] = df[['Combined_Sector']].apply(label_encoder.fit_transform)
        sector_map = {category: idx for idx, category in enumerate(label_encoder.classes_)}

        filepath = Path(f'{self.save_path}/sector_map.json')
        if os.path.isfile(filepath):
            pass
        else:
            save_json(sector_map, filepath)
        return df

    def pipeline(self, df):
        self.logger.info(
            'Running Initial Processing pipeline.')
        initial_shape = df.shape

        df = self.remove_data(df)
        df = self.map_categorical(df)
        df = self.encode_categorical(df)

        processed_shape = df.shape
        shape_diff = (initial_shape[0] - processed_shape[0], initial_shape[1] - processed_shape[1])
        self.logger.debug(
            'Initial Shape: %s, Processed Shape: %s, Shape Difference: %s (Rows Removed: %s, Columns Changed: %s)',
            initial_shape, processed_shape, shape_diff, shape_diff[0], shape_diff[1])
        self.logger.info(
            'Initial Processing pipeline complete.')
        return df


class FurtherProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def scale_data(self, df):
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df

    def pipeline(self, df, **feature_groups):
        self.logger.info(
            'Running FurtherProcessor pipeline. Data shape: %s', df.shape)
        df = df[feature_groups['continuous']]

        self.scale_data(df)

        self.logger.info(
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
