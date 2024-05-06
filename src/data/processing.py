from __future__ import annotations

import logging
import os
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from database.db_ops import DataBaseOps
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class InitialProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def remove_data(self, df):
        self.logger.debug('Removing data: Original shape: %s', df.shape)
        df = df.drop(columns=['No'])
        df = df.dropna()
        df = df[~df[['Company name']].duplicated(keep='first')]
        self.logger.debug('Removing data: Final shape: %s', df.shape)
        return df

    def map_categorical(self, df):
        self.logger.debug('Mapping categorical')
        to_map = ['Country', 'MScore.2020', 'MScore.2019',
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

        filepath = Path('reports/analysis/sector_map.json')
        if os.path.isfile(filepath):
            pass
        else:
            save_json(sector_map, filepath)
        return df

    def pipeline(self, df):
        df = self.remove_data(df)
        df = self.map_categorical(df)
        df = self.encode_categorical(df)
        return df


class FurtherProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def scale_data(self, df):
        scaler = StandardScaler()
        dates = ['2015', '2016', '2017', '2018', '2019', '2020']
        scale_cols = [col for col in df.columns if any(
            x in col for x in dates)]
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        return df

    def initial_processing(self, df):
        self.scale_data()
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
