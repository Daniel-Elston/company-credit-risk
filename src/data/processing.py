from __future__ import annotations

import logging
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from database.db_ops import DataBaseOps
from utils.file_handler import save_json
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class Processor:
    def __init__(self, df):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df = df

    def remove_data(self):
        self.logger.debug('Removing data: Original shape: %s', self.df.shape)
        self.df = self.df.drop(columns=['No'])
        self.df = self.df.dropna()
        self.df = self.df[~self.df[['Company name']].duplicated(keep='first')]
        self.logger.debug('Removing data: Final shape: %s', self.df.shape)

    def map_categorical(self):
        self.logger.debug('Mapping categorical')
        to_map = ['Country', 'MScore.2020', 'MScore.2019',
                  'MScore.2018', 'MScore.2017', 'MScore.2016', 'MScore.2015']
        mapping = {
            'Italy': 0, 'France': 1, 'Spain': 2, 'Germany': 3,
            'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6, 'CC': 7, 'C': 8, 'D': 9
        }
        self.df[to_map] = self.df[to_map].replace(mapping)

    def encode_categorical(self):
        self.logger.debug('Encoding categories')
        label_encoder = LabelEncoder()
        self.df['Combined_Sector'] = self.df['Sector 2'] + \
            "_" + self.df['Sector 1']
        self.df[['Combined_Sector']] = self.df[['Combined_Sector']].apply(
            label_encoder.fit_transform)
        # sector_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        sector_map = {category: idx for idx,
                      category in enumerate(label_encoder.classes_)}
        save_json(sector_map, Path('reports/analysis/sector_map.json'))

    def scale_data(self):
        scaler = StandardScaler()
        dates = ['2015', '2016', '2017', '2018', '2019', '2020']
        scale_cols = [col for col in self.df.columns if any(
            x in col for x in dates)]
        self.df[scale_cols] = scaler.fit_transform(self.df[scale_cols])

    def main(self):
        self.remove_data()
        self.map_categorical()
        self.encode_categorical()
        self.scale_data()


if __name__ == '__main__':
    Processor().main()

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
