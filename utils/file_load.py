from __future__ import annotations

import json
import logging
import pickle

import pandas as pd
import pyarrow.parquet as pq
import yaml
# import feather


class FileLoader:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_file(self, file_path, sheets=None):
        """
        Load file from local path
        """
        # self.logger.info('Loading file: %s', file_path)

        file_ext = file_path.split('.')[-1]

        try:
            if file_ext in ['xls', 'xlsx', 'xlsm', 'xlsb']:
                return self._load_excel(file_path, sheets)
            elif file_ext == 'csv':
                return self._load_csv(file_path)
            elif file_ext == 'json':
                return self._load_json(file_path)
            elif file_ext == 'pkl':
                return self._load_pickle(file_path)
            elif file_ext in ['parq', 'parquet']:
                return self._load_parquet(file_path)
            elif file_ext == 'yaml':
                return self._load_yaml(file_path)
            # elif file_ext == 'feather':
            #     return self._load_feather(file_path)
            else:
                raise ValueError('File extension not supported')

        except FileNotFoundError:
            self.logger.error('File not found: %s', file_path)
            return 'File not found'
        except PermissionError:
            self.logger.error('Permission denied: %s', file_path)
            return 'Permission denied'
        except KeyError:
            self.logger.error('Key not found: %s', file_path)
            return 'Key not found'
        except Exception as e:
            self.logger.error('Error loading file: %s', file_path, exc_info=e)
            raise e

    def _load_excel(self, file_path, sheets):
        try:
            if sheets == 'all':
                return pd.read_excel(file_path, sheet_name=None)
            elif isinstance(sheets, (list, str, int)):
                return pd.read_excel(file_path, sheet_name=sheets)
            else:
                return pd.read_excel(file_path)
        except Exception as e:
            self.logger.error('Error loading excel file: %s',
                              file_path, exc_info=e)
            raise e

    def _load_csv(self, file_path):
        return pd.read_csv(file_path)

    def _load_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def _load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def _load_parquet(self, file_path):
        return pq.read_table(file_path).to_pandas()

    def _load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    # def _load_feather(self, file_path):
        # """
        # Load feather file
        # """
        # return feather.read_dataframe(file_path)
