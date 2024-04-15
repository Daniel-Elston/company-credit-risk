from __future__ import annotations

import json
import logging
import pickle

import pyarrow.parquet as pq
import yaml

from utils.setup_env import setup_project_env
# import feather
project_dir, config, setup_logs = setup_project_env()


class FileSaver:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_file(self, data, file_path):
        """
        Save file to local path
        """
        file_ext = file_path.split('.')[-1]
        self.logger.info('Saving file: %s', file_path)

        try:
            if file_ext in ['xls', 'xlsx', 'xlsm', 'xlsb']:
                self._save_excel(data, file_path)
            elif file_ext == 'csv':
                self._save_csv(data, file_path)
            elif file_ext == 'json':
                self._save_json(data, file_path)
            elif file_ext == 'pkl':
                self._save_pkl(data, file_path)
            elif file_ext == 'parquet':
                self._save_parquet(data, file_path)
            elif file_ext == 'yaml':
                self._save_yaml(data, file_path)
            # elif file_ext == 'feather':
            #     self._save_feather(data, file_path)
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

    def _save_excel(self, data, file_path):
        data.to_excel(file_path)

    def _save_csv(self, data, file_path):
        data.to_csv(file_path)

    def _save_json(self, data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def _save_pkl(self, data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def _save_parquet(self, data, file_path):
        pq.write_table(data, file_path)

    def _save_yaml(self, data, file_path):
        with open(file_path, 'w') as file:
            yaml.dump(data, file)

    # def _save_feather(self, data, file_path):
    #     """
    #     Save feather file
    #     """
    #     feather.write_dataframe(data, file_path)
