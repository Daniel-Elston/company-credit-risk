from __future__ import annotations

import logging

import pandas as pd
import sqlalchemy

from database.db_ops import DataBaseOps
from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class LoadData:
    """Loading data from pgsql database as parquet"""

    def __init__(self):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_table_names(self):
        """
        Load table names from database
        """
        with engine.connect() as connect:
            result = connect.execute(sqlalchemy.text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            table_names = [row[0] for row in result]
            print(table_names)
            return table_names

    def load_data(self, table_name):
        """
        Load data from database
        """
        with engine.connect() as connect:
            result = connect.execute(
                sqlalchemy.text(f"SELECT * FROM {table_name}"))
            data = pd.DataFrame(result)
            return data


if __name__ == "__main__":
    ld = LoadData()
    table_names = ld.load_table_names()
    for table_name in table_names:
        data = ld.load_data(table_name)
        print(data)
