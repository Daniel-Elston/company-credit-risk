from __future__ import annotations

import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
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

    def table_to_parquet(self, export_tables):
        """Stream a database table directly to a Parquet file using PyArrow"""
        for table in export_tables:
            result = engine.connect().execution_options(stream_results=True).execute(
                sqlalchemy.text(f"SELECT * FROM {table}"))

            batches = []

            column_names = result.keys()
            first_batch = True

            while True:
                chunk = result.fetchmany(size=10000)
                if not chunk:
                    break

                # Generate schema from the column names and data types in the first chunk
                if first_batch:
                    schema = pa.schema(
                        [(name, pa.array([row[idx] for row in chunk]).type) for idx, name in enumerate(column_names)])
                    first_batch = False

                # Convert chunk (list of tuples) to PyArrow RecordBatch
                arrays = [pa.array([row[idx] for row in chunk])for idx in range(len(column_names))]
                batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
                batches.append(batch)

            arrow_table = pa.Table.from_batches(batches)

            file_path = f'data/sdo/{table}.parquet'
            pq.write_table(arrow_table, file_path)

    def create_pa_table(self):
        dataset_path = 'data/sdo/'
        dataset = ds.dataset(dataset_path, format='parquet')
        table = dataset.to_table()
        # print(table.schema)
        return table

    def create_pd_df(self, table):
        df = table.to_pandas()
        self.logger.debug('Dataframe shape: %s', df.shape)
        return df

    def create_pq_file(self):
        filepath = Path('data/interim/combined.parquet')
        if os.path.isfile(filepath):
            pass
        else:
            table = self.create_pa_table()
            pq.write_table(table, filepath)

    def pipeline(self, table_names: list):
        self.logger.info(
            'Beginning data export: PGSQL Table -> .parquet')

        self.table_to_parquet(table_names)
        table = self.create_pa_table()
        df = self.create_pd_df(table)

        self.logger.info(
            'Data export Complete. PA Table Saved to: ``data/sdo/*.parquet``')
        return table, df
