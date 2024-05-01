from __future__ import annotations

import logging
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy

from database.db_ops import DataBaseOps
from utils.setup_env import setup_project_env
# import psycopg2.pool
project_dir, config, setup_logs = setup_project_env()
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()


class LoadData:
    """Loading data from pgsql database as parquet"""

    def __init__(self):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def table_to_parquet(self, table_name):
        """Stream a database table directly to a Parquet file using PyArrow"""
        for table in table_name:
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
                arrays = [pa.array([row[idx] for row in chunk])
                          for idx in range(len(column_names))]
                batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
                batches.append(batch)

            arrow_table = pa.Table.from_batches(batches)

            file_path = f'data/sdo/{table}.parquet'
            pq.write_table(arrow_table, file_path)

    def main(self):
        export_tables = ['t1_20k', 't20k_40k', 't40k_60k',
                         't60k_80k', 't80k_100k', 't100k_121k', 'tdescription']

        self.logger.info(
            'Beginning data export: PGSQL Table -> .parquet')
        t1 = time.time()
        LoadData().table_to_parquet(export_tables)
        t2 = time.time()

        timed_sheet = round((t2-t1)/len(export_tables), 4)
        timed_total = round((t2-t1), 4)
        self.logger.info(
            'Time to export all Data: %s, ~ Time to export each sheet: %s', timed_total, timed_sheet)

    def test(self):
        export_tables = ['t1_20k', 't20k_40k', 't40k_60k',
                         't60k_80k', 't80k_100k', 't100k_121k', 'tdescription']
        for table in export_tables:
            file_path = f'data/sdo/{table}.parquet'
            df = pd.read_parquet(file_path)
            print(df)


if __name__ == "__main__":
    load = LoadData()
    load.main()
    # load.test()
