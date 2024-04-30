import pandas as pd
from pathlib import Path

from psycopg2 import pool
from database.db_ops import DataBaseOps
creds, pg_pool, engine, conn = DataBaseOps().ops_pipeline()

class DataBaseManagement(DataBaseOps):
    def __init__(self):
        super().__init__()
        self.data_path = Path(self.config['data_path'])
        
    def insert_data(self):
        self.logger.info('Begininning data insert')
        raw_data = pd.ExcelFile(self.data_path, engine='openpyxl')
        
        count = 0
        try:
            for sheet_name in raw_data.sheet_names:
                count += 1
                df = pd.read_excel(raw_data, sheet_name=sheet_name)
                df.to_sql(sheet_name.lower(), con=engine, index=False, if_exists='replace')
                self.logger.info('Inserted sheet %s: %s of %s', sheet_name, count, len(raw_data.sheet_names))

        finally:
            raw_data.close()
            
if __name__ == '__main__':
    db = DataBaseManagement()
    try:
        db.insert_data()
        db.logger.info('Data inserted successfully')
    finally:
        pg_pool.closeall()
        engine.dispose()
