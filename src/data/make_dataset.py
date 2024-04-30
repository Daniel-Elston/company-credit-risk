from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import json

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


class LoadData():
    def __init__(self):
        self.config = config
        self.data_path = Path(config['data_path'])
        self.data = pd.ExcelFile(self.data_path, engine='openpyxl')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        
    def xls_to_parq(self):
        xls = pd.ExcelFile(self.data_path)
        data = xls.parse(xls.sheet_names[0])
        df = pd.DataFrame(data)
        print(df)
        pass
        
        
    def create_mapping(self):
        self.logger.info('Creating Mappings...')
        dtype_mapping = {}
        
        for sheet_name in self.data.sheet_names[0:6]:
            df = self.data.parse(sheet_name)
            dtype_dict = {col: str(df[col].dtype) for col in df.columns}
            
            for col, dtype in dtype_dict.items():
                if dtype == 'int64':
                    dtype_dict[col] = 'int32'
                elif dtype == 'float64':
                    dtype_dict[col] = 'float32'
                else:
                    dtype_dict[col] = dtype
        
            dtype_mapping[sheet_name] = dtype_dict
            
        filepath = Path(self.config['mappings_path'])
        with open(filepath, 'w') as f:
            json.dump(dtype_mapping, f, indent=4)
        self.logger.info(f'Mappings created and saved to {filepath}')
            
        return dtype_mapping
    
    
    def test(self, mappings):
        self.logger.info('test')
        
        for sheet_name, dtype_dict in mappings.items():
            df = self.data.parse(sheet_name, dtype=dtype_dict)
            df.to_e
        
        return None
    
    
    def load_sheets(self, mappings):
        t0 = time.time()
        
        mappings = json.load(open('data/mapping/dtype_mappings.json', 'r'))
        
        t1 = time.time()
        self.logger.info('Time to load data: %s', t1 - t0)
        
        data_names = self.data.sheet_names[0:6]
        info_names = self.data.sheet_names[6]
        
        store = []
        # for i in data_names:
        #     frame = self.data.parse(i, dtype=mappings[i])
        #     store.append(frame)
        
        for sheet_name, dtype_dict in mappings.items():
            frame = self.data.parse(sheet_name, dtype=dtype_dict)
            store.append(frame)
        
        t2 = time.time()
        self.logger.info('Time to load sheets: %s', t2 - t1)
            
        df = pd.concat(store)
        df_info = self.data.parse(info_names)
        
        t3 = time.time()
        self.logger.info('Time to concat sheets: %s', t3 - t2)
        
        return df, df_info
        
    
    
    def pipeline(self):
        self.logger.info('Starting make_dataset.py Pipeline')
        
        mappings = self.create_mapping()
        df, df_info = self.load_sheets(mappings)
        
        print(df, df.shape, df.dtypes)
        print(df_info)
        print(df.info())

if __name__ == '__main__':
    ld = LoadData()
    
    # df, df_info = ld.load_sheets()
    print(ld.xls_to_parq())