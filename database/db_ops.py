import os
import logging

import psycopg2
import psycopg2.pool
from sqlalchemy import create_engine

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()

class DataBaseOps:
    def __init__(self):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.creds = self.db_creds()
        self.engine, self.conn = self.create_my_engine()
        self.pgsql_pool = self.create_my_pool()
        

    def db_creds(self):
        creds = {
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT'),
            'database': os.getenv('POSTGRES_DB')
        }
        return creds

    def create_my_pool(self):
        """Initialize connection pool"""
        pgsql_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            user=self.creds["user"],
            password=self.creds["password"],
            host=self.creds["host"],
            port=self.creds["port"],
            database=self.creds["database"])
        return pgsql_pool

    def create_my_engine(self):
        """Get the database engine."""
        engine = create_engine(
            f'postgresql+psycopg2://{self.creds["user"]}:{self.creds["password"]}@{self.creds["host"]}:{self.creds["port"]}/{self.creds["database"]}')
        conn = psycopg2.connect(
            f'dbname={self.creds["database"]} user={self.creds["user"]} host={self.creds["host"]} password={self.creds["password"]}')
        return engine, conn
    
    # def close_pool(self):
    #     """Close the connection pool on exit."""
    #     self.pgsql_pool.closeall()
    #     self.logger.info("Connection pool closed.")
        
    def ops_pipeline(self):
        self.logger.info('Starting db_ops.py Pipeline')
        creds = self.db_creds()
        pg_pool = self.create_my_pool()
        engine, conn = self.create_my_engine()
        return creds, pg_pool, engine, conn
        

