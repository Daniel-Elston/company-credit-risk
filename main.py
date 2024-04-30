from __future__ import annotations

from database.db_management import DataBaseManagement
from utils.setup_env import setup_project_env
# from src.pipeline import DataPipeline
# from src.tests import TestPipeline


if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    insert_data_to_db = DataBaseManagement().main()

    # DataPipeline(config).main()
    # TestPipeline(config).main()
