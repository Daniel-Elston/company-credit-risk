from __future__ import annotations

from src.pipeline import DataPipeline
# from database.db_management import DataBaseManagement


if __name__ == '__main__':
    # insert_data_to_db = DataBaseManagement().main()
    run_pipeline = DataPipeline().main()
