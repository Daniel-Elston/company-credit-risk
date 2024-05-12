from __future__ import annotations

from src.analysis import AnalysisPipeline
# from database.db_management import DataBaseManagement
# from src.pipeline import DataPipeline


if __name__ == '__main__':
    # insert_data_to_db = DataBaseManagement().main()
    # run_pipeline = DataPipeline().main()
    run_analysis = AnalysisPipeline().main()
