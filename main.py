from __future__ import annotations

import cProfile

from src.pipeline import DataPipeline
# from database.db_management import DataBaseManagement

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    DataPipeline().main()
    profiler.disable()
    # profiler.print_stats(sort='time')
