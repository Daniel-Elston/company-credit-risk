from __future__ import annotations

from src.pipeline import DataPipeline
from src.tests import TestPipeline
from utils.setup_env import setup_project_env

if __name__ == '__main__':
    project_dir, config, set_log = setup_project_env()
    pipeline = DataPipeline(config)
    test = TestPipeline(config)
    pipeline.main()
    # test.main()
