from __future__ import annotations

from config import DataState
from config import ModelConfig
from config import StatisticConfig
from src.data_pipeline import DataPipeline
from src.model_pipeline import ModelPipeline
from src.stats_pipeline import StatsPipeline
# from database.db_management import DataBaseManagement

if __name__ == '__main__':
    data_state = DataState()
    stats_config = StatisticConfig()
    model_config = ModelConfig()

    DataPipeline(data_state).main()
    StatsPipeline(data_state, stats_config).main()
    ModelPipeline(data_state, model_config).main()
