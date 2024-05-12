from __future__ import annotations

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


def amend_features(config):
    year = config['year']

    raw_feat = config['raw_features']
    growth_feat = config['growth_features']
    vol_feat = config['volatility_features']
    further_feat = config['further_features']

    add_year = [raw_feat, growth_feat, further_feat]

    new_feat = [[feat + year for feat in sublist] for sublist in add_year]
    return new_feat[0], new_feat[1], vol_feat, new_feat[2]
