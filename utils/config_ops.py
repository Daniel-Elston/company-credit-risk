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


def amend_features_all_years(config):
    years = ['2020', '2019', '2018', '2017', '2016', '2015']

    raw_feat = config['raw_features']
    growth_feat = config['growth_features']
    vol_feat = config['volatility_features']
    further_feat = config['further_features']

    add_year = [raw_feat, vol_feat, growth_feat, further_feat]

    new_feat = [[feat + year for feat in sublist]
                for sublist in add_year for year in years]
    all_feat = [feat for sublist in new_feat for feat in sublist]

    return all_feat


def continuous_discrete(config, df):
    cols = config['raw_features']+config['growth_features'] + \
        config['volatility_features']+config['further_features']
    discrete = df.columns[~df.columns.str.contains('|'.join(cols))]
    continuous = df.columns[~df.columns.isin(discrete)]
    return continuous, discrete
