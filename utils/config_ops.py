from __future__ import annotations

import re

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


def amend_col_lists(cont):
    volatil_cols = cont[cont.str.contains('volatility')]
    raw_cols = [
        'Turnover.2018', 'EBIT.2018', 'PLTax.2018',
        'Leverage.2018', 'ROE.2018', 'TAsset.2018']
    growth_cols = [
        'growth_Leverage2018', 'growth_EBIT2018', 'growth_TAsset2018',
        'growth_PLTax2018', 'growth_ROE2018', 'growth_Turnover2018']
    further_cols = [
        'fur_debt_to_eq2018', 'fur_op_marg2018', 'fur_asset_turnover2018', 'fur_roa2018']
    corr_cols = [
        'growth_MScore2018', 'MScore.2018', 'volatility_MScore']

    dist_store = [raw_cols, volatil_cols, growth_cols, further_cols]
    dist_names = ['raw', 'vol', 'grow', 'further']

    corr_store = [raw_cols, volatil_cols,
                  growth_cols, further_cols, corr_cols]
    corr_names = ['raw', 'vol', 'grow', 'further', 'corr']

    combined_cols = list(corr_store[0]) + list(corr_store[1]) + \
        list(corr_store[2]) + list(corr_store[3]) + list(corr_store[4])

    return dist_store, dist_names, corr_store, corr_names, combined_cols


def get_feature_columns(cont_cols, feature_name, dates, period):
    feature_name_escaped = re.escape(feature_name)

    if period:
        pattern = rf'\b{feature_name_escaped}\.'
    else:
        pattern = rf'{feature_name}'

    feature_cols = [col for col in cont_cols if re.search(pattern, col)]

    if dates == 'all':
        return feature_cols
    else:
        filtered = [col for col in feature_cols if dates in col]
    return filtered
