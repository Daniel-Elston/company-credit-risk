from __future__ import annotations

import re

from utils.setup_env import setup_project_env
project_dir, config, setup_logs = setup_project_env()


def stratified_random_sample(df):
    seed = 42
    df_strat = df.groupby('Sector 1', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(0.1 * len(x))), random_state=seed))
    return df_strat


def continuous_discrete(config, df):
    continuous = df.columns[df.columns.str.contains('|'.join(config['continuous']))]
    nominal = df.columns[df.columns.str.contains('|'.join(config['nominal']))]
    ordinal = df.columns[df.columns.str.contains('|'.join(config['ordinal']))]
    return continuous, nominal, ordinal


def select_feature(cont_cols, feature_name, dates, period):
    feature_name_escaped = re.escape(feature_name)
    pattern = rf'\b{feature_name_escaped}\.' if period else rf'{feature_name}'

    feature_cols = [col for col in cont_cols if re.search(pattern, col)]

    if dates == 'all':
        return feature_cols
    return [col for col in feature_cols if dates in col]


def group_features(config, df):
    continuous, nominal, ordinal = continuous_discrete(config, df)

    grow = select_feature(continuous, 'growth_', 'all', period=False)
    raw_mean = config['raw_means']
    vol = select_feature(continuous, 'volatility', 'all', period=False)
    fur = select_feature(continuous, 'fur_', 'all', period=False)
    msc = select_feature(ordinal, 'MScore', 'all', period=False)
    all_features = grow + raw_mean + vol + fur + msc
    groups = {
        'all': all_features,
        'fur': fur,
        'grow': grow,
        'raw_mean': raw_mean,
        'vol': vol}
    return {
        'continuous': continuous,
        'nominal': nominal,
        'ordinal': ordinal,
        'groups': groups}
