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
    cols = config['raw_features']+config['growth_features']+config['volatility_features']+config['further_features']
    discrete = df.columns[~df.columns.str.contains('|'.join(cols))]
    continuous = df.columns[~df.columns.isin(discrete)]
    return continuous, discrete


def select_feature(cont_cols, feature_name, dates, period):
    feature_name_escaped = re.escape(feature_name)
    pattern = rf'\b{feature_name_escaped}\.' if period else rf'{feature_name}'

    feature_cols = [col for col in cont_cols if re.search(pattern, col)]

    if dates == 'all':
        return feature_cols
    return [col for col in feature_cols if dates in col]


def group_features(continuous, discrete):
    grow = select_feature(continuous, 'growth_', config['year'], period=False)
    raw = select_feature(continuous, '', config['year'], period=True)
    vol = select_feature(continuous, 'volatility', 'all', period=False)
    fur = select_feature(continuous, 'fur_', config['year'], period=False)
    msc = select_feature(discrete, 'MScore', config['year'], period=False)+['volatility_MScore']
    all = grow + raw + vol + fur + msc
    return {
        'grow': grow,
        'raw': raw,
        'vol': vol,
        'fur': fur,
        'msc': msc,
        'all': all
    }


def grouped_features(config, df):
    """Get discrete/continuous columns and group features."""
    cont, disc = continuous_discrete(config, df)
    groups = group_features(cont, disc)
    return {
        'cont': cont,
        'disc': disc,
        'groups': groups
    }
