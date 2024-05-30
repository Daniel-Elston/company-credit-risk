from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq


def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)


def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)


def save_to_parquet(df, filepath):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath)


def load_from_parquet(filepath):
    table = pq.read_table(filepath)
    return table.to_pandas()


if __name__ == '__main__':
    print(load_from_parquet('reports/analysis/variance/exploration_2.parquet'))
