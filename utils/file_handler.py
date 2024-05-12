from __future__ import annotations

import json


def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)


def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)
