from __future__ import annotations

import json


def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)
