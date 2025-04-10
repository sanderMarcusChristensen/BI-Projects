import pandas as pd
import json

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    df = pd.json_normalize(data)
    return df
