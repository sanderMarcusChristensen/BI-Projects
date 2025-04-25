import pandas as pd
import json

def load_json(filepath):
    """
    Loads the JSON data from the given file path and returns it as a pandas DataFrame.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)

    df = pd.json_normalize(data)
    return df
