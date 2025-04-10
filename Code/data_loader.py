import pandas as pd
import json

def load_json(filepath):
    """
    Loads the JSON data from the given file path and returns it as a pandas DataFrame.
    
    Parameters:
    - filepath (str): Path to the JSON file.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the JSON data, flattened if needed.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Normalize the JSON data (flatten nested JSON structures)
    df = pd.json_normalize(data)
    return df
