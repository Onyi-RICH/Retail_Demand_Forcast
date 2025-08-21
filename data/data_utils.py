# data/data_utils.py

import pandas as pd
from app import config
import joblib
import pyarrow.parquet as pq

def load_data(data_path: str = config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)

def load_features(features_path: str = config.FEATURES_PATH): # -> pd.DataFrame:
    table = pq.read_table(features_path, columns=[])
    schema = pq.read_schema(features_path)
    
    # Get column names
    col_names = schema.names
    #print(col_names)
    
    return col_names #joblib.load(features_path)


#def load_features(features_path: str = config.FEATURES_PATH) -> pd.DataFrame:
#    return joblib.load(features_path)

