# data/data_utils.py

import pandas as pd
from app import config
import joblib

def load_data(data_path: str = config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)

def load_features(features_path: str = config.FEATURES_PATH) -> pd.DataFrame:
    return joblib.load(features_path)

#def load_features(path: str = config.FEATURES_PATH):
#    return joblib.load(path)
