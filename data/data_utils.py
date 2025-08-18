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


def filter_data222(
    data: pd.DataFrame,
    store_nbr: int,
    item_nbr: int,
    start_date=None,
    end_date=None
) -> pd.DataFrame:
    """
    Filter dataset by store number, item number, and optional date range.
    """
    df = data[(data['store_nbr'] == store_nbr) & (data['item_nbr'] == item_nbr)]

    # ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date'])

    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df



def filter_data444(
    data: pd.DataFrame,
    store_nbr: int,
    item_nbr: int,
    start_date=None,
    end_date=None
) -> pd.DataFrame:
    """
    Filter dataset by store number, item number, and optional date range.
    """
    df = data[(data['store_nbr'] == store_nbr) & (data['item_nbr'] == item_nbr)].copy()

    # ensure 'date' is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df





def filter_datavvvv(
    data: pd.DataFrame,
    store_nbr: int,
    item_nbr: int,
    start_date=None,
    n_days=None
) -> pd.DataFrame:
    """
    Filter dataset by store number, item number, and optional date + horizon.

    pd.DataFrame
        Filtered dataset.
    """
    # Slice & copy to avoid SettingWithCopyWarning
    df = data[(data['store_nbr'] == store_nbr) & (data['item_nbr'] == item_nbr)].copy()

    # ensure 'date' is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]

        # compute end_date if n_days is given
        if n_days is not None:
            end_date = start_date + pd.Timedelta(days=n_days)
            df = df[df['date'] <= end_date]

    return df






def filter_data(data: pd.DataFrame, store_nbr: int, item_nbr: int, start_date=None) -> pd.DataFrame:
    """
    Filter dataset by store, item, and optional start_date.
    """
    df = data[(data['store_nbr'] == store_nbr) & (data['item_nbr'] == item_nbr)]
    
    # ensure date column exists and is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
    
    return df.copy()



def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns and return features for prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataset.

    Returns
    -------
    pd.DataFrame
        Features ready for prediction.
    """
    cols_to_drop = ['unit_sales', 'unit_sales_log', 'date']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])
