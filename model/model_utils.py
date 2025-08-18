import joblib
import pandas as pd
import numpy as np
from app import config

def load_model(model_path: str = config.MODEL_PATH):
    """Load trained model from .pkl file."""
    return joblib.load(model_path)

def forecast(model, input_data: pd.DataFrame):
    """Generate single-step predictions from model."""
    return model.predict(input_data)



# data_utils.py

import pandas as pd



def forecast_recursive(model, input_data: pd.DataFrame, n_days: int, date_col: str = "date"):
    if date_col not in input_data.columns:
        raise KeyError("Input data must include a 'date' column.")

    df_future = input_data.copy()
    last_date = pd.to_datetime(df_future[date_col].iloc[-1])

    preds = []
    forecast_dates = []

    for i in range(n_days):
        # Prepare X (drop date + target column if exists)
        X = df_future.drop(columns=[date_col], errors="ignore")
        if "unit_sales" in X.columns:   # <--- drop target
            X = X.drop(columns=["unit_sales"])

        # Predict next step
        y_pred = model.predict(X.tail(1))[0]

        # Save prediction
        next_date = last_date + pd.Timedelta(days=1)
        preds.append(y_pred)
        forecast_dates.append(next_date)

        # Append row with new prediction
        new_row = df_future.tail(1).copy()
        new_row[date_col] = next_date
        if "unit_sales" in new_row.columns:
            new_row["unit_sales"] = y_pred
        df_future = pd.concat([df_future, new_row], ignore_index=True)

        last_date = next_date

    return preds, forecast_dates



def forecast_recursive999(model, input_data: pd.DataFrame, n_days: int):
    """
    Recursive forecast starting from last known date.
    """
    df_future = input_data.copy()
    if "date" not in df_future.columns:
        raise KeyError("Input data must include a 'date' column.")

    df_future["date"] = pd.to_datetime(df_future["date"])
    last_date = df_future["date"].iloc[-1]

    preds, forecast_dates = [], []

    for i in range(n_days):
        # use last row of features (excluding date)
        X_last = df_future.drop(columns=["date"]).iloc[[-1]]
        y_pred = model.predict(X_last)[0]

        preds.append(y_pred)
        next_date = last_date + pd.Timedelta(days=i + 1)
        forecast_dates.append(next_date)

        # create new row with same features but future date
        new_row = X_last.copy()
        new_row["date"] = next_date
        df_future = pd.concat([df_future, new_row], ignore_index=True)

    return preds, forecast_dates






def forecast_recursivewwww(model, input_data: pd.DataFrame, n_days: int):
    """
    Generate recursive multi-day forecasts.
    Parameters
    ----------
    model : trained model
    input_data : pd.DataFrame
        Feature matrix including a 'date' column
    n_days : int
        Number of days to forecast
    Returns
    -------
    preds : list
        Predictions for n_days
    forecast_dates : list
        Future dates corresponding to predictions
    """
    df_future = input_data.copy()
    df_future["date"] = pd.to_datetime(df_future["date"])

    # start from last date
    last_date = df_future["date"].iloc[-1]

    preds = []
    forecast_dates = []

    for i in range(n_days):
        # predict with last row
        X_last = df_future.drop(columns=["date"]).iloc[[-1]]
        y_pred = model.predict(X_last)[0]

        preds.append(y_pred)
        next_date = last_date + pd.Timedelta(days=i+1)
        forecast_dates.append(next_date)

        # append new row (date + same features except target update logic)
        new_row = X_last.copy()
        new_row["date"] = next_date
        df_future = pd.concat([df_future, new_row], ignore_index=True)

    return preds, forecast_dates




def forecast_recursiveaaaa(model, input_data: pd.DataFrame, n_days: int, date_col: str = "date"):
    """
    Generate recursive multi-day forecasts.
    """
    df_future = input_data.copy()

    # Ensure date column exists
    df_future[date_col] = pd.to_datetime(df_future[date_col])
    last_date = df_future[date_col].iloc[-1]

    preds = []
    forecast_dates = []

    for i in range(n_days):
        X = df_future.drop(columns=[date_col], errors="ignore")
        y_pred = model.predict(X.iloc[[-1]])[0]

        preds.append(y_pred)
        next_date = last_date + pd.Timedelta(days=i+1)
        forecast_dates.append(next_date)

        # Append prediction row for recursive feature generation
        new_row = X.iloc[[-1]].copy()
        new_row[date_col] = next_date
        # If your features include lag or rolling stats, you'd update them here
        df_future = pd.concat([df_future, new_row], ignore_index=True)

    return preds, forecast_dates




def forecast_recursive2222(model, input_data: pd.DataFrame, n_days: int, date_col: str = "date"):
    """
    Recursive multi-step forecast.
    
    Parameters:
    - model: trained XGBoost model
    - input_data: dataframe with the last available features (lags, rolling stats, etc.)
    - n_days: number of future days to predict
    - date_col: column containing the dates
    
    Returns:
    - preds: list of predictions
    - forecast_dates: list of forecast dates
    """
    preds = []
    forecast_dates = []

    # Make a copy so we can modify it
    df_future = input_data.copy()

    # Start from the last date in the dataset
    last_date = pd.to_datetime(df_future[date_col].iloc[-1])

    for i in range(1, n_days + 1):
        # Features for the next prediction
        X = df_future.drop(columns=[date_col], errors="ignore").iloc[[-1]]  # last row features only

        # Predict next step
        y_pred = model.predict(X)[0]
        preds.append(y_pred)

        # Create new row for the next day
        next_date = last_date + pd.Timedelta(days=i)
        forecast_dates.append(next_date)

        new_row = df_future.iloc[[-1]].copy()
        new_row[date_col] = next_date

        # ⚠️ update lag features here!
        if "lag_1" in new_row.columns:
            new_row["lag_1"] = y_pred  # predicted sales feeds back

        # Append new row for recursive prediction
        df_future = pd.concat([df_future, new_row], ignore_index=True)

    return preds, forecast_dates
