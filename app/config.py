# app/config.py

# Paths
MODEL_PATH = "model/xgboost_model.pkl"
DATA_PATH = "data/data_streamlit.csv"
FEATURES_PATH = "data/feature_columns.pkl" 

# MLflow local storage path
MLFLOW_TRACKING_URI = "mlflow_results"

# App settings
APP_TITLE = "Corporación Favorita Sales Forecasting"
APP_DESCRIPTION = """
This app forecasts retail demand using the best model from Sprint 3.
Upload your dataset to get predictions.
"""

# Forecast settings (you can adjust these as needed)
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"

# Default date format for parsing
DATE_FORMAT = "%Y-%m-%d"
