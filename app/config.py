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
This app forecasts retail demand for Corporación Favorita, one of Ecuador's largest grocery retail chains.
"""
