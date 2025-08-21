# app/config.py

# Paths
MODEL_PATH = "model/xgboost_model.pkl"
DATA_PATH = "data/data_streamlit.csv"
FEATURES_PATH = "data/feature_columns.parquet" 


ARTIFACT_PATH = "mlflow_results/296212850623942326/7530595e79af4a7aaa32ecab656b30cd/artifacts/hypertuned"
RUN_ID = "7530595e79af4a7aaa32ecab656b30cd" #'59ae7622fa6d491989efc2832fd807c3'

# App settings
APP_TITLE = "Corporación Favorita Sales Forecasting"
APP_DESCRIPTION = """
This app forecasts retail demand for Corporación Favorita, one of Ecuador's largest grocery retail chains.
"""
