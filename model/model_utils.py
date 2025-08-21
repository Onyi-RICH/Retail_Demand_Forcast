import joblib
import pandas as pd
import numpy as np
from app import config
import mlflow
import mlflow.xgboost


def load_model(run_id: str = config.RUN_ID, artifact_path: str = config.ARTIFACT_PATH):
    """
    Load a model from MLflow using the run_id and artifact_path.
    """
    #model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.xgboost.load_model(artifact_path)


#def load_model(model_path: str = config.MODEL_PATH):
#    """Load trained model from .pkl file."""
#    return joblib.load(model_path)

