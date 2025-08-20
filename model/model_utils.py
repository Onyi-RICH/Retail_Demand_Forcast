import joblib
import pandas as pd
import numpy as np
from app import config

def load_model(model_path: str = config.MODEL_PATH):
    """Load trained model from .pkl file."""
    return joblib.load(model_path)
