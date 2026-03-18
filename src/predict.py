"""Load a trained model and predict on new data."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import METRICS_PATH, MODEL_PATH, USE_LOG_TARGET


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def load_metrics(path: Path = METRICS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return predicted delay_min (real minutes)."""
    raw = model.predict(X)
    if USE_LOG_TARGET:
        return np.clip(np.expm1(raw), 0, None)
    return raw
