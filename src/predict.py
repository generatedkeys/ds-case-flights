"""Load trained models and predict on new data."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_COLUMNS_PATH,
    ARTIFACTS_DIR,
)


def load_best_model() -> Pipeline:
    return joblib.load(BEST_MODEL_PATH)


def load_all_models() -> dict[str, Pipeline]:
    comparison = load_model_comparison()
    models = {}
    for name in comparison:
        if name.startswith("_"):
            continue
        safe = name.lower().replace(" ", "_")
        path = ARTIFACTS_DIR / f"model_{safe}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def load_model_comparison() -> dict:
    with open(MODEL_COMPARISON_PATH) as f:
        return json.load(f)


def load_feature_meta() -> dict:
    with open(FEATURE_COLUMNS_PATH) as f:
        return json.load(f)


def build_feature_matrix(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Build the feature matrix from a processed-data DataFrame.

    Encodes categoricals using the label-encoder classes saved during
    training and selects columns in the correct order.
    """
    le_classes = meta["label_encoders"]
    out = df.copy()
    for col in meta["categorical_features"]:
        classes = le_classes[col]
        out[col + "_enc"] = (
            out[col]
            .fillna("UNKNOWN")
            .astype(str)
            .map({v: i for i, v in enumerate(classes)})
            .fillna(0)
            .astype(int)
        )
    return out[meta["feature_cols"]]


def predict_delay_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def predict_delay_class(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
