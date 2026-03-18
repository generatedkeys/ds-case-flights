"""Pipeline construction and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Ridge regression with OneHot categoricals + scaled numerics."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("impute", SimpleImputer(
                    strategy="constant", fill_value="__MISSING__")),
                ("encode", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of regression metrics."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }


def evaluate_by_group(y_true, y_pred, groups) -> dict[str, dict]:
    """Evaluate per unique value of *groups* (e.g. LSV = L/S)."""
    results = {}
    for g in sorted(set(groups)):
        mask = groups == g
        results[str(g)] = evaluate(y_true[mask], y_pred[mask])
    return results


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
