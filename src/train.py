"""Training entrypoint.  Run: python -m src.train"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    COL_ACTUAL_TIME,
    COL_DATE,
    COL_FLIGHT,
    COL_LSV,
    COL_SCHEDULED_TIME,
    DELAY_CAP_MIN,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RAW_CSV,
    TARGET,
    USE_LOG_TARGET,
)
from src.modeling import (
    build_pipeline,
    evaluate,
    evaluate_by_group,
    save_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# -- Data prep ----------------------------------------------------------------


def load_and_prepare() -> pd.DataFrame:
    """Load CSV, compute target + features, return clean dataframe."""
    df = pd.read_csv(RAW_CSV, on_bad_lines="skip")

    df["scheduled_dt"] = pd.to_datetime(
        df[COL_DATE] + " " + df[COL_SCHEDULED_TIME],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df["actual_dt"] = pd.to_datetime(
        df[COL_DATE] + " " + df[COL_ACTUAL_TIME],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    delay = (df["actual_dt"] - df["scheduled_dt"]).dt.total_seconds() / 60
    df[TARGET] = delay.clip(lower=0, upper=DELAY_CAP_MIN)

    df["hour"] = df["scheduled_dt"].dt.hour
    df["minute"] = df["scheduled_dt"].dt.minute
    df["day_of_week"] = df["scheduled_dt"].dt.dayofweek
    df["month"] = df["scheduled_dt"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["carrier"] = df[COL_FLIGHT].str.extract(r"^([A-Z]{2,3})", expand=False)
    df["gate_zone"] = df["GAT"].str.extract(r"^([A-Z])", expand=False)

    df = df.dropna(subset=[TARGET, "scheduled_dt"]).reset_index(drop=True)

    df = _add_congestion(df)
    return df


def _add_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Count movements in ±30m / ±60m windows (vectorised)."""
    df = df.sort_values("scheduled_dt").reset_index(drop=True)
    epoch = df["scheduled_dt"].values.astype("int64") // 10**9
    epoch = epoch.astype(np.float64)
    is_arr = (df[COL_LSV] == "L").values
    arr_cum = np.cumsum(is_arr)

    def _window(half_sec):
        lo = np.searchsorted(epoch, epoch - half_sec, side="left")
        hi = np.searchsorted(epoch, epoch + half_sec, side="right")
        total = hi - lo - 1
        hi_idx = np.clip(hi - 1, 0, len(arr_cum) - 1)
        lo_idx = np.clip(lo - 1, 0, len(arr_cum) - 1)
        a = arr_cum[hi_idx] - np.where(lo > 0, arr_cum[lo_idx], 0)
        a -= is_arr.astype(int)
        return total, a, total - a

    m30, a30, d30 = _window(1800)
    m60, _, _ = _window(3600)

    df["moves_30m"] = m30
    df["moves_60m"] = m60
    df["arr_30m"] = a30
    df["dep_30m"] = d30
    return df


# -- Train / eval -------------------------------------------------------------


def time_split(df, test_frac=0.2):
    df = df.sort_values("scheduled_dt").reset_index(drop=True)
    i = int(len(df) * (1 - test_frac))
    return df.iloc[:i], df.iloc[i:]


def main() -> None:
    log.info("Loading & preparing data ...")
    df = load_and_prepare()
    log.info("Rows: %d", len(df))

    train_df, test_df = time_split(df)
    log.info("Train: %d  |  Test: %d", len(train_df), len(test_df))

    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    if USE_LOG_TARGET:
        y_train = np.log1p(y_train)

    print("X_train: ")
    print(X_train.head(20))
    print("y_train: ")
    print(y_train.head(20))

    log.info("Fitting Ridge ...")
    pipeline = build_pipeline(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    pipeline.fit(X_train, y_train)

    y_pred_raw = pipeline.predict(X_test)
    if USE_LOG_TARGET:
        y_pred = np.clip(np.expm1(y_pred_raw), 0, None)
        y_true = np.array(y_test)
    else:
        y_pred = y_pred_raw
        y_true = np.array(y_test)

    metrics = {
        "overall": evaluate(y_true, y_pred),
        "by_lsv": evaluate_by_group(
            y_true, y_pred, np.array(test_df[COL_LSV])
        ),
    }
    log.info("Overall: %s", metrics["overall"])
    for g, m in metrics["by_lsv"].items():
        log.info("  %s: %s", g, m)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)
    log.info("Saved model + metrics → %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
