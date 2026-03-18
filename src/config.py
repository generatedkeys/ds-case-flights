"""Shared constants and paths for the modelling pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "schedule_airport.csv"
AIRPORTS_CSV = DATA_DIR / "airports-extended-clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

COL_DATE = "STD"
COL_FLIGHT = "FLT"
COL_SCHEDULED_TIME = "STA_STD_ltc"
COL_ACTUAL_TIME = "ATA_ATD_ltc"
COL_LSV = "LSV"
COL_TERMINAL = "TAR"
COL_GATE = "GAT"
COL_AIRCRAFT = "ACT"
COL_RUNWAY = "RWY"
COL_RUNWAY_CONFIG = "RWC"
COL_ORG_DES = "Org/Des"

TARGET = "delay_min"
DELAY_CAP_MIN = 20
USE_LOG_TARGET = True

CATEGORICAL_FEATURES = [
    COL_LSV,
    COL_TERMINAL,
    "gate_zone",
    COL_AIRCRAFT,
    COL_RUNWAY,
    COL_RUNWAY_CONFIG,
    COL_ORG_DES,
    "carrier",
]
NUMERIC_FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "moves_30m",
    "moves_60m",
    "arr_30m",
    "dep_30m",
]
