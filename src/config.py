"""Shared paths and column names (used by predict.py and Streamlit pages)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_FLIGHTS_CSV = DATA_DIR / "schedule_airport.csv"
WEATHER_CSV = DATA_DIR / "weather_zrh.csv"
AIRPORTS_CSV = DATA_DIR / "airports-extended-clean.csv"
PROCESSED_CSV = DATA_DIR / "flights_processed.csv"

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"

ZRH_LAT = 47.4647
ZRH_LON = 8.5492
