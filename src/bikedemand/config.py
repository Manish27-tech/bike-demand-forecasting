"""Paths, dataset pins and modelling constants for the demand-forecasting pipeline.

Two constants carry most of the methodological meaning:

* ``TEST_START`` -- a hard time boundary. Everything before it is training data,
  everything after is the hold-out period. No random shuffling is allowed
  anywhere in the pipeline.
* ``LAG_HOURS`` -- the lag structure. These are computed on a complete hourly
  grid, so a "1-hour lag" is genuinely one hour, not one observed row.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"
NOTEBOOKS = ROOT / "notebooks"

# --------------------------------------------------------------------------- #
# Dataset: UCI Machine Learning Repository #275, Bike Sharing Dataset
# --------------------------------------------------------------------------- #
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ARCHIVE_NAME = "bike.zip"
CSV_NAME = "hour.csv"
ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"

TARGET = "cnt"
# `casual` and `registered` sum to the target, so they are leakage, not features.
LEAKAGE_COLUMNS = ("casual", "registered")
ID_COLUMNS = ("instant",)

# Hold-out period: Q4 2012. The model never sees these hours during fitting.
TEST_START = "2012-10-01"

RANDOM_STATE = 42
LAG_HOURS = (1, 24, 168)
ROLLING_WINDOWS = (24, 168)
# Minimum observed hours required before lag features are considered valid.
MIN_HISTORY = 168

WEATHER_LABELS = {
    1: "Clear",
    2: "Mist / cloud",
    3: "Light rain / snow",
    4: "Heavy rain / snow",
}
SEASON_LABELS = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}

FIGURE_DPI = 140
PALETTE = {"actual": "#4C72B0", "predicted": "#C44E52"}
ACCENT = "#DD8452"
