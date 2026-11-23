"""Artifact loading for the dashboard. Everything is read from small saved files; nothing is computed
from raw data or models at start-up. Set PEAKWISE_DATA_DIR to point at another artifact folder."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PEAKWISE_DATA_DIR", ROOT / "dashboard" / "data"))

REQUIRED = ["metrics_validation.csv", "metrics_test.csv", "deltas_vs_baseline.csv",
            "bands_validation.csv", "bands_test.csv", "household_errors.csv",
            "model_costs_with_latency.csv", "permutation_importance.csv",
            "predictions_test_pseudonymized.csv.gz", "facts.json", "examples.json", "manifest.json", "best_specs.json"]

BASE = "base_roll7_median"
NAMES = {BASE: "7-day median (baseline)", "base_persistence": "Persistence (yesterday)",
         "base_seasonal_lag7": "Same weekday last week", "base_roll28_median": "28-day median",
         "base_expanding_median": "All-history median", "base_global_median": "Global median",
         "rf": "Random Forest", "xgb": "XGBoost", "xgb_alt": "XGBoost (log target)", "stack": "Stacking",
         "stack_equal_avg": "Equal-weight average"}
MAIN = [BASE, "rf", "xgb", "stack"]
ROLES = {BASE: "7-day median · baseline", "rf": "Random Forest · bagging", "xgb": "XGBoost · boosting", "stack": "Stacking · meta-learner"}


def missing_artifacts() -> list:
    return [f for f in REQUIRED if not (DATA_DIR / f).exists()]


@st.cache_data(show_spinner=False)
def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kw)


@st.cache_data(show_spinner=False)
def js(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def metrics(period: str) -> pd.DataFrame:
    return csv(f"metrics_{period}.csv", index_col="model")


def deltas() -> pd.DataFrame:
    return csv("deltas_vs_baseline.csv")


def bands(period: str) -> pd.DataFrame:
    return csv(f"bands_{period}.csv", index_col="band")


@st.cache_data(show_spinner=False)
def predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "predictions_test_pseudonymized.csv.gz", parse_dates=["date"])
