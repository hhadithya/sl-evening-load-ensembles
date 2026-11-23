#!/usr/bin/env python3
"""
PeakWise Phase 1: feature table, chronological splits and baselines for next-day evening forecasting.

Target
------
y(h, D) = household h's grid import in kWh over 18:30-22:30 Sri Lanka time on day D,
taken from the audited file `artifacts/audit/peakwise_target_clean.csv` (464,640 valid household-days).
The raw 45.6M-row pipeline is NOT rebuilt here.

Forecast setting
----------------
The forecast for day D is made at the end of day D-1, so a feature may only use information
from dates <= D - HORIZON_DAYS (HORIZON_DAYS = 1). Every feature below is computed from the
household's own valid evening values on a complete daily calendar, shifted by one day first,
so no value from day D or later can enter a feature of day D. `tests/check_leakage.py` tests this.

Design notes (why these features)
---------------------------------
* The EDA showed whole-day gaps (about 46% of quarter-hours missing, most of it entire dates),
  so features must survive missing history. Every rolling statistic uses min_periods=1 on a
  CALENDAR window, and `gap_days` / `roll*_n` tell a model how stale and how thin the history is.
* The target is strongly right-skewed with 8% exact zeros, so level features are medians, not
  means, and the share of zeros in the recent window is given explicitly.
* Households differ by more than an order of magnitude, so the level features are per household;
  no global scaling is applied and household_ID itself is not used as a model input.
* Weekday effect in the audited target is weak (medians 0.69-0.74 kWh across weekdays), so the
  calendar part is kept to dow / weekend / month rather than elaborate seasonal encodings.

Usage
-----
    python -I src/phase1.py --audit artifacts/audit --out artifacts/phase1
Writes: model_table.csv.gz, baseline_metrics.csv, baseline_predictions_sample.csv, phase1_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_DAYS = 1                      # forecast day D using information up to the end of D-1
SHORT_WINDOW, LONG_WINDOW = 7, 28     # calendar days

# chronological, non-overlapping, test period untouched until the very end
SPLIT_DATES = {"train": ("2023-10-01", "2024-09-30"),
               "val":   ("2024-10-01", "2024-11-10"),
               "test":  ("2024-11-11", "2024-12-22")}

FEATURES = ["lag1", "lag7", "last_value", "gap_days",
            "roll7_median", "roll28_median", "roll28_iqr", "roll28_frac_zero",
            "roll7_n", "roll28_n", "expanding_median",
            "dow", "is_weekend", "month"]

FEATURE_NOTES = {
    "lag1": "yesterday's evening kWh (NaN if yesterday is not a valid household-day)",
    "lag7": "same weekday last week, evening kWh (NaN if missing)",
    "last_value": "most recent valid evening kWh on or before D-1 (never NaN once any history exists)",
    "gap_days": "calendar days between D and that most recent valid evening (1 = yesterday)",
    "roll7_median": "median evening kWh over the 7 calendar days D-7..D-1 (observed days only)",
    "roll28_median": "median evening kWh over D-28..D-1: the household's recent level",
    "roll28_iqr": "p75-p25 of the same window: how variable the household has been",
    "roll28_frac_zero": "share of observed evenings in D-28..D-1 that are exactly 0 kWh",
    "roll7_n": "number of observed evenings in D-7..D-1 (0-7): recent coverage",
    "roll28_n": "number of observed evenings in D-28..D-1 (0-28): history depth",
    "expanding_median": "median of every valid evening the household has up to D-1: long-run level",
    "dow": "day of week of D (0 = Monday); known in advance",
    "is_weekend": "1 if D is Saturday or Sunday; known in advance",
    "month": "calendar month of D; known in advance",
}

BASELINES = {
    "global_median": "constant: median of the TRAINING targets",
    "persistence_lag1": "yesterday's evening value (only where yesterday exists)",
    "persistence_last": "last observed evening value, carried forward over gaps",
    "seasonal_lag7": "same weekday last week (only where it exists)",
    "roll7_median": "median of the last 7 calendar days' observed evenings",
    "roll28_median": "median of the last 28 calendar days' observed evenings",
    "expanding_median": "median of all of the household's past evenings",
}


# --------------------------------------------------------------------------------------- loading
def read_audit_csv(path: Path) -> pd.DataFrame:
    """Audit CSVs may be space-padded if an editor aligned them; strip rather than trust."""
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = d.columns.str.strip()
    for c in d.select_dtypes("object"):
        d[c] = d[c].str.strip()
    return d


def load_target(audit_dir: Path) -> pd.DataFrame:
    t = read_audit_csv(Path(audit_dir) / "peakwise_target_clean.csv")
    t = t.rename(columns={"peak_kwh_1830_2230": "y"})
    t["date"] = pd.to_datetime(t["date"])
    return t.sort_values(["household_ID", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------- features
def _stack(df: pd.DataFrame) -> pd.Series:
    """DataFrame -> long Series keeping NaN cells (pandas renamed the option in 2.1)."""
    try:
        return df.stack(future_stack=True)
    except TypeError:                                    # pandas < 2.1
        return df.stack(dropna=False)


def build_features(target: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """One row per valid household-day, with features that use only dates <= D - horizon.

    Works on a households x calendar-days matrix: every rolling window is a CALENDAR window on a
    series that has already been shifted by `horizon` days, so a feature of day D physically
    cannot see day D (or anything after it).
    """
    piv = target.pivot(index="date", columns="household_ID", values="y")
    cal = pd.date_range(piv.index.min(), piv.index.max(), freq="D")
    piv = piv.reindex(cal)                       # missing days are explicit NaN, never 0
    prev = piv.shift(horizon)                    # information available when day D is forecast

    f = {
        "lag1": piv.shift(horizon),
        "lag7": piv.shift(horizon + 6),          # with horizon=1 this is D-7, the same weekday
        "roll7_median": prev.rolling(SHORT_WINDOW, min_periods=1).median(),
        "roll28_median": prev.rolling(LONG_WINDOW, min_periods=1).median(),
        "roll7_n": prev.rolling(SHORT_WINDOW, min_periods=1).count(),
        "roll28_n": prev.rolling(LONG_WINDOW, min_periods=1).count(),
        "expanding_median": prev.expanding(min_periods=1).median(),
        "last_value": prev.ffill(),
    }
    q75 = prev.rolling(LONG_WINDOW, min_periods=1).quantile(0.75)
    q25 = prev.rolling(LONG_WINDOW, min_periods=1).quantile(0.25)
    f["roll28_iqr"] = q75 - q25
    zeros = prev.eq(0).where(prev.notna())       # NaN stays NaN, so unobserved days do not count
    f["roll28_frac_zero"] = zeros.rolling(LONG_WINDOW, min_periods=1).sum() / f["roll28_n"]

    # days between D and the most recent observation on or before D-horizon (1 = yesterday)
    day_no = pd.DataFrame(np.tile(np.arange(len(cal), dtype=float)[:, None], (1, piv.shape[1])),
                          index=cal, columns=piv.columns)
    f["gap_days"] = day_no - (day_no - horizon).where(prev.notna()).ffill()

    long = pd.concat({k: _stack(v) for k, v in f.items()}, axis=1)
    long.index.names = ["date", "household_ID"]
    out = target.merge(long.reset_index(), on=["household_ID", "date"], how="left")
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["month"] = out["date"].dt.month
    return out


def assign_split(df: pd.DataFrame, split_dates: dict = None) -> pd.DataFrame:
    s = split_dates or SPLIT_DATES
    df = df.copy()
    df["split"] = pd.NA
    for name, (a, b) in s.items():
        df.loc[(df["date"] >= a) & (df["date"] <= b), "split"] = name
    return df


def apply_inclusion(df: pd.DataFrame, min_history: int = 1) -> pd.DataFrame:
    """Keep rows whose household has at least `min_history` observed evenings in the previous 28 days."""
    return df[df["roll28_n"].fillna(0) >= min_history].reset_index(drop=True)


def cv_folds(df: pd.DataFrame, n_folds: int = 3, val_days: int = 41) -> list:
    """Expanding-window chronological folds INSIDE the training period, for Phase 2 tuning.

    Fold k trains on everything up to a cut date and validates on the next `val_days` days, so a
    model is never tuned on data that precedes its own training window. Returns
    [(train_index, valid_index), ...] of positional indices into `df`.
    """
    tr = df[df["split"] == "train"]
    end = pd.Timestamp(SPLIT_DATES["train"][1])
    folds = []
    for k in range(n_folds, 0, -1):
        v_start = end - pd.Timedelta(days=val_days * k - 1)
        v_end = v_start + pd.Timedelta(days=val_days - 1)
        a = np.flatnonzero((df["split"] == "train") & (df["date"] < v_start))
        b = np.flatnonzero((df["split"] == "train") & (df["date"] >= v_start) & (df["date"] <= v_end))
        if len(a) and len(b):
            folds.append((a, b))
    assert len(tr), "no training rows"
    return folds


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """For models without native NaN support (e.g. RandomForestRegressor): fill every feature from
    the SAME ROW's level features and flag what was filled. No statistic is computed across rows,
    so this cannot leak. The result is guaranteed to contain no NaN in FEATURES.
    """
    out = df.copy()
    for c in ("lag1", "lag7", "roll7_median"):
        out[f"{c}_missing"] = out[c].isna().astype("int8")
        out[c] = out[c].fillna(out["last_value"]).fillna(out["roll28_median"])
    out["roll28_iqr"] = out["roll28_iqr"].fillna(0.0)
    out["roll28_frac_zero"] = out["roll28_frac_zero"].fillna(0.0)
    missing = [c for c in FEATURES if out[c].isna().any()]
    if missing:                                       # only possible if the inclusion rule was skipped
        raise ValueError(f"cannot fill {missing} from the same row: apply_inclusion() first")
    return out


FILLED_FEATURES = FEATURES + ["lag1_missing", "lag7_missing", "roll7_median_missing"]


# --------------------------------------------------------------------------------------- baselines
def baseline_predictions(df: pd.DataFrame, train_median: float) -> pd.DataFrame:
    return pd.DataFrame({
        "global_median": np.full(len(df), train_median),
        "persistence_lag1": df["lag1"].to_numpy(),
        "persistence_last": df["last_value"].to_numpy(),
        "seasonal_lag7": df["lag7"].to_numpy(),
        "roll7_median": df["roll7_median"].to_numpy(),
        "roll28_median": df["roll28_median"].to_numpy(),
        "expanding_median": df["expanding_median"].to_numpy(),
    }, index=df.index)


def metrics(y: np.ndarray, p: np.ndarray, households: np.ndarray = None) -> dict:
    """MAE/RMSE in kWh plus per-household MAE summaries (mean and median over households), which are
    not dominated by big consumers. (Earlier versions called the median one `macro_MAE`.)"""
    ok = np.isfinite(p) & np.isfinite(y)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "coverage": 0.0, "MAE": np.nan, "RMSE": np.nan, "MedAE": np.nan,
                "hh_MAE_mean": np.nan, "hh_MAE_median": np.nan}
    e = p[ok] - y[ok]
    out = {"n": n, "coverage": n / len(y), "MAE": float(np.abs(e).mean()),
           "RMSE": float(np.sqrt((e ** 2).mean())), "MedAE": float(np.median(np.abs(e)))}
    if households is not None:
        per = pd.Series(np.abs(e)).groupby(pd.Series(households[ok]).to_numpy()).mean()
        out["hh_MAE_mean"] = float(per.mean())        # mean over households of each household's own MAE
        out["hh_MAE_median"] = float(per.median())    # median over households: the "typical household"
        out["n_households"] = int(per.size)
    return out


def evaluate(df: pd.DataFrame, train_median: float, restrict_common: bool = False) -> pd.DataFrame:
    preds = baseline_predictions(df, train_median)
    rows = []
    sub = df
    if restrict_common:                                  # same rows for every baseline
        keep = preds.notna().all(axis=1)
        sub, preds = df[keep], preds[keep]
    y, hh = sub["y"].to_numpy(), sub["household_ID"].to_numpy()
    for name in preds.columns:
        rows.append({"baseline": name, **metrics(y, preds[name].to_numpy(), hh)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=Path("artifacts/audit"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/phase1"))
    ap.add_argument("--min-history", type=int, default=1)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    target = load_target(a.audit)
    feats = assign_split(build_features(target))
    kept = apply_inclusion(feats, a.min_history)

    summary = {"target_rows": len(target), "households": int(target.household_ID.nunique()),
               "date_range": [str(target.date.min().date()), str(target.date.max().date())],
               "horizon_days": HORIZON_DAYS, "inclusion_rule": f"roll28_n >= {a.min_history}",
               "rows_dropped_by_inclusion": len(feats) - len(kept),
               "rows_outside_split_windows": int(feats["split"].isna().sum()),
               "history_threshold_counts": {str(k): int((feats["roll28_n"].fillna(0) >= k).sum())
                                            for k in (1, 3, 7, 14)},
               "splits": {}, "features": FEATURE_NOTES, "baselines": BASELINES}

    train = kept[kept.split == "train"]
    train_median = float(train["y"].median())
    summary["train_median_kwh"] = train_median

    for name in ("train", "val", "test"):
        s = kept[kept.split == name]
        summary["splits"][name] = {
            "dates": list(SPLIT_DATES[name]), "rows": len(s), "households": int(s.household_ID.nunique()),
            "y_mean": float(s.y.mean()), "y_median": float(s.y.median()),
            "y_zero_share": float((s.y == 0).mean()),
            "missing_lag1_share": float(s.lag1.isna().mean()),
            "missing_lag7_share": float(s.lag7.isna().mean()),
            "median_gap_days": float(s.gap_days.median())}
    seen = set(train.household_ID)
    for name in ("val", "test"):
        s = kept[kept.split == name]
        summary["splits"][name]["households_unseen_in_train"] = int(s.household_ID.nunique() - s.household_ID.isin(seen).groupby(s.household_ID).any().sum())
        summary["splits"][name]["rows_from_unseen_households"] = int((~s.household_ID.isin(seen)).sum())

    res = []
    for name in ("val", "test"):
        s = kept[kept.split == name]
        for common in (False, True):
            r = evaluate(s, train_median, restrict_common=common)
            r.insert(0, "split", name); r.insert(1, "subset", "common_rows" if common else "all_rows")
            res.append(r)
    results = pd.concat(res, ignore_index=True)

    summary["cv_folds_inside_train"] = [
        {"train_rows": int(len(i)), "valid_rows": int(len(j)),
         "valid_dates": [str(kept["date"].iloc[j].min().date()), str(kept["date"].iloc[j].max().date())]}
        for i, j in cv_folds(kept)]

    cols = ["household_ID", "date", "split", "y", "tr2_check"] + FEATURES
    kept[cols].to_csv(a.out / "model_table.csv.gz", index=False)
    results.to_csv(a.out / "baseline_metrics.csv", index=False)
    (a.out / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("target_rows", "rows_dropped_by_inclusion", "splits")},
                     indent=2, default=str))
    print(results.round(4).to_string(index=False))
    return kept, results, summary


if __name__ == "__main__":
    main()
