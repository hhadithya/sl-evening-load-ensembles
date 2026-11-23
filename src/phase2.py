#!/usr/bin/env python3
"""
PeakWise Phase 2: bagging (Random Forest), boosting (XGBoost) and stacking, compared with the
historical baselines on the Phase 1 model table.

Protocol (fixed before any model is fitted; see notebooks/02_ensembles.ipynb for the reasoning)
----------------------------------------------------------------------------------------------
* Data:      artifacts/phase1/model_table.csv.gz, features and splits from phase1.py (unchanged).
* Tuning:    the 3 expanding-window chronological folds inside the TRAIN period only
             (`phase1.cv_folds`). The validation period and the test period are not seen by tuning.
* Selection: model families are compared on the VALIDATION period (models fitted on train only).
* Test:      read exactly once, at the very end, with models REFITTED on train+validation using the
             hyperparameters chosen above. No setting is changed after the test period is opened.
* Stacking:  base models give genuinely out-of-fold predictions. For fold k a base model is fitted
             on rows dated before the fold's validation window and predicts only that window. The
             validation period acts as a fourth fold. The meta-learner is a non-negative linear
             combination fitted on those out-of-fold predictions only:
               - evaluating on validation: meta fitted on folds 1-3        (all dated <= 2024-09-30)
               - evaluating on test:       meta fitted on folds 1-3 + validation (all dated <= 2024-11-10)
* Comparison population: every model, baseline included, gets a prediction for EVERY row
             (baselines use an explicit fallback chain, see `baseline_frame`). Headline numbers use
             rows of households that have training data; rows of unseen households are reported apart.

Usage:  python src/phase2.py tune | final | all      (artifacts in artifacts/phase2/)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import phase1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/phase2"
MODEL_DIR = ROOT / "artifacts/models"          # trained models, published with the repository
SEED = 42
N_JOBS = 10
XGB_CHECKPOINTS = list(range(50, 1501, 50))     # cap raised from 600: abs-error models were still improving at 600
VAL_END = pd.Timestamp(phase1.SPLIT_DATES["val"][1])
TRAIN_END = pd.Timestamp(phase1.SPLIT_DATES["train"][1])


# ------------------------------------------------------------------------------------ data
def load_table(include_test: bool, path: Path = None) -> pd.DataFrame:
    """Phase 1 model table. With include_test=False the test rows are not even loaded into memory,
    so a tuning mistake that peeks at the test period is structurally impossible."""
    m = pd.read_csv(path or (ROOT / "artifacts/phase1/model_table.csv.gz"), parse_dates=["date"])
    if not include_test:
        m = m[m["split"] != "test"]
    return m.sort_values(["date", "household_ID"]).reset_index(drop=True)


def design(m: pd.DataFrame, kind: str) -> np.ndarray:
    """Feature matrix. XGBoost keeps NaN (native missing-value handling, learned default direction);
    Random Forest gets the row-local, flagged fill from phase1.fill_missing."""
    if kind == "rf":
        return phase1.fill_missing(m)[phase1.FILLED_FEATURES].to_numpy(dtype=np.float32)
    return m[phase1.FEATURES].to_numpy(dtype=np.float32)


def to_target(y: np.ndarray, mode: str) -> np.ndarray:
    return np.log1p(y) if mode == "log" else y


def from_target(p: np.ndarray, mode: str) -> np.ndarray:
    return np.clip(np.expm1(p) if mode == "log" else p, 0.0, None)      # consumption cannot be negative


# ------------------------------------------------------------------------------------ baselines
BASELINE_RULES = {
    "global_median": "training median (fitted on the training rows of the fit window only)",
    "persistence": "yesterday's evening; if yesterday is missing, the last observed evening",
    "seasonal_lag7": "same weekday last week; if missing, the last observed evening",
    "roll7_median": "median of the last 7 calendar days; if none observed, the last observed evening",
    "roll28_median": "median of the last 28 calendar days (always defined after inclusion)",
    "expanding_median": "median of all of the household's past evenings (always defined)",
}


def baseline_frame(m: pd.DataFrame, train_median: float) -> pd.DataFrame:
    """Every baseline answers on every row, through an explicit fallback, so that all comparisons
    run on the same rows. The fallbacks only ever use information available at forecast time."""
    return pd.DataFrame({
        "global_median": np.full(len(m), train_median),
        "persistence": m["lag1"].fillna(m["last_value"]).to_numpy(),
        "seasonal_lag7": m["lag7"].fillna(m["last_value"]).to_numpy(),
        "roll7_median": m["roll7_median"].fillna(m["last_value"]).to_numpy(),
        "roll28_median": m["roll28_median"].to_numpy(),
        "expanding_median": m["expanding_median"].to_numpy(),
    }, index=m.index)


# ------------------------------------------------------------------------------------ models
def build_model(spec: dict, n_estimators: int = None):
    p = spec["params"]
    if spec["kind"] == "rf":
        return RandomForestRegressor(n_estimators=n_estimators or spec["n_estimators"],
                                     min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                     max_samples=p.get("max_samples", 0.5), n_jobs=N_JOBS, random_state=SEED)
    return xgb.XGBRegressor(objective=spec["objective"], n_estimators=n_estimators or spec["rounds"],
                            learning_rate=p.get("learning_rate", 0.05), max_depth=p["max_depth"],
                            min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            n_jobs=N_JOBS, random_state=SEED, verbosity=0)


def fit_model(spec: dict, m_train: pd.DataFrame, n_estimators: int = None):
    model = build_model(spec, n_estimators)
    model.fit(design(m_train, spec["kind"]), to_target(m_train["y"].to_numpy(), spec["target"]))
    return model


def predict_model(spec: dict, model, m: pd.DataFrame, iteration: int = None) -> np.ndarray:
    X = design(m, spec["kind"])
    if spec["kind"] == "xgb" and iteration is not None:
        raw = model.predict(X, iteration_range=(0, iteration))
    else:
        raw = model.predict(X)
    return from_target(raw, spec["target"])


def _scores(y, p):
    e = p - y
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


# ------------------------------------------------------------------------------------ tuning
def xgb_spec(name, objective, target, depth, mcw, lam, rounds=300):
    return {"name": name, "kind": "xgb", "objective": objective, "target": target, "rounds": rounds,
            "params": {"max_depth": depth, "min_child_weight": mcw, "reg_lambda": lam, "learning_rate": 0.05}}


def rf_spec(name, target, msl, mf, n_estimators=100):
    return {"name": name, "kind": "rf", "target": target, "n_estimators": n_estimators,
            "params": {"min_samples_leaf": msl, "max_features": mf, "max_samples": 0.5}}


def tune_xgb(m: pd.DataFrame, folds, specs: list, log=print) -> pd.DataFrame:
    """One fit per (config, fold) at the maximum round count, then the fold error is read off at every
    checkpoint with `iteration_range`, so the number of boosting rounds is tuned almost for free."""
    rows = []
    for spec in specs:
        t0 = time.time()
        per_fold = []
        for k, (a, b) in enumerate(folds, 1):
            mod = fit_model(spec, m.iloc[a], n_estimators=max(XGB_CHECKPOINTS))
            yv = m["y"].to_numpy()[b]
            for r in XGB_CHECKPOINTS:
                mae, rmse = _scores(yv, predict_model(spec, mod, m.iloc[b], iteration=r))
                per_fold.append({"fold": k, "rounds": r, "MAE": mae, "RMSE": rmse})
        pf = pd.DataFrame(per_fold).groupby("rounds")[["MAE", "RMSE"]].mean()
        best_r = int(pf["MAE"].idxmin())
        rows.append({"name": spec["name"], "kind": "xgb", "objective": spec["objective"], "target": spec["target"],
                     **spec["params"], "rounds": best_r, "cv_MAE": float(pf.loc[best_r, "MAE"]),
                     "cv_RMSE": float(pf.loc[best_r, "RMSE"]), "seconds": time.time() - t0})
        log(f"  {spec['name']:<26} rounds={best_r:>3}  cv MAE {rows[-1]['cv_MAE']:.4f}  RMSE {rows[-1]['cv_RMSE']:.4f}  ({rows[-1]['seconds']:.0f}s)")
    return pd.DataFrame(rows)


def tune_rf(m: pd.DataFrame, folds, specs: list, log=print) -> pd.DataFrame:
    rows = []
    for spec in specs:
        t0 = time.time(); maes, rmses = [], []
        for a, b in folds:
            mod = fit_model(spec, m.iloc[a])
            mae, rmse = _scores(m["y"].to_numpy()[b], predict_model(spec, mod, m.iloc[b]))
            maes.append(mae); rmses.append(rmse)
        rows.append({"name": spec["name"], "kind": "rf", "objective": "squared error", "target": spec["target"],
                     **spec["params"], "rounds": spec["n_estimators"], "cv_MAE": float(np.mean(maes)),
                     "cv_RMSE": float(np.mean(rmses)), "seconds": time.time() - t0})
        log(f"  {spec['name']:<26} cv MAE {rows[-1]['cv_MAE']:.4f}  RMSE {rows[-1]['cv_RMSE']:.4f}  ({rows[-1]['seconds']:.0f}s)")
    return pd.DataFrame(rows)


def run_tuning(out: Path = OUT, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = load_table(include_test=False)                       # the test rows are never loaded here
    folds = phase1.cv_folds(m)
    log(f"tuning on {len(m):,} train+val rows, but folds come from the TRAIN period only: "
        f"{[(len(a), len(b)) for a, b in folds]}")
    tr = m[m["split"] == "train"]
    base = baseline_frame(m, float(tr["y"].median()))["roll7_median"].to_numpy()
    bcv = [_scores(m["y"].to_numpy()[b], base[b]) for a, b in folds]
    ref = {"cv_MAE": float(np.mean([x[0] for x in bcv])), "cv_RMSE": float(np.mean([x[1] for x in bcv]))}
    log(f"reference: roll7_median baseline on the same folds: cv MAE {ref['cv_MAE']:.4f}  RMSE {ref['cv_RMSE']:.4f}")

    # --- XGBoost, stage 1: the choices that change what the model optimises (loss/target) and its capacity
    log("XGBoost stage 1: loss/target x depth")
    s1 = [xgb_spec(f"xgb_{tag}_d{d}", obj, tgt, d, 50, 1.0)
          for tag, obj, tgt in [("sq", "reg:squarederror", "raw"), ("sqlog", "reg:squarederror", "log"),
                                ("abs", "reg:absoluteerror", "raw")] for d in (3, 6)]
    r1 = tune_xgb(m, folds, s1, log)
    best1 = r1.sort_values("cv_MAE").iloc[0]
    # --- XGBoost, stage 2: regularisation around the stage-1 winner, for both loss families
    log("XGBoost stage 2: regularisation around the winners")
    s2 = []
    for tag, obj, tgt in [("abs", "reg:absoluteerror", "raw"), ("sqlog", "reg:squarederror", "log")]:
        w = r1[(r1.objective == obj) & (r1.target == tgt)].sort_values("cv_MAE").iloc[0]
        for mcw, lam, depth in [(10, 1.0, int(w.max_depth)), (200, 1.0, int(w.max_depth)),
                                (50, 10.0, int(w.max_depth)), (50, 1.0, 9 if int(w.max_depth) == 6 else 5)]:
            s2.append(xgb_spec(f"xgb_{tag}_d{depth}_mcw{mcw}_l{int(lam)}", obj, tgt, depth, mcw, lam))
    r2 = tune_xgb(m, folds, s2, log)
    rx = pd.concat([r1, r2], ignore_index=True)

    # --- Random Forest: leaf size and feature subsampling are the bias/variance and correlation knobs
    log("Random Forest: min_samples_leaf x max_features x target")
    s3 = [rf_spec(f"rf_{t}_msl{msl}_mf{mf}", t, msl, mf)
          for t in ("raw", "log") for msl in (10, 50) for mf in (0.4, 1.0)]
    rr = tune_rf(m, folds, s3, log)

    res = pd.concat([rx, rr], ignore_index=True)
    res.to_csv(out / "tuning_results.csv", index=False)
    best = {}
    xr = rx.sort_values("cv_MAE")
    row = xr.iloc[0]
    best["xgb"] = xgb_spec("xgb", row.objective, row.target, int(row.max_depth), float(row.min_child_weight),
                           float(row.reg_lambda), int(row.rounds))
    other = xr[(xr.objective != row.objective) | (xr.target != row.target)]
    other = other[other.objective == ("reg:squarederror" if row.objective == "reg:absoluteerror" else "reg:absoluteerror")]
    o = other.iloc[0]
    best["xgb_alt"] = xgb_spec("xgb_alt", o.objective, o.target, int(o.max_depth), float(o.min_child_weight),
                               float(o.reg_lambda), int(o.rounds))
    rrow = rr.sort_values("cv_MAE").iloc[0]
    best["rf"] = rf_spec("rf", rrow.target, int(rrow.min_samples_leaf), float(rrow.max_features), n_estimators=300)
    best["reference_baseline_cv"] = ref
    (out / "best_specs.json").write_text(json.dumps(best, indent=2))
    log("selected: " + json.dumps({k: (v.get("objective"), v.get("target"), v["params"]) for k, v in best.items() if k != "reference_baseline_cv"}))
    return best


# ------------------------------------------------------------------------------------ stacking
STACK_INPUTS = ["rf", "xgb", "xgb_alt", "baseline_roll7"]


def oof_predictions(m: pd.DataFrame, spec: dict, folds) -> tuple:
    """Out-of-fold predictions on the chronological folds. Returns (pred array with NaN outside the
    folds' validation windows, audit list of (train_max_date, valid_min_date, n_train, n_valid)).
    The model that predicts window k has been fitted ONLY on rows dated before window k."""
    pred = np.full(len(m), np.nan)
    audit = []
    for a, b in folds:
        mod = fit_model(spec, m.iloc[a])
        pred[b] = predict_model(spec, mod, m.iloc[b])
        audit.append((m["date"].iloc[a].max(), m["date"].iloc[b].min(), len(a), len(b)))
    return pred, audit


# numpy 2.x on macOS (Accelerate BLAS) emits spurious "divide by zero / overflow / invalid value in
# matmul" RuntimeWarnings for perfectly finite inputs. The results are verified independently in
# tests/check_phase2.py (least squares on the active columns reproduces the weights), so only this message
# is silenced, and only around the meta-learner.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def fit_meta(P: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Non-negative linear combination without intercept: interpretable weights, cannot extrapolate
    wildly, and with ~130k out-of-fold rows and 4 inputs there is no real risk of overfitting it."""
    meta = LinearRegression(positive=True, fit_intercept=False)
    return meta.fit(np.ascontiguousarray(P, dtype=np.float64), np.asarray(y, dtype=np.float64))


def stack_pipeline(m_nontest: pd.DataFrame, specs: dict, m_test: pd.DataFrame = None) -> dict:
    """The complete leakage-safe stacking procedure.

    m_nontest: train+validation rows (never test). m_test (optional): test rows, used ONLY for
    predicting, after everything else has been fitted. Returns predictions and an audit trail.
    """
    folds = phase1.cv_folds(m_nontest)
    train_idx = np.flatnonzero((m_nontest["split"] == "train").to_numpy())
    val_idx = np.flatnonzero((m_nontest["split"] == "val").to_numpy())
    base_names = ["rf", "xgb", "xgb_alt"]
    tr_med = float(m_nontest.iloc[train_idx]["y"].median())

    oof_cv, oof_val, val_pred, audit = {}, {}, {}, {}
    for n in base_names:
        oof_cv[n], audit[n] = oof_predictions(m_nontest, specs[n], folds)
        mod = fit_model(specs[n], m_nontest.iloc[train_idx])              # fourth "fold": train -> validation
        val_pred[n] = predict_model(specs[n], mod, m_nontest.iloc[val_idx])
    base_all = baseline_frame(m_nontest, tr_med)["roll7_median"].to_numpy()

    cv_rows = np.sort(np.unique(np.concatenate([b for _, b in folds])))
    y = m_nontest["y"].to_numpy()
    P_cv = np.column_stack([oof_cv[n][cv_rows] for n in base_names] + [base_all[cv_rows]])
    meta_val = fit_meta(P_cv, y[cv_rows])                                # folds 1-3 only
    P_val = np.column_stack([val_pred[n] for n in base_names] + [base_all[val_idx]])
    out = {"folds": folds, "audit": audit, "cv_rows": cv_rows, "val_idx": val_idx, "train_idx": train_idx,
           "meta_val": meta_val, "stack_val": np.clip(meta_val.predict(P_val), 0, None),
           "val_base": {n: val_pred[n] for n in base_names}, "P_cv": P_cv, "P_val": P_val,
           "meta_train_dates": (m_nontest["date"].iloc[cv_rows].min(), m_nontest["date"].iloc[cv_rows].max())}
    if m_test is not None:
        meta_test = fit_meta(np.vstack([P_cv, P_val]), np.concatenate([y[cv_rows], y[val_idx]]))
        full = m_nontest                                                 # base models refit on train+val
        test_base = {}
        for n in base_names:
            mod = fit_model(specs[n], full)
            test_base[n] = predict_model(specs[n], mod, m_test)
        tb = baseline_frame(m_test, float(full["y"].median()))["roll7_median"].to_numpy()
        P_test = np.column_stack([test_base[n] for n in base_names] + [tb])
        out.update({"meta_test": meta_test, "stack_test": np.clip(meta_test.predict(P_test), 0, None),
                    "test_base": test_base, "P_test": P_test,
                    "meta_test_dates": (full["date"].iloc[np.concatenate([cv_rows, val_idx])].min(),
                                        full["date"].iloc[np.concatenate([cv_rows, val_idx])].max())})
    return out


# ------------------------------------------------------------------------------------ evaluation
def seen_mask(eval_df: pd.DataFrame, fit_df: pd.DataFrame) -> np.ndarray:
    return eval_df["household_ID"].isin(set(fit_df["household_ID"])).to_numpy()


def metric_table(y, preds: dict, hh) -> pd.DataFrame:
    rows = [{"model": k, **phase1.metrics(y, np.asarray(v, dtype=float), hh)} for k, v in preds.items()]
    return pd.DataFrame(rows).set_index("model")


def paired_household_bootstrap(y, p_model, p_base, hh, n_boot: int = 400, seed: int = SEED) -> dict:
    """Resample HOUSEHOLDS (not rows: rows of one household are strongly dependent) and recompute the
    paired difference in MAE and RMSE, model minus baseline. Negative = model better."""
    y, a, b = (np.asarray(v, dtype=float) for v in (y, p_model, p_base))
    codes, uniq = pd.factorize(pd.Series(hh))
    H = len(uniq)
    def per_hh(v):
        return np.bincount(codes, weights=v, minlength=H)
    n = np.bincount(codes, minlength=H).astype(float)
    ae_m, ae_b = per_hh(np.abs(a - y)), per_hh(np.abs(b - y))
    se_m, se_b = per_hh((a - y) ** 2), per_hh((b - y) ** 2)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, H, H), minlength=H) for _ in range(n_boot)]).astype(float)
    N = W @ n
    d_mae = (W @ ae_m) / N - (W @ ae_b) / N
    d_rmse = np.sqrt((W @ se_m) / N) - np.sqrt((W @ se_b) / N)
    point_mae = ae_m.sum() / n.sum() - ae_b.sum() / n.sum()
    point_rmse = np.sqrt(se_m.sum() / n.sum()) - np.sqrt(se_b.sum() / n.sum())
    lo, hi = np.percentile(d_mae, [2.5, 97.5]); rlo, rhi = np.percentile(d_rmse, [2.5, 97.5])
    return {"d_MAE": float(point_mae), "d_MAE_lo": float(lo), "d_MAE_hi": float(hi),
            "d_RMSE": float(point_rmse), "d_RMSE_lo": float(rlo), "d_RMSE_hi": float(rhi), "n_households": int(H)}


def band_labels(y, train_y) -> pd.Series:
    q = np.quantile(train_y, [0.25, 0.75, 0.95, 0.99])
    return pd.cut(pd.Series(y), [-0.01, 0, q[0], q[1], q[2], q[3], np.inf],
                  labels=["exactly 0", "low (<P25)", "typical (P25-P75)", "high (P75-P95)",
                          "very high (P95-P99)", "extreme (>P99)"])


# ------------------------------------------------------------------------------------ reporting tables
MODELS = ["rf", "xgb", "xgb_alt", "stack"]
REFERENCE = "base_roll7_median"                 # the strongest historical baseline (best on validation AND test in Phase 1)
ALL_COLUMNS = ["base_global_median", "base_persistence", "base_seasonal_lag7", "base_roll28_median",
               "base_expanding_median", REFERENCE, "rf", "xgb", "xgb_alt", "stack", "stack_equal_avg"]
LABELS = {"base_global_median": "baseline: global median", "base_persistence": "baseline: persistence",
          "base_seasonal_lag7": "baseline: same weekday last week", "base_roll28_median": "baseline: 28-day median",
          "base_expanding_median": "baseline: all-history median", REFERENCE: "baseline: 7-day median (reference)",
          "rf": "Random Forest (bagging)", "xgb": "XGBoost, absolute-error loss (boosting)",
          "xgb_alt": "XGBoost, squared-error on log1p (boosting)", "stack": "Stacking (non-negative linear meta)",
          "stack_equal_avg": "equal-weight average (diagnostic only)"}


def metrics_table(pred: pd.DataFrame, mask=None) -> pd.DataFrame:
    s = pred if mask is None else pred[mask]
    t = metric_table(s["y"].to_numpy(), {k: s[k] for k in ALL_COLUMNS}, s["household_ID"].to_numpy())
    t["MAE_vs_ref_%"] = 100 * (t["MAE"] / t.loc[REFERENCE, "MAE"] - 1)
    t["RMSE_vs_ref_%"] = 100 * (t["RMSE"] / t.loc[REFERENCE, "RMSE"] - 1)
    return t


def delta_table(pred: pd.DataFrame, mask=None, models=MODELS, ref: str = REFERENCE, n_boot: int = 400) -> pd.DataFrame:
    """Paired household-bootstrap differences (model minus reference). Negative = model better."""
    s = pred if mask is None else pred[mask]
    if len(s) == 0:
        return pd.DataFrame()
    rows = [{"model": k, **paired_household_bootstrap(s["y"], s[k], s[ref], s["household_ID"], n_boot)} for k in models]
    return pd.DataFrame(rows).set_index("model")


def band_table(pred: pd.DataFrame, train_y: np.ndarray, models=("base_roll7_median", "rf", "xgb", "stack")) -> pd.DataFrame:
    s = pred.copy()
    s["band"] = band_labels(s["y"].to_numpy(), train_y).to_numpy()
    rows = []
    for b, g in s.groupby("band", observed=True):
        r = {"band": b, "n": len(g), "households": g["household_ID"].nunique()}
        for k in models:
            r[f"MAE {k}"] = float(np.abs(g[k] - g["y"]).mean())
            r[f"bias {k}"] = float((g[k] - g["y"]).mean())          # mean signed error: negative = under-prediction
        rows.append(r)
    out = pd.DataFrame(rows).set_index("band")
    order = ["exactly 0", "low (<P25)", "typical (P25-P75)", "high (P75-P95)", "very high (P95-P99)", "extreme (>P99)"]
    return out.loc[[b for b in order if b in out.index]]


def household_table(pred: pd.DataFrame, models=("rf", "xgb", "stack"), ref: str = REFERENCE) -> pd.DataFrame:
    e = {k: (pred[k] - pred["y"]).abs() for k in (ref,) + tuple(models)}
    per = pd.DataFrame(e).groupby(pred["household_ID"]).mean()
    for k in models:
        per[f"d_{k}"] = per[k] - per[ref]
    return per


def measure_latency(spec: dict, model, sample: pd.DataFrame, n_single: int = 30) -> dict:
    """Prediction cost. Random Forest is timed with 1 and with all threads, because thread start-up
    dominates for a single row."""
    out = {}
    variants = [("", None)] if spec["kind"] == "xgb" else [("_1thread", 1), ("_allthreads", N_JOBS)]
    X, one = sample.head(20000), sample.head(1)
    for tag, nj in variants:
        if nj is not None:
            model.set_params(n_jobs=nj)
        t0 = time.time(); predict_model(spec, model, X); batch = time.time() - t0
        lat = []
        for _ in range(n_single):
            t0 = time.time(); predict_model(spec, model, one); lat.append(time.time() - t0)
        out[f"ms_per_1000_rows{tag}"] = batch / len(X) * 1e6
        out[f"single_row_ms{tag}"] = float(np.median(lat) * 1000)
    if spec["kind"] == "rf":
        model.set_params(n_jobs=N_JOBS)
    return out


# ------------------------------------------------------------------------------------ costs
def model_costs(spec: dict, model, m_fit_rows: int, m_sample: pd.DataFrame, path: Path, train_seconds: float) -> dict:
    X = m_sample.head(20000)
    t0 = time.time(); predict_model(spec, model, X); batch = time.time() - t0
    one = m_sample.head(1); lat = []
    for _ in range(30):
        t0 = time.time(); predict_model(spec, model, one); lat.append(time.time() - t0)
    if spec["kind"] == "xgb":
        model.save_model(str(path.with_suffix(".ubj")))
        size = path.with_suffix(".ubj").stat().st_size
        trees = int(spec["rounds"])
    else:
        joblib.dump(model, path.with_suffix(".joblib"), compress=3)
        size = path.with_suffix(".joblib").stat().st_size
        trees = len(model.estimators_)
    return {"train_seconds": train_seconds, "predict_ms_per_1000_rows": 1000 * batch / len(X) * 1000,
            "single_row_latency_ms": float(np.median(lat) * 1000), "size_MB": size / 1e6, "n_trees": trees,
            "fit_rows": m_fit_rows}


# ------------------------------------------------------------------------------------ final run
def run_final(out: Path = OUT, log=print) -> dict:
    mdir = MODEL_DIR if out == OUT else out / "models"
    out.mkdir(parents=True, exist_ok=True); mdir.mkdir(parents=True, exist_ok=True)
    specs = json.loads((out / "best_specs.json").read_text())
    m_all = load_table(include_test=True)
    nontest = m_all[m_all["split"] != "test"].reset_index(drop=True)
    test = m_all[m_all["split"] == "test"].reset_index(drop=True)
    log(f"fitting stack pipeline: {len(nontest):,} train+val rows; test rows used for prediction only: {len(test):,}")
    t0 = time.time()
    st = stack_pipeline(nontest, specs, test)
    log(f"stack pipeline done in {time.time()-t0:.0f}s; meta weights (val-eval) {dict(zip(STACK_INPUTS, st['meta_val'].coef_.round(3)))}")
    val = nontest.iloc[st["val_idx"]].reset_index(drop=True)
    tr = nontest[nontest["split"] == "train"]

    # --- train-only individual models for the validation table + the cost table (timed here)
    costs = {}
    final_models = {}
    for n in ("rf", "xgb", "xgb_alt"):
        t0 = time.time(); mod = fit_model(specs[n], tr); ts = time.time() - t0
        costs[n] = model_costs(specs[n], mod, len(tr), val, mdir / f"{n}_trainonly_tmp", ts)
        for f in (mdir).glob(f"{n}_trainonly_tmp.*"):
            f.unlink()                                                   # the dashboard uses the refit models
    for n in ("rf", "xgb", "xgb_alt"):                                    # refit on train+val: what the dashboard loads
        mod = fit_model(specs[n], nontest)
        final_models[n] = mod
        if specs[n]["kind"] == "xgb":
            mod.save_model(str(mdir / f"{n}.ubj"))
        else:
            joblib.dump(mod, mdir / f"{n}.joblib", compress=3)
    joblib.dump({"weights": dict(zip(STACK_INPUTS, st["meta_test"].coef_)), "inputs": STACK_INPUTS,
                 "meta": st["meta_test"]}, mdir / "stack_meta.joblib")
    manifest = {"features": phase1.FEATURES, "filled_features": phase1.FILLED_FEATURES, "specs": specs,
                "stack_inputs": STACK_INPUTS, "stack_weights_for_test": dict(zip(STACK_INPUTS, map(float, st["meta_test"].coef_))),
                "stack_weights_for_validation": dict(zip(STACK_INPUTS, map(float, st["meta_val"].coef_))),
                "fit_window": [str(nontest["date"].min().date()), str(nontest["date"].max().date())],
                "versions": {"xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # --- predictions tables
    def frame(df, base_preds, stack, tr_med):
        f = df[["household_ID", "date", "split", "y", "tr2_check"]].copy().reset_index(drop=True)
        bf = baseline_frame(df.reset_index(drop=True), tr_med)
        for c in bf: f[f"base_{c}"] = bf[c].to_numpy()
        for n in ("rf", "xgb", "xgb_alt"): f[n] = base_preds[n]
        f["stack"] = stack
        f["stack_equal_avg"] = np.mean([base_preds[n] for n in ("rf", "xgb", "xgb_alt")] + [bf["roll7_median"].to_numpy()], axis=0)
        return f
    pv = frame(val, st["val_base"], st["stack_val"], float(tr["y"].median()))
    pt = frame(test, st["test_base"], st["stack_test"], float(nontest["y"].median()))
    pv["seen_in_fit"] = seen_mask(val, tr); pt["seen_in_fit"] = seen_mask(test, nontest)
    pv.to_csv(out / "predictions_val.csv.gz", index=False); pt.to_csv(out / "predictions_test.csv.gz", index=False)
    pd.DataFrame(costs).T.to_csv(out / "model_costs.csv")
    cv_pred = pd.DataFrame({"row": st["cv_rows"], **{n: st["P_cv"][:, i] for i, n in enumerate(STACK_INPUTS)},
                            "y": nontest["y"].to_numpy()[st["cv_rows"]], "date": nontest["date"].iloc[st["cv_rows"]].to_numpy()})
    cv_pred.to_csv(out / "oof_predictions_cv.csv.gz", index=False)
    log("artifacts written to " + str(out))
    return {"val": pv, "test": pt, "costs": costs, "audit": st["audit"], "meta_dates": st["meta_train_dates"]}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("tune", "all"):
        run_tuning()
    if what in ("final", "all"):
        run_final()
