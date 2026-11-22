#!/usr/bin/env python3
"""
Build the small, pseudonymised artifact set that the PeakWise dashboard reads.

The dashboard itself needs no raw data, no models and no retraining: everything it shows is copied or
aggregated from the saved Phase 1 / Phase 2 outputs by this script, which you only re-run if those change.

    python build_dashboard_data.py            # writes dashboard/data/

What it does
  * copies saved metric / delta / band / household / cost / tuning tables unchanged,
  * pseudonymises household identifiers with a salted hash (e.g. "H-3F9A1C"). The salt is secret and never stored:
    set PEAKWISE_SALT to make labels reproducible, otherwise a random salt is drawn on every build (labels then change
    per build). The mapping is not stored. Labels are pseudonyms, not strong anonymisation: the source data are public,
    so someone holding them could still match a series by its values,
  * keeps only what the dashboard uses: unused columns and the household label of the household-error table are dropped,
  * writes the test-period predictions the Forecast Explorer reads,
  * computes permutation importance ONCE for the saved Random Forest and XGBoost on the TEST period
    (descriptive only; nothing was selected or tuned with it),
  * records the facts shown on the Overview (audit and split sizes).
No model is retrained and no number is typed in by hand.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import phase1
import phase2

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "outputs/phase2"
P1 = ROOT / "outputs/phase1"
OUT = ROOT / "dashboard/data"
SALT = os.environ.get("PEAKWISE_SALT") or secrets.token_hex(16)     # secret; never written to disk
MODELS = ["rf", "xgb", "xgb_alt", "stack"]
COPY = ["metrics_validation.csv", "metrics_test.csv", "deltas_vs_baseline.csv", "deltas_validation_tr2_agree.csv",
        "bands_validation.csv", "bands_test.csv", "household_summary.csv", "model_costs_with_latency.csv",
        "tuning_results.csv", "best_specs.json"]


def anon(ids) -> dict:
    """Salted pseudonyms (stable within one build). Collisions are resolved, the mapping itself is never written."""
    m, used = {}, set()
    for h in sorted(set(ids)):
        digest = hashlib.sha256(f"{SALT}:{h}".encode()).hexdigest().upper()
        n = 6
        while "H-" + digest[:n] in used:          # extend the label until it is unique (deterministic)
            n += 1
        m[h] = "H-" + digest[:n]; used.add(m[h])
    return m


def permutation_importance(model, spec, X: np.ndarray, y: np.ndarray, groups: dict, repeats: int = 3, seed: int = 7):
    """Increase in test MAE (kWh) when one feature (or a feature and its missing-flag, together) is shuffled
    across rows. Shuffling breaks the feature's link to the target while keeping its marginal distribution."""
    rng = np.random.default_rng(seed)
    base = np.abs(phase2.from_target(model.predict(X), spec["target"]) - y).mean()
    rows = []
    for name, cols in groups.items():
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, cols] = X[perm][:, cols]
            inc.append(np.abs(phase2.from_target(model.predict(Xp), spec["target"]) - y).mean() - base)
        rows.append({"feature": name, "mae_increase": float(np.mean(inc)), "mae_increase_sd": float(np.std(inc))})
    out = pd.DataFrame(rows)
    out["baseline_mae"] = float(base)
    out["relative_increase_pct"] = 100 * out["mae_increase"] / base
    return out.sort_values("mae_increase", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy(P2 / f, OUT / f)
    shutil.copy(P2 / "models/manifest.json", OUT / "manifest.json")

    # ---- predictions (test) with pseudonymised ids
    pt = pd.read_csv(P2 / "predictions_test.csv.gz", parse_dates=["date"])
    pv = pd.read_csv(P2 / "predictions_val.csv.gz", parse_dates=["date"])
    amap = anon(pd.concat([pt.household_ID, pv.household_ID]))
    keep = ["date", "y", "base_roll7_median", "rf", "xgb", "stack"]          # only what the Forecast Explorer reads
    t = pt[keep].copy(); t.insert(0, "household", pt.household_ID.map(amap))
    t.sort_values(["household", "date"]).to_csv(OUT / "predictions_test_pseudonymized.csv.gz", index=False, float_format="%.4f")

    # ---- per-household MAE for the "is the gain widespread?" view: test period only, NO household label
    d = pt[pt.seen_in_fit]
    e = pd.DataFrame({k: (d[k] - d.y).abs() for k in ["base_roll7_median"] + MODELS}).groupby(d.household_ID).mean()
    e["n_days"] = d.groupby(d.household_ID).size()
    e.insert(0, "period", "test")
    e.sample(frac=1, random_state=0).reset_index(drop=True).to_csv(OUT / "household_errors.csv", index=False, float_format="%.5f")

    # ---- curated example households for the explorer (deterministic rules on the TEST predictions)
    s = pt[pt.seen_in_fit]
    g = s.groupby("household_ID").agg(n=("y", "size"), mean_y=("y", "mean"), zero_share=("y", lambda v: (v == 0).mean()),
                                     e_base=("base_roll7_median", lambda v: 0.0))
    ee = pd.DataFrame({k: (s[k] - s.y).abs() for k in ["base_roll7_median", "rf"]}).groupby(s.household_ID).mean()
    g["d_rf"] = ee["rf"] - ee["base_roll7_median"]
    g = g[g.n >= 30]
    pick = lambda q: (g.mean_y - g.mean_y.quantile(q)).abs().idxmin()
    ex = {"Typical consumption": pick(0.5), "Low consumption": pick(0.15), "High consumption": pick(0.9),
          "Often exactly zero": g.zero_share.idxmax(), "Largest Random Forest gain": g.d_rf.idxmin(),
          "Largest Random Forest loss": g.d_rf.idxmax()}
    (OUT / "examples.json").write_text(json.dumps({k: amap[v] for k, v in ex.items()}, indent=2))

    # ---- permutation importance on the TEST period (saved final models, no retraining)
    specs = json.loads((P2 / "best_specs.json").read_text())
    m = phase2.load_table(include_test=True)
    test = m[m.split == "test"].reset_index(drop=True)
    y = test["y"].to_numpy()
    imp = []
    rf = joblib.load(P2 / "models/rf.joblib"); rf.set_params(n_jobs=phase2.N_JOBS)
    Xr = phase2.design(test, "rf"); cols = phase1.FILLED_FEATURES
    flag = {"lag1": "lag1_missing", "lag7": "lag7_missing", "roll7_median": "roll7_median_missing"}
    groups = {f: [cols.index(f)] + ([cols.index(flag[f])] if f in flag else []) for f in phase1.FEATURES}
    imp.append(permutation_importance(rf, specs["rf"], Xr, y, groups).assign(model="rf"))
    xm = xgb.XGBRegressor(); xm.load_model(str(P2 / "models/xgb.ubj"))
    Xx = phase2.design(test, "xgb")
    imp.append(permutation_importance(xm, specs["xgb"], Xx, y, {f: [i] for i, f in enumerate(phase1.FEATURES)}).assign(model="xgb"))
    pd.concat(imp).to_csv(OUT / "permutation_importance.csv", index=False)

    # ---- facts for the Overview
    p1 = json.loads((P1 / "phase1_summary.json").read_text())
    audit = json.loads((ROOT / "audit_out/audit_report.json").read_text())
    tr2 = pd.crosstab(pd.concat([pv.assign(period="validation"), pt.assign(period="test")], ignore_index=True)["period"],
                      pd.concat([pv, pt], ignore_index=True)["tr2_check"])
    facts = {
        "raw_rows": audit["pipeline"]["rows_total"], "households_in_raw": audit["coverage"]["households_in_15min_data"],
        "valid_household_days": audit["peak"]["valid"], "households_with_valid": audit["coverage"]["households_with_valid"],
        "missing_quarter_hours_pct": 100 * audit["span"]["missing_slots_within_span"] / audit["span"]["expected_slots_within_span"],
        "first_date": audit["coverage"]["first_valid_date"], "last_date": audit["coverage"]["last_valid_date"],
        "model_rows": int(sum(v["rows"] for v in p1["splits"].values())),
        "splits": {k: {"dates": v["dates"], "rows": v["rows"], "households": v["households"]} for k, v in p1["splits"].items()},
        "cv_folds": p1["cv_folds_inside_train"], "features": p1["features"],
        "test_rows": int(len(pt)), "test_households": int(pt.household_ID.nunique()),
        "val_rows": int(len(pv)), "val_unseen_rows": int((~pv.seen_in_fit).sum()),
        "val_unseen_households": int(pv.loc[~pv.seen_in_fit, "household_ID"].nunique()),
        "test_unseen_rows": int((~pt.seen_in_fit).sum()),
        "tr2_by_period": {k: {c: int(v) for c, v in r.items()} for k, r in tr2.iterrows()},
        "target_median_kwh": float(pt.y.median()), "target_zero_share_test": float((pt.y == 0).mean()),
        "cv_reference": specs["reference_baseline_cv"], "license": "CC BY 4.0 (Zenodo record 15023048)",
    }
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2, default=str))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    print("pseudonymised households:", len(amap), "| example households:", json.loads((OUT / 'examples.json').read_text()))


if __name__ == "__main__":
    main()
