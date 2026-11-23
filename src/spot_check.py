#!/usr/bin/env python3
"""Independent spot check: recompute the 18:30-22:30 consumption for randomly chosen household-days straight from the
raw 15-minute CSVs (no shared code with peakwise_audit.py besides pandas) and compare with household_days_peak.csv.
Usage: python -I src/spot_check.py --zip data.zip --audit artifacts/audit --n 40 --seed 7"""
import argparse, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd

MEMBERS = [f"data/consumption_data/smart_meter/15min_interval/smart_15min_{k}.csv" for k in (1, 2, 3)]
WIN = [f"{h:02d}:{m:02d}:00" for h, m in [(18, 30), (18, 45), (19, 0), (19, 15), (19, 30), (19, 45), (20, 0), (20, 15), (20, 30),
                                          (20, 45), (21, 0), (21, 15), (21, 30), (21, 45), (22, 0), (22, 15), (22, 30)]]
ap = argparse.ArgumentParser(); ap.add_argument("--zip", type=Path, required=True); ap.add_argument("--audit", type=Path, required=True)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
hd = pd.read_csv(a.audit / "household_days_peak.csv")
rng = np.random.default_rng(a.seed)
pick = pd.concat([hd[hd.status == "valid"].sample(a.n, random_state=a.seed),
                  hd[hd.status != "valid"].groupby("status").head(a.n // 8).sample(frac=1, random_state=a.seed)])
want = {(r.household_ID, r.date) for r in pick.itertuples()}
hhs = {h for h, _ in want}
rows = []
zf = zipfile.ZipFile(a.zip)
for m in MEMBERS:
    with zf.open(m) as f:
        for ch in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), usecols=["household_ID", "date", "time", "importkwh(kwh)"],
                              dtype=str, keep_default_na=False, chunksize=1_000_000):
            ch = ch[ch["household_ID"].isin(hhs) & ch["time"].isin(WIN)]
            rows.append(ch[[(h, d) in want for h, d in zip(ch["household_ID"], ch["date"])]])
raw = pd.concat(rows); raw["v"] = raw["importkwh(kwh)"].astype(float)
bad = 0
for r in pick.itertuples():
    g = raw[(raw.household_ID == r.household_ID) & (raw.date == r.date)].drop_duplicates(["time"], keep=False) if False else raw[(raw.household_ID == r.household_ID) & (raw.date == r.date)]
    full = g["time"].nunique() == 17 and len(g) == 17
    steps = g.set_index("time").reindex(WIN)["v"].diff().dropna() if full else None
    ok_rule = full and (steps >= 0).all() and (steps <= 10).all()
    manual = (g.set_index("time").loc["22:30:00", "v"] - g.set_index("time").loc["18:30:00", "v"]) if full else np.nan
    expect_valid = bool(ok_rule)
    is_valid = r.status == "valid"
    match = (expect_valid == is_valid) and (not is_valid or abs(manual - r.peak_kwh_1830_2230) < 1e-5)
    bad += (not match)
    if not match:
        print("MISMATCH", r.household_ID, r.date, r.status, r.peak_kwh_1830_2230, manual, len(g))
print(f"checked {len(pick)} household-days ({int((pick.status=='valid').sum())} valid, {int((pick.status!='valid').sum())} rejected); mismatches: {bad}")
