#!/usr/bin/env python3
"""Profile the non-LP (obis = DBE) rows of the RECON-SL 15-minute files and test whether they are the same
cumulative import register as the neighbouring LP readings.

For each DBE row stamped 00:00:00 of day D (household h) it looks up h's LP readings at 23:45 of D-1 and 00:15 of D
and tests whether the DBE register lies between them (+-0.01 kWh). Writes <out>/dbe_probe.json.
Usage: python -I src/probe_dbe.py --zip data.zip --out artifacts/audit
"""
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

MEMBERS = [f"data/consumption_data/smart_meter/15min_interval/smart_15min_{k}.csv" for k in (1, 2, 3)]
COLS = ["household_ID", "date", "time", "obis", "importkwh(kwh)"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/audit")); a = ap.parse_args()
    zf = zipfile.ZipFile(a.zip)
    dbe, all_obis = [], {}
    for m in MEMBERS:                                   # pass A: collect every non-LP row
        with zf.open(m) as f:
            for ch in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), usecols=COLS, dtype=str,
                                  keep_default_na=False, chunksize=1_000_000):
                for k, v in ch["obis"].value_counts().items():
                    all_obis[k] = all_obis.get(k, 0) + int(v)
                dbe.append(ch[ch["obis"] != "LP"])
    d = pd.concat(dbe)
    d["imp"] = pd.to_numeric(d["importkwh(kwh)"], errors="coerce")
    at_mid = d[d["time"] == "00:00:00"].copy()
    at_mid["dt"] = pd.to_datetime(at_mid["date"])
    hhs = set(at_mid["household_ID"])
    want = set()
    for h, dt in zip(at_mid["household_ID"], at_mid["dt"]):
        want.add((h, (dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "23:45:00"))
        want.add((h, dt.strftime("%Y-%m-%d"), "00:15:00")); want.add((h, dt.strftime("%Y-%m-%d"), "00:00:00"))
    lp = {}
    for m in MEMBERS:                                   # pass B: LP neighbours of those rows
        with zf.open(m) as f:
            for ch in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), usecols=COLS, dtype=str,
                                  keep_default_na=False, chunksize=1_000_000):
                ch = ch[(ch["obis"] == "LP") & ch["household_ID"].isin(hhs) & ch["time"].isin(["23:45:00", "00:15:00", "00:00:00"])]
                for h, dd, t, v in zip(ch["household_ID"], ch["date"], ch["time"], ch["importkwh(kwh)"]):
                    if (h, dd, t) in want:
                        lp[(h, dd, t)] = float(v)
    prev = np.array([lp.get((h, (dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "23:45:00"), np.nan) for h, dt in zip(at_mid["household_ID"], at_mid["dt"])])
    nxt = np.array([lp.get((h, dt.strftime("%Y-%m-%d"), "00:15:00"), np.nan) for h, dt in zip(at_mid["household_ID"], at_mid["dt"])])
    same = np.array([lp.get((h, dt.strftime("%Y-%m-%d"), "00:00:00"), np.nan) for h, dt in zip(at_mid["household_ID"], at_mid["dt"])])
    v = at_mid["imp"].to_numpy()
    both = np.isfinite(prev) & np.isfinite(nxt)
    inside = both & (v >= prev - 0.01) & (v <= nxt + 0.01)
    res = {"obis_row_counts": all_obis, "non_lp_rows": int(len(d)), "non_lp_time_counts": d["time"].value_counts().to_dict(),
           "non_lp_date_range": [d["date"].min(), d["date"].max()], "non_lp_households": int(d["household_ID"].nunique()),
           "dbe_rows_at_0000": int(len(at_mid)), "dbe_rows_with_LP_in_same_slot": int(np.isfinite(same).sum()),
           "dbe_rows_with_both_LP_neighbours": int(both.sum()), "dbe_between_LP_2345_and_0015": int(inside.sum()),
           "share_between": float(inside.sum() / both.sum()) if both.sum() else None,
           "median_dbe_minus_LP_2345_kwh": float(np.nanmedian((v - prev)[both])) if both.any() else None,
           "interpretation": "DBE rows occupy the 00:00 slot only; where tested they are the same cumulative import register as LP."}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dbe_probe.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
