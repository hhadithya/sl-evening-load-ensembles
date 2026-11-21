#!/usr/bin/env python3
"""
PeakWise data audit for RECON-SL 15-minute smart-meter data.

Reads the three 15-minute CSVs straight out of Zenodo's data.zip (record 15023048,
md5 dcdac5809b4da6b6362519d33112a5f9) or from extracted CSV files, audits them, and extracts
the PeakWise target: household grid import (kWh) over 18:30-22:30 Sri Lanka time.

Semantics this script relies on (verified in the audit report; re-checked on every run):
  * importkwh(kwh) / exportkwh(kwh) are cumulative energy registers (kWh), not interval energy.
  * date + time are Sri Lanka local time (UTC+05:30). `timestamp` is epoch milliseconds rounded
    to 1e7 ms (~2.8 h), so it cannot be used for ordering; it is only cross-checked.
  * A row stamped t holds the register at t; the interval it closes is (t-15min, t]
    (avgimportkw at t matches the register increase from t-15 to t).
  => Peak energy for day D = R(D 22:30) - R(D 18:30), built from the 17 readings
     18:30, 18:45, ..., 22:30, i.e. exactly 16 complete 15-minute intervals.

Nothing is imputed. Every household-day that has at least one reading appears in the output
with a status (valid or the first failing rule), and every dropped or flagged row is counted.

Usage:
  python peakwise_audit.py --zip data.zip --out audit_out
  python peakwise_audit.py --csv smart_15min_1.csv smart_15min_2.csv smart_15min_3.csv --out audit_out
Options: --verify-md5, --households ID0001,ID0002 (debug subset), --cap-kwh 10, --chunksize 2000000
Requirements: Python >= 3.9, pandas >= 1.5, numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_MD5 = "dcdac5809b4da6b6362519d33112a5f9"
MEMBERS = [f"data/consumption_data/smart_meter/15min_interval/smart_15min_{k}.csv" for k in (1, 2, 3)]
EXPECTED_HEADER = ("household_ID,timestamp,date,time,obis,importkwh(kwh),exportkwh(kwh),importkvarh(kvarh),"
                   "exportkvarh(kvarh),powerfactor,avgcurrent(v),avgexportkw(kw),avgimportkw(kw),avgvoltage(v),"
                   "phaseainstcurrent(a),phaseainstvoltage(v),phasebinstcurrent(a),phasebinstvoltage(v),"
                   "phasecinstcurrent(a),phasecinstvoltage(v)")
USECOLS = ["household_ID", "timestamp", "date", "time", "obis", "importkwh(kwh)", "exportkwh(kwh)",
           "avgexportkw(kw)", "avgimportkw(kw)"]
SLT_OFFSET_MS = 19_800_000           # UTC+05:30
TS_TOL_MS = 5_000_000                # half of the 1e7 ms rounding unit
P0, P1 = 74, 90                      # slots of 18:30 and 22:30 (slot = hour*4 + minute/15)
BIN_EDGES = [0, 0.01, 0.1, 0.5, 1, 2.5, 5, 10, 25]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def open_sources(args):
    """Yield (label, opener) where opener() returns a binary file object."""
    if args.zip:
        zf = zipfile.ZipFile(args.zip)
        names = set(zf.namelist())
        for m in MEMBERS:
            if m not in names:
                sys.exit(f"missing zip member: {m}")
            yield Path(m).name, (lambda m=m: zf.open(m))
    else:
        for p in args.csv:
            yield Path(p).name, (lambda p=p: open(p, "rb"))


def count_lines(opener) -> tuple[int, str]:
    n, header, first = 0, "", True
    with opener() as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            if first:
                header = block.split(b"\n", 1)[0].decode().rstrip("\r")
                first = False
            n += block.count(b"\n")
    return n, header


def bin_counts(x: np.ndarray) -> dict:
    out = {"eq0": int((x == 0).sum())}
    edges = BIN_EDGES + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[f">{lo:g}"] = int(((x > lo) & (x <= hi)).sum()) if np.isfinite(hi) else int((x > lo).sum())
    return out


HH_CODES: dict[str, int] = {}


def encode_hh(values: np.ndarray) -> np.ndarray:
    """Map household IDs to stable int32 codes (keeps 45M-row frames small)."""
    for h in pd.unique(values):
        if h not in HH_CODES:
            HH_CODES[h] = len(HH_CODES)
    return pd.Series(values).map(HH_CODES).to_numpy(dtype="int32")


def read_file(label, opener, args, file_stats):
    n_newlines, header = count_lines(opener)
    st = {"header_matches_expected": header == EXPECTED_HEADER, "header": header,
          "newline_count": n_newlines, "rows": 0, "obis": {}, "min_date": None, "max_date": None,
          "bad_datetime": 0, "offgrid_time": 0, "ts_local": 0, "ts_utc_like": 0, "ts_other": 0,
          "ts_max_abs_dev_local_ms": 0.0, "ts_multiple_of_1e7": 0, "import_empty": 0, "export_empty": 0,
          "avgimport_empty": 0, "import_non_numeric": 0, "import_raw_negative": 0, "export_raw_negative": 0}
    parts = []
    hh_filter = set(args.households.split(",")) if args.households else None
    with opener() as fh:
        reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", newline=""), usecols=USECOLS, dtype=str,
                             keep_default_na=False, chunksize=args.chunksize, on_bad_lines="skip")
        row0 = 0
        for ch in reader:
            st["rows"] += len(ch)
            idx = np.arange(row0, row0 + len(ch)); row0 += len(ch)
            if hh_filter is not None:
                keep = ch["household_ID"].isin(hh_filter).to_numpy()
                ch, idx = ch[keep], idx[keep]
                if ch.empty:
                    continue
            for k, v in ch["obis"].value_counts().items():
                st["obis"][k] = st["obis"].get(k, 0) + int(v)
            d = pd.to_datetime(ch["date"], format="%Y-%m-%d", errors="coerce")
            tparts = ch["time"].str.extract(r"^(\d{2}):(\d{2}):(\d{2})$").astype(float)
            H, M, S = tparts[0].to_numpy(), tparts[1].to_numpy(), tparts[2].to_numpy()
            bad = d.isna().to_numpy() | np.isnan(H)
            st["bad_datetime"] += int(bad.sum())
            dmin, dmax = ch["date"][~bad].min(), ch["date"][~bad].max()
            st["min_date"] = dmin if st["min_date"] is None or dmin < st["min_date"] else st["min_date"]
            st["max_date"] = dmax if st["max_date"] is None or dmax > st["max_date"] else st["max_date"]
            day = (d.to_numpy().astype("datetime64[D]").astype("int64"))
            secs = H * 3600 + M * 60 + S
            ts = pd.to_numeric(ch["timestamp"], errors="coerce").to_numpy()
            dev = ts - (day * 86_400_000 + secs * 1000 - SLT_OFFSET_MS)
            okb = ~bad
            loc = okb & (np.abs(dev) <= TS_TOL_MS)
            utc = okb & ~loc & (np.abs(dev - SLT_OFFSET_MS) <= TS_TOL_MS)
            st["ts_local"] += int(loc.sum()); st["ts_utc_like"] += int(utc.sum())
            st["ts_other"] += int((okb & ~loc & ~utc).sum())
            if loc.any():
                st["ts_max_abs_dev_local_ms"] = max(st["ts_max_abs_dev_local_ms"], float(np.abs(dev[loc]).max()))
            st["ts_multiple_of_1e7"] += int((np.mod(ts, 1e7) == 0).sum())
            imp_s, exp_s = ch["importkwh(kwh)"], ch["exportkwh(kwh)"]
            st["import_empty"] += int((imp_s == "").sum()); st["export_empty"] += int((exp_s == "").sum())
            st["avgimport_empty"] += int((ch["avgimportkw(kw)"] == "").sum())
            imp = pd.to_numeric(imp_s, errors="coerce").to_numpy(); ex = pd.to_numeric(exp_s, errors="coerce").to_numpy()
            st["import_non_numeric"] += int(((imp_s != "") & np.isnan(imp)).sum())
            st["import_raw_negative"] += int((imp < 0).sum()); st["export_raw_negative"] += int((ex < 0).sum())
            offgrid = okb & ((M % 15 != 0) | (S != 0) | (H > 23))
            st["offgrid_time"] += int(offgrid.sum())
            keep = okb & ~offgrid
            parts.append(pd.DataFrame({
                "hh": encode_hh(ch["household_ID"].to_numpy()[keep]),
                "day": day[keep].astype("int32"),
                "slot": (H[keep] * 4 + M[keep] // 15).astype("int16"),
                "imp": imp[keep], "exp": ex[keep],
                "ai": pd.to_numeric(ch["avgimportkw(kw)"], errors="coerce").to_numpy()[keep].astype("float32"),
                "ae": pd.to_numeric(ch["avgexportkw(kw)"], errors="coerce").to_numpy()[keep].astype("float32"),
                "src": np.int8(len(file_stats)), "row": idx[keep].astype("int64")}))
    st["malformed_rows_skipped"] = (st["newline_count"] - 1) - st["rows"] if hh_filter is None else None
    file_stats[label] = st
    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path)
    src.add_argument("--csv", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("audit_out"))
    ap.add_argument("--verify-md5", action="store_true")
    ap.add_argument("--households", default=None, help="comma-separated IDs (debug subset)")
    ap.add_argument("--cap-kwh", type=float, default=10.0,
                    help="15-min increase above this is flagged physically implausible (default 10 kWh = 40 kW)")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = {"inputs": {}, "rules": {"peak_window": "R(22:30)-R(18:30), 17 readings, 16 intervals",
                                 "cap_kwh_per_15min": args.cap_kwh, "timezone": "Asia/Colombo (UTC+05:30)",
                                 "interval_label": "row t closes interval (t-15min, t]"}}
    if args.zip:
        R["inputs"]["zip"] = str(args.zip); R["inputs"]["zip_bytes"] = args.zip.stat().st_size
        if args.verify_md5:
            m = md5sum(args.zip); R["inputs"]["zip_md5"] = m; R["inputs"]["zip_md5_ok"] = (m == ZIP_MD5)
        with zipfile.ZipFile(args.zip) as zf:
            R["inputs"]["members"] = {i.filename: {"compressed": i.compress_size, "uncompressed": i.file_size}
                                      for i in zf.infolist() if i.filename in MEMBERS}
    file_stats, frames = {}, []
    for label, opener in open_sources(args):
        print(f"reading {label} ...", flush=True)
        df = read_file(label, opener, args, file_stats)
        if df is not None:
            frames.append(df)
    R["files"] = file_stats
    df = pd.concat(frames, ignore_index=True)
    del frames
    hh_names = np.array(sorted(HH_CODES, key=HH_CODES.get), dtype=object)
    df["t"] = df["day"].astype("int64") * 96 + df["slot"]
    df.sort_values(["hh", "t", "src", "row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # ---------- duplicates on (household, 15-min slot) ----------
    key_first = ~df.duplicated(["hh", "t"], keep="first")
    grp = key_first.cumsum().to_numpy() - 1
    first_imp = df["imp"].to_numpy()[key_first.to_numpy()][grp]
    first_exp = df["exp"].to_numpy()[key_first.to_numpy()][grp]
    imp_a, exp_a = df["imp"].to_numpy(), df["exp"].to_numpy()
    same = (((imp_a == first_imp) | (np.isnan(imp_a) & np.isnan(first_imp))) &
            ((exp_a == first_exp) | (np.isnan(exp_a) & np.isnan(first_exp))))
    dup = ~key_first.to_numpy()
    conflict_row = dup & ~same
    conflict_grp = np.zeros(grp.max() + 1, dtype=bool); conflict_grp[grp[conflict_row]] = True
    R["duplicates"] = {"duplicate_rows": int(dup.sum()), "exact_duplicate_rows": int((dup & same).sum()),
                       "conflicting_duplicate_rows": int(conflict_row.sum()),
                       "conflicting_slots": int(conflict_grp.sum()),
                       "examples": [[hh_names[a], str(np.datetime64(int(b), "D")), int(c), float(d_), float(e_)] for a, b, c, d_, e_ in
                                    df.loc[conflict_row, ["hh", "day", "slot", "imp", "exp"]].head(10).itertuples(index=False)]}
    u = df[key_first].copy()                     # one row per household-slot (first occurrence)
    u["conf"] = conflict_grp
    u["usable"] = ~u["conf"] & np.isfinite(u["imp"]) & np.isfinite(u["exp"])
    R["unusable_readings"] = int((~u["usable"]).sum())

    # ---------- consecutive-reading analysis ----------
    hh_codes = u["hh"].to_numpy()
    t = u["t"].to_numpy(); same_hh = np.r_[False, hh_codes[1:] == hh_codes[:-1]]
    gap = np.r_[0, np.diff(t)]
    g = gap[same_hh]
    R["gaps"] = {"1": int((g == 1).sum()), "2-4": int(((g >= 2) & (g <= 4)).sum()),
                 "5-96": int(((g >= 5) & (g <= 96)).sum()), ">96": int((g > 96).sum())}
    span = u.groupby("hh")["t"].agg(["min", "max", "size"])
    R["span"] = {"households": int(len(span)), "expected_slots_within_span": int((span["max"] - span["min"] + 1).sum()),
                 "present_slots": int(span["size"].sum())}
    R["span"]["missing_slots_within_span"] = R["span"]["expected_slots_within_span"] - R["span"]["present_slots"]
    usable = u["usable"].to_numpy()
    pair = same_hh & (gap == 1)
    pu = pair & usable & np.r_[False, usable[:-1]]
    imp_u, exp_u = u["imp"].to_numpy(), u["exp"].to_numpy()
    dI = np.r_[np.nan, np.diff(imp_u)]; dE = np.r_[np.nan, np.diff(exp_u)]
    ai, ae = u["ai"].to_numpy().astype(float), u["ae"].to_numpy().astype(float)
    ai0, ae0 = np.r_[np.nan, ai[:-1]], np.r_[np.nan, ae[:-1]]
    slot_u = u["slot"].to_numpy()
    di, de = dI[pu], dE[pu]
    neg_by_slot = np.bincount(slot_u[pu][di < 0], minlength=96)
    big_by_slot = np.bincount(slot_u[pu][di > args.cap_kwh], minlength=96)
    neg_idx = np.flatnonzero(pu & (dI < 0))[:12]
    R["intervals"] = {"pairs_15min": int(pair.sum()), "pairs_usable": int(pu.sum()),
                      "import_negative": int((di < 0).sum()), "export_negative": int((de < 0).sum()),
                      "import_gt_cap": int((di > args.cap_kwh).sum()), "export_gt_cap": int((de > args.cap_kwh).sum()),
                      "import_bins": bin_counts(di[di >= 0]), "export_bins": bin_counts(de[de >= 0]),
                      "import_negative_by_end_slot": neg_by_slot.tolist(), "import_gt_cap_by_end_slot": big_by_slot.tolist(),
                      "negative_examples": [[hh_names[u["hh"].iat[i]], str(np.datetime64(int(t[i]) * 15, "m")),
                                             float(imp_u[i - 1]), float(imp_u[i]), float(dI[i])] for i in neg_idx]}
    # interval-label test: does avg power at t (interval-ending) or at t-15 (interval-starting) explain R(t)-R(t-15)?
    m = pu & (dI >= 0) & (dI <= 5) & np.isfinite(ai) & np.isfinite(ai0)
    e_end, e_start = np.abs(dI[m] - 0.25 * ai[m]), np.abs(dI[m] - 0.25 * ai0[m])
    mE = pu & (dE >= 0) & (dE <= 5) & np.isfinite(ae) & np.isfinite(ae0) & ((ae > 0) | (ae0 > 0) | (dE > 0))
    f_end, f_start = np.abs(dE[mE] - 0.25 * ae[mE]), np.abs(dE[mE] - 0.25 * ae0[mE])
    R["interval_label_test"] = {
        "import": {"n": int(m.sum()), "mae_if_interval_ending": float(e_end.mean()) if m.any() else None,
                   "mae_if_interval_starting": float(e_start.mean()) if m.any() else None,
                   "share_within_0.01kWh_ending": float((e_end <= 0.01).mean()) if m.any() else None,
                   "share_within_0.01kWh_starting": float((e_start <= 0.01).mean()) if m.any() else None},
        "export": {"n": int(mE.sum()), "mae_if_interval_ending": float(f_end.mean()) if mE.any() else None,
                   "mae_if_interval_starting": float(f_start.mean()) if mE.any() else None,
                   "share_within_0.01kWh_ending": float((f_end <= 0.01).mean()) if mE.any() else None,
                   "share_within_0.01kWh_starting": float((f_start <= 0.01).mean()) if mE.any() else None}}

    # ---------- household-day table and peak target ----------
    hd = u.groupby(["hh", "day"]).size().rename("n_readings_day").reset_index()
    R["household_days"] = int(len(hd))
    R["readings_per_day_hist"] = np.bincount(hd["n_readings_day"].to_numpy(), minlength=97).tolist()
    w = u[(u["slot"] >= P0) & (u["slot"] <= P1)]
    hd_index = pd.MultiIndex.from_frame(hd[["hh", "day"]])
    row = hd_index.get_indexer(pd.MultiIndex.from_arrays([w["hh"], w["day"]]))
    col = w["slot"].to_numpy() - P0
    n = len(hd)
    IMP = np.full((n, 17), np.nan); PRES = np.zeros((n, 17), bool); CONF = np.zeros((n, 17), bool)
    AI = np.full((n, 17), np.nan)
    IMP[row, col] = w["imp"].to_numpy(); PRES[row, col] = True; CONF[row, col] = w["conf"].to_numpy()
    AI[row, col] = w["ai"].to_numpy()
    miss = 17 - PRES.sum(1)
    conf_any = CONF.any(1)
    nan_any = (PRES & ~np.isfinite(IMP)).any(1)
    d16 = np.diff(IMP, axis=1)
    with np.errstate(invalid="ignore"):
        neg_any = (d16 < 0).any(1); big_any = (d16 > args.cap_kwh).any(1)
    status = np.full(n, "valid", dtype=object)
    status[big_any] = "implausible_interval"; status[neg_any] = "negative_interval"
    status[nan_any] = "nan_value"; status[conf_any] = "conflicting_duplicate"; status[miss > 0] = "missing_reading"
    valid = status == "valid"
    peak = np.where(valid, IMP[:, 16] - IMP[:, 0], np.nan)
    avg16 = 0.25 * AI[:, 1:].sum(1)
    avg_ok = valid & np.isfinite(AI[:, 1:]).all(1)
    err = np.abs(peak - avg16)
    out = pd.DataFrame({"household_ID": hh_names[hd["hh"].to_numpy()], "date": pd.to_datetime(hd["day"], unit="D").dt.strftime("%Y-%m-%d"),
                        "n_readings_day": hd["n_readings_day"], "n_window_readings": PRES.sum(1), "status": status,
                        "peak_kwh_1830_2230": np.round(peak, 6),
                        "avgpower_check_kwh": np.where(avg_ok, np.round(avg16, 6), np.nan)})
    out.to_csv(args.out / "household_days_peak.csv", index=False)
    vals = peak[valid]
    R["peak"] = {"evaluated_household_days": int(n), "valid": int(valid.sum()),
                 "status_counts": {k: int(v) for k, v in pd.Series(status).value_counts().items()},
                 "missing_window_readings_hist": np.bincount(miss, minlength=18).tolist(),
                 "zero_kwh_valid_days": int((vals == 0).sum()),
                 "quantiles_kwh": {str(q): float(np.quantile(vals, q, method="lower")) for q in (0, .01, .25, .5, .75, .99, 1)} if vals.size else {},
                 "avgpower_crosscheck": {"n": int(avg_ok.sum()),
                                         **{k: int(c) for k, c in zip(["<0.01", "<0.05", "<0.1", "<0.5", ">=0.5"],
                                                                      np.histogram(err[avg_ok], [0, .01, .05, .1, .5, np.inf])[0])}}}
    v = out[out["status"] == "valid"]
    per_hh = v.groupby("household_ID").size()
    R["coverage"] = {"households_with_valid": int(per_hh.size), "hh_ge30": int((per_hh >= 30).sum()),
                     "hh_ge60": int((per_hh >= 60).sum()), "hh_ge100": int((per_hh >= 100).sum()),
                     "hh_ge200": int((per_hh >= 200).sum()),
                     "by_month": {k: int(c) for k, c in v.groupby(v["date"].str[:7]).size().items()},
                     "dates_with_valid": int(v["date"].nunique()),
                     "first_valid_date": v["date"].min() if len(v) else None, "last_valid_date": v["date"].max() if len(v) else None}
    per_hh.rename("valid_days").to_csv(args.out / "valid_days_per_household.csv")
    v.groupby("date").size().rename("valid_households").to_csv(args.out / "valid_households_per_date.csv")
    with open(args.out / "audit_report.json", "w") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(json.dumps({k: R[k] for k in ("household_days", "peak", "coverage")}, indent=2, default=str))


if __name__ == "__main__":
    main()
