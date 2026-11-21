#!/usr/bin/env python3
"""
PeakWise data audit for RECON-SL 15-minute smart-meter data (v2, memory-bounded).

Reads the three 15-minute CSVs straight out of Zenodo's data.zip (record 15023048,
md5 dcdac5809b4da6b6362519d33112a5f9) or from extracted CSV files, audits them, and extracts
the PeakWise target: household grid import (kWh) over 18:30-22:30 Sri Lanka time.

Semantics this script relies on (verified in the audit report; re-checked on every run):
  * importkwh(kwh) / exportkwh(kwh) are cumulative energy registers (kWh), not interval energy.
  * date + time are Sri Lanka local time (UTC+05:30). `timestamp` is epoch milliseconds rounded
    to 1e7 ms (~2.8 h), so it cannot be used for ordering; it is only cross-checked.
  * A row stamped t holds the register at t; the interval it closes is (t-15min, t].
  => Peak energy for day D = R(D 22:30) - R(D 18:30), built from the 17 readings
     18:30, 18:45, ..., 22:30, i.e. exactly 16 complete 15-minute intervals.

Pipeline (peak memory is bounded by the chunk size and the largest bucket, never by the dataset):
  pass 1  stream every CSV in chunks, validate each row, count every rejection, and append one
          compact fixed-width record per usable row to an on-disk bucket chosen by
          crc32(household_ID) % buckets. A household therefore lives in exactly one bucket, with
          its rows from all files (so file boundaries and row order are irrelevant).
  pass 2  load one bucket at a time, sort by (household, 15-min slot, file, row), apply the audit
          rules, emit that bucket's household-day rows, and fold its additive counters into the
          report. Per-bucket household-day files are k-way merged into one globally sorted CSV.

Nothing is imputed. Every household-day that has at least one reading appears in the output
with a status (valid or the first failing rule), and every dropped or flagged row is counted.

Usage:
  python peakwise_audit.py --zip data.zip --verify-md5 --out audit_out
  python peakwise_audit.py --csv smart_15min_1.csv smart_15min_2.csv smart_15min_3.csv --out audit_out
Options: --verify-md5 (aborts with exit code 2 on mismatch), --expected-md5, --households ID0001,ID0002
         (debug subset), --cap-kwh 10, --chunksize 500000, --buckets 64, --keep-temp,
         --no-tr2-check (skip the cross-check against the six-hour TR2_TOTAL_IMPORT register),
         --csv6 (six-hour CSVs when using --csv)
Requirements: Python >= 3.9, pandas >= 1.5, numpy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import resource
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_MD5 = "dcdac5809b4da6b6362519d33112a5f9"
MEMBERS = [f"data/consumption_data/smart_meter/15min_interval/smart_15min_{k}.csv" for k in (1, 2, 3)]
MEMBER_BYTES = {MEMBERS[0]: 948_902_706, MEMBERS[1]: 1_696_953_202, MEMBERS[2]: 2_015_227_030}
EXPECTED_HEADER = ("household_ID,timestamp,date,time,obis,importkwh(kwh),exportkwh(kwh),importkvarh(kvarh),"
                   "exportkvarh(kvarh),powerfactor,avgcurrent(v),avgexportkw(kw),avgimportkw(kw),avgvoltage(v),"
                   "phaseainstcurrent(a),phaseainstvoltage(v),phasebinstcurrent(a),phasebinstvoltage(v),"
                   "phasecinstcurrent(a),phasecinstvoltage(v)")
USECOLS = ["household_ID", "timestamp", "date", "time", "obis", "importkwh(kwh)", "exportkwh(kwh)",
           "avgexportkw(kw)", "avgimportkw(kw)"]
H6_MEMBERS = [f"data/consumption_data/smart_meter/6hour_interval/smart_6hour_{k}.csv" for k in (1, 2, 3, 4, 5)]
H6_COLS = ["household_ID", "DATE", "TIME", "TOTAL_IMPORT (kWh)", "TR2_TOTAL_IMPORT (kWh)"]
BIG = 10_000_000                     # key = household_code * BIG + slot_index (slot_index < 2.5e6 for any real date)
TR2_AGREE_TOL_KWH = 0.01             # 15-min peak and meter-booked TR2 peak "agree" within this
PLATEAU_TOL_KWH = 0.002              # TR2 is flat outside the peak period; allow float-rounding noise
SHIFTS = tuple(range(-4, 5))         # window shifts (in 15-min slots) used to test the 18:30-22:30 alignment
SLT_OFFSET_MS = 19_800_000           # UTC+05:30
TS_TOL_MS = 5_000_000                # half of the 1e7 ms rounding unit
P0, P1 = 74, 90                      # slots of 18:30 and 22:30 (slot = hour*4 + minute/15)
BIN_EDGES = [0, 0.01, 0.1, 0.5, 1, 2.5, 5, 10, 25]
PEAK_HIST_EDGES = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10]
QUANTILES = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)
STATUSES = ("valid", "missing_reading", "conflicting_duplicate", "nan_value", "negative_interval",
            "implausible_interval")

# one compact record per usable six-hour reading (tau = day*96 + seconds_of_day/900, i.e. in 15-min slot units)
REC6 = np.dtype([("hh", "<i4"), ("src", "i1"), ("row", "<i8"), ("tau", "<f8"), ("ti", "<f8"), ("tr2", "<f8")])
# one compact record per usable reading (packed, 43 bytes)
REC = np.dtype([("hh", "<i4"), ("day", "<i4"), ("slot", "<i2"), ("src", "i1"), ("obis", "i1"), ("row", "<i8"),
                ("imp", "<f8"), ("exp", "<f8"), ("ai", "<f4"), ("ae", "<f4")])


# ----------------------------------------------------------------------------- helpers
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


def open_sources6(args):
    if args.zip:
        zf = zipfile.ZipFile(args.zip)
        names = set(zf.namelist())
        for m in H6_MEMBERS:
            if m in names:
                yield Path(m).name, (lambda m=m: zf.open(m))
    else:
        for p in (args.csv6 or []):
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


def classify_times(date_s: pd.Series, time_s: pd.Series):
    """Strictly validate the date and time strings of a chunk.

    reason: 0 ok, 1 malformed/impossible date, 2 time not HH:MM:SS, 3 time fields out of range
    (hour > 23, minute > 59 or second > 59). offgrid marks valid times that are not on a whole
    quarter hour. Returns (day_since_epoch int64, H, M, S float arrays, reason int8, offgrid bool).
    """
    fmt_ok = date_s.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).to_numpy(dtype=bool)
    d = pd.to_datetime(date_s.where(fmt_ok, ""), format="%Y-%m-%d", errors="coerce")
    bad_date = d.isna().to_numpy()
    tp = time_s.str.extract(r"^(\d{2}):(\d{2}):(\d{2})$").astype(float)
    H, M, S = tp[0].to_numpy(), tp[1].to_numpy(), tp[2].to_numpy()
    fmt_bad = np.isnan(H)
    with np.errstate(invalid="ignore"):
        range_bad = ~fmt_bad & ((H > 23) | (M > 59) | (S > 59))
    reason = np.zeros(len(date_s), dtype=np.int8)
    reason[range_bad] = 3
    reason[fmt_bad] = 2
    reason[bad_date] = 1
    with np.errstate(invalid="ignore"):
        offgrid = (reason == 0) & ((M % 15 != 0) | (S != 0))
    day = np.zeros(len(date_s), dtype="int64")
    ok = ~bad_date
    day[ok] = d[ok].to_numpy().astype("datetime64[D]").astype("int64")
    return day, H, M, S, reason, offgrid


def bucket_of(name: str, k: int) -> int:
    return zlib.crc32(name.encode()) % k


class Households:
    """Stable int32 codes for household IDs (a few thousand strings; the only O(households) state)."""
    def __init__(self, k):
        self.code, self.names, self.bucket, self.k = {}, [], [], k

    def encode(self, values: np.ndarray) -> np.ndarray:
        for h in pd.unique(values):
            if h not in self.code:
                self.code[h] = len(self.names)
                self.names.append(h)
                self.bucket.append(bucket_of(h, self.k))
        return pd.Series(values).map(self.code).to_numpy(dtype="int32")


class BucketWriter:
    def __init__(self, directory: Path, k: int, prefix: str = "bucket"):
        self.dir, self.k, self.prefix, self.handles, self.rows = directory, k, prefix, {}, np.zeros(k, dtype="int64")
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, b):
        return self.dir / f"{self.prefix}_{b:03d}.bin"

    def write(self, rec: np.ndarray, buckets: np.ndarray):
        if len(rec) == 0:
            return
        order = np.argsort(buckets, kind="stable")
        rec, buckets = rec[order], buckets[order]
        cuts = np.flatnonzero(np.diff(buckets)) + 1
        for part, b in zip(np.split(rec, cuts), buckets[np.r_[0, cuts]]):
            if int(b) not in self.handles:
                self.handles[int(b)] = open(self.path(int(b)), "wb")
            self.handles[int(b)].write(part.tobytes())
            self.rows[int(b)] += len(part)

    def close(self):
        for fh in self.handles.values():
            fh.close()
        self.handles = {}


# ----------------------------------------------------------------------------- pass 1
def read_file(label, opener, args, file_stats, src_index, hh, writer):
    n_newlines, header = count_lines(opener)
    st = {"header_matches_expected": header == EXPECTED_HEADER, "header": header,
          "newline_count": n_newlines, "rows": 0, "rows_kept_for_analysis": 0, "obis": {},
          "min_date": None, "max_date": None,
          "bad_datetime": 0, "bad_date_format": 0, "bad_time_format": 0, "invalid_time_range": 0,
          "offgrid_time": 0, "ts_local": 0, "ts_utc_like": 0, "ts_other": 0,
          "ts_max_abs_dev_local_ms": 0.0, "ts_multiple_of_1e7": 0, "import_empty": 0, "export_empty": 0,
          "avgimport_empty": 0, "import_non_numeric": 0, "import_raw_negative": 0, "export_raw_negative": 0}
    hh_filter = set(args.households.split(",")) if args.households else None
    st["max_chunk_rows"] = 0
    with opener() as fh:
        reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", newline=""), usecols=USECOLS, dtype=str,
                             keep_default_na=False, chunksize=args.chunksize, on_bad_lines="skip")
        row0 = 0
        for ch in reader:
            st["rows"] += len(ch)
            st["max_chunk_rows"] = max(st["max_chunk_rows"], len(ch))
            idx = np.arange(row0, row0 + len(ch)); row0 += len(ch)
            if hh_filter is not None:
                keep = ch["household_ID"].isin(hh_filter).to_numpy()
                ch, idx = ch[keep], idx[keep]
                if ch.empty:
                    continue
            for k, v in ch["obis"].value_counts().items():
                st["obis"][k] = st["obis"].get(k, 0) + int(v)
            day, H, M, S, reason, offgrid = classify_times(ch["date"], ch["time"])
            okb = reason == 0
            st["bad_date_format"] += int((reason == 1).sum()); st["bad_time_format"] += int((reason == 2).sum())
            st["invalid_time_range"] += int((reason == 3).sum()); st["bad_datetime"] += int((reason != 0).sum())
            if okb.any():
                dmin, dmax = ch["date"][okb].min(), ch["date"][okb].max()
                st["min_date"] = dmin if st["min_date"] is None or dmin < st["min_date"] else st["min_date"]
                st["max_date"] = dmax if st["max_date"] is None or dmax > st["max_date"] else st["max_date"]
            secs = H * 3600 + M * 60 + S
            ts = pd.to_numeric(ch["timestamp"], errors="coerce").to_numpy()
            with np.errstate(invalid="ignore"):
                dev = ts - (day * 86_400_000 + secs * 1000 - SLT_OFFSET_MS)
                loc = okb & (np.abs(dev) <= TS_TOL_MS)
                utc = okb & ~loc & (np.abs(dev - SLT_OFFSET_MS) <= TS_TOL_MS)
            st["ts_local"] += int(loc.sum()); st["ts_utc_like"] += int(utc.sum())
            st["ts_other"] += int((okb & ~loc & ~utc).sum())
            if loc.any():
                st["ts_max_abs_dev_local_ms"] = max(st["ts_max_abs_dev_local_ms"], float(np.abs(dev[loc]).max()))
            with np.errstate(invalid="ignore"):
                st["ts_multiple_of_1e7"] += int((np.mod(ts, 1e7) == 0).sum())
            imp_s, exp_s = ch["importkwh(kwh)"], ch["exportkwh(kwh)"]
            st["import_empty"] += int((imp_s == "").sum()); st["export_empty"] += int((exp_s == "").sum())
            st["avgimport_empty"] += int((ch["avgimportkw(kw)"] == "").sum())
            imp = pd.to_numeric(imp_s, errors="coerce").to_numpy(); ex = pd.to_numeric(exp_s, errors="coerce").to_numpy()
            st["import_non_numeric"] += int(((imp_s != "") & np.isnan(imp)).sum())
            with np.errstate(invalid="ignore"):
                st["import_raw_negative"] += int((imp < 0).sum()); st["export_raw_negative"] += int((ex < 0).sum())
            st["offgrid_time"] += int(offgrid.sum())
            keep = okb & ~offgrid
            n = int(keep.sum())
            rec = np.empty(n, dtype=REC)
            codes = hh.encode(ch["household_ID"].to_numpy()[keep])
            rec["hh"], rec["day"] = codes, day[keep]
            rec["slot"] = (H[keep] * 4 + M[keep] // 15)
            rec["src"], rec["row"] = src_index, idx[keep]
            rec["obis"] = np.where(ch["obis"].to_numpy()[keep] == "LP", 0, np.where(ch["obis"].to_numpy()[keep] == "DBE", 1, 2))
            rec["imp"], rec["exp"] = imp[keep], ex[keep]
            rec["ai"] = pd.to_numeric(ch["avgimportkw(kw)"], errors="coerce").to_numpy()[keep]
            rec["ae"] = pd.to_numeric(ch["avgexportkw(kw)"], errors="coerce").to_numpy()[keep]
            writer.write(rec, np.asarray(hh.bucket, dtype="int16")[codes] if n else np.empty(0, "int16"))
            st["rows_kept_for_analysis"] += n
    st["malformed_rows_skipped"] = (st["newline_count"] - 1) - st["rows"] if hh_filter is None else None
    file_stats[label] = st


def read_file6(label, opener, args, file_stats, src_index, hh, writer):
    """Pass 1 for a six-hour file: validate rows, count rejections, bucket TOTAL_IMPORT and TR2_TOTAL_IMPORT."""
    n_newlines, header = count_lines(opener)
    st = {"header": header, "newline_count": n_newlines, "rows": 0, "rows_kept_for_analysis": 0,
          "min_date": None, "max_date": None, "bad_datetime": 0, "bad_date_format": 0, "bad_time_format": 0,
          "invalid_time_range": 0, "total_import_empty": 0, "tr2_empty": 0, "total_import_non_numeric": 0,
          "tr2_non_numeric": 0, "tr2_raw_negative": 0, "total_import_raw_negative": 0, "max_chunk_rows": 0}
    hh_filter = set(args.households.split(",")) if args.households else None
    with opener() as fh:
        reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", newline=""), usecols=H6_COLS, dtype=str,
                             keep_default_na=False, chunksize=args.chunksize, on_bad_lines="skip")
        row0 = 0
        for ch in reader:
            st["rows"] += len(ch); st["max_chunk_rows"] = max(st["max_chunk_rows"], len(ch))
            idx = np.arange(row0, row0 + len(ch)); row0 += len(ch)
            if hh_filter is not None:
                keep = ch["household_ID"].isin(hh_filter).to_numpy()
                ch, idx = ch[keep], idx[keep]
                if ch.empty:
                    continue
            day, H, M, S, reason, _ = classify_times(ch["DATE"], ch["TIME"])
            ok = reason == 0
            st["bad_date_format"] += int((reason == 1).sum()); st["bad_time_format"] += int((reason == 2).sum())
            st["invalid_time_range"] += int((reason == 3).sum()); st["bad_datetime"] += int((reason != 0).sum())
            if ok.any():
                dmin, dmax = ch["DATE"][ok].min(), ch["DATE"][ok].max()
                st["min_date"] = dmin if st["min_date"] is None or dmin < st["min_date"] else st["min_date"]
                st["max_date"] = dmax if st["max_date"] is None or dmax > st["max_date"] else st["max_date"]
            ti_s, tr_s = ch["TOTAL_IMPORT (kWh)"], ch["TR2_TOTAL_IMPORT (kWh)"]
            ti, tr2 = pd.to_numeric(ti_s, errors="coerce").to_numpy(), pd.to_numeric(tr_s, errors="coerce").to_numpy()
            st["total_import_empty"] += int((ti_s == "").sum()); st["tr2_empty"] += int((tr_s == "").sum())
            st["total_import_non_numeric"] += int(((ti_s != "") & np.isnan(ti)).sum())
            st["tr2_non_numeric"] += int(((tr_s != "") & np.isnan(tr2)).sum())
            with np.errstate(invalid="ignore"):
                st["tr2_raw_negative"] += int((tr2 < 0).sum()); st["total_import_raw_negative"] += int((ti < 0).sum())
            keep = ok & (np.isfinite(ti) | np.isfinite(tr2))
            n = int(keep.sum())
            rec = np.empty(n, dtype=REC6)
            codes = hh.encode(ch["household_ID"].to_numpy()[keep])
            rec["hh"], rec["src"], rec["row"] = codes, src_index, idx[keep]
            rec["tau"] = day[keep] * 96 + (H[keep] * 3600 + M[keep] * 60 + S[keep]) / 900.0
            rec["ti"], rec["tr2"] = ti[keep], tr2[keep]
            writer.write(rec, np.asarray(hh.bucket, dtype="int16")[codes] if n else np.empty(0, "int16"))
            st["rows_kept_for_analysis"] += n
    st["malformed_rows_skipped"] = (st["newline_count"] - 1) - st["rows"] if hh_filter is None else None
    file_stats[label] = st


# ----------------------------------------------------------------------------- pass 2

def _first_dup_stats(df, keys, cols):
    """Exact / conflicting duplicate counts for df already sorted so the first row of each key group is kept."""
    kf = ~df.duplicated(keys, keep="first")
    grp = kf.cumsum().to_numpy() - 1
    same = np.ones(len(df), dtype=bool)
    for c in cols:
        a = df[c].to_numpy(); f = a[kf.to_numpy()][grp]
        same &= (a == f) | (np.isnan(a) & np.isnan(f))
    dup = ~kf.to_numpy()
    return kf.to_numpy(), int(dup.sum()), int((dup & same).sum()), int((dup & ~same).sum())


def _plateau_flat(tr2, lo, hi, has, max_readings=64):
    """True where every TR2 reading in [lo, hi] lies within PLATEAU_TOL_KWH of the others (so TR2 was flat).
    Plateaus with more than `max_readings` readings are conservatively reported as not flat."""
    n = len(tr2)
    ok = np.zeros(len(lo), dtype=bool)
    if not has.any() or n == 0:
        return ok
    span = np.where(has, hi - lo, 0)
    ok = has & (span < max_readings)
    mn, mx = np.full(len(lo), np.inf), np.full(len(lo), -np.inf)
    for k in range(int(min(span.max(), max_readings - 1)) + 1):
        idx = np.clip(lo + k, 0, n - 1)
        use = has & (k <= span)
        v = tr2[idx]
        mn = np.where(use, np.minimum(mn, v), mn)
        mx = np.where(use, np.maximum(mx, v), mx)     # NaN propagates through maximum/minimum
    with np.errstate(invalid="ignore"):
        return ok & np.isfinite(mn) & np.isfinite(mx) & ((mx - mn) <= PLATEAU_TOL_KWH)


def tr2_crosscheck(u, hd, valid, peak, h6, cap):
    """Independent check of the 15-min peak against the six-hour meter registers.

    TR2_TOTAL_IMPORT is the cumulative *peak-period* register, so it only rises during the ToU peak
    (18:30-22:30). It is therefore flat from 22:30 of the previous day to 18:30 of day D (plateau B) and from
    22:30 of D to 18:30 of D+1 (plateau A). For each valid household-day we take the six-hour reading nearest
    to 18:30 inside plateau B and nearest to 22:30 inside plateau A; their difference is the peak energy the
    meter itself booked. We also test (i) the window alignment by shifting the 15-min window by -4..+4 slots and
    (ii) whether 15-min importkwh is the same register as six-hour TOTAL_IMPORT (bracketing test).
    """
    T = {}
    uk = u["hh"].to_numpy().astype("int64") * BIG + u["t"].to_numpy()
    uimp, uuse = u["imp"].to_numpy(), u["usable"].to_numpy()

    def reg(hh, t):
        key = np.asarray(hh, dtype="int64") * BIG + np.asarray(t, dtype="int64")
        pos = np.minimum(np.searchsorted(uk, key), len(uk) - 1)
        ok = (uk[pos] == key) & uuse[pos]
        return np.where(ok, uimp[pos], np.nan)

    h6 = h6.sort_values(["hh", "tau", "src", "row"], kind="mergesort").reset_index(drop=True)
    kf, nd, ne, nc = _first_dup_stats(h6, ["hh", "tau"], ["ti", "tr2"])
    T["h6_readings"], T["h6_duplicate_rows"], T["h6_exact_duplicates"], T["h6_conflicting_duplicates"] = len(h6), nd, ne, nc
    h = h6[kf].reset_index(drop=True)
    hh6, tau6, tr2, ti = h["hh"].to_numpy().astype("int64"), h["tau"].to_numpy(), h["tr2"].to_numpy(), h["ti"].to_numpy()
    keys6 = hh6 * float(BIG) + tau6
    same_hh = np.r_[False, hh6[1:] == hh6[:-1]]
    with np.errstate(invalid="ignore"):
        T["h6_tr2_decreasing_steps"] = int((same_hh & (np.r_[np.nan, np.diff(tr2)] < -PLATEAU_TOL_KWH)).sum())
        T["h6_total_import_decreasing_steps"] = int((same_hh & (np.r_[np.nan, np.diff(ti)] < -PLATEAU_TOL_KWH)).sum())

    # ---- (A) TR2 rise versus the 15-min extracted peak, on valid household-days
    vi = np.flatnonzero(valid)
    T["valid_days_considered"] = int(len(vi))
    rows = arr = None
    if len(vi):
        hhv, dayv = hd["hh"].to_numpy()[vi].astype("int64"), hd["day"].to_numpy()[vi].astype("int64")
        base = hhv * float(BIG)
        s0, s1 = dayv * 96 + 74, dayv * 96 + 90
        lo_b, hi_b = np.searchsorted(keys6, base + (dayv - 1) * 96 + 90, "left"), np.searchsorted(keys6, base + s0, "right") - 1
        lo_a, hi_a = np.searchsorted(keys6, base + s1, "left"), np.searchsorted(keys6, base + (dayv + 1) * 96 + 74, "right") - 1
        n6 = len(keys6)
        has_b, has_a = (hi_b >= lo_b) & (hi_b >= 0) & (lo_b < n6), (hi_a >= lo_a) & (hi_a >= 0) & (lo_a < n6)
        g = lambda a, i: a[np.clip(i, 0, max(n6 - 1, 0))]
        before, after = g(tr2, hi_b), g(tr2, lo_a)
        plat_b, plat_a = _plateau_flat(tr2, lo_b, hi_b, has_b), _plateau_flat(tr2, lo_a, hi_a, has_a)
        fin = np.isfinite(before) & np.isfinite(after)
        both = has_b & has_a & fin
        compared = both & plat_b & plat_a
        peak15, tr2_peak = peak[vi], after - before
        diff = peak15 - tr2_peak
        T.update({"valid_days_with_bracket": int(both.sum()), "valid_days_plateau_consistent_bracket": int(compared.sum()),
                  "valid_days_bracket_but_plateau_inconsistent": int((both & ~(plat_b & plat_a)).sum()),
                  "tr2_peak_negative_days": int((both & (tr2_peak < -PLATEAU_TOL_KWH)).sum())})
        ad = np.abs(diff[compared])
        T["abs_diff_bins"] = {k: int(c) for k, c in zip(["<=0.005", "<=0.01", "<=0.05", "<=0.1", "<=0.5", ">0.5"],
                              np.histogram(ad, [-1, 0.005, 0.01, 0.05, 0.1, 0.5, np.inf])[0])}
        T["sum_signed_diff"] = float(diff[compared].sum())
        T["shift"] = {}
        for d in SHIFTS:
            pd_ = reg(hhv, s1 + d) - reg(hhv, s0 + d)
            e = np.abs(pd_[compared] - tr2_peak[compared]); e = e[np.isfinite(e)]
            T["shift"][f"{d:+d}"] = {"n": int(e.size), "sum_abs": float(e.sum()), "within_0.01": int((e <= 0.01).sum()),
                                     "within_0.05": int((e <= 0.05).sum())}
        tb, ta = tau6[np.clip(hi_b, 0, n6 - 1)], tau6[np.clip(lo_a, 0, n6 - 1)]
        rows = pd.DataFrame({"hh": hhv[both], "day": dayv[both], "peak_15min_kwh": np.round(peak15[both], 6),
                             "tr2_before_kwh": before[both], "tr2_after_kwh": after[both],
                             "tr2_peak_kwh": np.round(tr2_peak[both], 6), "diff_kwh": np.round(diff[both], 6),
                             "plateau_consistent": (plat_b & plat_a)[both],
                             "before_gap_min": np.round((s0[both] - tb[both]) * 15, 1),
                             "after_gap_min": np.round((ta[both] - s1[both]) * 15, 1)})
        arr = {"absdiff": ad, "peak15": peak15[compared], "tr2": tr2_peak[compared]}

    # ---- (B) is 15-min importkwh the same register as six-hour TOTAL_IMPORT?
    m = np.isfinite(ti)
    f = np.floor(tau6[m]).astype("int64"); hm = hh6[m]
    Rf, Rc, tim, tm = reg(hm, f), reg(hm, f + 1), ti[m], tau6[m]
    with np.errstate(invalid="ignore"):
        pair = np.isfinite(Rf) & np.isfinite(Rc) & (Rc - Rf >= 0) & (Rc - Rf <= cap)
        brk = pair & (tim >= Rf - 0.01) & (tim <= Rc + 0.01)
        resid = (tim - (Rf + (tm - f) * (Rc - Rf)))[pair]
    T["identity"] = {"h6_total_import_readings": int(m.sum()), "with_adjacent_15min_registers": int(pair.sum()),
                     "bracketed_by_15min_registers": int(brk.sum()),
                     "abs_resid_bins": {k: int(c) for k, c in zip(["<=0.01", "<=0.05", "<=0.1", "<=0.5", "<=1", ">1"],
                                        np.histogram(np.abs(resid), [-1, 0.01, 0.05, 0.1, 0.5, 1, np.inf])[0])}}
    return T, rows, arr, resid


def analyze_bucket(df: pd.DataFrame, hh_names, cap: float, h6=None):
    """Apply the audit rules to one bucket (whole households, all files). Returns (S, out_frame, peaks)."""
    S: dict = {}
    df["t"] = df["day"].astype("int64") * 96 + df["slot"]
    df.sort_values(["hh", "t", "src", "row"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # ---------- duplicates on (household, 15-min slot)
    key_first = ~df.duplicated(["hh", "t"], keep="first")
    kf = key_first.to_numpy()
    grp = key_first.cumsum().to_numpy() - 1
    imp_a, exp_a = df["imp"].to_numpy(), df["exp"].to_numpy()
    first_imp, first_exp = imp_a[kf][grp], exp_a[kf][grp]
    same = (((imp_a == first_imp) | (np.isnan(imp_a) & np.isnan(first_imp))) &
            ((exp_a == first_exp) | (np.isnan(exp_a) & np.isnan(first_exp))))
    dup = ~kf
    conflict_row = dup & ~same
    conflict_grp = np.zeros(grp.max() + 1, dtype=bool); conflict_grp[grp[conflict_row]] = True
    first_src = df["src"].to_numpy()[kf][grp]
    S["duplicates"] = {"duplicate_rows": int(dup.sum()), "exact_duplicate_rows": int((dup & same).sum()),
                       "conflicting_duplicate_rows": int(conflict_row.sum()),
                       "conflicting_slots": int(conflict_grp.sum()),
                       "cross_file_duplicate_rows": int((dup & (df["src"].to_numpy() != first_src)).sum())}
    S["_dup_examples"] = [(hh_names[a], str(np.datetime64(int(b), "D")), int(c), float(d_), float(e_))
                          for a, b, c, d_, e_ in df.loc[conflict_row, ["hh", "day", "slot", "imp", "exp"]]
                          .head(10).itertuples(index=False)]
    u = df[key_first].copy()                     # one row per household-slot (first occurrence)
    u["conf"] = conflict_grp
    u["usable"] = ~u["conf"] & np.isfinite(u["imp"]) & np.isfinite(u["exp"])
    S["unusable_readings"] = int((~u["usable"]).sum())

    # ---------- consecutive-reading analysis
    hh_codes = u["hh"].to_numpy()
    t = u["t"].to_numpy(); same_hh = np.r_[False, hh_codes[1:] == hh_codes[:-1]]
    gap = np.r_[0, np.diff(t)]
    g = gap[same_hh]
    S["gaps"] = {"1": int((g == 1).sum()), "2-4": int(((g >= 2) & (g <= 4)).sum()),
                 "5-96": int(((g >= 5) & (g <= 96)).sum()), ">96": int((g > 96).sum())}
    span = u.groupby("hh")["t"].agg(["min", "max", "size"])
    S["span"] = {"households": int(len(span)),
                 "expected_slots_within_span": int((span["max"] - span["min"] + 1).sum()),
                 "present_slots": int(span["size"].sum())}
    S["non_lp_rows_in_register_series"] = int((u["obis"] != 0).sum())
    usable = u["usable"].to_numpy()
    pair = same_hh & (gap == 1)
    pu = pair & usable & np.r_[False, usable[:-1]]
    imp_u, exp_u = u["imp"].to_numpy(), u["exp"].to_numpy()
    dI = np.r_[np.nan, np.diff(imp_u)]; dE = np.r_[np.nan, np.diff(exp_u)]
    ai, ae = u["ai"].to_numpy().astype(float), u["ae"].to_numpy().astype(float)
    ai0, ae0 = np.r_[np.nan, ai[:-1]], np.r_[np.nan, ae[:-1]]
    slot_u = u["slot"].to_numpy()
    di, de = dI[pu], dE[pu]
    nz = lambda a: a[np.isfinite(a)]
    bins_i, bins_e = bin_counts(di[di >= 0]), bin_counts(de[de >= 0])
    S["intervals"] = {"pairs_15min": int(pair.sum()), "pairs_usable": int(pu.sum()),
                      "import_negative": int((di < 0).sum()), "export_negative": int((de < 0).sum()),
                      "import_gt_cap": int((di > cap).sum()), "export_gt_cap": int((de > cap).sum()),
                      "import_bins": bins_i, "export_bins": bins_e,
                      "import_negative_by_end_slot": np.bincount(slot_u[pu][di < 0], minlength=96),
                      "import_gt_cap_by_end_slot": np.bincount(slot_u[pu][di > cap], minlength=96)}
    neg_idx = np.flatnonzero(pu & (dI < 0))[:12]
    S["_neg_examples"] = [(hh_names[u["hh"].iat[i]], str(np.datetime64(int(t[i]) * 15, "m")),
                           float(imp_u[i - 1]), float(imp_u[i]), float(dI[i])) for i in neg_idx]
    m = pu & (dI >= 0) & (dI <= 5) & np.isfinite(ai) & np.isfinite(ai0)
    e_end, e_start = np.abs(dI[m] - 0.25 * ai[m]), np.abs(dI[m] - 0.25 * ai0[m])
    mE = pu & (dE >= 0) & (dE <= 5) & np.isfinite(ae) & np.isfinite(ae0) & ((ae > 0) | (ae0 > 0) | (dE > 0))
    f_end, f_start = np.abs(dE[mE] - 0.25 * ae[mE]), np.abs(dE[mE] - 0.25 * ae0[mE])
    S["interval_label_test"] = {
        "import": {"n": int(m.sum()), "sum_err_end": float(e_end.sum()), "sum_err_start": float(e_start.sum()),
                   "n_within_end": int((e_end <= 0.01).sum()), "n_within_start": int((e_start <= 0.01).sum())},
        "export": {"n": int(mE.sum()), "sum_err_end": float(f_end.sum()), "sum_err_start": float(f_start.sum()),
                   "n_within_end": int((f_end <= 0.01).sum()), "n_within_start": int((f_start <= 0.01).sum())}}

    # ---------- household-day table and peak target
    hd = u.groupby(["hh", "day"]).size().rename("n_readings_day").reset_index()
    n = len(hd)
    S["household_days"] = n
    n_lp = u.assign(_lp=(u["obis"] == 0)).groupby(["hh", "day"])["_lp"].sum().to_numpy()
    S["household_days_without_any_lp_row"] = int((n_lp == 0).sum())
    S["readings_per_day_hist"] = np.bincount(hd["n_readings_day"].to_numpy(), minlength=97)
    sp = df.groupby(["hh", "day"])["src"].agg(["min", "max"])
    S["days_spanning_files"] = int((sp["max"] != sp["min"]).sum())
    w = u[(u["slot"] >= P0) & (u["slot"] <= P1)]
    hd_index = pd.MultiIndex.from_frame(hd[["hh", "day"]])
    row = hd_index.get_indexer(pd.MultiIndex.from_arrays([w["hh"], w["day"]]))
    col = w["slot"].to_numpy() - P0
    IMP = np.full((n, 17), np.nan); PRES = np.zeros((n, 17), bool); CONF = np.zeros((n, 17), bool)
    AI = np.full((n, 17), np.nan)
    IMP[row, col] = w["imp"].to_numpy(); PRES[row, col] = True; CONF[row, col] = w["conf"].to_numpy()
    AI[row, col] = w["ai"].to_numpy()
    miss = 17 - PRES.sum(1)
    conf_any = CONF.any(1)
    nan_any = (PRES & ~np.isfinite(IMP)).any(1)
    d16 = np.diff(IMP, axis=1)
    with np.errstate(invalid="ignore"):
        neg_any = (d16 < 0).any(1); big_any = (d16 > cap).any(1)
    status = np.full(n, "valid", dtype=object)
    status[big_any] = "implausible_interval"; status[neg_any] = "negative_interval"
    status[nan_any] = "nan_value"; status[conf_any] = "conflicting_duplicate"; status[miss > 0] = "missing_reading"
    valid = status == "valid"
    peak = np.where(valid, IMP[:, 16] - IMP[:, 0], np.nan)
    avg16 = 0.25 * AI[:, 1:].sum(1)
    avg_ok = valid & np.isfinite(AI[:, 1:]).all(1)
    err = np.abs(peak - avg16)
    out = pd.DataFrame({"household_ID": np.asarray(hh_names, dtype=object)[hd["hh"].to_numpy()],
                        "date": pd.to_datetime(hd["day"], unit="D").dt.strftime("%Y-%m-%d"),
                        "n_readings_day": hd["n_readings_day"], "n_lp_readings_day": n_lp.astype(int),
                        "n_window_readings": PRES.sum(1), "status": status,
                        "peak_kwh_1830_2230": np.round(peak, 6),
                        "avgpower_check_kwh": np.where(avg_ok, np.round(avg16, 6), np.nan),
                        "reg_1830_kwh": np.where(valid, IMP[:, 0], np.nan),
                        "reg_2230_kwh": np.where(valid, IMP[:, 16], np.nan)})
    out.sort_values(["household_ID", "date"], inplace=True, kind="mergesort")
    vals = peak[valid]
    S["peak"] = {"evaluated_household_days": n, "valid": int(valid.sum()),
                 "status_counts": {k: int((status == k).sum()) for k in STATUSES},
                 "any_rule_flags": {"missing_reading": int((miss > 0).sum()), "conflicting_duplicate": int(conf_any.sum()),
                                    "nan_value": int(nan_any.sum()), "negative_interval": int(neg_any.sum()),
                                    "implausible_interval": int(big_any.sum())},
                 "missing_window_readings_hist": np.bincount(miss, minlength=18),
                 "zero_kwh_valid_days": int((vals == 0).sum()),
                 "avgpower_crosscheck": {"n": int(avg_ok.sum()),
                                         **{k: int(c) for k, c in zip(["<0.01", "<0.05", "<0.1", "<0.5", ">=0.5"],
                                                                      np.histogram(err[avg_ok], [0, .01, .05, .1, .5, np.inf])[0])}}}
    tr2_rows = tr2_arr = tr2_resid = None
    if h6 is not None and len(h6):
        S["tr2"], tr2_rows, tr2_arr, tr2_resid = tr2_crosscheck(u, hd, valid, peak, h6, cap)
        if tr2_rows is not None:
            tr2_rows.insert(0, "household_ID", np.asarray(hh_names, dtype=object)[tr2_rows.pop("hh").to_numpy()])
            tr2_rows.insert(1, "date", pd.to_datetime(tr2_rows.pop("day"), unit="D").dt.strftime("%Y-%m-%d"))
            tr2_rows.sort_values(["household_ID", "date"], inplace=True, kind="mergesort")
    return S, out, vals, (tr2_rows, tr2_arr, tr2_resid)


def merge_add(dst: dict, src: dict):
    for k, v in src.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            merge_add(dst.setdefault(k, {}), v)
        elif isinstance(v, np.ndarray):
            dst[k] = dst[k] + v if k in dst else v.copy()
        else:
            dst[k] = dst.get(k, 0) + v


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


# ----------------------------------------------------------------------------- driver
def verify_md5_or_exit(args):
    if not args.zip:
        sys.exit("--verify-md5 only applies to --zip input")
    m = md5sum(args.zip)
    if m != args.expected_md5:
        print(f"MD5 MISMATCH: {args.zip} is {m}, expected {args.expected_md5}. Aborting; nothing was audited.",
              file=sys.stderr)
        sys.exit(2)
    return m


def run_audit(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    R = {"inputs": {}, "rules": {"peak_window": "R(22:30)-R(18:30), 17 readings, 16 intervals",
                                 "cap_kwh_per_15min": args.cap_kwh, "timezone": "Asia/Colombo (UTC+05:30)",
                                 "interval_label": "row t closes interval (t-15min, t]"}}
    if args.zip:
        R["inputs"]["zip"] = str(args.zip); R["inputs"]["zip_bytes"] = args.zip.stat().st_size
        if args.verify_md5:
            R["inputs"]["zip_md5"] = verify_md5_or_exit(args); R["inputs"]["zip_md5_ok"] = True
        with zipfile.ZipFile(args.zip) as zf:
            R["inputs"]["members"] = {i.filename: {"compressed": i.compress_size, "uncompressed": i.file_size}
                                      for i in zf.infolist() if i.filename in MEMBERS}
            R["inputs"]["member_sizes_match_release"] = all(
                R["inputs"]["members"].get(m, {}).get("uncompressed") == b for m, b in MEMBER_BYTES.items())
    tmp = args.out / "_tmp_buckets"
    if tmp.exists():
        shutil.rmtree(tmp)
    K = args.buckets
    hh, writer, file_stats = Households(K), BucketWriter(tmp, K), {}

    # ---------- pass 1
    for i, (label, opener) in enumerate(open_sources(args)):
        print(f"pass 1: reading {label} ...", flush=True)
        read_file(label, opener, args, file_stats, i, hh, writer)
    writer.close()
    R["files"] = file_stats
    writer6, files6 = BucketWriter(tmp, K, prefix="h6"), {}
    if args.tr2_check:
        for i, (label, opener) in enumerate(open_sources6(args)):
            print(f"pass 1: reading six-hour file {label} ...", flush=True)
            read_file6(label, opener, args, files6, i, hh, writer6)
    writer6.close()
    R["files_six_hour"] = files6

    # ---------- pass 2
    agg, hd_parts, valid_peaks = {}, [], []
    dup_ex, neg_ex = [], []
    per_hh, by_month, per_date, hh_first_last = {}, {}, {}, {}
    max_bucket, max_bucket6 = 0, 0
    tr2_parts, tr2_absdiff, tr2_p15, tr2_t2, tr2_resid = [], [], [], [], []
    for b in range(K):
        p = writer.path(b)
        if not p.exists():
            continue
        arr = np.fromfile(p, dtype=REC)
        max_bucket = max(max_bucket, len(arr))
        df = pd.DataFrame({name: arr[name] for name in REC.names})
        del arr
        h6df = None
        if writer6.path(b).exists():
            a6 = np.fromfile(writer6.path(b), dtype=REC6)
            max_bucket6 = max(max_bucket6, len(a6))
            h6df = pd.DataFrame({name: a6[name] for name in REC6.names})
            del a6
        S, out, vals, (t_rows, t_arr, t_resid) = analyze_bucket(df, hh.names, args.cap_kwh, h6df)
        merge_add(agg, S)
        if t_rows is not None and len(t_rows):
            tp = args.out / f"_tr2_part_{b:03d}.csv"; t_rows.to_csv(tp, index=False); tr2_parts.append(tp)
        if t_arr is not None:
            tr2_absdiff.append(t_arr["absdiff"]); tr2_p15.append(t_arr["peak15"]); tr2_t2.append(t_arr["tr2"])
        if t_resid is not None:
            tr2_resid.append(t_resid)
        dup_ex = sorted(dup_ex + S["_dup_examples"])[:10]
        neg_ex = sorted(neg_ex + S["_neg_examples"])[:12]
        valid_peaks.append(vals)
        part = args.out / f"_hd_part_{b:03d}.csv"
        out.to_csv(part, index=False)
        hd_parts.append(part)
        v = out[out["status"] == "valid"]
        for k, c in v.groupby("household_ID").size().items():
            per_hh[k] = per_hh.get(k, 0) + int(c)
        for k, c in v.groupby(v["date"].str[:7]).size().items():
            by_month[k] = by_month.get(k, 0) + int(c)
        for k, c in v.groupby("date").size().items():
            per_date[k] = per_date.get(k, 0) + int(c)
        for k, grp_ in v.groupby("household_ID")["date"]:
            hh_first_last[k] = (grp_.min(), grp_.max())
        p.unlink()
        if writer6.path(b).exists():
            writer6.path(b).unlink()
    if not hd_parts:
        R["note"] = "no usable rows"
        (args.out / "audit_report.json").write_text(json.dumps(R, indent=2))
        shutil.rmtree(tmp, ignore_errors=True)
        return R

    # ---------- merge the TR2 cross-check parts; keep one small flag per household-day for the target file
    tr2_flag, hh_cmp, hh_bad, tr2_big = {}, {}, {}, []
    if tr2_parts:
        files2 = [open(p, newline="") for p in tr2_parts]
        rds = [csv.reader(f_) for f_ in files2]
        hdr2 = [next(r_) for r_ in rds][0]
        i_d, i_pl = hdr2.index("diff_kwh"), hdr2.index("plateau_consistent")
        with open(args.out / "tr2_crosscheck.csv", "w", newline="") as f2:
            w2 = csv.writer(f2); w2.writerow(hdr2)
            for r_ in heapq.merge(*rds, key=lambda r_: (r_[0], r_[1])):
                w2.writerow(r_)
                d_ = float(r_[i_d])
                if r_[i_pl] != "True":
                    tr2_flag[(r_[0], r_[1])] = "plateau_inconsistent"
                    continue
                tr2_flag[(r_[0], r_[1])] = "agree" if abs(d_) <= TR2_AGREE_TOL_KWH else "disagree"
                hh_cmp[r_[0]] = hh_cmp.get(r_[0], 0) + 1
                if abs(d_) > 0.05:
                    hh_bad[r_[0]] = hh_bad.get(r_[0], 0) + 1
                if abs(d_) > 0.5:
                    tr2_big.append((abs(d_), r_[0], r_[1], float(r_[2]), float(r_[5])))
        for f_ in files2:
            f_.close()
        for p in tr2_parts:
            p.unlink()
        tr2_big = sorted(tr2_big, reverse=True)

    # ---------- k-way merge of per-bucket household-day files into one globally sorted file
    header, files, readers = None, [], []
    for p in hd_parts:
        fh = open(p, newline=""); files.append(fh)
        rd = csv.reader(fh); header = next(rd); readers.append(rd)
    n_out = n_valid = 0
    with open(args.out / "household_days_peak.csv", "w", newline="") as fa, \
            open(args.out / "peakwise_target_clean.csv", "w", newline="") as fc:
        wa, wc = csv.writer(fa), csv.writer(fc)
        wa.writerow(header); wc.writerow(["household_ID", "date", "peak_kwh_1830_2230", "tr2_check"])
        nochk = "not_checked" if not args.tr2_check else "no_bracket"
        iv = header.index("status"); ip = header.index("peak_kwh_1830_2230")
        for r in heapq.merge(*readers, key=lambda r: (r[0], r[1])):
            wa.writerow(r); n_out += 1
            if r[iv] == "valid":
                wc.writerow([r[0], r[1], r[ip], tr2_flag.get((r[0], r[1]), nochk)]); n_valid += 1
    for fh in files:
        fh.close()
    for p in hd_parts:
        p.unlink()
    # ---------- assemble the report
    dups = dict(agg["duplicates"]); dups["examples"] = [list(e) for e in dup_ex]
    R["duplicates"] = dups
    R["unusable_readings"] = agg["unusable_readings"]
    R["gaps"] = agg["gaps"]
    sp = agg["span"]
    R["span"] = {**sp, "missing_slots_within_span": sp["expected_slots_within_span"] - sp["present_slots"]}
    iv_ = dict(agg["intervals"]); iv_["negative_examples"] = [list(e) for e in neg_ex]
    R["intervals"] = iv_
    lt = {}
    for kind, d in agg["interval_label_test"].items():
        nn = d["n"]
        lt[kind] = {"n": nn, "mae_if_interval_ending": d["sum_err_end"] / nn if nn else None,
                    "mae_if_interval_starting": d["sum_err_start"] / nn if nn else None,
                    "share_within_0.01kWh_ending": d["n_within_end"] / nn if nn else None,
                    "share_within_0.01kWh_starting": d["n_within_start"] / nn if nn else None}
    R["interval_label_test"] = lt
    R["household_days"] = agg["household_days"]
    R["days_spanning_files"] = agg["days_spanning_files"]
    R["household_days_without_any_lp_row"] = agg["household_days_without_any_lp_row"]
    R["non_lp_rows_in_register_series"] = agg["non_lp_rows_in_register_series"]
    R["readings_per_day_hist"] = agg["readings_per_day_hist"]
    vals = np.concatenate(valid_peaks) if valid_peaks else np.empty(0)
    pk = agg["peak"]
    if vals.size:
        pk["quantiles_kwh"] = {str(q): float(np.quantile(vals, q, method="lower")) for q in QUANTILES}
        pk["mean_kwh"], pk["std_kwh"] = float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        counts, _ = np.histogram(vals, PEAK_HIST_EDGES + [np.inf])
        labels = [f"[{lo:g},{hi:g})" for lo, hi in zip(PEAK_HIST_EDGES[:-1], PEAK_HIST_EDGES[1:])] + [f">={PEAK_HIST_EDGES[-1]:g}"]
        pk["histogram_kwh"] = {l: int(c) for l, c in zip(labels, counts)}
    else:
        pk["quantiles_kwh"] = {}
    R["peak"] = pk
    ph = np.array(list(per_hh.values())) if per_hh else np.empty(0)
    dates = sorted(per_date)
    miss_ranges = []
    if dates:
        allv = pd.date_range(dates[0], dates[-1]); gone = allv.difference(pd.to_datetime(dates))
        if len(gone):
            grp_ = (pd.Series(gone).diff() != pd.Timedelta(days=1)).cumsum()
            miss_ranges = [[str(g.min().date()), str(g.max().date()), int(len(g))] for _, g in pd.Series(gone).groupby(grp_)]
    R["coverage"] = {"households_in_15min_data": agg["span"]["households"], "households_with_valid": len(per_hh),
                     "date_ranges_without_valid_days": miss_ranges,
                     "hh_ge30": int((ph >= 30).sum()), "hh_ge60": int((ph >= 60).sum()),
                     "hh_ge100": int((ph >= 100).sum()), "hh_ge200": int((ph >= 200).sum()),
                     "valid_days_per_household_quantiles": ({str(q): float(np.quantile(ph, q, method="lower")) for q in (0, .25, .5, .75, 1)} if ph.size else {}),
                     "by_month": dict(sorted(by_month.items())), "dates_with_valid": len(dates),
                     "first_valid_date": dates[0] if dates else None, "last_valid_date": dates[-1] if dates else None,
                     "median_valid_households_per_date": float(np.median(list(per_date.values()))) if per_date else None}
    pd.Series(per_hh, name="valid_days").sort_index().rename_axis("household_ID").to_csv(args.out / "valid_days_per_household.csv")
    pd.Series(per_date, name="valid_households").sort_index().rename_axis("date").to_csv(args.out / "valid_households_per_date.csv")
    if "tr2" in agg:
        R["tr2_crosscheck"] = assemble_tr2(agg["tr2"], tr2_absdiff, tr2_p15, tr2_t2, tr2_resid)
        if hh_cmp:
            frac = {k: hh_bad.get(k, 0) / v for k, v in hh_cmp.items()}
            R["tr2_crosscheck"]["disagreement_by_household"] = {
                "households_compared": len(hh_cmp), "households_with_any_day_gt_0.05kWh": len(hh_bad),
                "households_with_majority_days_gt_0.05kWh": int(sum(1 for k in hh_bad if frac[k] > 0.5)),
                "days_gt_0.05kWh": int(sum(hh_bad.values())), "days_gt_0.5kWh": len(tr2_big),
                "worst_days": [[h_, d_, a_, b_] for _, h_, d_, a_, b_ in tr2_big[:10]],
                "worst_days_columns": ["household_ID", "date", "peak_15min_kwh", "tr2_peak_kwh"],
                "top_households_by_days_gt_0.05kWh": sorted(([k, v, hh_cmp[k]] for k, v in hh_bad.items()),
                                                           key=lambda x: -x[1])[:10]}
    R["pipeline"] = {"buckets": K, "max_six_hour_bucket_rows": int(max_bucket6), "chunksize": args.chunksize, "max_bucket_rows": int(max_bucket),
                     "max_chunk_rows": max(s["max_chunk_rows"] for s in file_stats.values()),
                     "rows_total": sum(s["rows"] for s in file_stats.values()),
                     "rows_kept_for_analysis": sum(s["rows_kept_for_analysis"] for s in file_stats.values()),
                     "household_day_rows_written": n_out, "valid_rows_written": n_valid,
                     "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                          (1 << 20 if sys.platform == "darwin" else 1 << 10), 1)}
    if not args.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(args.out / "audit_report.json", "w") as fh:
        json.dump(jsonable(R), fh, indent=2, default=str)
    return R


def assemble_tr2(T, absdiff, p15, t2, resid):
    out = {k: v for k, v in T.items() if k not in ("shift", "identity", "abs_diff_bins", "sum_signed_diff")}
    n = out.get("valid_days_plateau_consistent_bracket", 0)
    ad = np.concatenate(absdiff) if absdiff else np.empty(0)
    a, b = (np.concatenate(p15), np.concatenate(t2)) if p15 else (np.empty(0), np.empty(0))
    out["compared_days"] = int(n)
    if n:
        out["abs_diff_bins"] = T["abs_diff_bins"]
        out["share_within_0.01kWh"] = T["abs_diff_bins"]["<=0.005"] / n + T["abs_diff_bins"]["<=0.01"] / n
        out["abs_diff_quantiles_kwh"] = {str(q): float(np.quantile(ad, q, method="lower")) for q in (0.5, 0.9, 0.95, 0.99, 1.0)}
        out["mean_signed_diff_15min_minus_tr2_kwh"] = T["sum_signed_diff"] / n
        if n > 2 and a.std() > 0 and b.std() > 0:
            out["pearson_r"] = float(np.corrcoef(a, b)[0, 1])
            out["ols_slope_15min_on_tr2"] = float(np.polyfit(b, a, 1)[0])
            out["spearman_r"] = float(np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[0, 1])
            kept = ad <= 0.5
            if kept.sum() > 2:
                out["pearson_r_excluding_abs_diff_gt_0.5"] = float(np.corrcoef(a[kept], b[kept])[0, 1])
                out["ols_slope_excluding_abs_diff_gt_0.5"] = float(np.polyfit(b[kept], a[kept], 1)[0])
        out["window_shift_sensitivity"] = {d: {"n": v["n"], "mean_abs_diff_kwh": v["sum_abs"] / v["n"] if v["n"] else None,
                                               "share_within_0.01": v["within_0.01"] / v["n"] if v["n"] else None,
                                               "share_within_0.05": v["within_0.05"] / v["n"] if v["n"] else None}
                                           for d, v in T["shift"].items()}
        out["window_shift_note"] = "shift s moves both endpoints by s*15 min; s=0 is the audited 18:30-22:30 window"
    idn = dict(T["identity"])
    r = np.concatenate(resid) if resid else np.empty(0)
    if r.size:
        idn["median_signed_resid_kwh"] = float(np.median(r)); idn["median_abs_resid_kwh"] = float(np.median(np.abs(r)))
        idn["share_bracketed"] = idn["bracketed_by_15min_registers"] / idn["with_adjacent_15min_registers"]
    out["register_identity_total_import_vs_importkwh"] = idn
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path)
    src.add_argument("--csv", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("audit_out"))
    ap.add_argument("--verify-md5", action="store_true", help="abort with exit code 2 if the zip md5 differs")
    ap.add_argument("--expected-md5", default=ZIP_MD5)
    ap.add_argument("--households", default=None, help="comma-separated IDs (debug subset)")
    ap.add_argument("--cap-kwh", type=float, default=10.0,
                    help="15-min increase above this is flagged physically implausible (default 10 kWh = 40 kW)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--buckets", type=int, default=64, help="on-disk household buckets (more = lower peak memory)")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--no-tr2-check", dest="tr2_check", action="store_false",
                    help="skip the six-hour TR2_TOTAL_IMPORT cross-check")
    ap.add_argument("--csv6", nargs="*", default=None, help="six-hour CSVs (with --csv input)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_md5 and not args.zip:
        sys.exit("--verify-md5 only applies to --zip input")
    R = run_audit(args)
    if "peak" in R:
        print(json.dumps(jsonable({k: R[k] for k in ("household_days", "peak", "coverage", "pipeline", "tr2_crosscheck") if k in R}), indent=2, default=str))
    return R


if __name__ == "__main__":
    main()
