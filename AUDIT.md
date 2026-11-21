# PeakWise – RECON-SL 15-minute data audit (status: 21 Nov 2026, 10:40 SLT)

## 1. What was acquired

| Item | Value | How verified |
|---|---|---|
| GitHub repo `LIRNEasia/SL_electricity_consumption` (commit 98cc584, 16 Oct 2025) | Docs and notebooks only, **no data files** | `git clone`, file listing |
| Zenodo record 15023048 (DOI 10.21227/n1dk-q860), published 2025-02-28 | One file: `data.zip`, **1,184,468,558 bytes**, md5 `dcdac5809b4da6b6362519d33112a5f9`, licence **CC BY 4.0** | Zenodo API |
| `smart_15min_1.csv` | 190,191,137 B compressed / **948,902,706 B** uncompressed | zip central directory (HTTP Range) |
| `smart_15min_2.csv` | 340,053,399 B / **1,696,953,202 B** | same |
| `smart_15min_3.csv` | 390,952,928 B / **2,015,227,030 B** | same |
| Kaggle mirror | Not checked (needs account and API token) | — |

The full files could not be copied into the analysis workspace: the workspace's network can't reach Zenodo, and folder access on the laptop was declined. The audit therefore ran in two places:

* A full-data streaming scan (`browser_full_scan.js`) inside the Claude desktop app's browser, reading `data.zip` straight from Zenodo. **Its results could not be collected:** the laptop disconnected while the scan was in file 2 of 3.
* `peakwise_audit.py`, tested on (a) 39 real rows decompressed from the head of each Zenodo file and (b) a synthetic edge-case fixture with known answers. Python and the JS scan produce identical counts on the fixture (`tests/`).

## 2. Schema (real header, all three files)

`household_ID, timestamp, date, time, obis, importkwh(kwh), exportkwh(kwh), importkvarh(kvarh), exportkvarh(kvarh), powerfactor, avgcurrent(v), avgexportkw(kw), avgimportkw(kw), avgvoltage(v), phaseainstcurrent(a), phaseainstvoltage(v), phasebinstcurrent(a), phasebinstvoltage(v), phasecinstcurrent(a), phasecinstvoltage(v)`

20 columns. The package readme shows only 5, and the data dictionary inside the zip gives no description for `importkwh(kwh)`. `obis` = `LP` (load profile) in every sampled row.

## 3. Semantics – evidence, not assumption

| Question | Finding | Evidence |
|---|---|---|
| Is `importkwh(kwh)` cumulative or per-interval? | **Cumulative lifetime register (kWh).** | (1) GitHub data dictionary: "Total Cumulative consumed energy units". (2) Values are large and slowly rising (e.g. 21,553.312; 88,506.68). (3) ID1114 on 2024-03-02 holds 6,857.743 flat for seven readings while exporting, then steps to 6,857.750. (4) The 6-hour file's `TOTAL_IMPORT` equals `TR1+TR2+TR3` exactly (2198.669 = 1247.141 + 484.38 + 467.148), so it is a ToU-split cumulative register. (5) LIRNEasia's own EDA differences the column. |
| Units | **kWh** registers; `avg*kw` columns are kW. The readme's "kilowatts (kW)" is a documentation error. | `avgexportkw × 0.25 h` matches the register increase to ±0.0005 kWh |
| Timestamp convention | `date` + `time` are **Sri Lanka local time (UTC+05:30)**. `timestamp` is epoch ms **rounded to 10⁷ ms (~2.8 h)**, so it is useless for ordering. | All 39 real rows lie within 4.9×10⁶ ms of local→UTC; ≥1.54×10⁷ ms off if read as UTC |
| Interval label | A row stamped **t closes the interval (t−15 min, t]**. | Real rows, 11 export pairs: MAE 0.0008 kWh if interval-ending vs 0.158 kWh if interval-starting; 100% vs 0% within 0.01 kWh |
| Row order | Files are ordered by date, **not by time within a day** (ID4147: 09:00 before 08:30). | Head scan of all three files: 0 rows with a date below the running maximum (~520k rows each) |

**Target definition that follows:** peak(D) = R(D 22:30) − R(D 18:30), using the 17 readings 18:30, 18:45, …, 22:30 = **16 complete 15-minute intervals** ending 18:45…22:30. A household-day is valid only if all 17 readings exist, none is a conflicting duplicate or non-numeric, and no interval is negative or above the cap (default 10 kWh per 15 min = 40 kW average). Nothing is imputed.

## 4. Data-quality issues known so far

| Issue | Count / status | Source |
|---|---|---|
| Rows | 9,435,755 + 16,425,676 + 19,751,849 = 45,613,280 | readme; LIRNEasia notebook output |
| Empty `importkwh` / `exportkwh` | 0 / 0 (dictionary); the LIRNEasia notebook later shows NaNs after differencing | dictionary |
| Negative 15-min increments (import / export) | 4,249 / 3,529 across 121 households, after LIRNEasia's naive diff (no gap check) | LIRNEasia notebook output; **re-count pending** |
| Register flips of ±5,000–30,000 kWh (ID0457 at 00:00/00:15, ID5569) | Present; look like import/export register swaps | LIRNEasia notebook output |
| Midnight (00:00) increments abnormal | LIRNEasia overwrite the 00:00 mean as "noise" | notebook; **quantify by slot (scan computes it)** |
| Duplicate household-timestamps, gaps, off-grid times, valid peak days | **Pending full scan** | — |
| Trailing-digit artefacts (235.1001, 14144.1391) | Lossy float conversion upstream; ≤~1e-4 kWh in observed cases | real rows |

## 5. Unresolved issues

1. **Full-data counts are not yet available.** Run `peakwise_audit.py` on `data.zip` (Section 6), or reconnect the laptop so the browser scan's results can be read.
2. **Meter ToU boundaries vs 18:30–22:30:** a cross-check against the 6-hour `TR2_TOTAL_IMPORT` (peak-period register) was started but not collected. If TR2's daily increase equals R(22:30) − R(18:30), both the window and the register identity are confirmed.
3. **Whether 15-min `importkwh` is the same register as 6-hour `TOTAL_IMPORT`.** Not yet matched for the same household-time.
4. **The ±kWh register flips:** cause unknown (meter replacement, register swap, firmware). The script flags them; it does not repair them.
5. **The 10 kWh per 15-min cap** is a judgement call. Counts above it are reported, and the cap is a parameter.
6. **File-boundary days** (2024-03-01/02, 2024-08-01/02): readme says no overlap. The scan counts any late or overlapping rows; not yet confirmed.
7. **Kaggle copy** not compared with Zenodo.

## 6. How to run on the full data (laptop)

```bash
# 1. Download data.zip (1.18 GB) from https://zenodo.org/records/15023048
python3 -m pip install "pandas>=1.5" numpy
python3 peakwise_audit.py --zip ~/Downloads/data.zip --verify-md5 --out audit_out
# outputs: audit_out/audit_report.json, household_days_peak.csv (every household-day with status),
#          valid_days_per_household.csv, valid_households_per_date.csv
python3 peakwise_audit.py --zip ~/Downloads/data.zip --households ID1114,ID4147 --out audit_debug   # quick subset
```
Expect ~2–3 GB working memory for the full 45.6M rows (int-coded IDs, float32 power columns); 16 GB RAM recommended.

## 7. Files

* `peakwise_audit.py`: the reproducible audit and target extraction.
* `browser_full_scan.js`: the same rules as a streaming scan over Zenodo's zip (used for the full-data run).
* `tests/make_synthetic_fixture.py`, `tests/run_js_scan_node.mjs`: synthetic edge cases (PASS) and the Python/JS parity check (identical).
* `real_sample/zenodo_head_rows.csv`: 39 real rows from Zenodo used for the semantics checks; `real_sample/out/`: script output on them.
