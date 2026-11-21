# PeakWise – RECON-SL 15-minute data audit, v2 (full Zenodo run, 21 Nov 2026)

Target: household grid import in kWh over 18:30–22:30 Sri Lanka time, for the next-day peak forecast. No model has been trained.

## 1. What changed from v1 (the three review findings)

| Finding | Fix | Regression test |
|---|---|---|
| Whole dataset concatenated and sorted in memory | Two-pass pipeline. Pass 1 streams each CSV in chunks, validates and counts every row, and appends compact records to 64 on-disk buckets chosen by `crc32(household_ID) % 64`. Pass 2 loads one bucket at a time (all files for those households), sorts by (household, slot, file, row), applies the rules and folds the counters into the report. Per-bucket outputs are k-way merged into one sorted file. File boundaries cannot matter because a household's rows from every file land in the same bucket. | `MemoryBoundedPipeline` (7 tests): identical report and CSVs for chunk sizes 7…10M and 1…64 buckets; no `pd.concat` larger than ¼ of the data; household-day split across files; unsorted rows; cross-file duplicates; empty input |
| `--verify-md5` recorded a mismatch and carried on | Mismatch prints `MD5 MISMATCH` and exits with code 2 before any audit output is written. `--verify-md5` with `--csv` input is rejected. `--expected-md5` lets another release be checked. | `Md5Verification` (3 tests) |
| Timestamp validation accepted impossible times | Date must match `YYYY-MM-DD` and be a real date; time must match `HH:MM:SS` with hour ≤ 23, minute ≤ 59, second ≤ 59. v1 mapped `18:60:00` to slot 76 (19:00) because 60 % 15 = 0, so an invalid row could fill a missing reading. Rejections are counted separately: `bad_date_format`, `bad_time_format`, `invalid_time_range`. | `TimestampValidation` (2 tests) |
| TR2 never checked | Added the six-hour cross-check (section 4). | `test_tr2.py` (8 tests) |

All earlier rules are unchanged: 17 readings 18:30…22:30, nothing imputed, cap 10 kWh per 15 min, first failing rule recorded, conflicting duplicates rejected. The original synthetic fixture still gives the same known answers. Added a regression test with distinct increments in every interval so that any one-slot alignment error would change the answer.

Test run: `python -m unittest discover -s tests -v` → **24 tests, all OK** (`test_results.txt`).

## 2. Data acquired and verified

* Zenodo record 15023048 `data.zip`, 1,184,468,558 bytes, **md5 dcdac5809b4da6b6362519d33112a5f9 verified** (matches the record), `testzip` clean, CC BY 4.0. The three 15-minute members have the sizes reported earlier (948,902,706 / 1,696,953,202 / 2,015,227,030 bytes).
* Full run: 128 s wall time, **peak memory 912 MB** (about 5.7 GB of CSV, 15-minute and six-hour), largest bucket 913,673 rows.
* Rows: 9,435,755 + 16,425,676 + 19,751,849 = **45,613,280**, matching the published count. Date coverage: file 1 2023-10-01…2024-03-01, file 2 2024-03-02…2024-08-01, file 3 2024-08-02…2024-12-23.

## 3. Audit results (15-minute data)

**Row-level rejections.** Malformed lines 0; invalid date or time fields 0 in all three files; empty or non-numeric `importkwh` 0; negative raw register values 0. **7 rows dropped as off-grid times (`00:00:01`)**, all of them `DBE` rows (see below).

**Timestamp convention.** `date`+`time` are Sri Lanka local time: 45,613,280 of 45,613,280 rows have `timestamp` within 5×10⁶ ms of local→UTC, 0 look UTC. Interval label: a row stamped *t* closes (*t*−15 min, *t*]. Mean absolute error of `avgimportkw×0.25` against the register increase is 0.0033 kWh under that reading and 0.0166 kWh if the row opened the interval (45.2M pairs); within 0.01 kWh in 97.0% vs 68.9% of pairs.

**Duplicates.** Duplicate household-timestamp rows: **0** (so 0 exact, 0 conflicting, 0 across files). No household-day spans two files (files split at day boundaries).

**`obis`.** 45,585,961 rows are `LP`; **27,319 are `DBE`**. DBE rows exist only for 2024-03-28…2024-11-01, 1,205 households, and only at 00:00 (27,312 at `00:00:00`, 7 at `00:00:01`). None shares a slot with an LP row. Where both neighbours exist (8,029 cases), the DBE import value lies between the LP readings at 23:45 and 00:15 in 7,996 (99.6%), so it is the same cumulative register (`probe_dbe.py`, `audit_out/dbe_probe.json`). They are kept in the register series. They are outside 18:30–22:30 and cannot change the target. They do create **18,522 household-days that have only a DBE row** (no LP reading); those appear in the table as `missing_reading` and are counted separately in the report.

**Missing intervals.** 2,938 households. Inside each household's first-to-last span, 98,294,338 quarter-hour slots are expected and 45,613,273 present, so **52,681,065 (53.6%) are missing**. Consecutive-reading gaps: 45,509,741 of one slot, 54,616 of 2–4 slots, 32,182 of 5–96 slots, 13,796 longer than a day. Nothing is filled.

**Anomalies (reported, not repaired).** Negative 15-minute import steps: 4,175 (export 3,478). Steps above the 10 kWh cap: 4,100 (export 3,562). The largest single cluster is at the 00:00 slot (281 negative, 283 above cap), for example ID0457 jumping from 5,287 kWh to 0.285 kWh at midnight on consecutive days (a register swap or reset). Cause unknown.

**Household-days and target.** 505,398 household-days have at least one reading.

| Status | Household-days |
|---|---|
| **valid** | **464,640** |
| missing_reading | 40,064 (29,826 of them have no reading at all inside the window) |
| negative_interval | 519 |
| implausible_interval (>10 kWh in 15 min) | 175 |
| conflicting_duplicate / nan_value | 0 / 0 |

(Non-exclusive counts of days that trip each rule: negative 531, implausible 475.)

**Coverage of the 464,640 valid days.** 2,931 of 2,938 households have at least one valid day; median 162 valid days per household (quartiles 128 / 202, max 342); ≥30 days: 2,885 households, ≥60: 2,771, ≥100: 2,398, ≥200: 768. 398 distinct dates from 2023-10-01 to 2024-12-22, median 1,322 valid households per date. **No valid days at all from 2024-01-12 to 2024-02-29 (49 days)**, plus 2023-12-09 and 2024-01-09; the source has almost no readings then (3 household-days in February 2024). Valid days by month: 2023-10 10,963; 11 51,849; 12 30,020; 2024-01 2,817; 02 0; 03 3,282; 04 32,997; 05 60,923; 06 19,260; 07 50,272; 08 20,468; 09 40,350; 10 61,979; 11 50,457; 12 29,003.

**Distribution of the target (kWh in 18:30–22:30, valid days).** mean 1.383, sd 3.495; quantiles 0 / 1% / 5% / 10% / 25% / 50% / 75% / 90% / 95% / 99% / 100% = 0 / 0 / 0 / 0.030 / 0.334 / **0.721** / 1.292 / 2.674 / 4.379 / 12.171 / 122.95. **37,219 valid days (8.0%) are exactly 0.000 kWh** and 6,426 (1.4%) are ≥10 kWh. Both tails are legitimate under the audit rules and have **not** been trimmed. They are what a model has to handle (empty homes or export-only; large or multi-unit loads).

## 4. Independent verification against the six-hour meter registers

`TR2_TOTAL_IMPORT` is the cumulative **peak-period** register (data dictionary: "Day / Peak / Off-Peak"). Six-hour readings are at irregular times (for example 00:40:14, 18:53:17), not on a fixed grid, so the check uses the plateaus: TR2 is flat from 22:30 of the previous day to 18:30, and from 22:30 to 18:30 of the next day. For each valid household-day I take the six-hour reading nearest 18:30 in the first plateau and nearest 22:30 in the second, require TR2 to be flat within 0.002 kWh across every reading of both plateaus, and compare the TR2 rise with the 15-minute value.

* Six-hour data: 4,441,782 readings, 0 duplicates; 11 rows with an invalid time field; 19,329 decreasing TR2 steps (register flips).
* Valid days with a usable bracket on both sides: 337,899 of 464,640; 335,343 have flat plateaus and are compared (2,556 are excluded as not flat). The other 126,741 valid days have no six-hour reading in one of the plateaus and are **unverified**.
* **99.58% of compared days agree within 0.01 kWh** (median absolute difference 0.0001 kWh; 99th percentile 0.0019 kWh). Excluding the 616 days that differ by more than 0.5 kWh, Pearson r = 0.999996 and slope 1.000000. Over all days the plain Pearson r is only 0.546 because of a few register flips of ±1,767 kWh.
* **The window alignment is confirmed.** Moving the window by ±1 slot (15 minutes) in either direction drops agreement within 0.01 kWh from 99.6% to 38–39% and raises the mean absolute error from 0.023 to 0.078 kWh; ±4 slots gives 19–21%. Zero shift is the clear minimum (`window_shift_sensitivity`).
* **Same register:** of 735,236 six-hour `TOTAL_IMPORT` readings with adjacent 15-minute registers, 98.7% lie between the 15-minute registers on either side of them (median absolute residual 0.0005 kWh). So `importkwh` is the same cumulative register as `TOTAL_IMPORT`, and the six-hour clock is on the same local time.
* **Disagreeing days.** 974 compared days (0.29%) differ by more than 0.05 kWh, in 49 of 2,909 households; 8 households disagree on most of their days (for example ID2420, 60 of 119 days), which may be a different tariff-period configuration, cause unknown. They are **flagged, not removed**: `peakwise_target_clean.csv` has a `tr2_check` column (`agree` 334,280; `disagree` 1,063; `plateau_inconsistent` 2,556; `no_bracket` 126,741).
* Independent spot check (`spot_check.py`, shares no code with the audit): 55 random household-days recomputed from the raw CSVs, 40 valid and 15 rejected, **0 mismatches**.

## 5. What could not be verified

1. **126,741 valid days (27%) have no TR2 bracket**, so the meter cross-check says nothing about them.
2. **Cause of the register flips** (midnight resets, ±1,700–11,700 kWh swaps, ~19,000 decreasing steps in the six-hour data) is unknown. Days containing a flip inside the window are rejected by rule; flips outside it are not looked at.
3. **Why 8 households disagree with their own TR2 register** on most days (tariff configuration, meter setup) is unknown.
4. **The 10 kWh per 15-minute cap** is a judgement call and was not changed. 175 days fail it; the maximum valid target is 122.95 kWh.
5. **Kaggle copy** was not compared with Zenodo. The GitHub data dictionary was not re-read in this run; cumulative semantics rest on the register-identity test above and the zip's own dictionary for TOTAL_IMPORT.
6. **The ToU boundary 18:30–22:30** is confirmed empirically by the shift test for the 335,343 compared days, not from a written specification.
7. **`DBE` meaning** is inferred (midnight register snapshot); the zip's dictionary does not define `obis`.
8. `browser_full_scan.js` and the old `real_sample/` output belong to v1 and are superseded.

## 6. Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy
.venv/bin/python -I peakwise_audit.py --zip data.zip --verify-md5 --out audit_out   # ~2 min, ~1 GB RAM
.venv/bin/python -I probe_dbe.py --zip data.zip --out audit_out
.venv/bin/python -I spot_check.py --zip data.zip --audit audit_out --n 40 --seed 7
.venv/bin/python -m unittest discover -s tests -v
```

Outputs in `audit_out/`: `audit_report.json`, `household_days_peak.csv` (every household-day with status), `peakwise_target_clean.csv` (valid days only: household_ID, date, peak_kwh_1830_2230, tr2_check), `tr2_crosscheck.csv`, `valid_days_per_household.csv`, `valid_households_per_date.csv`, `dbe_probe.json`.
