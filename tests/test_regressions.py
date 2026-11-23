"""Regression tests for peakwise_audit.py. Run from the project root:

    python -m unittest discover -s tests -v

All data here is synthetic with known answers (no real meter data).
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import peakwise_audit as pa          # noqa: E402
import synth                         # noqa: E402
from synth import day_rows, row, build_zip  # noqa: E402


def audit(zip_path, out, *extra):
    """Run the audit CLI in-process; returns (report dict, household-day DataFrame or None)."""
    with contextlib.redirect_stdout(io.StringIO()):
        pa.main(["--zip", str(zip_path), "--out", str(out), *map(str, extra)])
    rep = json.loads((Path(out) / "audit_report.json").read_text())
    f = Path(out) / "household_days_peak.csv"
    return rep, (pd.read_csv(f) if f.exists() else None)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


class OriginalSyntheticSuite(Tmp):
    """The pre-existing edge-case fixture and its known answers must still hold exactly."""

    def test_known_answers(self):
        z, e = self.tmp / "s.zip", self.tmp / "e.json"
        subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic_fixture.py"), str(z), str(e)],
                       check=True, capture_output=True)
        exp = json.loads(e.read_text())
        rep, hd = audit(z, self.tmp / "o")
        self.assertEqual(rep["peak"]["valid"], exp["valid"])
        st = dict(zip(hd["household_ID"], hd["status"]))
        for h, s in exp["status"].items():
            self.assertEqual(st[h], s, h)
        for h, p in exp["peaks"].items():
            self.assertAlmostEqual(float(hd.loc[hd.household_ID == h, "peak_kwh_1830_2230"].iloc[0]), p, places=6)
        self.assertEqual(rep["files"]["smart_15min_1.csv"]["offgrid_time"], exp["offgrid"])
        self.assertEqual(rep["duplicates"]["exact_duplicate_rows"], exp["exact_dup"])
        self.assertEqual(rep["duplicates"]["conflicting_duplicate_rows"], exp["conflict_dup"])


class AlignmentAndValues(Tmp):
    def test_peak_is_register_difference_1830_to_2230_with_distinct_increments(self):
        # increments differ in every interval, so any off-by-one slot shift changes the answer
        inc = lambda s: 0.001 * s
        z = build_zip(self.tmp / "a.zip", f1=day_rows("IDX", "2024-05-05", 1000.0, inc))
        rep, hd = audit(z, self.tmp / "o")
        want = sum(inc(s) for s in range(75, 91))          # intervals ending 18:45 ... 22:30
        self.assertAlmostEqual(float(hd["peak_kwh_1830_2230"].iloc[0]), want, places=6)
        shifted_early = sum(inc(s) for s in range(74, 90))  # (18:15, 22:15] must NOT be used
        self.assertNotAlmostEqual(want, shifted_early, places=6)
        self.assertEqual(rep["peak"]["valid"], 1)


class MemoryBoundedPipeline(Tmp):
    def _many_households(self):
        rows = []
        for i in range(60):
            for d in ("2024-06-01", "2024-06-02"):
                rows += day_rows(f"ID{i:04d}", d, 100.0 * (i + 1), lambda s: 0.01 + 0.0001 * s)
        return rows

    def test_results_identical_for_any_chunksize_and_bucket_count(self):
        z = build_zip(self.tmp / "m.zip", f1=self._many_households())
        base, hd0 = audit(z, self.tmp / "o0", "--chunksize", 10_000_000, "--buckets", 1)
        for i, (cs, k) in enumerate([(37, 5), (7, 64), (1000, 16)], 1):
            rep, hd = audit(z, self.tmp / f"o{i}", "--chunksize", cs, "--buckets", k)
            for key in ("files", "duplicates", "gaps", "span", "intervals", "household_days", "peak", "coverage",
                        "interval_label_test", "readings_per_day_hist", "unusable_readings"):
                a, b = json.loads(json.dumps(base[key])), json.loads(json.dumps(rep[key]))
                if key == "files":      # chunk bookkeeping legitimately differs
                    for f in (a, b):
                        for st in f.values():
                            st.pop("max_chunk_rows", None)
                self.assertEqual(a, b, f"{key} differs for chunksize={cs}, buckets={k}")
            pd.testing.assert_frame_equal(hd0, hd)
            self.assertEqual((self.tmp / "o0" / "peakwise_target_clean.csv").read_bytes(),
                             (self.tmp / f"o{i}" / "peakwise_target_clean.csv").read_bytes())
        self.assertEqual(base["peak"]["valid"], 120)

    def test_memory_is_bounded_by_chunk_and_bucket_not_by_dataset(self):
        z = build_zip(self.tmp / "m.zip", f1=self._many_households())
        rep, _ = audit(z, self.tmp / "o", "--chunksize", 500, "--buckets", 32)
        pl = rep["pipeline"]
        self.assertEqual(pl["rows_total"], 60 * 2 * 96)
        self.assertLessEqual(pl["max_chunk_rows"], 500)
        self.assertLess(pl["max_bucket_rows"], pl["rows_kept_for_analysis"] / 4)

    def test_no_full_dataset_concatenation(self):
        """pd.concat must never see a frame close to the dataset size (the v1 bug)."""
        z = build_zip(self.tmp / "m.zip", f1=self._many_households())
        sizes, real = [], pd.concat

        def spy(objs, *a, **k):
            out = real(objs, *a, **k)
            sizes.append(len(out))
            return out
        pd.concat = spy
        try:
            audit(z, self.tmp / "o", "--chunksize", 500, "--buckets", 32)
        finally:
            pd.concat = real
        self.assertTrue(all(n < 60 * 2 * 96 / 4 for n in sizes), sizes)

    def test_household_day_split_across_files_and_unsorted_rows(self):
        inc = lambda s: 0.002 * s
        full = day_rows("IDS", "2024-07-01", 500.0, inc)                 # 96 rows, slots in order
        f1, f2 = full[:81], full[81:]                                    # file boundary inside 18:30-22:30
        zrev = day_rows("IDR", "2024-07-01", 800.0, inc)[::-1]           # rows in reverse time order
        late = day_rows("IDL", "2024-07-01", 900.0, inc)                 # date appears in file 2 ...
        early = day_rows("IDL", "2024-06-30", 880.0, inc)                # ... earlier date after it
        z = build_zip(self.tmp / "b.zip", f1=f1 + zrev, f2=f2 + late, f3=early)
        rep, hd = audit(z, self.tmp / "o", "--buckets", 8, "--chunksize", 11)
        want = sum(inc(s) for s in range(75, 91))
        got = hd.set_index(["household_ID", "date"])
        for key in (("IDS", "2024-07-01"), ("IDR", "2024-07-01"), ("IDL", "2024-07-01"), ("IDL", "2024-06-30")):
            self.assertEqual(got.loc[key, "status"], "valid", key)
            self.assertAlmostEqual(got.loc[key, "peak_kwh_1830_2230"], want, places=6, msg=str(key))
        self.assertEqual(rep["days_spanning_files"], 1)
        self.assertEqual(list(hd["household_ID"]), sorted(hd["household_ID"]))   # globally sorted output

    def test_duplicates_across_file_boundary(self):
        inc = lambda s: 0.01
        d = day_rows("IDD", "2024-08-01", 10.0, inc)
        dup_exact = d[80]                                               # slot 80 (20:00) repeated in file 2
        hh, mm = divmod(84 * 15, 60)                                    # slot 84 repeated with another value
        dup_conf = row("IDD", "2024-08-01", hh, mm, 99.0)
        z = build_zip(self.tmp / "d.zip", f1=d, f2=[dup_exact, dup_conf])
        rep, hd = audit(z, self.tmp / "o")
        self.assertEqual(rep["duplicates"]["duplicate_rows"], 2)
        self.assertEqual(rep["duplicates"]["exact_duplicate_rows"], 1)
        self.assertEqual(rep["duplicates"]["conflicting_duplicate_rows"], 1)
        self.assertEqual(rep["duplicates"]["cross_file_duplicate_rows"], 2)
        self.assertEqual(hd["status"].iloc[0], "conflicting_duplicate")

    def test_empty_input_does_not_crash(self):
        z = build_zip(self.tmp / "e.zip")
        with contextlib.redirect_stdout(io.StringIO()):
            pa.main(["--zip", str(z), "--out", str(self.tmp / "o")])
        rep = json.loads((self.tmp / "o" / "audit_report.json").read_text())
        self.assertEqual(rep["note"], "no usable rows")


class Md5Verification(Tmp):
    def _zip(self):
        return build_zip(self.tmp / "z.zip", f1=day_rows("IDM", "2024-09-01", 1.0, lambda s: 0.01))

    def test_mismatch_terminates_before_any_audit_output(self):
        z, out = self._zip(), self.tmp / "o"
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()) as err:
            pa.main(["--zip", str(z), "--out", str(out), "--verify-md5"])      # default md5 = the real release
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("MD5 MISMATCH", err.getvalue())
        self.assertFalse((out / "audit_report.json").exists())
        self.assertFalse((out / "household_days_peak.csv").exists())

    def test_match_proceeds_and_is_recorded(self):
        z = self._zip()
        rep, _ = audit(z, self.tmp / "o", "--verify-md5", "--expected-md5", synth.md5(z))
        self.assertTrue(rep["inputs"]["zip_md5_ok"])
        self.assertEqual(rep["inputs"]["zip_md5"], synth.md5(z))

    def test_verify_md5_with_csv_input_is_rejected(self):
        csv = self.tmp / "a.csv"
        csv.write_text(synth.HEADER + "\n")
        with self.assertRaises(SystemExit) as cm:
            pa.main(["--csv", str(csv), "--out", str(self.tmp / "o"), "--verify-md5"])
        self.assertNotEqual(cm.exception.code, 0)


class TimestampValidation(Tmp):
    def test_classify_times_rejects_invalid_hours_minutes_seconds(self):
        dates = pd.Series(["2024-01-01"] * 11 + ["2024-02-30", "2024-1-5", "2024-13-01", "not-a-date"])
        times = pd.Series(["18:30:00", "24:00:00", "25:00:00", "18:60:00", "18:75:00", "18:30:60", "18:30:99",
                           "18:3:00", "18:30", "18:30:00", "18:37:00", "18:30:00", "18:30:00", "18:30:00",
                           "18:30:00"])
        day, H, M, S, reason, offgrid = pa.classify_times(dates, times)
        self.assertEqual(reason.tolist(), [0, 3, 3, 3, 3, 3, 3, 2, 2, 0, 0, 1, 1, 1, 1])
        self.assertEqual(offgrid.tolist(), [False] * 10 + [True] + [False] * 4)   # 18:37 is valid but off-grid

    def test_invalid_time_is_counted_and_cannot_pose_as_a_valid_slot(self):
        # v1 mapped 18:60:00 to slot 76 (19:00) because 60 % 15 == 0 and so filled a genuinely missing reading
        d = "2024-10-01"
        rows = day_rows("IDT", d, 50.0, lambda s: 0.02, skip={76})        # 19:00 reading missing
        rows.append(row("IDT", d, 18, 45, 123.0, time_str="18:60:00"))   # impostor for 19:00
        rows += [row("IDT", d, 12, 0, 1.0, time_str="25:00:00"), row("IDT", d, 12, 0, 1.0, time_str="12:30:61"),
                 row("IDT", d, 12, 0, 1.0, time_str="12:3:00"), row("IDT", d, 12, 0, 1.0, date_str="2024-02-30"),
                 row("IDT", d, 12, 0, 1.0, date_str="2024-1-5"), row("IDT", d, 12, 7, 1.0)]
        z = build_zip(self.tmp / "t.zip", f1=rows)
        rep, hd = audit(z, self.tmp / "o")
        f = rep["files"]["smart_15min_1.csv"]
        self.assertEqual(f["invalid_time_range"], 3)        # 18:60:00, 25:00:00, 12:30:61
        self.assertEqual(f["bad_time_format"], 1)           # 12:3:00
        self.assertEqual(f["bad_date_format"], 2)           # 2024-02-30, 2024-1-5
        self.assertEqual(f["bad_datetime"], 6)
        self.assertEqual(f["offgrid_time"], 1)              # 12:07:00
        self.assertEqual(hd["status"].iloc[0], "missing_reading")
        self.assertEqual(rep["peak"]["valid"], 0)


class ObisAndCoverage(Tmp):
    def _dbe_row(self, h, d):
        return row(h, d, 0, 0, 500.0).replace(",LP,", ",DBE,")

    def test_dbe_midnight_rows_are_counted_and_do_not_touch_the_peak(self):
        inc = lambda s: 0.002 * s
        rows = day_rows("IDA", "2024-04-01", 100.0, inc, skip={0})        # LP day, midnight reading absent
        rows.append(self._dbe_row("IDA", "2024-04-01"))                    # replaced by a DBE snapshot
        rows.append(self._dbe_row("IDB", "2024-04-01"))                    # a household-day that has ONLY a DBE row
        z = build_zip(self.tmp / "o.zip", f1=rows)
        rep, hd = audit(z, self.tmp / "o")
        self.assertEqual(rep["files"]["smart_15min_1.csv"]["obis"], {"LP": 95, "DBE": 2})
        self.assertEqual(rep["household_days_without_any_lp_row"], 1)
        self.assertEqual(rep["non_lp_rows_in_register_series"], 2)
        a = hd[hd.household_ID == "IDA"].iloc[0]
        self.assertEqual(a["status"], "valid")
        self.assertAlmostEqual(a["peak_kwh_1830_2230"], sum(inc(s) for s in range(75, 91)), places=6)
        b = hd[hd.household_ID == "IDB"].iloc[0]
        self.assertEqual((b["n_readings_day"], b["n_lp_readings_day"], b["status"]), (1, 0, "missing_reading"))

    def test_household_counts_exclude_six_hour_only_households_and_date_gaps_are_listed(self):
        inc = lambda s: 0.01
        rows = day_rows("IDA", "2024-04-01", 100.0, inc) + day_rows("IDA", "2024-04-05", 110.0, inc)
        h6 = [synth.row6("IDZ", "2024-04-01", "10:00:00", 5.0, 1.0)]      # household that never appears in 15-min data
        z = build_zip(self.tmp / "g.zip", f1=rows, h6=h6)
        rep, _ = audit(z, self.tmp / "o")
        self.assertEqual(rep["coverage"]["households_in_15min_data"], 1)
        self.assertEqual(rep["coverage"]["date_ranges_without_valid_days"], [["2024-04-02", "2024-04-04", 3]])
        self.assertEqual(rep["coverage"]["dates_with_valid"], 2)

    def test_clean_target_carries_tr2_check_flag_and_drops_nothing(self):
        inc = lambda s: 0.01
        rows = day_rows("IDA", "2024-04-01", 100.0, inc)
        z = build_zip(self.tmp / "f.zip", f1=rows)                         # no six-hour data at all
        audit(z, self.tmp / "o")
        t = pd.read_csv(self.tmp / "o" / "peakwise_target_clean.csv")
        self.assertEqual(list(t.columns), ["household_ID", "date", "peak_kwh_1830_2230", "tr2_check"])
        self.assertEqual(t["tr2_check"].tolist(), ["no_bracket"])
        audit(z, self.tmp / "o2", "--no-tr2-check")
        self.assertEqual(pd.read_csv(self.tmp / "o2" / "peakwise_target_clean.csv")["tr2_check"].tolist(), ["not_checked"])


if __name__ == "__main__":
    unittest.main()
