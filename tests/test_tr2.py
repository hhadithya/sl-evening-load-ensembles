"""Tests for the six-hour TR2_TOTAL_IMPORT cross-check (synthetic data with known answers).

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import peakwise_audit as pa            # noqa: E402
from synth import day_rows, row6, build_zip  # noqa: E402

D, D1 = "2024-05-05", "2024-05-06"
START = 1000.0
inc = lambda s: 0.001 * s                      # distinct increment in every 15-min interval
R = lambda s: START + sum(inc(k) for k in range(1, s + 1))   # register at slot s of day D


def reg_at_seconds(sec):
    """15-min register linearly interpolated at `sec` seconds after midnight of D (what a 6-h TOTAL_IMPORT should show)."""
    s, frac = divmod(sec / 900.0, 1)
    s = int(s)
    return R(s) + frac * (R(s + 1) - R(s))


def hms(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Tr2Tests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(); self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def run_audit(self, h6, name="x"):
        z = build_zip(self.tmp / f"{name}.zip", f1=day_rows("IDX", D, START, inc), h6=h6)
        out = self.tmp / name
        with contextlib.redirect_stdout(io.StringIO()):
            pa.main(["--zip", str(z), "--out", str(out)])
        rep = json.loads((out / "audit_report.json").read_text())
        return rep, out

    def readings(self, tr2_before, tr2_after, ti_offset=0.0, extra=()):
        sec_b, sec_a = 12 * 3600 + 190, 22 * 3600 + 3000      # 12:03:10 and 22:50:00
        rows = [row6("IDX", D, hms(sec_b), round(reg_at_seconds(sec_b) + ti_offset, 4), tr2_before),
                row6("IDX", D, "18:10:00", round(reg_at_seconds(18 * 3600 + 600) + ti_offset, 4), tr2_before),
                row6("IDX", D, hms(sec_a), round(reg_at_seconds(sec_a) + ti_offset, 4), tr2_after)]
        return rows + list(extra)

    def test_matching_tr2_confirms_extraction_and_alignment(self):
        peak = sum(inc(s) for s in range(75, 91))
        rep, out = self.run_audit(self.readings(200.0, round(200.0 + peak, 6)))
        t = rep["tr2_crosscheck"]
        self.assertEqual(t["valid_days_considered"], 1)
        self.assertEqual(t["compared_days"], 1)
        self.assertEqual(t["abs_diff_bins"]["<=0.005"], 1)
        self.assertAlmostEqual(t["mean_signed_diff_15min_minus_tr2_kwh"], 0.0, places=5)
        sh = t["window_shift_sensitivity"]
        best = min(sh, key=lambda k: sh[k]["mean_abs_diff_kwh"])
        self.assertEqual(best, "+0")
        self.assertGreater(sh["-1"]["mean_abs_diff_kwh"], 0.01)       # a shifted window is distinguishable
        df = pd.read_csv(out / "tr2_crosscheck.csv")
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["diff_kwh"].iloc[0], 0.0, places=5)

    def test_tr2_booked_one_slot_earlier_is_detected(self):
        early = R(89) - R(73)                                          # (18:15, 22:15]
        rep, _ = self.run_audit(self.readings(200.0, round(200.0 + early, 6)))
        t = rep["tr2_crosscheck"]
        sh = t["window_shift_sensitivity"]
        self.assertEqual(min(sh, key=lambda k: sh[k]["mean_abs_diff_kwh"]), "-1")
        self.assertGreater(abs(t["mean_signed_diff_15min_minus_tr2_kwh"]), 0.01)

    def test_inconsistent_plateau_is_not_used(self):
        peak = sum(inc(s) for s in range(75, 91))
        # TR2 dips at 14:00 (200 -> 199.5 -> 200): not flat before the peak, so that day must be excluded even
        # though the first and last readings of the plateau agree
        rows = self.readings(200.0, round(200.0 + peak, 6), extra=[row6("IDX", D, "14:00:00", 1.0, 199.5)])
        rep, _ = self.run_audit(rows)
        t = rep["tr2_crosscheck"]
        self.assertEqual(t["valid_days_with_bracket"], 1)
        self.assertEqual(t["valid_days_bracket_but_plateau_inconsistent"], 1)
        self.assertEqual(t["compared_days"], 0)

    def test_missing_bracket_is_reported_not_guessed(self):
        rep, out = self.run_audit([row6("IDX", D, "12:03:10", 1000.0, 200.0)])      # nothing after 22:30
        t = rep["tr2_crosscheck"]
        self.assertEqual(t["valid_days_with_bracket"], 0)
        self.assertEqual(t["compared_days"], 0)
        self.assertFalse((out / "tr2_crosscheck.csv").exists())

    def test_total_import_register_identity(self):
        peak = sum(inc(s) for s in range(75, 91))
        good, _ = self.run_audit(self.readings(200.0, round(200.0 + peak, 6)), "good")
        gi = good["tr2_crosscheck"]["register_identity_total_import_vs_importkwh"]
        self.assertEqual(gi["with_adjacent_15min_registers"], 3)
        self.assertEqual(gi["bracketed_by_15min_registers"], 3)
        bad, _ = self.run_audit(self.readings(200.0, round(200.0 + peak, 6), ti_offset=5.0), "bad")
        bi = bad["tr2_crosscheck"]["register_identity_total_import_vs_importkwh"]
        self.assertEqual(bi["bracketed_by_15min_registers"], 0)
        self.assertAlmostEqual(bi["median_signed_resid_kwh"], 5.0, places=2)

    def test_six_hour_row_validation_and_duplicates_are_counted(self):
        peak = sum(inc(s) for s in range(75, 91))
        rows = self.readings(200.0, round(200.0 + peak, 6), extra=[
            row6("IDX", D, "25:10:00", 1.0, 1.0), row6("IDX", D, "10:75:00", 1.0, 1.0),
            row6("IDX", D, "18:10:00", 1.0, 999.0)])                       # conflicting duplicate of the 18:10:00 reading
        rep, _ = self.run_audit(rows)
        f = rep["files_six_hour"]["smart_6hour_1.csv"]
        self.assertEqual(f["invalid_time_range"], 2)
        self.assertEqual(rep["tr2_crosscheck"]["h6_conflicting_duplicates"], 1)

    def test_target_flag_agree_vs_disagree(self):
        peak = sum(inc(s) for s in range(75, 91))
        _, out = self.run_audit(self.readings(200.0, round(200.0 + peak, 6)), "agree")
        self.assertEqual(pd.read_csv(out / "peakwise_target_clean.csv")["tr2_check"].tolist(), ["agree"])
        rep, out = self.run_audit(self.readings(200.0, round(200.0 + peak + 1.0, 6)), "dis")
        self.assertEqual(pd.read_csv(out / "peakwise_target_clean.csv")["tr2_check"].tolist(), ["disagree"])
        self.assertEqual(len(pd.read_csv(out / "peakwise_target_clean.csv")), 1)            # flagged, never dropped
        d = rep["tr2_crosscheck"]["disagreement_by_household"]
        self.assertEqual((d["days_gt_0.05kWh"], d["days_gt_0.5kWh"]), (1, 1))
        self.assertEqual(d["worst_days"][0][:2], ["IDX", D])

    def test_tr2_check_can_be_skipped(self):
        z = build_zip(self.tmp / "s.zip", f1=day_rows("IDX", D, START, inc), h6=self.readings(1.0, 2.0))
        with contextlib.redirect_stdout(io.StringIO()):
            pa.main(["--zip", str(z), "--out", str(self.tmp / "o"), "--no-tr2-check"])
        rep = json.loads((self.tmp / "o" / "audit_report.json").read_text())
        self.assertNotIn("tr2_crosscheck", rep)


if __name__ == "__main__":
    unittest.main()
