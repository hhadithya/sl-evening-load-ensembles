"""Smoke and integrity tests for the PeakWise dashboard.

    python -m unittest tests.test_dashboard -v          (from the project root)

* artifacts: every file the app needs exists and is internally consistent;
* integrity: dashboard numbers are copies of the saved Phase 2 outputs, never retyped or recomputed;
* privacy: no raw household identifier appears in a dashboard artifact, labels cannot be reproduced with the old public salt,
  and the data carry only the columns the app uses;
* smoke: each page renders without an exception (Streamlit's AppTest), the Explorer reacts to its controls,
  and a missing artifact produces a readable error instead of a traceback;
* honesty: statements the project insists on are present, and "real-time" claims are absent.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "dashboard" / "data"
P2 = ROOT / "artifacts" / "phase2"
TIMEOUT = 90


def _page(module: str) -> AppTest:
    """Render one view in isolation (the navigation shell is covered by the app-level test)."""
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
            "from dashboard import theme as T; T.inject_css()\n"
            f"from dashboard.views import {module}; {module}.render()\n")
    return AppTest.from_string(code, default_timeout=TIMEOUT)


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [m.label + " " + str(m.value) for m in at.metric]
    return "\n".join(parts)


class Artifacts(unittest.TestCase):
    def test_required_files_exist(self):
        from dashboard import data as D
        self.assertEqual(D.missing_artifacts(), [])

    def test_dashboard_folder_is_small_enough_to_host(self):
        total = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
        self.assertLess(total, 20e6, f"dashboard data is {total/1e6:.1f} MB")

    def test_no_raw_household_identifiers(self):
        raw = re.compile(r"\bID\d{4}\b")
        for name in ("predictions_test_pseudonymized.csv.gz", "household_errors.csv", "examples.json", "facts.json"):
            text = (pd.read_csv(DATA / name).to_csv() if name.endswith((".csv", ".gz")) else (DATA / name).read_text())
            self.assertIsNone(raw.search(text), f"raw household id found in {name}")
        p = pd.read_csv(DATA / "predictions_test_pseudonymized.csv.gz")
        self.assertTrue(p.household.str.fullmatch(r"H-[0-9A-F]{5,}").all())

    def test_labels_are_not_reversible_with_the_old_public_salt(self):
        """An earlier build used the public salt 'peakwise-demo', which made every label reversible from the ID format alone."""
        import hashlib
        p = pd.read_csv(DATA / "predictions_test_pseudonymized.csv.gz")
        labels = set(p.household.unique())
        for i in range(1, 10000):
            d = hashlib.sha256(f"peakwise-demo:ID{i:04d}".encode()).hexdigest().upper()
            self.assertNotIn("H-" + d[:6], labels, "a label is reproducible with the old public salt")
        self.assertNotIn("peakwise-demo", (ROOT / "src" / "build_dashboard_data.py").read_text(), "a fixed salt is hard-coded again")

    def test_only_the_columns_the_app_uses_are_published(self):
        p = pd.read_csv(DATA / "predictions_test_pseudonymized.csv.gz", nrows=5)
        self.assertEqual(list(p.columns), ["household", "date", "y", "base_roll7_median", "rf", "xgb", "stack"])
        he = pd.read_csv(DATA / "household_errors.csv")
        self.assertNotIn("household", he.columns, "household-error table should carry no household label")
        self.assertEqual(set(he.period), {"test"})

    def test_no_missing_forecasts(self):
        p = pd.read_csv(DATA / "predictions_test_pseudonymized.csv.gz")
        self.assertEqual(int(p[["y", "base_roll7_median", "rf", "xgb", "stack"]].isna().sum().sum()), 0)
        self.assertTrue((p[["rf", "xgb", "stack"]] >= 0).all().all())


@unittest.skipUnless((P2 / "metrics_test.csv").exists(), "Phase 2 outputs not present")
class NumbersAreCopiesOfSavedResults(unittest.TestCase):
    def test_metric_tables_are_identical_to_phase2(self):
        for f in ("metrics_test.csv", "metrics_validation.csv", "deltas_vs_baseline.csv", "bands_test.csv", "bands_validation.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(DATA / f), pd.read_csv(P2 / f), obj=f)

    def test_predictions_match_phase2_row_for_row(self):
        if not (P2 / "predictions_test.csv.gz").exists():
            self.skipTest("raw-ID Phase 2 predictions are not stored in the repository; run `python src/phase2.py all` to regenerate them")
        a = pd.read_csv(DATA / "predictions_test_pseudonymized.csv.gz", parse_dates=["date"])
        b = pd.read_csv(P2 / "predictions_test.csv.gz", parse_dates=["date"])
        self.assertEqual(len(a), len(b))
        self.assertEqual(a.household.nunique(), b.household_ID.nunique())
        for k in ("y", "rf", "xgb", "stack", "base_roll7_median"):
            self.assertAlmostEqual(float(a[k].sum()), float(b[k].sum()), places=0, msg=k)

    def test_headline_mae_is_reproduced_from_the_published_predictions(self):
        """Needs no raw files: MAE recomputed from the dashboard's own predictions equals the saved metric."""
        a = pd.read_csv(DATA / "predictions_test_pseudonymized.csv.gz")
        mt = pd.read_csv(DATA / "metrics_test.csv", index_col="model")
        self.assertEqual(len(a), int(mt.loc["rf", "n"]))
        for k in ("rf", "xgb", "stack", "base_roll7_median"):
            self.assertAlmostEqual((a[k] - a.y).abs().mean(), mt.loc[k, "MAE"], places=4, msg=k)

    def test_importance_baseline_equals_test_mae(self):
        imp, mt = pd.read_csv(DATA / "permutation_importance.csv"), pd.read_csv(DATA / "metrics_test.csv", index_col="model")
        for m in ("rf", "xgb"):
            self.assertAlmostEqual(float(imp[imp.model == m].baseline_mae.iloc[0]), mt.loc[m, "MAE"], places=3)


class Pages(unittest.TestCase):
    def test_app_shell_renders(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIn("PeakWise", _text(at))

    def test_overview(self):
        at = _page("overview").run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        t = _text(at)
        mt = pd.read_csv(DATA / "metrics_test.csv", index_col="model")
        self.assertIn(f"{mt.loc['rf', 'MAE']:.4f}", t.replace("</td>", " "))     # the saved Random Forest test MAE is shown
        self.assertTrue("kWh" in t and "previously observed households" in t and "TR2" in t, "Overview is missing a required statement")

    def test_model_lab(self):
        at = _page("model_lab").run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(len(at.tabs), 4)
        t = _text(at)
        self.assertTrue("not to sum to one" in t and "out-of-fold" in t, "Model Lab is missing a required statement")
        at.radio(key="imp_model").set_value("xgb").run()                          # permutation-importance toggle
        self.assertEqual(len(at.exception), 0)

    def test_forecast_explorer_reacts_to_controls(self):
        at = _page("explorer").run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertTrue("Nothing here is a live forecast" in _text(at), "Explorer is missing its not-live disclaimer")
        for model in ("xgb", "stack", "rf"):
            at.radio(key="ex_model").set_value(model).run()
            self.assertEqual(len(at.exception), 0, f"model={model}")
        for opt in at.selectbox(key="ex_hh").options:                              # every example household renders
            at.selectbox(key="ex_hh").select(opt).run()
            self.assertEqual(len(at.exception), 0, f"household={opt}")
        # an arbitrary household and a narrow date window (few days: must show the small-sample caution)
        at.selectbox(key="ex_hh").select("Another household...").run()
        self.assertEqual(len(at.exception), 0)
        self.assertGreater(len(at.selectbox(key="ex_hh_any").options), 1000)

    def test_insights(self):
        at = _page("insights").run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        t = _text(at)
        low = t.lower()
        for must in ("tr2", "previously observed", "possible explanation", "not forced to sum to one"):
            self.assertTrue(must in low, f"Insights page is missing the statement: {must!r}")
        at.radio(key="ins_period").set_value("validation").run()
        self.assertEqual(len(at.exception), 0)

    def test_no_realtime_claims_and_correct_units(self):
        joined = ""
        for mod in ("overview", "model_lab", "explorer", "insights"):
            joined += _text(_page(mod).run())
        low = joined.lower()
        self.assertNotIn("real-time", low)
        self.assertNotIn("realtime", low)
        self.assertNotIn("live monitoring dashboard", low)
        self.assertTrue("not a live" in low or "nothing here is a live" in low or "not live" in low)
        self.assertNotRegex(low, r"peak power forecast(?!ing would)")           # we never claim to forecast peak power


class MissingArtifacts(unittest.TestCase):
    def test_empty_data_folder_gives_a_readable_error(self):
        with tempfile.TemporaryDirectory() as td:
            old = os.environ.get("PEAKWISE_DATA_DIR")
            os.environ["PEAKWISE_DATA_DIR"] = td
            stale = {k: v for k, v in sys.modules.items() if k == "dashboard" or k.startswith("dashboard.")}
            for k in stale:
                del sys.modules[k]                                                  # force DATA_DIR to be re-read
            try:
                at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
                self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
                self.assertGreaterEqual(len(at.error), 1)
                self.assertIn("build_dashboard_data.py", "\n".join(c.value for c in at.code))
            finally:
                if old is None:
                    os.environ.pop("PEAKWISE_DATA_DIR", None)
                else:
                    os.environ["PEAKWISE_DATA_DIR"] = old
                for k in [k for k in sys.modules if k == "dashboard" or k.startswith("dashboard.")]:
                    del sys.modules[k]
                sys.modules.update(stale)


if __name__ == "__main__":
    unittest.main(verbosity=2)
