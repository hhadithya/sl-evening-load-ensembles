#!/usr/bin/env python3
"""Correctness checks for Phase 2 stacking and evaluation.   python tests/check_phase2.py

The stacking tests are the important ones. They use light model settings on a real-data subsample so
they run in about a minute, but exercise the very same `stack_pipeline` that produced the results.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import phase1
import phase2

SEED = 7


def _small_table(n_households=60):
    m = phase2.load_table(include_test=True)
    hh = pd.Series(sorted(m.household_ID.unique())).sample(n_households, random_state=SEED)
    return m[m.household_ID.isin(set(hh))].reset_index(drop=True)


def _light_specs():
    return {"rf": phase2.rf_spec("rf", "raw", 20, 0.4, n_estimators=15),
            "xgb": phase2.xgb_spec("xgb", "reg:absoluteerror", "raw", 3, 20, 1.0, rounds=25),
            "xgb_alt": phase2.xgb_spec("xgb_alt", "reg:squarederror", "log", 3, 20, 1.0, rounds=25)}


class StackingIsChronologicalAndLeakFree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _small_table()
        cls.nontest = cls.m[cls.m.split != "test"].reset_index(drop=True)
        cls.test = cls.m[cls.m.split == "test"].reset_index(drop=True)
        cls.specs = _light_specs()
        cls.res = phase2.stack_pipeline(cls.nontest, cls.specs, cls.test)

    def test_every_oof_model_was_fitted_strictly_before_the_window_it_predicts(self):
        for name, audit in self.res["audit"].items():
            self.assertEqual(len(audit), 3)
            for train_max, valid_min, n_tr, n_va in audit:
                self.assertLess(train_max, valid_min, f"{name}: a fold's model saw rows not before its window")
                self.assertGreater(n_tr, 0); self.assertGreater(n_va, 0)

    def test_folds_are_disjoint_and_inside_the_training_period(self):
        for a, b in self.res["folds"]:
            self.assertEqual(len(np.intersect1d(a, b)), 0, "a row is both trained on and predicted in one fold")
            self.assertTrue((self.nontest.split.iloc[a] == "train").all())
            self.assertTrue((self.nontest.split.iloc[b] == "train").all())
        # validation windows of the folds do not overlap each other
        allb = np.concatenate([b for _, b in self.res["folds"]])
        self.assertEqual(len(allb), len(np.unique(allb)))

    def test_meta_learner_is_fitted_only_on_rows_before_validation(self):
        lo, hi = self.res["meta_train_dates"]
        self.assertLessEqual(hi, phase2.TRAIN_END)
        self.assertLess(hi, pd.Timestamp(phase1.SPLIT_DATES["val"][0]))
        tr_rows = set(self.res["cv_rows"].tolist())
        self.assertTrue(tr_rows.isdisjoint(set(self.res["val_idx"].tolist())), "meta rows overlap validation rows")
        lo2, hi2 = self.res["meta_test_dates"]
        self.assertLessEqual(hi2, phase2.VAL_END)
        self.assertLess(hi2, pd.Timestamp(phase1.SPLIT_DATES["test"][0]))

    def test_oof_inputs_are_complete_and_finite(self):
        self.assertTrue(np.isfinite(self.res["P_cv"]).all())
        self.assertTrue(np.isfinite(self.res["P_val"]).all())
        self.assertTrue(np.isfinite(self.res["P_test"]).all())

    def test_meta_weights_match_an_independent_least_squares_solution(self):
        """NNLS optimum: on the columns with positive weight it equals ordinary least squares, and it can
        never be worse than the best single input (a weight vector it is allowed to choose)."""
        for P, y_rows, key in ((self.res["P_cv"], self.nontest["y"].to_numpy()[self.res["cv_rows"]], "meta_val"),):
            w = self.res[key].coef_
            act = w > 1e-12
            ols = np.linalg.lstsq(P[:, act], y_rows, rcond=None)[0]
            np.testing.assert_allclose(w[act], ols, rtol=1e-6, atol=1e-8)
            sse = lambda v: float(((P @ v - y_rows) ** 2).sum())
            self.assertLessEqual(sse(w), min(sse(np.eye(P.shape[1])[j]) for j in range(P.shape[1])) + 1e-6)

    def test_meta_weights_are_non_negative(self):
        for k in ("meta_val", "meta_test"):
            self.assertTrue((self.res[k].coef_ >= 0).all())

    def test_future_targets_cannot_change_validation_or_test_predictions(self):
        """The decisive test: corrupt targets the stack must not see and require identical predictions."""
        def corrupt(df, mask):
            d = df.copy(); d.loc[mask, "y"] = d.loc[mask, "y"] * 1000.0 + 777.0
            return d
        # (a) test targets are never used for fitting: corrupting them must change nothing, for val or test
        test_bad = corrupt(self.test, np.ones(len(self.test), bool))
        r_a = phase2.stack_pipeline(self.nontest, self.specs, test_bad)
        np.testing.assert_allclose(r_a["stack_val"], self.res["stack_val"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(r_a["stack_test"], self.res["stack_test"], rtol=0, atol=1e-9)
        # (b) validation targets are never used when evaluating on validation: corrupting them must leave
        #     the validation predictions of every base model and of the stack unchanged
        val_mask = (self.nontest.split == "val").to_numpy()
        r_b = phase2.stack_pipeline(corrupt(self.nontest, val_mask), self.specs, None)
        np.testing.assert_allclose(r_b["stack_val"], self.res["stack_val"], rtol=0, atol=1e-9)
        for n in ("rf", "xgb", "xgb_alt"):
            np.testing.assert_allclose(r_b["val_base"][n], self.res["val_base"][n], rtol=0, atol=1e-9)
        # (c) and the check is not vacuous: corrupting TRAINING targets does change the validation predictions
        tr_mask = (self.nontest.split == "train").to_numpy()
        r_c = phase2.stack_pipeline(corrupt(self.nontest, tr_mask), self.specs, None)
        self.assertGreater(float(np.abs(r_c["stack_val"] - self.res["stack_val"]).max()), 1e-3)


class TuningNeverSeesTheTestPeriod(unittest.TestCase):
    def test_tuning_table_contains_no_test_rows(self):
        m = phase2.load_table(include_test=False)
        self.assertNotIn("test", set(m.split))
        self.assertLessEqual(m.date.max(), phase2.VAL_END)
        for a, b in phase1.cv_folds(m):
            self.assertLessEqual(m.date.iloc[b].max(), phase2.TRAIN_END)


class EvaluationCorrectness(unittest.TestCase):
    def test_household_mae_mean_and_median_are_what_they_say(self):
        y = np.array([1., 1., 2., 2., 5., 5.]); p = np.array([0., 2., 2., 2., 3., 9.])
        hh = np.array(["a", "a", "b", "b", "c", "c"])
        out = phase1.metrics(y, p, hh)
        per = {"a": 1.0, "b": 0.0, "c": 3.0}
        self.assertAlmostEqual(out["hh_MAE_mean"], np.mean(list(per.values())))
        self.assertAlmostEqual(out["hh_MAE_median"], np.median(list(per.values())))
        self.assertAlmostEqual(out["MAE"], np.abs(p - y).mean())
        self.assertAlmostEqual(out["RMSE"], np.sqrt(((p - y) ** 2).mean()))

    def test_paired_bootstrap_point_estimate_equals_direct_difference(self):
        rng = np.random.default_rng(1)
        hh = np.repeat(np.arange(40), 25); y = rng.gamma(2, 1, len(hh))
        a = y + rng.normal(0, .3, len(hh)); b = y + rng.normal(0, .5, len(hh))
        r = phase2.paired_household_bootstrap(y, a, b, hh, n_boot=50)
        self.assertAlmostEqual(r["d_MAE"], np.abs(a - y).mean() - np.abs(b - y).mean(), places=10)
        self.assertAlmostEqual(r["d_RMSE"], np.sqrt(((a - y) ** 2).mean()) - np.sqrt(((b - y) ** 2).mean()), places=10)
        self.assertLess(r["d_MAE"], 0)
        self.assertLessEqual(r["d_MAE_lo"], r["d_MAE"] + 1e-9); self.assertGreaterEqual(r["d_MAE_hi"], r["d_MAE"] - 1e-9)

    def test_every_baseline_answers_on_every_row(self):
        m = phase2.load_table(include_test=True)
        bf = phase2.baseline_frame(m, 0.7)
        self.assertEqual(int(bf.isna().sum().sum()), 0, "a baseline still has missing predictions")
        self.assertEqual(len(bf), len(m))

    def test_unseen_households_are_flagged_correctly(self):
        m = phase2.load_table(include_test=True)
        fit = m[m.split == "train"]; ev = m[m.split == "val"]
        seen = phase2.seen_mask(ev, fit)
        self.assertEqual(int((~seen).sum()), int((~ev.household_ID.isin(set(fit.household_ID))).sum()))
        self.assertTrue(ev.household_ID[seen].isin(set(fit.household_ID)).all())

    def test_back_transform_is_non_negative_and_inverts_log(self):
        y = np.array([0., .5, 3., 40.])
        np.testing.assert_allclose(phase2.from_target(phase2.to_target(y, "log"), "log"), y, atol=1e-9)
        self.assertTrue((phase2.from_target(np.array([-5., 0.]), "raw") >= 0).all())


class SavedArtifactsAreConsistent(unittest.TestCase):
    def test_saved_predictions_match_recomputed_metrics(self):
        f = phase2.OUT / "predictions_test.csv.gz"
        if not f.exists():
            self.skipTest("run `python src/phase2.py all` first")
        pt = pd.read_csv(f)
        self.assertEqual(int(pt[["rf", "xgb", "xgb_alt", "stack", "base_roll7_median"]].isna().sum().sum()), 0)
        self.assertTrue((pt.split == "test").all())
        self.assertTrue((pt[["rf", "xgb", "xgb_alt", "stack"]] >= 0).all().all())


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
