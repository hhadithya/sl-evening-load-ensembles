#!/usr/bin/env python3
"""Executable leakage and correctness tests for the PeakWise Phase 1 feature table.

    python -I -m unittest check_leakage -v        (or: python -I check_leakage.py)

These are tests, not assertions in prose: each one would fail loudly if a feature of day D could
see day D or anything after it.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import phase1

AUDIT = Path(__file__).resolve().parent / "audit_out"
SEED = 7


def _toy_target(n_households=6, days=120, seed=0, missing=0.35):
    """Synthetic panel with gaps, zeros and a heavy tail - the same awkward shapes as the real data."""
    rng = np.random.default_rng(seed)
    cal = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for i in range(n_households):
        level = 10 ** rng.uniform(-1.5, 0.8)
        for d in cal:
            if rng.random() < missing:
                continue                                    # whole day absent
            v = 0.0 if rng.random() < 0.1 else level * rng.lognormal(0, 0.5)
            rows.append((f"ID{i:04d}", d, round(v, 3)))
    return pd.DataFrame(rows, columns=["household_ID", "date", "y"])


class FutureCannotReachThePast(unittest.TestCase):
    """The decisive test: corrupt the future, and nothing in the past may move."""

    def _corrupt_after(self, target, cutoff, factor=1000.0, add=777.0):
        bad = target.copy()
        m = bad["date"] > cutoff
        bad.loc[m, "y"] = bad.loc[m, "y"] * factor + add
        return bad

    def _assert_past_unchanged(self, target, cutoff):
        a = phase1.build_features(target)
        b = phase1.build_features(self._corrupt_after(target, cutoff))
        key = ["household_ID", "date"]
        a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
        past = a["date"] <= cutoff
        self.assertGreater(int(past.sum()), 0)
        for c in phase1.FEATURES:
            x, y = a.loc[past, c].to_numpy(dtype=float), b.loc[past, c].to_numpy(dtype=float)
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            self.assertTrue(same.all(), f"feature '{c}' changed for {int((~same).sum())} past rows "
                                        f"when only future targets were altered")

    def test_synthetic_panel(self):
        t = _toy_target()
        self._assert_past_unchanged(t, pd.Timestamp("2024-03-01"))

    def test_real_target_sample(self):
        t = phase1.load_target(AUDIT)
        hh = pd.Series(sorted(t.household_ID.unique())).sample(60, random_state=SEED)
        t = t[t.household_ID.isin(set(hh))]
        self._assert_past_unchanged(t, pd.Timestamp("2024-06-30"))

    def test_row_cannot_see_its_own_target(self):
        t = _toy_target(n_households=30, seed=3)
        a = phase1.build_features(t)
        # at most one corrupted row per household, otherwise one corrupted row would legitimately
        # appear in another corrupted row's history and the comparison would be meaningless
        pick = (a.dropna(subset=["roll28_median"]).sample(frac=1, random_state=SEED)
                .groupby("household_ID", as_index=False).head(1))
        bad = t.copy()
        key = set(zip(pick.household_ID, pick.date))
        m = [(h, d) in key for h, d in zip(bad.household_ID, bad.date)]
        bad.loc[m, "y"] = 12345.0
        b = phase1.build_features(bad)
        j = a.merge(b, on=["household_ID", "date"], suffixes=("_a", "_b"))
        j = j[[(h, d) in key for h, d in zip(j.household_ID, j.date)]]
        for c in phase1.FEATURES:
            x, y = j[f"{c}_a"].to_numpy(dtype=float), j[f"{c}_b"].to_numpy(dtype=float)
            self.assertTrue(((x == y) | (np.isnan(x) & np.isnan(y))).all(), f"'{c}' saw its own target")


class FeaturesMatchAnAsOfRecomputation(unittest.TestCase):
    """Brute-force check: recompute features for sampled rows by plain date filtering."""

    @classmethod
    def setUpClass(cls):
        cls.t = phase1.load_target(AUDIT)
        hh = pd.Series(sorted(cls.t.household_ID.unique())).sample(40, random_state=SEED)
        cls.t = cls.t[cls.t.household_ID.isin(set(hh))].reset_index(drop=True)
        cls.f = phase1.build_features(cls.t)

    def test_sampled_rows(self):
        sample = self.f.sample(250, random_state=SEED)
        hist_by_hh = {h: g.set_index("date")["y"].sort_index() for h, g in self.t.groupby("household_ID")}
        for r in sample.itertuples():
            s = hist_by_hh[r.household_ID]
            past = s[s.index <= r.date - pd.Timedelta(days=1)]          # strictly before day D
            w28 = past[past.index >= r.date - pd.Timedelta(days=28)]
            w7 = past[past.index >= r.date - pd.Timedelta(days=7)]
            exp = {
                "lag1": s.get(r.date - pd.Timedelta(days=1), np.nan),
                "lag7": s.get(r.date - pd.Timedelta(days=7), np.nan),
                "last_value": past.iloc[-1] if len(past) else np.nan,
                "gap_days": (r.date - past.index[-1]).days if len(past) else np.nan,
                "roll7_median": w7.median() if len(w7) else np.nan,
                "roll28_median": w28.median() if len(w28) else np.nan,
                "roll7_n": float(len(w7)), "roll28_n": float(len(w28)),
                "roll28_iqr": (w28.quantile(0.75) - w28.quantile(0.25)) if len(w28) else np.nan,
                "roll28_frac_zero": float((w28 == 0).mean()) if len(w28) else np.nan,
                "expanding_median": past.median() if len(past) else np.nan,
            }
            for k, want in exp.items():
                got = float(getattr(r, k))
                if np.isnan(want):
                    self.assertTrue(np.isnan(got), f"{k}: expected NaN, got {got} ({r.household_ID} {r.date})")
                else:
                    self.assertAlmostEqual(got, float(want), places=9,
                                           msg=f"{k} mismatch for {r.household_ID} {r.date}")

    def test_no_feature_equals_the_target(self):
        f = self.f.dropna(subset=["roll28_median"])
        for c in ("lag1", "lag7", "last_value", "roll7_median", "roll28_median", "expanding_median"):
            same = np.isclose(f[c].to_numpy(dtype=float), f["y"].to_numpy(dtype=float), equal_nan=False)
            self.assertLess(same.mean(), 0.5, f"'{c}' equals the target on {100*same.mean():.0f}% of rows")


class SplitsAreChronological(unittest.TestCase):
    def test_boundaries_do_not_overlap(self):
        d = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in phase1.SPLIT_DATES.items()}
        self.assertLess(d["train"][1], d["val"][0])
        self.assertLess(d["val"][1], d["test"][0])

    def test_assigned_rows_respect_the_order(self):
        t = phase1.load_target(AUDIT)
        f = phase1.assign_split(t.assign(**{c: np.nan for c in phase1.FEATURES}))
        mx = f.groupby("split")["date"].agg(["min", "max"])
        self.assertLess(mx.loc["train", "max"], mx.loc["val", "min"])
        self.assertLess(mx.loc["val", "max"], mx.loc["test", "min"])
        self.assertEqual(int(f["split"].isna().sum()), 0, "some target rows fall outside every split window")


class FillingDoesNotInventInformation(unittest.TestCase):
    def test_filled_table_has_no_missing_values_left(self):
        """A plain RandomForestRegressor must be able to consume the filled table as-is."""
        for src in (_toy_target(seed=11), phase1.load_target(AUDIT).head(40000)):
            f = phase1.apply_inclusion(phase1.build_features(src))
            g = phase1.fill_missing(f)
            self.assertTrue(np.isfinite(g[phase1.FILLED_FEATURES].to_numpy(dtype=float)).all(),
                            f"NaN left in: {[c for c in phase1.FILLED_FEATURES if g[c].isna().any()]}")

    def test_fill_is_row_local_and_flagged(self):
        t = _toy_target(seed=5)
        f = phase1.apply_inclusion(phase1.build_features(t))
        g = phase1.fill_missing(f)
        self.assertEqual(int(g["lag1"].isna().sum()), 0)
        self.assertTrue(((g["lag1_missing"] == 1) == f["lag1"].isna().to_numpy()).all())
        filled = g.loc[f["lag1"].isna().to_numpy(), "lag1"].to_numpy()
        src = f.loc[f["lag1"].isna().to_numpy(), "last_value"].to_numpy()
        self.assertTrue(np.allclose(filled, src, equal_nan=True), "lag1 was filled from something else")
        untouched = f["lag1"].notna().to_numpy()
        self.assertTrue(np.allclose(g.loc[untouched, "lag1"], f.loc[untouched, "lag1"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
