"""Overview: what PeakWise predicts, the data, and the headline findings (without overstating)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data as D
from .. import theme as T
from ..tables import fmt_pct, results_table


def split_timeline(facts: dict, height: int = 210) -> go.Figure:
    fig = go.Figure()
    sp = facts["splits"]
    t0 = pd.Timestamp(facts["first_date"])
    spans = [("Training", t0, pd.Timestamp(sp["train"]["dates"][1]), T.COLORS["rf"]),
             ("Validation", pd.Timestamp(sp["val"]["dates"][0]), pd.Timestamp(sp["val"]["dates"][1]), T.COLORS["xgb"]),
             ("Test", pd.Timestamp(sp["test"]["dates"][0]), pd.Timestamp(sp["test"]["dates"][1]), T.BRONZE)]
    for label, a, b, c in spans:
        fig.add_trace(go.Bar(y=[label], x=[(b - a).days * 86400000 + 86400000], base=[a], orientation="h", marker_color=c,
                             hovertemplate=f"{a:%d %b %Y} to {b:%d %b %Y}<extra></extra>", showlegend=False, width=0.55))
    fig.update_layout(xaxis=dict(type="date", tickformat="%b %Y"), yaxis=dict(autorange="reversed", showgrid=False), barmode="overlay",
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


def render():
    f = D.js("facts.json")
    mt, mv = D.metrics("test"), D.metrics("validation")
    dl = D.deltas().set_index(["period", "model"])
    man = D.js("manifest.json")

    T.header("Overview", "Forecasting tomorrow evening's electricity use",
             "Can ensemble learning improve next-day evening electricity forecasts for Sri Lankan households? "
             "This dashboard presents a chronological, leakage-tested experiment on the RECON-SL smart-meter dataset, "
             "and what it actually found.")

    T.stats([
        ("Target", "18:30 – 22:30", "Grid import over four hours, in <b>kWh</b> (energy, not instantaneous power in kW)"),
        ("Horizon", "1 day ahead", "Forecast made at the end of day D−1 for the evening of day D"),
        ("Data", f"{f['valid_household_days']:,}", f"audited household-days, {f['households_with_valid']:,} households, "
                                                  f"{pd.Timestamp(f['first_date']):%b %Y} to {pd.Timestamp(f['last_date']):%b %Y}"),
        ("Test period", f"{pd.Timestamp(f['splits']['test']['dates'][0]):%d %b} – {pd.Timestamp(f['splits']['test']['dates'][1]):%d %b}",
         f"{pd.Timestamp(f['splits']['test']['dates'][1]):%Y}: {f['test_rows']:,} household-days from {f['test_households']:,} households seen in training"),
    ])

    st.markdown("## What goes in, what comes out")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            f"**Source.** RECON-SL, published by LIRNEasia: 15-minute smart-meter registers for {f['households_in_raw']:,} Sri Lankan households "
            f"({f['raw_rows']:,} readings). The meter register is cumulative, so each 15-minute consumption is the difference between consecutive readings. "
            f"A data audit kept only evenings with all 17 readings between 18:30 and 22:30 and no negative or implausible steps. "
            f"About {f['missing_quarter_hours_pct']:.0f}% of quarter-hours inside household histories are missing, mostly whole days, so nothing is imputed.")
        st.markdown(
            f"**Inputs.** {len(f['features'])} features computed only from the household's own past evenings up to the end of D−1: "
            "yesterday's value, the same weekday last week, 7- and 28-day medians, spread, share of zero evenings, how many days of history exist, "
            "and the calendar. No weather, tariff or occupancy information is available.")
        st.markdown("**Output.** One number per household-day: forecast energy in kWh for 18:30 to 22:30.")
    with c2:
        st.markdown('<div class="pw-eyebrow">Chronological design</div>', unsafe_allow_html=True)
        T.plot(split_timeline(f), 215, key="ov_split")
        st.caption("Training: fit and tune. Validation: compare model families. Test: opened once, with models refitted on training plus validation. Models never see data after the period they predict.")

    st.markdown("## Four predictors, compared on the untouched test period")
    cols = [("MAE", "MAE (kWh)", lambda v: f"{v:.4f}", "metric"), ("RMSE", "RMSE (kWh)", lambda v: f"{v:.4f}", "metric"),
            ("MedAE", "Median error (kWh)", lambda v: f"{v:.4f}", "metric"),
            ("MAE_vs_ref_%", "MAE vs baseline", fmt_pct, "delta"), ("RMSE_vs_ref_%", "RMSE vs baseline", fmt_pct, "delta")]
    st.markdown(results_table(mt, D.MAIN, cols), unsafe_allow_html=True)
    st.caption(f"{int(mt.loc['rf', 'n']):,} test household-days. MAE: average absolute miss. RMSE: penalises large misses more. "
               "Lower is better. Negative percentages mean better than the 7-day median baseline.")

    rf_t, xg_t, st_t = mt.loc["rf"], mt.loc["xgb"], mt.loc["stack"]
    rf_ci = dl.loc[("test", "rf")]
    w = man["stack_weights_for_test"]
    st.markdown("## Key findings")
    st.markdown(f"""
- **A 7-day rolling median is already a strong forecaster.** Its error is about {mt.loc[D.BASE, 'MAE']:.2f} kWh on a household-evening whose median is
  {f['target_median_kwh']:.2f} kWh. Any model has to beat this, not a naive guess.
- **Random Forest improves on it only modestly:** test MAE {fmt_pct(rf_t['MAE_vs_ref_%'])} and RMSE {fmt_pct(rf_t['RMSE_vs_ref_%'])}. The 95% interval for the MAE difference
  ({rf_ci.d_MAE_lo:+.4f} to {rf_ci.d_MAE_hi:+.4f} kWh) touches zero; the RMSE gain is the clearer one.
- **XGBoost does not reliably beat the baseline:** test MAE {fmt_pct(xg_t['MAE_vs_ref_%'])} and RMSE {fmt_pct(xg_t['RMSE_vs_ref_%'])}, although it has the lowest median error.
- **Stacking adds essentially nothing over Random Forest** (test MAE {st_t['MAE']:.4f} vs {rf_t['MAE']:.4f}). The meta-learner put {w['rf']:.2f} of its weight on Random Forest.
""")
    T.note("<b>Scope.</b> This is a retrospective experiment on saved historical data, not a live forecasting or monitoring service. "
           "It measures forecasting the future of <b>previously observed households</b>; it does not demonstrate performance for new households. "
           "Test-period targets could not be cross-checked against the meter's own TR2 register (no six-hour data after 31 Oct 2024).")
