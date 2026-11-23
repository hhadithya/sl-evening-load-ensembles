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


def change_chart(mt: pd.DataFrame) -> go.Figure:
    """Percentage change in MAE and RMSE against the 7-day median (negative = better)."""
    keys = ["rf", "xgb", "stack"]
    cols = [T.COLORS[k] for k in keys]
    fig = go.Figure()
    for metric, col, pattern in (("MAE", "MAE_vs_ref_%", None), ("RMSE", "RMSE_vs_ref_%", "/")):
        v = [float(mt.loc[k, col]) for k in keys]
        marker = dict(color=cols) if pattern is None else dict(color=cols, pattern=dict(shape=pattern, bgcolor="#FFFFFF", size=6))
        fig.add_trace(go.Bar(y=[D.ROLES[k] for k in keys], x=v, orientation="h", name=metric, marker=marker, text=[f"{metric} {x:+.1f}%" for x in v],
                             textposition="outside", cliponaxis=False, hovertemplate=metric + " %{x:+.1f}% vs 7-day median<extra></extra>"))
    lim = max(abs(float(mt.loc[k, c])) for k in keys for c in ("MAE_vs_ref_%", "RMSE_vs_ref_%"))
    fig.add_vline(x=0, line_color=T.NAVY, line_width=1)
    fig.update_layout(barmode="group", bargap=.3, showlegend=False, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(range=[-lim - 6, lim + 9], title="Change vs the 7-day median (%): left of zero = better", zeroline=False),
                      yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def decision_row(q: str, key: str, verdict: str, evidence: str) -> str:
    return (f'<div class="pw-q"><div class="q">{q}</div><div class="a">{T.dot(key)}<b>{verdict}</b>'
            f'<span class="e">{evidence}</span></div></div>')


def render():
    f = D.js("facts.json")
    mt, mv, bt = D.metrics("test"), D.metrics("validation"), D.bands("test")
    dl = D.deltas().set_index(["period", "model"])
    man = D.js("manifest.json")
    B = D.BASE

    T.header("Overview", "Forecasting tomorrow evening's electricity use",
             "Do ensemble models beat a simple rolling median at forecasting Sri Lankan households' next-day evening consumption? "
             "A chronological, leakage-tested experiment on the RECON-SL smart-meter data.")

    T.stats([
        ("Target", "18:30 – 22:30", "Grid import over four hours, in <b>kWh</b> (energy, not power in kW)"),
        ("Horizon", "1 day ahead", "Forecast made at the end of day D−1 for the evening of day D"),
        ("Data", f"{f['valid_household_days']:,}", f"audited household-days, {f['households_with_valid']:,} households, "
                                                  f"{pd.Timestamp(f['first_date']):%b %Y} to {pd.Timestamp(f['last_date']):%b %Y}"),
        ("Test period", f"{pd.Timestamp(f['splits']['test']['dates'][0]):%d %b} – {pd.Timestamp(f['splits']['test']['dates'][1]):%d %b}",
         f"{pd.Timestamp(f['splits']['test']['dates'][1]):%Y}: {f['test_rows']:,} household-days from {f['test_households']:,} households seen in training"),
    ])

    rf_t, xg_t, st_t = mt.loc["rf"], mt.loc["xgb"], mt.loc["stack"]
    rf_ci = dl.loc[("test", "rf")]
    w = man["stack_weights_for_test"]
    zero, typ, ext = bt.loc["exactly 0"], bt.loc["typical (P25-P75)"], bt.loc["extreme (>P99)"]

    st.markdown("## Which model, and why?")
    T.note(f"<b>Short answer.</b> Random Forest is the most consistent model, but it improves on a 7-day rolling median only slightly "
           f"({mt.loc[B, 'MAE'] - rf_t['MAE']:.4f} kWh of MAE on a typical miss of {mt.loc[B, 'MAE']:.2f} kWh), and XGBoost and stacking add no clear gain. "
           "Figures are for the test period; the baseline is the median of the household's own last 7 days.")
    rows = [
        ("Which model is best overall?", "rf", "Random Forest (bagging).",
         f"Better than the baseline on both MAE and RMSE in both periods (test MAE {fmt_pct(rf_t['MAE_vs_ref_%'])}, RMSE {fmt_pct(rf_t['RMSE_vs_ref_%'])}; "
         f"validation {fmt_pct(mv.loc['rf', 'MAE_vs_ref_%'])} and {fmt_pct(mv.loc['rf', 'RMSE_vs_ref_%'])}). The MAE gain is small: its 95% interval "
         f"({rf_ci.d_MAE_lo:+.4f} to {rf_ci.d_MAE_hi:+.4f} kWh) touches zero."),
        ("When is the 7-day median preferable?", B, "When simplicity matters, and on very quiet evenings.",
         f"It needs no training, trails Random Forest by only {mt.loc[B, 'MAE'] - rf_t['MAE']:.4f} kWh MAE, beats XGBoost on test MAE and RMSE, and beats "
         f"Random Forest and stacking on exactly-zero evenings (MAE {zero[f'MAE {B}']:.3f} vs {zero['MAE rf']:.3f} for Random Forest)."),
        ("Does XGBoost offer a meaningful advantage?", "xgb", "Only on typical and zero-consumption evenings.",
         f"Lowest median error ({xg_t['MedAE']:.4f} vs {mt.loc[B, 'MedAE']:.4f} kWh) and lowest MAE on typical evenings ({typ['MAE xgb']:.3f} vs {typ[f'MAE {B}']:.3f}), "
         f"but overall test MAE {fmt_pct(xg_t['MAE_vs_ref_%'])} and RMSE {fmt_pct(xg_t['RMSE_vs_ref_%'])} are worse than the baseline."),
        ("Did stacking improve on Random Forest?", "stack", "No.",
         f"Test MAE {st_t['MAE']:.4f} vs {rf_t['MAE']:.4f} and RMSE {st_t['RMSE']:.4f} vs {rf_t['RMSE']:.4f}. The meta-learner put {w['rf']:.2f} of its weight on Random Forest, "
         "so the stack is essentially Random Forest."),
        ("Where do the models struggle?", B, "Unusually high evenings.",
         f"Above the 99th percentile every model under-forecasts (mean error {ext[f'bias {B}']:+.1f} kWh baseline, {ext['bias rf']:+.1f} Random Forest, {ext['bias xgb']:+.1f} XGBoost), "
         f"but that is only {int(ext.n):,} household-days from {int(ext.households)} households, so the size of the gap is uncertain; the direction is the warning."),
    ]
    st.markdown("".join(decision_row(*r) for r in rows), unsafe_allow_html=True)

    st.markdown("## The evidence: four predictors on the untouched test period")
    cols = [("MAE", "MAE (kWh)", lambda v: f"{v:.4f}", "metric"), ("RMSE", "RMSE (kWh)", lambda v: f"{v:.4f}", "metric"),
            ("MedAE", "Median error (kWh)", lambda v: f"{v:.4f}", "metric"),
            ("MAE_vs_ref_%", "MAE vs baseline", fmt_pct, "delta"), ("RMSE_vs_ref_%", "RMSE vs baseline", fmt_pct, "delta")]
    st.markdown(results_table(mt, D.MAIN, cols, labels=D.ROLES), unsafe_allow_html=True)
    st.caption(f"{int(rf_t['n']):,} test household-days. Lower is better. MAE is the typical miss in kWh; RMSE penalises large misses more.")
    T.plot(change_chart(mt), 250, key="ov_change")
    st.caption("Solid bars: MAE. Hatched bars: RMSE. The baseline is the zero line.")
    T.note("<b>Scope.</b> A retrospective experiment on saved historical data, not a live forecasting service. It measures forecasting for <b>previously observed households</b>, "
           "not new ones. Test-period targets could not be cross-checked against the meter's TR2 register (no six-hour data after 31 Oct 2024). "
           f"The test-evening median is {f['target_median_kwh']:.2f} kWh; the full audited record (Oct 2023 – Dec 2024) has a median of {f['audited_median_kwh']:.2f} kWh.")

    st.markdown("## Data and design")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            f"**Source.** RECON-SL (LIRNEasia): 15-minute smart-meter registers for {f['households_in_raw']:,} Sri Lankan households. The register is cumulative, so consumption is the "
            f"difference between readings. An audit kept only evenings with all 17 readings from 18:30 to 22:30; about {f['missing_quarter_hours_pct']:.0f}% of quarter-hours are missing, "
            "mostly whole days, and nothing is imputed.")
        st.markdown(
            f"**Inputs.** {len(f['features'])} features from the household's own past evenings up to D−1 (yesterday, same weekday last week, 7- and 28-day medians, spread, "
            "history length, calendar). No weather, tariff or occupancy data. **Output:** forecast energy in kWh for the evening.")
    with c2:
        st.markdown('<div class="pw-eyebrow">Chronological design</div>', unsafe_allow_html=True)
        T.plot(split_timeline(f), 215, key="ov_split")
        st.caption("Train and tune, compare on validation, open the test period once (models refitted on train + validation).")
    st.markdown('<div class="pw-next"><b>Next:</b> <i>Model Lab</i> explains bagging, boosting and stacking and has the full metrics · '
                '<i>Forecast Explorer</i> shows saved forecasts against what was observed · <i>Insights and Limitations</i> shows where the models help or fail.</div>',
                unsafe_allow_html=True)

