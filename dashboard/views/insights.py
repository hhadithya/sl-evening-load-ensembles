"""Insights and Limitations: where the models help or fail, how widespread the gain is, and what the data cannot say."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data as D
from .. import theme as T

KEYS = [D.BASE, "rf", "xgb", "stack"]


def band_chart(b: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = [i.replace(" (", "<br>(") + f"<br>n={int(r.n):,}<br>{int(r.households):,} hh" for i, r in b.iterrows()]
    for k in KEYS:
        fig.add_trace(go.Bar(x=x, y=b[f"MAE {k}"] / b[f"MAE {D.BASE}"], name=D.NAMES[k], marker_color=T.COLORS[k],
                             customdata=b[f"MAE {k}"], hovertemplate="%{customdata:.3f} kWh MAE<br>%{y:.2f} x baseline<extra>" + D.NAMES[k] + "</extra>"))
    fig.add_hline(y=1, line_color=T.NAVY, line_width=1)
    fig.update_layout(barmode="group", bargap=.25, yaxis=dict(title="MAE as a multiple of the 7-day median's MAE"), xaxis=dict(tickfont=dict(size=11)))
    return fig


def render():
    f, man = D.js("facts.json"), D.js("manifest.json")
    T.header("Insights and limitations", "Where the models help, where they do not",
             "What the saved results show about different evenings and households, separate from explanations that are only suggestions.")

    # ------------------------------------------------------------------ by consumption range
    st.markdown("## Different consumption ranges")
    period = st.radio("Period", ["test", "validation"], horizontal=True, key="ins_period",
                      format_func=lambda p: "Test" if p == "test" else "Validation")
    b = D.bands(period)
    T.plot(band_chart(b), 400, key=f"ins_band_{period}")
    st.caption("Bands use training-period percentiles. Below 1 = better than the 7-day median. n = household-days, hh = households.")
    z, lo, ty, ex = b.loc["exactly 0"], b.loc["low (<P25)"], b.loc["typical (P25-P75)"], b.loc["extreme (>P99)"]
    st.markdown(f"""
**Observed ({period}).**
- *Typical evenings (middle 50%):* all three ensembles beat the baseline (MAE {ty['MAE rf']:.3f} Random Forest, {ty['MAE xgb']:.3f} XGBoost, {ty['MAE stack']:.3f} stacking, against {ty[f'MAE {D.BASE}']:.3f}).
- *Exactly-zero and low evenings:* Random Forest and stacking are worse than the baseline ({z['MAE rf']:.3f} and {z['MAE stack']:.3f} against {z[f'MAE {D.BASE}']:.3f} on zero days); XGBoost does best ({z['MAE xgb']:.3f}).
- *Unusually high evenings (above the 99th percentile):* every model under-forecasts heavily (mean error {ex[f'bias {D.BASE}']:+.1f} kWh baseline, {ex['bias rf']:+.1f} Random Forest, {ex['bias xgb']:+.1f} XGBoost) and the ensembles are no better (MAE {ex[f'MAE {D.BASE}']:.2f}, {ex['MAE rf']:.2f}, {ex['MAE xgb']:.2f}). Only {int(ex.n):,} household-days from {int(ex.households)} households: the direction is the warning, not the size.
""")
    T.note("<b>Possible explanation (not tested).</b> Very high evenings are rare and probably not predictable from a household's own recent past, so models predict a typical level for them. "
           "Models also cannot commit to exactly zero, which may explain the small positive forecasts on empty evenings.")

    # ------------------------------------------------------------------ household breadth
    st.markdown("## Is the improvement widespread across households?")
    he = D.csv("household_errors.csv")
    he = he[he.period == "test"].copy()
    for k in ("rf", "xgb", "stack"):
        he[f"d_{k}"] = he[k] - he[D.BASE]
        he[f"tot_{k}"] = he[f"d_{k}"] * he.n_days            # change in summed absolute error for the household, in kWh
    fig = go.Figure()
    lim = float(he["d_rf"].quantile([.01, .99]).abs().max())
    for k in ("rf", "xgb"):
        fig.add_trace(go.Histogram(x=he[f"d_{k}"].clip(-lim, lim), name=D.NAMES[k], marker_color=T.COLORS[k], opacity=.65,
                                   xbins=dict(start=-lim, end=lim, size=lim / 35), hovertemplate="%{y} households<extra>" + D.NAMES[k] + "</extra>"))
    fig.add_vline(x=0, line_color=T.NAVY, line_width=1)
    fig.update_layout(barmode="overlay", xaxis_title="Household MAE: model minus 7-day median (kWh); negative = model better (clipped at the 1st and 99th percentile)",
                      yaxis_title="Households")
    T.plot(fig, 340, key="ins_hh")
    share = {k: 100 * (he[f"d_{k}"] < 0).mean() for k in ("rf", "xgb", "stack")}
    worst10, best10 = he.tot_rf.nlargest(10).sum(), he.tot_rf.nsmallest(10).sum()
    st.markdown(f"""
**Observed (test, {len(he):,} households).** {share['rf']:.0f}% of households improve with Random Forest, {share['xgb']:.0f}% with XGBoost and {share['stack']:.0f}% with stacking; the median household's MAE changes by {he.d_rf.median():+.4f} kWh (Random Forest). 
The ten households Random Forest serves worst add {worst10:.0f} kWh of absolute error, more than the ten it serves best remove ({-best10:.0f} kWh). The gain is broad but thin.
""")

    # ------------------------------------------------------------------ stacking
    st.markdown("## Why stacking adds so little")
    mt, mv = D.metrics("test"), D.metrics("validation")
    wv, wt = man["stack_weights_for_validation"], man["stack_weights_for_test"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
**Observed.**
- Test MAE {mt.loc['stack', 'MAE']:.4f} for stacking vs {mt.loc['rf', 'MAE']:.4f} for Random Forest ({mv.loc['stack', 'MAE']:.4f} vs {mv.loc['rf', 'MAE']:.4f} on validation).
- The meta-learner put {wt['rf']:.2f} (test fit) and {wv['rf']:.2f} (validation fit) of its weight on Random Forest. Weights are non-negative and not forced to sum to one (sums {sum(wt.values()):.3f} and {sum(wv.values()):.3f}).
- An equal-weight average (not a candidate) has the lowest validation MAE of any row ({mv.loc['stack_equal_avg', 'MAE']:.4f}) but not on test ({mt.loc['stack_equal_avg', 'MAE']:.4f}).
""")
    with c2:
        st.markdown("""
**Possible explanations (not tested).**
- All base models see the same few features, so their errors are highly correlated and a second level has little to combine.
- The meta-learner minimises squared error, favouring Random Forest, while the score is mainly MAE. An absolute-error meta-learner was not tried.
""")

    # ------------------------------------------------------------------ limitations
    st.markdown("## Data and method limitations")
    tr = f["tr2_by_period"]
    rows = [
        ("Missing and irregular data",
         f"About {f['missing_quarter_hours_pct']:.0f}% of quarter-hours are missing, mostly whole days (almost none from mid-January to March 2024). Nothing is imputed.",
         "Probably collection outages. Not verified."),
        ("No meter cross-check on test",
         f"The meter's TR2 register has no usable bracket after 31 Oct 2024, so all {tr['test'].get('no_bracket', 0):,} test rows are unverified. "
         f"On validation, {tr['validation'].get('agree', 0):,} rows agree, but that subset is confounded with the month.",
         "A change in the six-hour feed. Not investigated."),
        ("Previously observed households only",
         f"Only households in the fitting data are forecast. {f['val_unseen_rows']} validation rows ({f['val_unseen_households']} households) were unseen; the test period has none.",
         "Performance for new households is unknown."),
        ("Short evaluation windows",
         "Validation and test are six weeks each, recent and fairly dense: no seasonality, probably easier than average.",
         "A longer record would give more honest intervals."),
        ("Extreme evenings",
         "The highest household-days pass every audit rule but are few and concentrated in a few households.",
         "Possibly shared or commercial-like loads. Not investigated."),
        ("Model complexity",
         f"Random Forest changes test MAE by {mt.loc['rf', 'MAE_vs_ref_%']:+.1f}% against the baseline, and stacking runs three models.",
         "A rolling median may be more defensible where simplicity matters."),
        ("What is forecast",
         "Next-day energy over four hours in kWh, not instantaneous peak power in kW, and not a live signal.",
         "Peak-power forecasting needs different data."),
    ]
    body = "".join(f"<tr><td><b>{a}</b></td><td>{b_}</td><td>{c_}</td></tr>" for a, b_, c_ in rows)
    st.markdown(f'<table class="pw-table pw-two"><thead><tr><th style="width:18%">Topic</th><th>What the results show</th><th style="width:28%">Possible explanation or consequence</th></tr></thead>'
                f"<tbody>{body}</tbody></table>", unsafe_allow_html=True)

    st.markdown("## Provenance and privacy")
    st.markdown(f"""
- **Data:** RECON-SL residential electricity dataset (LIRNEasia), {f['license']}. The dashboard reads about 1 MB of saved results; the raw data and models are not needed.
- **Everything shown is a saved experiment output** produced by the code and tests in this repository; no figure is typed in or recomputed at start-up.
- **Privacy:** household identifiers are replaced by salted-hash labels (pseudonymised, not anonymous); no addresses or survey data are included, and individual histories are shown only for the test period.
- **Implementation:** written with AI assistance (Claude Code) under the project owner's direction; decisions are recorded in `artifacts/phase2/TUNING_AND_AI_LOG.md`.
""")
