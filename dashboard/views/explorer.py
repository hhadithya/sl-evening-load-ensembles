"""Forecast Explorer: browse the SAVED test-period predictions against the observed evening consumption."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data as D
from .. import theme as T

CHOICES = {"rf": "Random Forest", "xgb": "XGBoost", "stack": "Stacking"}
ANOTHER = "Another household..."


def household_chart(h: pd.DataFrame, model: str) -> go.Figure:
    cal = pd.date_range(h.date.min(), h.date.max(), freq="D")
    h = h.set_index("date").reindex(cal)              # missing days stay missing: the line breaks, it is never drawn as zero
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h[D.BASE], name=D.NAMES[D.BASE], mode="lines", line=dict(color=T.COLORS[D.BASE], width=1.6, dash="dot"),
                             connectgaps=False, hovertemplate="%{y:.3f} kWh<extra>7-day median</extra>"))
    fig.add_trace(go.Scatter(x=h.index, y=h[model], name=f"{CHOICES[model]} forecast", mode="lines", line=dict(color=T.COLORS[model], width=2),
                             connectgaps=False, hovertemplate="%{y:.3f} kWh<extra>" + CHOICES[model] + "</extra>"))
    fig.add_trace(go.Scatter(x=h.index, y=h["y"], name="Observed", mode="lines+markers", line=dict(color=T.NAVY, width=1.6), marker=dict(size=5),
                             connectgaps=False, hovertemplate="%{y:.3f} kWh<extra>Observed</extra>"))
    fig.update_layout(yaxis=dict(title="Energy 18:30–22:30 (kWh)", rangemode="tozero"), xaxis=dict(tickformat="%d %b"), hovermode="x unified")
    return fig


def error_chart(h: pd.DataFrame, model: str) -> go.Figure:
    e = h.assign(err=h[model] - h["y"], err_base=h[D.BASE] - h["y"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e.date, y=e.err_base, name="7-day median error", marker_color="#CBD5E1", hovertemplate="%{y:+.3f} kWh<extra>7-day median</extra>"))
    fig.add_trace(go.Bar(x=e.date, y=e.err, name=f"{CHOICES[model]} error", marker_color=T.COLORS[model], opacity=.9,
                         hovertemplate="%{y:+.3f} kWh<extra>" + CHOICES[model] + "</extra>"))
    fig.add_hline(y=0, line_color=T.NAVY, line_width=1)
    fig.update_layout(barmode="group", bargap=.25, yaxis=dict(title="Forecast minus observed (kWh)"), xaxis=dict(tickformat="%d %b"), hovermode="x unified")
    return fig


def render():
    T.header("Forecast Explorer", "Saved test-period forecasts against what was observed",
             "Each point is a historical household-evening from the test period. Nothing here is a live forecast.")
    p = D.predictions()
    ex = D.js("examples.json")
    d0, d1 = p.date.min().date(), p.date.max().date()
    T.note(f"Test period {d0:%d %b} – {d1:%d %b %Y}. Household labels are pseudonymised. Days without an audited evening are absent: lines break instead of drawing a zero.")

    c1, c2, c3 = st.columns([1.1, 1.3, 1.4], gap="large")
    model = c1.radio("Model", list(CHOICES), format_func=CHOICES.get, key="ex_model")
    options = [f"{k} ({v})" for k, v in ex.items()] + [ANOTHER]
    choice = c2.selectbox("Household", options, key="ex_hh")
    if choice == ANOTHER:
        hh = c2.selectbox("Pseudonymised household", sorted(p.household.unique()), key="ex_hh_any")
    else:
        hh = ex[choice.split(" (")[0]]
    lo, hi = c3.slider("Date range", min_value=d0, max_value=d1, value=(d0, d1), format="DD MMM", key="ex_range")

    win = p[(p.date.dt.date >= lo) & (p.date.dt.date <= hi)]
    tab1, tab2 = st.tabs(["One household", "All test households"])

    with tab1:
        h = win[win.household == hh].sort_values("date")
        if h.empty:
            st.info("This household has no audited evenings in the selected date range.")
        else:
            e_m, e_b = (h[model] - h.y).abs(), (h[D.BASE] - h.y).abs()
            m = st.columns(4)
            m[0].metric("Days shown", f"{len(h)}")
            m[1].metric("Mean observed", f"{h.y.mean():.2f} kWh", help="Average energy over 18:30-22:30 on the days shown.")
            m[2].metric(f"{CHOICES[model]} MAE", f"{e_m.mean():.3f} kWh", delta=f"{e_m.mean() - e_b.mean():+.3f} vs baseline", delta_color="inverse")
            m[3].metric("Mean error (bias)", f"{(h[model] - h.y).mean():+.3f} kWh", help="Forecast minus observed, averaged. Negative = under-forecasting on average.")
            if len(h) < 20:
                T.note(f"Only {len(h)} days shown: error summaries this small are noisy.", "caution")
            T.plot(household_chart(h, model), 360, key="ex_hh_chart")
            with st.expander("Forecast errors for each day"):
                T.plot(error_chart(h, model), 280, key="ex_err_chart")
                st.caption("Positive = forecast above observed. Grey bars: the 7-day median baseline.")
            st.caption("One household is an illustration, not evidence about the model; across households the gain over the baseline is small (see Insights).")

    with tab2:
        if win.empty:
            st.info("No test rows in the selected date range.")
        else:
            e_m, e_b = (win[model] - win.y).abs(), (win[D.BASE] - win.y).abs()
            m = st.columns(4)
            m[0].metric("Household-days", f"{len(win):,}")
            m[1].metric("Mean observed", f"{win.y.mean():.2f} kWh")
            m[2].metric(f"{CHOICES[model]} MAE", f"{e_m.mean():.3f} kWh", delta=f"{e_m.mean() - e_b.mean():+.3f} vs baseline", delta_color="inverse")
            m[3].metric("Median error", f"{e_m.median():.3f} kWh", delta=f"{e_m.median() - e_b.median():+.3f} vs baseline", delta_color="inverse")
            day = win.groupby("date").agg(Observed=("y", "mean"), model=(model, "mean"), base=(D.BASE, "mean"), n=("y", "size"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=day.index, y=day.base, name=D.NAMES[D.BASE], mode="lines", line=dict(color=T.COLORS[D.BASE], width=1.6, dash="dot")))
            fig.add_trace(go.Scatter(x=day.index, y=day.model, name=f"{CHOICES[model]} forecast", mode="lines", line=dict(color=T.COLORS[model], width=2)))
            fig.add_trace(go.Scatter(x=day.index, y=day.Observed, name="Observed", mode="lines", line=dict(color=T.NAVY, width=1.8),
                                     customdata=day.n, hovertemplate="%{y:.3f} kWh over %{customdata:,} households<extra>Observed</extra>"))
            fig.update_layout(yaxis=dict(title="Mean energy 18:30–22:30 (kWh per household)", rangemode="tozero"), xaxis=dict(tickformat="%d %b"), hovermode="x unified")
            T.plot(fig, 330, key="ex_agg")
            st.caption("Daily average across households. Which households report changes from day to day (see hover), so movements partly reflect who is reporting.")
            cap = float(np.quantile(np.concatenate([e_m, e_b]), 0.99))
            hist = go.Figure()
            for lab, err, col in ((D.NAMES[D.BASE], e_b, T.COLORS[D.BASE]), (CHOICES[model], e_m, T.COLORS[model])):
                hist.add_trace(go.Histogram(x=err.clip(upper=cap), name=lab, marker_color=col, opacity=.7, xbins=dict(start=0, end=cap, size=cap / 40),
                                            histnorm="percent", hovertemplate="%{x:.2f} kWh: %{y:.1f}%<extra>" + lab + "</extra>"))
            hist.update_layout(barmode="overlay", xaxis_title="Absolute forecast error (kWh); the last bar includes the largest 1%", yaxis_title="Share of household-days (%)")
            st.markdown("**How large are the errors?**")
            T.plot(hist, 300, key="ex_hist")
            st.caption("Most evenings are forecast within a fraction of a kWh by both methods; the gap between them is small next to the spread of errors.")
