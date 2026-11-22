"""Model Lab: how the three ensemble ideas differ, how they were evaluated, and what they cost."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import data as D
from .. import diagrams, theme as T
from ..tables import fmt_pct, results_table

FEATURE_LABELS = {"lag1": "yesterday", "lag7": "same weekday last week", "last_value": "last observed evening",
                  "gap_days": "days since last observation", "roll7_median": "7-day median", "roll28_median": "28-day median",
                  "roll28_iqr": "28-day spread (IQR)", "roll28_frac_zero": "share of zero evenings (28 d)",
                  "roll7_n": "days observed (7 d)", "roll28_n": "days observed (28 d)", "expanding_median": "all-history median",
                  "dow": "day of week", "is_weekend": "weekend", "month": "month"}


def fold_timeline(f: dict) -> go.Figure:
    t0 = pd.Timestamp(f["first_date"])
    rows = []
    for i, fo in enumerate(f["cv_folds"], 1):
        v0 = pd.Timestamp(fo["valid_dates"][0]); v1 = pd.Timestamp(fo["valid_dates"][1])
        rows.append((f"Tuning fold {i}", t0, v0 - pd.Timedelta(days=1), v0, v1, T.COLORS["xgb"]))
    sp = f["splits"]
    rows.append(("Validation", t0, pd.Timestamp(sp["train"]["dates"][1]), pd.Timestamp(sp["val"]["dates"][0]),
                 pd.Timestamp(sp["val"]["dates"][1]), T.COLORS["xgb"]))
    rows.append(("Final test", t0, pd.Timestamp(sp["val"]["dates"][1]), pd.Timestamp(sp["test"]["dates"][0]),
                 pd.Timestamp(sp["test"]["dates"][1]), T.BRONZE))
    fig = go.Figure()
    ms = 86400000
    for j, (lab, a, b, c, d, col) in enumerate(rows):
        fig.add_trace(go.Bar(y=[lab], x=[(b - a).days * ms], base=[a], orientation="h", marker_color="#CBD5E1", width=0.5,
                             name="models fitted on", showlegend=(j == 0), legendgroup="fit",
                             hovertemplate=f"fitted on {a:%d %b %Y} to {b:%d %b %Y}<extra></extra>"))
        fig.add_trace(go.Bar(y=[lab], x=[(d - c).days * ms + ms], base=[c], orientation="h", marker_color=col, width=0.5,
                             name="predicted" if j == 0 else None, showlegend=(j == 0), legendgroup="pred",
                             hovertemplate=f"predicted {c:%d %b %Y} to {d:%d %b %Y}<extra></extra>"))
    fig.update_layout(barmode="overlay", xaxis=dict(type="date", tickformat="%b %Y"), yaxis=dict(autorange="reversed", showgrid=False),
                      margin=dict(l=8, r=8, t=30, b=8))
    return fig


def delta_chart(dl: pd.DataFrame, period: str) -> go.Figure:
    d = dl[dl.period == period].set_index("model")
    keys = ["rf", "xgb", "stack"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("Change in MAE vs baseline (kWh)", "Change in RMSE vs baseline (kWh)"))
    for col, m in ((1, "MAE"), (2, "RMSE")):
        for k in keys:
            r = d.loc[k]
            fig.add_trace(go.Scatter(x=[r[f"d_{m}"]], y=[D.NAMES[k]], mode="markers", marker=dict(color=T.COLORS[k], size=10),
                                     error_x=dict(type="data", symmetric=False, array=[r[f"d_{m}_hi"] - r[f"d_{m}"]],
                                                  arrayminus=[r[f"d_{m}"] - r[f"d_{m}_lo"]], color=T.COLORS[k], thickness=1.6, width=5),
                                     showlegend=False, hovertemplate=f"{D.NAMES[k]}<br>{r[f'd_{m}']:+.4f} kWh "
                                                                     f"(95% {r[f'd_{m}_lo']:+.4f} to {r[f'd_{m}_hi']:+.4f})<extra></extra>"), 1, col)
        fig.add_vline(x=0, line_color="#14213D", line_width=1, row=1, col=col)
        fig.update_xaxes(zeroline=False, row=1, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#EEF2F6")
    for a in fig.layout.annotations:
        a.font = dict(size=12.5, color=T.NAVY)
    return fig


def render():
    f, man, specs = D.js("facts.json"), D.js("manifest.json"), D.js("best_specs.json")
    T.header("Model Lab", "Bagging, boosting and stacking",
             "Three ways of combining many decision trees, and how they compare.")
    t1, t2, t3, t4 = st.tabs(["How the models work", "Evaluation design", "Accuracy", "Cost and drivers"])

    with t1:
        rf, xg, xa = specs["rf"], specs["xgb"], specs["xgb_alt"]
        c = st.columns(3, gap="large")
        with c[0]:
            st.markdown('<div class="pw-eyebrow">Random Forest</div>', unsafe_allow_html=True); st.markdown("### Bagging")
            st.markdown(diagrams.bagging(), unsafe_allow_html=True)
            st.markdown("Many deep trees, each grown on a random sample of the history, then **averaged**. Averaging cancels each tree's noise (lower *variance*).")
            st.caption(f"Here: {rf['n_estimators']} trees, minimum leaf {rf['params']['min_samples_leaf']} rows, "
                       f"{int(rf['params']['max_features'] * 100)}% of features per split.")
        with c[1]:
            st.markdown('<div class="pw-eyebrow">XGBoost</div>', unsafe_allow_html=True); st.markdown("### Boosting")
            st.markdown(diagrams.boosting(), unsafe_allow_html=True)
            st.markdown("Shallow trees added **one after another**, each fitted to what the previous ones still get wrong (lower *bias*, but it can overfit).")
            st.caption(f"Here: absolute-error loss (targets the median), depth {xg['params']['max_depth']}, {xg['rounds']:,} rounds, "
                       f"learning rate {xg['params']['learning_rate']}.")
        with c[2]:
            st.markdown('<div class="pw-eyebrow">Meta-learner</div>', unsafe_allow_html=True); st.markdown("### Stacking")
            st.markdown(diagrams.stacking(), unsafe_allow_html=True)
            st.markdown("A second-level model **learns how much to trust** each predictor, using *out-of-fold* predictions: made by models that had not seen the rows they predict.")
            st.caption("Here: a non-negative linear combination of Random Forest, two XGBoost variants and the 7-day median.")

        st.markdown("### What the stacking meta-learner learned")
        wv, wt = man["stack_weights_for_validation"], man["stack_weights_for_test"]
        names = {"rf": "Random Forest", "xgb": "XGBoost (absolute error)", "xgb_alt": "XGBoost (log target)", "baseline_roll7": "7-day median"}
        rows = "".join(f"<tr><td>{names[k]}</td><td>{wv[k]:.3f}</td><td>{wt[k]:.3f}</td></tr>" for k in names)
        rows += f"<tr><td><b>Sum of weights</b></td><td><b>{sum(wv.values()):.3f}</b></td><td><b>{sum(wt.values()):.3f}</b></td></tr>"
        c1, c2 = st.columns([1, 1.25], gap="large")
        with c1:
            st.markdown(f'<table class="pw-table"><thead><tr><th>Input</th><th>Weights used on validation</th><th>Weights used on test</th></tr></thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown("Weights are constrained to be **non-negative but not to sum to one**, so a sum slightly above one is not an error. "
                        "Almost all weight sits on Random Forest, so the stack behaves like Random Forest with a small adjustment. "
                        "The test fit also uses the validation period.")

    with t2:
        st.markdown("### Chronological training, validation and testing")
        T.plot(fold_timeline(f), 330, key="ml_folds")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**What keeps the comparison honest**")
            st.markdown("""
- Features of day D use only dates up to D−1; a test fails if corrupting future targets changes any past feature.
- Tuning used only three chronological folds *inside the training period*.
- Models were compared on validation; the test period was read once, by models refitted on train + validation.
- Stacking uses out-of-fold predictions from the chronological folds.
""")
        with c2:
            st.markdown("**What this design does not prove**")
            st.markdown(f"""
- Test baselines were already visible earlier, so the test period is not blind for the baselines.
- Only households seen in training are evaluated ({f['val_unseen_rows']} validation rows from {f['val_unseen_households']} unseen households are reported separately).
- Validation and test are six-week windows: no seasonality.
""")

    with t3:
        period = st.radio("Period", ["test", "validation"], horizontal=True, format_func=lambda p: {"test": "Test (models refitted on train + validation)",
                                                                                                   "validation": "Validation (models fitted on train only)"}[p])
        m = D.metrics(period)
        cols = [("MAE", "MAE (kWh)", lambda v: f"{v:.4f}", "metric"), ("RMSE", "RMSE (kWh)", lambda v: f"{v:.4f}", "metric"),
                ("MedAE", "Median error (kWh)", lambda v: f"{v:.4f}", "metric"),
                ("MAE_vs_ref_%", "MAE vs baseline", fmt_pct, "delta"), ("RMSE_vs_ref_%", "RMSE vs baseline", fmt_pct, "delta")]
        st.markdown(results_table(m, D.MAIN, cols), unsafe_allow_html=True)
        st.caption(f"{int(m.loc['rf', 'n']):,} household-days from {int(m.loc['rf', 'n_households']):,} households. Bold = lowest in the column. "
                   "MAE is the typical miss in kWh, RMSE penalises large misses, median error ignores the tail.")
        st.markdown("### Could the difference be chance?")
        T.plot(delta_chart(D.deltas(), period), 300, key=f"ml_delta_{period}")
        st.caption("Model minus the 7-day median, with 95% intervals from resampling **households**. Left of the line = better; an interval crossing the line is not distinguishable from no change.")
        with st.expander("More detail: household-level errors, the second XGBoost and other baselines"):
            more = ["xgb_alt", "base_global_median", "base_persistence", "base_seasonal_lag7", "base_roll28_median", "base_expanding_median", D.BASE]
            cols2 = cols[:3] + [("hh_MAE_mean", "Household MAE, mean", lambda v: f"{v:.4f}", "metric"),
                                ("hh_MAE_median", "Household MAE, median", lambda v: f"{v:.4f}", "metric")] + cols[3:]
            st.markdown(results_table(m, more, cols2), unsafe_allow_html=True)
            st.caption("Household MAE is each household's own average miss, averaged (mean) or taken at the middle (median), so large consumers cannot dominate it. "
                       "Every baseline answers on every row through an explicit fallback. Energy in kWh over four hours divided by four is the average power in kW; these are not peak-power forecasts.")

    with t4:
        st.markdown("### Cost and complexity")
        c = D.csv("model_costs_with_latency.csv", index_col="model")
        rows = ""
        for k, label, one, batch in (("rf", "Random Forest", c.loc["rf", "single_row_ms_1thread"], c.loc["rf", "ms_per_1000_rows_1thread"]),
                                     ("xgb", "XGBoost (absolute error)", c.loc["xgb", "single_row_ms"], c.loc["xgb", "ms_per_1000_rows"]),
                                     ("xgb_alt", "XGBoost (log target)", c.loc["xgb_alt", "single_row_ms"], c.loc["xgb_alt", "ms_per_1000_rows"])):
            r = c.loc[k]
            rows += (f"<tr><td>{T.dot(k)}{label}</td><td>{int(r['trees']):,}</td><td>{r['train seconds']:.1f}</td><td>{r['tuning seconds per config (3 folds)']:.0f}</td>"
                     f"<td>{r['size on disk (MB)']:.1f}</td><td>{batch:.1f}</td><td>{one:.2f}</td></tr>")
        rows += (f'<tr class="ref"><td>{T.dot("stack")}Stacking</td><td colspan="6" style="text-align:left">Runs all three models and a four-weight linear combination: '
                 "roughly the sum of the rows above. Not timed separately.</td></tr>")
        st.markdown('<table class="pw-table"><thead><tr><th>Model</th><th>Trees</th><th>Training (s)</th><th>Tuning per setting (s)</th><th>Saved size (MB)</th>'
                    f'<th>Predict 1,000 rows (ms)</th><th>Predict 1 row (ms)</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        st.caption("Measured on one laptop; training time is the fit on the training period (315,079 rows). Random Forest latency is single-threaded. "
                   "Speed is not an issue for a dashboard; the practical cost is complexity.")

        st.markdown("### What the models rely on")
        imp = D.csv("permutation_importance.csv")
        which = st.radio("Model", ["rf", "xgb"], horizontal=True, format_func=lambda k: D.NAMES[k], key="imp_model")
        g = imp[imp.model == which].sort_values("mae_increase")
        fig = go.Figure(go.Bar(y=[FEATURE_LABELS[x] for x in g.feature], x=g.mae_increase, orientation="h", marker_color=T.COLORS[which],
                               error_x=dict(type="data", array=g.mae_increase_sd, color="#94A3B8", thickness=1),
                               hovertemplate="%{y}<br>+%{x:.4f} kWh MAE when shuffled<extra></extra>"))
        fig.update_layout(xaxis_title="Increase in test MAE when the feature is shuffled (kWh)", yaxis=dict(showgrid=False), margin=dict(l=8, r=8, t=8, b=8))
        T.plot(fig, 430, key=f"imp_{which}")
        st.caption(f"Permutation importance on the **test period** ({f['test_rows']:,} household-days), saved final models, baseline MAE {g.baseline_mae.iloc[0]:.4f} kWh. "
                   "Descriptive only: nothing was tuned with it.")
        T.note("<b>Reading it carefully.</b> The recent-level features (7-day median, last observed evening, 28-day median, yesterday) carry almost all the signal. "
               "They are strongly correlated, so single bars understate the <i>group</i> and are not causal effects. Calendar features contribute almost nothing.")
