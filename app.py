"""PeakWise dashboard entry point:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PeakWise", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

from dashboard import data as D          # noqa: E402  (after set_page_config)
from dashboard import theme as T         # noqa: E402

T.inject_css()
BOLT = ('<svg class="pw-bolt" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M13.5 1.5 4 13.6h6.4L9.3 22.5 20 9.6h-6.6z"/></svg>')
st.markdown(f'<div class="pw-brand">{BOLT}<b>PeakWise</b><span>Ensemble learning for next-day evening electricity forecasts</span></div>', unsafe_allow_html=True)

missing = D.missing_artifacts()
if missing:
    st.error("The dashboard's saved result files were not found, so it cannot start.")
    st.markdown(f"Looked in `{D.DATA_DIR}`. Missing: {', '.join(f'`{m}`' for m in missing)}.")
    st.markdown("Create them from the saved Phase 1 and Phase 2 outputs with:")
    st.code("python src/build_dashboard_data.py", language="bash")
    st.stop()

from dashboard.views import explorer, insights, model_lab, overview   # noqa: E402

pages = [st.Page(overview.render, title="Overview", url_path="overview", default=True),
         st.Page(model_lab.render, title="Model Lab", url_path="model-lab"),
         st.Page(explorer.render, title="Forecast Explorer", url_path="forecast-explorer"),
         st.Page(insights.render, title="Insights and Limitations", url_path="insights")]
st.markdown('<span class="pw-banner">Retrospective experiment on saved historical data. Not a live forecast or monitoring service.</span>', unsafe_allow_html=True)
st.navigation(pages, position="top").run()

st.markdown('<div class="pw-foot">PeakWise is an academic ensemble-learning experiment on the RECON-SL dataset (LIRNEasia, CC BY 4.0). '
            "Household labels are pseudonymised. Results describe forecasting for previously observed households and are not a basis for operational decisions.</div>",
            unsafe_allow_html=True)
