"""PeakWise design system: palette, CSS, Plotly template and small HTML helpers."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, INK, SLATE, MUTED, LINE, SURFACE = "#14213D", "#1E293B", "#475569", "#8795A8", "#E2E8F0", "#F5F7FA"
TEAL, TEAL_SOFT, BRONZE, AMBER_BG = "#1F7A70", "#E6F2F0", "#B7863B", "#FFF8EC"
COLORS = {"base_roll7_median": "#94A3B8", "rf": "#1F3A5F", "xgb": "#2A9D8F", "xgb_alt": "#8CCBC3",
          "stack": BRONZE, "stack_equal_avg": "#C7CED8"}
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp div[data-testid="stMarkdownContainer"] {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"] {{ font-family: "Material Symbols Rounded", "Material Icons" !important; }}
.stApp {{ background: #FFFFFF; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 4.2rem; padding-bottom: 4rem; }}
h1 {{ font-size: 2.0rem; font-weight: 650; letter-spacing: -0.02em; color: {NAVY}; margin: .2rem 0 .4rem; }}
h2 {{ font-size: 1.35rem; font-weight: 620; letter-spacing: -0.01em; color: {NAVY}; margin-top: 2.2rem; }}
h3 {{ font-size: 1.05rem; font-weight: 620; color: {NAVY}; margin-top: 1.4rem; }}
p, li {{ line-height: 1.62; }}
.pw-brand {{ display:flex; align-items:baseline; gap:.7rem; padding-bottom:.55rem; border-bottom:1px solid {LINE}; margin-bottom:.4rem; }}
.pw-brand b {{ font-size:1.15rem; letter-spacing:-0.01em; color:{NAVY}; font-weight:700; }}
.pw-brand span {{ color:{MUTED}; font-size:.82rem; }}
.pw-eyebrow {{ text-transform:uppercase; letter-spacing:.09em; font-size:.72rem; font-weight:600; color:{TEAL}; margin:.2rem 0 .2rem; }}
.pw-lead {{ font-size:1.08rem; color:{SLATE}; max-width:64ch; line-height:1.62; margin:.2rem 0 1.2rem; }}
.pw-stats {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:1.4rem; margin:1.2rem 0 .6rem; }}
.pw-stat {{ border-top:2px solid {NAVY}; padding-top:.65rem; }}
.pw-stat .k {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; color:{MUTED}; font-weight:600; }}
.pw-stat .v {{ font-size:1.55rem; font-weight:650; color:{NAVY}; letter-spacing:-0.02em; line-height:1.25; }}
.pw-stat .s {{ font-size:.84rem; color:{SLATE}; line-height:1.4; }}
.pw-note {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:6px; padding:.8rem 1rem; color:{SLATE}; font-size:.93rem; margin:.8rem 0; }}
.pw-note b {{ color:{NAVY}; }}
.pw-caution {{ background:{AMBER_BG}; border:1px solid #F0DDB8; border-radius:6px; padding:.8rem 1rem; color:#5B4A1E; font-size:.93rem; margin:.8rem 0; }}
.pw-banner {{ font-size:.8rem; color:{SLATE}; background:{SURFACE}; border:1px solid {LINE}; border-radius:4px; padding:.35rem .7rem; display:inline-block; margin:.1rem 0 .9rem; }}
.pw-table {{ border-collapse:collapse; width:100%; font-size:.92rem; margin:.4rem 0 .8rem; font-variant-numeric:tabular-nums; }}
.pw-table th {{ text-align:right; font-weight:600; color:{SLATE}; font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; padding:.5rem .7rem; border-bottom:2px solid {NAVY}; }}
.pw-table th:first-child, .pw-table td:first-child {{ text-align:left; min-width:11.5rem; white-space:nowrap; }}
.pw-table td {{ text-align:right; padding:.55rem .7rem; border-bottom:1px solid {LINE}; color:{INK}; }}
.pw-table tr.ref td {{ color:{SLATE}; background:{SURFACE}; }}
.pw-table td.best {{ font-weight:700; color:{NAVY}; }}
.pw-table td.good {{ color:{TEAL}; font-weight:600; }}
.pw-table td.bad {{ color:#9A3412; font-weight:600; }}
.pw-dot {{ display:inline-block; width:.62rem; height:.62rem; border-radius:50%; margin-right:.5rem; vertical-align:baseline; }}
.pw-cols2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1.6rem; }}
.pw-two th {{ text-align:left !important; }} .pw-two td {{ text-align:left !important; vertical-align:top; }}
.pw-foot {{ margin-top:3rem; padding-top:.9rem; border-top:1px solid {LINE}; color:{MUTED}; font-size:.78rem; line-height:1.55; }}
div[data-testid="stTabs"] button {{ font-weight:560; }}
div[data-testid="stMetric"] {{ border-top:2px solid {NAVY}; padding-top:.5rem; }}
@media (max-width: 800px) {{ h1 {{ font-size:1.55rem; }} .pw-stats {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} .pw-cols2 {{ grid-template-columns:1fr; }} }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, lead: str = ""):
    st.markdown(f'<div class="pw-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if lead:
        st.markdown(f'<p class="pw-lead">{lead}</p>', unsafe_allow_html=True)


def note(text: str, kind: str = "note"):
    st.markdown(f'<div class="pw-{kind}">{text}</div>', unsafe_allow_html=True)


def stats(items: list):
    cells = "".join(f'<div class="pw-stat"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
                    for k, v, s in items)
    st.markdown(f'<div class="pw-stats">{cells}</div>', unsafe_allow_html=True)


def dot(model_key: str) -> str:
    return f'<span class="pw-dot" style="background:{COLORS.get(model_key, MUTED)}"></span>'


def plot(fig: go.Figure, height: int = 380, key: str = None):
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=12.5, color=INK), paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=8, r=8, t=36, b=8), colorway=[COLORS["rf"], COLORS["xgb"], COLORS["stack"], COLORS["base_roll7_median"]],
        xaxis=dict(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE, zeroline=False, title_font=dict(size=12, color=SLATE)),
        yaxis=dict(gridcolor="#EEF2F6", linecolor="rgba(0,0,0,0)", zeroline=False, title_font=dict(size=12, color=SLATE)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="white", bordercolor=LINE), title=dict(font=dict(size=13, color=NAVY)))
    return t


pio.templates["peakwise"] = _template()
pio.templates.default = "peakwise"
