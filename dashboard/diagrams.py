"""Three small, consistent SVG diagrams of how bagging, boosting and stacking combine trees."""
from __future__ import annotations

NAVY, TEAL, SLATE, LINE, BRONZE, SURF = "#14213D", "#2A9D8F", "#475569", "#CBD5E1", "#B7863B", "#F5F7FA"


def _wrap(body: str) -> str:
    return (f'<svg viewBox="0 0 330 170" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Inter, sans-serif" font-size="12">'
            f'<defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0 0L8 4L0 8z" fill="{SLATE}"/></marker></defs>{body}</svg>')


def _box(x, y, w, h, label, fill=SURF, stroke=LINE, color=NAVY, weight=500):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 3.5}" text-anchor="middle" fill="{color}" font-weight="{weight}">{label}</text>')


def _arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{SLATE}" stroke-width="1.1" marker-end="url(#a)"/>'


def bagging() -> str:
    b = _box(6, 62, 62, 40, "History", SURF)
    for i, y in enumerate((14, 66, 118)):
        b += _box(100, y, 62, 32, f"Sample {i + 1}") + _arrow(68, 82, 100, y + 16)
        b += _box(190, y, 52, 32, f"Tree {i + 1}", "#E6F2F0", TEAL, NAVY) + _arrow(162, y + 16, 190, y + 16)
        b += _arrow(242, y + 16, 270, 80)
    b += _box(270, 62, 54, 40, "Average", NAVY, NAVY, "white", 600)
    return _wrap(b)


def boosting() -> str:
    b = ""
    xs = (10, 124, 238)
    for i, x in enumerate(xs):
        b += _box(x, 40, 82, 46, f"Tree {i + 1}", "#E6F2F0", TEAL, NAVY, 600)
        if i < 2:
            b += _arrow(x + 82, 63, xs[i + 1], 63)
            b += f'<text x="{x + 98}" y="52" text-anchor="middle" fill="{SLATE}" font-size="10">fits what is</text>'
            b += f'<text x="{x + 98}" y="94" text-anchor="middle" fill="{SLATE}" font-size="10">still wrong</text>'
    b += f'<line x1="51" y1="86" x2="51" y2="128" stroke="{LINE}"/><line x1="165" y1="86" x2="165" y2="128" stroke="{LINE}"/>'
    b += f'<line x1="279" y1="86" x2="279" y2="128" stroke="{LINE}"/><line x1="51" y1="128" x2="279" y2="128" stroke="{LINE}"/>'
    b += _arrow(165, 128, 165, 142)
    b += _box(95, 142, 140, 24, "Sum of small steps", NAVY, NAVY, "white", 600)
    return _wrap(b)


def stacking() -> str:
    b = ""
    for i, (y, t) in enumerate(((10, "Random Forest"), (46, "XGBoost"), (82, "XGBoost (log)"), (118, "7-day median"))):
        b += _box(6, y, 92, 30, t, "#E6F2F0" if i < 3 else SURF, TEAL if i < 3 else LINE) + _arrow(98, y + 15, 138, 82)
    b += _box(138, 52, 92, 60, "Meta-learner<tspan x='184' dy='13' font-weight='400' font-size='10'>non-negative weights</tspan>",
              "#FBF4E6", BRONZE, NAVY, 600)
    b += _arrow(230, 82, 262, 82) + _box(262, 66, 62, 32, "Forecast", NAVY, NAVY, "white", 600)
    b += f'<text x="165" y="160" text-anchor="middle" fill="{SLATE}" font-size="10">meta-learner sees only out-of-fold predictions</text>'
    return _wrap(b)
