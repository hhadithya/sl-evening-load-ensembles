"""Small HTML table helper so the key result tables share one typographic style."""
from __future__ import annotations

import pandas as pd

from .data import BASE, NAMES
from .theme import dot


def fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def results_table(m: pd.DataFrame, rows: list, cols: list, ref: str = BASE) -> str:
    """cols: list of (column, header, formatter, kind) with kind in {'metric', 'delta'}.
    Best (lowest) value per metric column among `rows` is bold; deltas are coloured by sign."""
    best = {c: m.loc[rows, c].min() for c, _, _, k in cols if k == "metric"}
    head = "".join(f"<th>{h}</th>" for _, h, _, _ in cols)
    body = ""
    for r in rows:
        cells = ""
        for c, _, f, kind in cols:
            v = m.loc[r, c]
            cls = ""
            if kind == "metric" and abs(v - best[c]) < 1e-12:
                cls = "best"
            if kind == "delta" and r != ref:
                cls = "good" if v < -0.05 else ("bad" if v > 0.05 else "")
            cells += f'<td class="{cls}">{"–" if (kind == "delta" and r == ref) else f(v)}</td>'
        body += f'<tr class="{"ref" if r == ref else ""}"><td>{dot(r)}{NAMES[r]}</td>{cells}</tr>'
    return f'<table class="pw-table"><thead><tr><th>Model</th>{head}</tr></thead><tbody>{body}</tbody></table>'
