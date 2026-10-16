"""Shared pieces for the book-format write-up (writeup_ch*.py, writeup_build.py).  New file; reads finished runs and bars, changes nothing.

Every item of the book's Contents becomes one dict:
  id        'Figure 2.1', 'Snippet 3.1', 'Equation 32', ... exactly as the Contents names it
  title     a short title (our wording)
  basis     'data' (computed on our Kraken / Alpaca bars or run outputs), 'formula' (closed form or simulation with the book's own inputs), 'code' (a Snippet: where it lives and how it was checked),
            'diagram' (the book's item is a drawing with no data; we show our nearest analogue or say so), 'text' (a section)
  fig       path of a png (optional), table: {'cols': [...], 'rows': [[...]]} (optional), text: str (optional), note: str (what differs from the book / what the book's data would need)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "book_writeup"
FIG = OUT / "fig"
RUNS = {"Kraken": ROOT / "runtime" / "book_bot3" / "kraken_20261015T034703Z", "Alpaca": ROOT / "runtime" / "book_bot3" / "alpaca_20261015T034703Z"}
BARS = {"Kraken": ROOT / "runtime" / "dollar_bars" / "kraken", "Alpaca": ROOT / "runtime" / "dollar_bars" / "alpaca"}
COST = {"Kraken": 12.0, "Alpaca": 10.0}
SYM = {"Kraken": "ETH", "Alpaca": "AAPL"}
VENUES = ("Kraken", "Alpaca")
DAY = 86400

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.grid": False,
                     "axes.prop_cycle": plt.cycler(color=["0.15", "0.45", "0.7", "0.3", "0.55"])})


def fig_path(name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    return str(FIG / f"{name}.png")


def save(fig, name: str) -> str:
    p = fig_path(name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def item(id_, title, basis, fig=None, table=None, text=None, note=None) -> dict:
    return {"id": id_, "title": title, "basis": basis, "fig": fig, "table": table, "text": text, "note": note}


def tbl(cols, rows) -> dict:
    return {"cols": [str(c) for c in cols], "rows": [[("" if v is None else v) for v in r] for r in rows]}


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:,.{nd}f}"
    return str(x)


_cache: dict = {}


def run_json(v: str, name: str):
    return json.loads((RUNS[v] / name).read_text())


def run_csv(v: str, name: str, **kw) -> pd.DataFrame:
    key = (v, name, repr(sorted(kw.items())))
    if key not in _cache:
        p = RUNS[v] / name
        if not p.exists() and name == "events.csv":
            parts = sorted(RUNS[v].glob("events_part*.csv.gz"))
            _cache[key] = pd.concat([pd.read_csv(q, keep_default_na=False, na_values=[""], **kw) for q in parts], ignore_index=True)
        else:
            _cache[key] = pd.read_csv(p, keep_default_na=False, na_values=[""], **kw)
    return _cache[key]


def bars(v: str, sym: str | None = None):
    import bar_loader as BL
    key = ("bars", v, sym)
    if key not in _cache:
        man = json.loads((BARS[v] / "manifest.json").read_text())
        _cache[key] = BL.load_one(BARS[v], sym or SYM[v], int(man["break_hours"]) * 3600)
    return _cache[key]


def symbols(v: str) -> list:
    import bar_loader as BL
    return BL.list_symbols(BARS[v])


def daily_close(v: str, sym: str) -> pd.Series:
    """Last dollar-bar close of each UTC day (the daily series the bot's FFD / SADF / entropy features use)."""
    b = bars(v, sym)
    s = pd.Series(b.close, index=b.idx)
    return s.groupby(s.index.floor("D")).last()


def two_panel(draw, name, w=7.2, h=2.6, titles=None, **kw):
    """draw(ax, venue) for Kraken (left) and Alpaca (right)."""
    fig, axs = plt.subplots(1, 2, figsize=(w, h), **kw)
    for ax, v in zip(axs, VENUES):
        draw(ax, v)
        ax.set_title(f"{v}  ({SYM[v]})" if titles is None else titles[v])
    fig.tight_layout()
    return save(fig, name)
