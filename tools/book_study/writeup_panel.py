"""Daily simple-return panel per venue, exactly as the bot builds it (book_bot_v2 main), cached to runtime/book_writeup/panel_<venue>.pkl.  New file."""
from __future__ import annotations
import json, sys
import pandas as pd
import bar_loader as BL
import ports_common as PCm
import writeup_core as W

MAX_GAP_DAYS = {"Kraken": 1, "Alpaca": 5}


def panel(v: str) -> pd.DataFrame:
    p = W.OUT / f"panel_{v}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    man = json.loads((W.BARS[v] / "manifest.json").read_text())
    syms = None
    bars, _skipped = BL.load_all(W.BARS[v], syms, 0.5, log=lambda m: None)
    dc = PCm.trading_days(PCm.daily_closes(bars), .8)
    rets = PCm.returns_within_gap(dc, MAX_GAP_DAYS[v], "simple")
    rets = rets[rets.index >= dc.index[0]]
    rets.to_pickle(p)
    return rets


if __name__ == "__main__":
    for v in sys.argv[1:] or W.VENUES:
        r = panel(v)
        print(v, r.shape, r.index[0], r.index[-1], flush=True)
