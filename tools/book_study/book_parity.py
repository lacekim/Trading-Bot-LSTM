"""Parity check: the book's printed Snippets 3.1-3.8 and 4.1-4.2 (verbatim ports in book_port_e.py) run on the SAME bars and the SAME CUSUM events the bot used,
against what the bot's own (vectorised) event code saved in events.csv.  New file; reads a finished run, changes nothing.
Usage: python -W ignore tools/book_study/book_parity.py <run folder> <bars dir> <symbol> <out.txt> [max_events]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import bar_loader as BL
import book_port_e as E


def main(run, bars_dir, sym, out, max_events=3000):
    run, bars_dir = Path(run), Path(bars_dir)
    man = json.loads((bars_dir / "manifest.json").read_text())
    b = BL.load_one(bars_dir, sym, int(man["break_hours"]) * 3600)
    close = pd.Series(b.close, index=b.idx)
    ev = pd.read_csv(run / "events.csv", keep_default_na=False, na_values=[""])
    ev = ev[ev["symbol"] == sym].sort_values("t0").head(max_events).reset_index(drop=True)
    t0 = pd.DatetimeIndex(pd.to_datetime(ev["t0"].to_numpy(np.int64), unit="s"))
    L = []
    # Snippet 3.1 (verbatim) on the bot's own bars
    trgt = E.getDailyVol(close)
    L.append(f"events compared: {len(ev)} ({sym}); bars {len(close)}")
    tg = trgt.reindex(t0).to_numpy()
    d_trgt = np.abs(tg - ev["trgt"].to_numpy())
    L.append(f"Snippet 3.1 daily vol  : verbatim vs bot  max|diff| {np.nanmax(d_trgt):.3e}  events differing >1e-9: {int((d_trgt > 1e-9).sum())}")
    # Snippet 3.4 vertical barrier (4 days), 3.3 triple barrier with side=None, 3.5-3.7 bins
    t1v = E.addVerticalBarrier(close, t0, 4)
    events = E.getEvents(close, t0, [2.0, 2.0], trgt, 0, 1, t1=t1v, side=None)
    bins = E.getBins(events, close)
    bot_t1 = pd.DatetimeIndex(pd.to_datetime(ev["t1"].to_numpy(np.int64), unit="s"))
    m = pd.DataFrame({"t1_book": events["t1"], "ret_book": bins["ret"], "bin_book_snip": bins["bin"]}).reindex(t0)
    m["t1_bot"] = bot_t1
    m["ret_bot"] = ev["ret_side"].to_numpy()
    m["bin_bot"] = ev["bin_book"].to_numpy()
    both = m.dropna(subset=["t1_book"])
    L.append(f"events the verbatim snippets keep: {len(both)} of {len(m)}")
    same_t1 = (both["t1_book"] == both["t1_bot"])
    L.append(f"Snippets 3.3/3.4 exit time: identical for {int(same_t1.sum())} of {len(both)}  ({100 * same_t1.mean():.2f}%)")
    same_bin = (both["bin_book_snip"] == both["bin_bot"])
    L.append(f"Snippets 3.5-3.7 label (sign of return): identical for {int(same_bin.sum())} of {len(both)}  ({100 * same_bin.mean():.2f}%)")
    d = (both["ret_book"] - both["ret_bot"]).abs()
    L.append(f"return at exit: max|diff| {d.max():.3e}; differing >1e-9: {int((d > 1e-9).sum())}")
    bad = both[~same_t1]
    if len(bad):
        L.append("first 10 exit-time disagreements (verbatim t1 / bot t1):")
        for i, r in bad.head(10).iterrows():
            L.append(f"   {i}  {r['t1_book']}  /  {r['t1_bot']}")
    # Second pass: the same verbatim snippets, but fed the bot's own Snippet-3.1 target (isolates the volatility estimate from everything else)
    trgt_bot = pd.Series(ev["trgt"].to_numpy(), index=t0)
    events2 = E.getEvents(close, t0, [2.0, 2.0], trgt_bot, 0, 1, t1=t1v, side=None)
    bins2 = E.getBins(events2, close)
    m2 = pd.DataFrame({"t1_book": events2["t1"], "ret_book": bins2["ret"], "bin_book_snip": bins2["bin"]}).reindex(t0)
    m2["t1_bot"], m2["ret_bot"], m2["bin_bot"] = bot_t1, ev["ret_side"].to_numpy(), ev["bin_book"].to_numpy()
    b2 = m2.dropna(subset=["t1_book"])
    L.append("--- same snippets, given the bot's own target (trgt) ---")
    L.append(f"exit time identical: {int((b2['t1_book'] == b2['t1_bot']).sum())} of {len(b2)}; label identical: {int((b2['bin_book_snip'] == b2['bin_bot']).sum())} of {len(b2)}; "
             f"return at exit max|diff| {(b2['ret_book'] - b2['ret_bot']).abs().max():.3e}")
    bad2 = b2[b2["t1_book"] != b2["t1_bot"]]
    for i, r in bad2.head(10).iterrows():
        L.append(f"   {i}  verbatim {r['t1_book']}  bot {r['t1_bot']}")
    # Snippets 4.1-4.2: uniqueness and sample weights from the verbatim ports vs the bot's u and w_raw
    ev1 = events.dropna(subset=["t1"]).copy()
    nco, tw = E.getTW(ev1, close, 1)
    w = E.getSampleW(ev1, nco, close, 1)
    mm = pd.DataFrame({"u_book": tw, "w_book": w}).reindex(t0)
    mm["u_bot"] = ev["u"].to_numpy()
    mm["w_bot"] = ev["w_raw"].to_numpy()
    ok = mm.dropna()
    ok = ok.iloc[: max(1, len(ok) - 300)]               # drop the last 300 events: the bot counts overlaps with events beyond this sample
    du = (ok["u_book"] - ok["u_bot"]).abs()
    L.append(f"Snippet 4.2 average uniqueness: verbatim vs bot max|diff| {du.max():.3e}; events differing >1e-9: {int((du > 1e-9).sum())} of {len(ok)}")
    ratio = (ok["w_book"] / ok["w_bot"]).replace([np.inf, -np.inf], np.nan).dropna()
    L.append(f"Snippet 4.10 attribution weight: verbatim / bot ratio min {ratio.min():.6f} max {ratio.max():.6f} (a constant ratio = the same weights up to the scaling to the sum)")
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3000)
