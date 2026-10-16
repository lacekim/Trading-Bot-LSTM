"""Writes the Chapter 14 one-year statistics table from the saved ch14_year.json of both venues.  Usage: python book_ch14_report.py <kraken run> <alpaca run> <out.md>"""
import json
import sys
from pathlib import Path

k, a = (json.loads((Path(x) / "ch14_year.json").read_text()) for x in sys.argv[1:3])


def f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


rows = [("Period", f"{k['start']} to {k['end']}", f"{a['start']} to {a['end']}"), ("Gain over the year, net of costs, %", f(k["gain_pct"], 2), f(a["gain_pct"], 4)),
        ("Largest absolute account position (share of capital)", f(k["max_abs_position"]), f(a["max_abs_position"], 5)), ("Mean absolute account position", f(k["mean_abs_position"]), f(a["mean_abs_position"], 5)),
        ("Days with a non-zero position", str(k["days_with_nonzero_position"]), str(a["days_with_nonzero_position"])), ("Bets (Snippet 14.1)", str(k["bets_14_1"]), str(a["bets_14_1"])),
        ("Average holding period, days (14.2)", f(k["avg_holding_period_days_14_2"], 1), f(a["avg_holding_period_days_14_2"], 1)), ("Return of each bet, % (14.1)", ", ".join(f"{v:.4f}" for v in k["bet_returns_pct"]), ", ".join(f"{v:.4f}" for v in a["bet_returns_pct"])),
        ("HHI of positive-return bets (14.3)", f(k["hhi_positive_returns_14_3"]), f(a["hhi_positive_returns_14_3"])), ("HHI of negative-return bets (14.3)", f(k["hhi_negative_returns_14_3"]), f(a["hhi_negative_returns_14_3"])),
        ("HHI of bets per month (14.3)", f(k["hhi_bets_per_month_14_3"]), f(a["hhi_bets_per_month_14_3"])), ("Drawdowns: count / max % / 95th pct % (14.4)", f"{k['drawdowns_14_4']['count']} / {k['drawdowns_14_4']['max_pct']:.3f} / {k['drawdowns_14_4']['p95_pct']:.3f}", f"{a['drawdowns_14_4']['count']} / {a['drawdowns_14_4']['max_pct']:.4f} / {a['drawdowns_14_4']['p95_pct']:.4f}"),
        ("Time under water, years: max / 95th pct (14.4)", f"{f(k['time_under_water_years_14_4']['max'], 3)} / {f(k['time_under_water_years_14_4']['p95'], 3)}", f"{f(a['time_under_water_years_14_4']['max'], 3)} / {f(a['time_under_water_years_14_4']['p95'], 3)}")]
L = ["# Chapter 14 backtest statistics, last 365 days, Kraken and Alpaca", "",
     "Book strategy, walk-forward path, net of costs, HRP allocation. Series: the account's daily target position (sum over instruments of HRP weight x averaged bet signal at the day's last bar) and its daily net return. "
     "Snippets 14.1-14.4 as ported. 'n/a' = the snippet returns nothing (a HHI needs more than two bets, and negative-return bets must exist).", "", "| | Kraken | Alpaca |", "|---|---|---|"]
L += [f"| {a_} | {b_} | {c_} |" for a_, b_, c_ in rows]
L += ["", "Not computed: the other Chapter 14 measures (average AUM, capacity, leverage, turnover, hit ratio, implementation shortfall) are described in the book's text, not given as snippets in the extract.", ""]
Path(sys.argv[3]).write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
