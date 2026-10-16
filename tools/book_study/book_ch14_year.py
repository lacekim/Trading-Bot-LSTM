"""Chapter 14 backtest statistics (Snippets 14.1 bet timing, 14.2 holding period, 14.3 HHI concentration, 14.4 drawdown and time under water; book_port_c as ported) for the last 365 days of the book strategy, one venue per run.
Series used: the account's daily target position (sum over instruments of HRP weight x that instrument's averaged bet signal at the day's last bar; signed, in [-1, 1]) and the account's daily net return, both for the walk-forward path of the book_final run.
The daily return series is recomputed here with book_final's own functions and must equal the saved equity curve.  New file; nothing audited is edited.
Usage (from the project root): python tools/book_study/book_ch14_year.py <book_final run folder>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import bet_averaging as BA  # noqa: E402
import book_final as BF  # noqa: E402
import book_otr_exit as OX  # noqa: E402
import book_port_c as PC  # noqa: E402
import meta_events as E  # noqa: E402
import meta_model as MM  # noqa: E402
import ports_common as PCm  # noqa: E402
import run_bet_averaging as RBA  # noqa: E402

DAY = 86400


def main(run: Path) -> int:
    meta = json.loads((run / "meta.json").read_text())
    src, ports, bar_dir = Path(meta["source_run"]), Path(meta["ports_run"]), Path(meta["bar_dir"])
    summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    venue, p = summ["meta"]["venue"], summ["params"]
    step, cost_frac = float(p["step"]), float(p["cost_frac"])
    pt, sl, K = meta["otr"]["pt"], meta["otr"]["sl"], meta["otr"]["K"]
    V = PCm.Venue(venue, src, bar_dir, 40)
    ev0 = V.events.copy()
    r_long = ev0["ret_side"].to_numpy(float) * ev0["side"].to_numpy(float)
    keep = r_long != 0.0
    ev = ev0[keep].reset_index(drop=True)
    ev["ret_side"], ev["side"] = r_long[keep], 1.0
    ev["bin"] = (ev["ret_side"] > 0).astype(np.int8)
    bars = V.bars(sorted(ev["symbol"].unique()))
    closes = {s: pd.Series(b.close, index=b.idx) for s, b in bars.items()}
    RBA.check_alignment(ev, closes)
    E.FEATURES = list(BF.FEATURES_BOOK)
    wpred = pd.read_csv(run / "walk_forward_predictions.csv")
    w_idx, pp = wpred["event"].to_numpy(np.int64), wpred["p"].to_numpy(float)
    side = np.where(pp > 0.5, 1.0, -1.0)
    size = MM.bet_size(np.maximum(pp, 1.0 - pp), step)
    t1pos, _ = OX.otr_exits(bars, ev, w_idx, side, pt, sl, K)
    sig = side * size
    # ---- weights exactly as book_final ----
    dc = PCm.trading_days(PCm.daily_closes(bars), .8)
    rets = PCm.returns_within_gap(dc, V.max_gap_days, "simple")
    rets = rets[rets.index >= dc.index[0]]
    nets = BF.instrument_nets(ev, closes, w_idx, side, size, t1pos, cost_frac / 2.0)
    all_days = [d for (days, _n, _a) in nets.values() for d in (int(days.min()), int(days.max()))]
    last_day, first_day_all = max(all_days), min(all_days)
    first_event_day = int(ev["t0"].min() // DAY)
    sched = BF.weight_schedule(rets, min(first_day_all, first_event_day) + BF.REB_WINDOW, last_day, False)
    lo = last_day - 364
    r, act = BF.account(nets, sched, lo, last_day)
    saved = pd.read_csv(run / "equity_curves_walk_forward.csv")
    saved = saved[(saved["span"] == "last_365_days")]
    assert len(saved) == len(r) and np.allclose(saved["ret"].to_numpy(), r, rtol=0, atol=1e-11), "recomputed daily returns differ from the saved equity curve"
    # ---- the account's daily target position ----
    keys = sorted(sched)
    pos_ev = ev["pos"].to_numpy(np.int64)
    sym_ev = ev["symbol"].to_numpy()
    E_day = np.zeros(last_day - lo + 1)
    for s, c in closes.items():
        sel = np.flatnonzero(sym_ev[w_idx] == s)
        if not len(sel):
            continue
        ts = c.index.astype("datetime64[s]").astype(np.int64)
        avg, _C = BA.average_positions(len(c), pos_ev[w_idx][sel], t1pos[sel], sig[sel])
        day = ts // DAY
        last_of_day = pd.Series(avg).groupby(day).last()                     # position at the day's last bar
        pos_d = last_of_day.reindex(np.arange(min(lo, int(day.min())), last_day + 1)).ffill().fillna(0.0)
        for k, st in enumerate(keys):
            en = keys[k + 1] if k + 1 < len(keys) else last_day + 1
            w = float(sched[st].get(s, 0.0)) if len(sched[st]) else 0.0
            if w == 0.0:
                continue
            a_, b_ = max(st, lo), min(en, last_day + 1)
            if a_ < b_:
                E_day[a_ - lo:b_ - lo] += w * pos_d.loc[a_:b_ - 1].to_numpy()
    idx = pd.to_datetime(np.arange(lo, last_day + 1) * DAY, unit="s")
    tPos = pd.Series(E_day, index=idx)
    ret_d = pd.Series(r, index=idx)
    equity = (1.0 + ret_d).cumprod()
    # ---- Snippets 14.1 - 14.4 ----
    bets = PC.getBets(tPos)
    hp = PC.getHoldingPeriod(tPos)
    prev = tPos.index[0]
    bet_ret = []
    for b in bets:
        seg = ret_d.loc[(ret_d.index > prev) & (ret_d.index <= b)] if prev != tPos.index[0] else ret_d.loc[ret_d.index <= b]
        bet_ret.append((b, float((1.0 + seg).prod() - 1.0)))
        prev = b
    bet_ret = pd.Series([v for _, v in bet_ret], index=[b for b, _ in bet_ret])
    hhi_pos, hhi_neg, t_hhi = PC.getHHIs(bet_ret)
    dd, tuw = PC.computeDD_TuW(equity)
    res = {"venue": venue, "start": str(idx[0].date()), "end": str(idx[-1].date()), "days": int(len(idx)), "gain_pct": 100.0 * float(equity.iloc[-1] - 1.0),
           "days_with_nonzero_position": int((tPos != 0).sum()), "max_abs_position": float(tPos.abs().max()), "mean_abs_position": float(tPos.abs().mean()),
           "bets_14_1": int(len(bets)), "bet_end_dates": [str(b.date()) for b in bets][:50], "avg_holding_period_days_14_2": None if not np.isfinite(hp) else float(hp),
           "bet_returns_pct": [round(100 * v, 6) for v in bet_ret.tolist()][:50], "hhi_positive_returns_14_3": None if not np.isfinite(hhi_pos) else float(hhi_pos),
           "hhi_negative_returns_14_3": None if not np.isfinite(hhi_neg) else float(hhi_neg), "hhi_bets_per_month_14_3": None if not np.isfinite(t_hhi) else float(t_hhi),
           "drawdowns_14_4": {"count": int(len(dd)), "max_pct": 100.0 * float(dd.max()) if len(dd) else 0.0, "p95_pct": 100.0 * float(np.percentile(dd, 95)) if len(dd) else 0.0},
           "time_under_water_years_14_4": {"count": int(len(tuw)), "max": float(tuw.max()) if len(tuw) else None, "p95": float(np.percentile(tuw, 95)) if len(tuw) else None}}
    tPos.rename("target_position").to_csv(run / "ch14_target_position.csv", float_format="%.12g")
    ret_d.rename("net_return").to_csv(run / "ch14_daily_return.csv", float_format="%.12g")
    (run / "ch14_year.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
