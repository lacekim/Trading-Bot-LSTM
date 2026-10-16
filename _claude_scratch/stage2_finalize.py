#!/usr/bin/env python3
"""Stage 2: combine the 5 long + 5 short ensemble member checkpoints, apply
the predeclared agreement rule and threshold selection, run the holdout
simulation (with and without the rolling pull-out/re-entry rule), break the
holdout into 3 windows with the June-2026 overlap flag, and check the
promotion bar. No training happens here -- purely combining checkpoints.

PREDECLARED:
- Ensemble: signal fires only if >=4/5 vote positive AND mean ensemble proba
  clears a threshold selected by max-PF scan on calibration.
- Promotion bar: PF>=1.30 AND positive return in EVERY holdout sub-window.
- Pull-out/re-entry: rolling PF over last N=20 closed trades; stop opening
  new trades if rolling PF<1.0; resume once rolling PF recovers >=1.1.
- June 2026 AI-correction window (May15-Jul1 2026) reported separately.

FIX APPLIED (predeclared before this run, in response to the first run's
zero-trades result): "vote positive" no longer means proba>0.5. The
triple-barrier label makes wins the minority outcome (stop 3x ATR vs target
20x ATR), so every model's calibrated output sat under 0.5 everywhere --
a correctly-calibrated model could never cross that line, regardless of
signal quality. Each model now votes positive when its predicted proba on a
row exceeds ITS OWN base rate (sigmoid(base_score_), i.e. the win-rate it
saw in its training bootstrap) -- flagging rows it thinks beat its own
prior, not rows where it's "more likely than not" in an absolute sense that
the label distribution makes nearly impossible to reach. Nothing else
(labeling, split, AGREE_MIN, PF-max threshold scan structure, promotion bar,
pullout rule) changed.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_ENSEMBLE = 5
AGREE_MIN = 4
MIN_CAL_TRADES = 30
MIN_WINDOW_TRADES = 10
ROLL_N = 20
PULLOUT_PF = 1.0
REENTRY_PF = 1.1
JUNE_START = pd.Timestamp("2026-05-15")
JUNE_END = pd.Timestamp("2026-07-01")


def pf_of(returns):
    g = returns[returns > 0].sum(); l = -returns[returns < 0].sum()
    return float(g / l) if l > 0 else (float("inf") if g > 0 else 0.0)


def load_ensemble(direction):
    cal_probs, hold_probs, trees_used, bag_ns, base_rates = [], [], [], [], []
    for i in range(N_ENSEMBLE):
        d = np.load(HERE / f"member_{direction}_{i}.npz")
        cal_probs.append(d["cal_proba"])
        hold_probs.append(d["hold_proba"])
        trees_used.append(int(d["best_n_trees"]))
        bag_ns.append(int(d["bag_n"]))
        base_rates.append(float(d["base_rate"]))
    cal_probs = np.stack(cal_probs, axis=0)
    hold_probs = np.stack(hold_probs, axis=0)
    base_rates = np.array(base_rates)
    return cal_probs, hold_probs, trees_used, bag_ns, base_rates


def main():
    data = np.load(HERE / "data_splits.npz", allow_pickle=True)
    net_cal = {"long": data["net_cal_long"], "short": data["net_cal_short"]}
    net_hold = {"long": data["net_hold_long"], "short": data["net_hold_short"]}
    hold_ts = pd.to_datetime(data["hold_timestamp"])
    hold_sym = data["hold_symbol"]

    results = {}
    ensemble_info = {}
    for direction in ["long", "short"]:
        cal_probs, hold_probs, trees_used, bag_ns, base_rates = load_ensemble(direction)
        # vote positive = this model's proba beats ITS OWN base rate (see FIX APPLIED above)
        cal_votes = (cal_probs > base_rates[:, None]).sum(axis=0)
        cal_p = cal_probs.mean(axis=0)
        hold_votes = (hold_probs > base_rates[:, None]).sum(axis=0)
        hold_p = hold_probs.mean(axis=0)
        ensemble_info[direction] = {"cal_p": cal_p, "cal_votes": cal_votes,
                                     "hold_p": hold_p, "hold_votes": hold_votes,
                                     "trees_used": trees_used, "bag_ns": bag_ns,
                                     "base_rates": base_rates.tolist()}
        print(f"[{direction}] per-model base_rates={np.round(base_rates, 4).tolist()}")

        cal_net = net_cal[direction]
        best_thr, best_pf = None, -1.0
        for thr in np.arange(0.05, 0.96, 0.01):
            mask = (cal_votes >= AGREE_MIN) & (cal_p >= thr) & ~np.isnan(cal_net)
            if mask.sum() < MIN_CAL_TRADES:
                continue
            sel = cal_net[mask]
            gains = sel[sel > 0].sum(); losses = -sel[sel < 0].sum()
            pf = gains / losses if losses > 0 else (np.inf if gains > 0 else 0.0)
            if sel.sum() > 0 and pf > best_pf:
                best_pf, best_thr = pf, thr
        if best_thr is None:
            best_thr = 1.0  # unreachable sentinel -- no threshold ever met MIN_CAL_TRADES with positive sum
        n_cal_fire = int(((cal_votes >= AGREE_MIN) & (cal_p >= best_thr)).sum())
        results[direction] = {"threshold": float(best_thr), "cal_pf": float(best_pf),
                               "cal_trades": n_cal_fire, "trees_used": trees_used, "bag_ns": bag_ns}
        print(f"[{direction}] trees_used per member={trees_used} bag_n per member={bag_ns}")
        print(f"[{direction}] selected threshold={best_thr:.2f} cal_pf={best_pf:.3f} cal_trades={n_cal_fire}")

    long_thr = results["long"]["threshold"]; short_thr = results["short"]["threshold"]
    long_fire = (ensemble_info["long"]["hold_votes"] >= AGREE_MIN) & (ensemble_info["long"]["hold_p"] >= long_thr)
    short_fire = (ensemble_info["short"]["hold_votes"] >= AGREE_MIN) & (ensemble_info["short"]["hold_p"] >= short_thr)
    both = long_fire & short_fire
    long_fire = long_fire & ~both
    short_fire = short_fire & ~both

    n = len(hold_ts)
    trade_return = np.where(long_fire, net_hold["long"], np.where(short_fire, net_hold["short"], np.nan))
    direction_arr = np.where(long_fire, "long", np.where(short_fire, "short", "flat"))

    hold_df = pd.DataFrame({
        "timestamp": hold_ts, "symbol": hold_sym, "direction": direction_arr, "trade_return": trade_return,
    })
    trades = hold_df[hold_df["direction"] != "flat"].dropna(subset=["trade_return"]).reset_index(drop=True)
    print(f"\ntotal holdout rows={n}, trades fired={len(trades)} "
          f"(long={int(long_fire.sum())}, short={int(short_fire.sum())})")

    # rolling pull-out / re-entry
    active = True
    taken = []
    roll_returns = []
    for _, row in trades.iterrows():
        if active:
            taken.append(True)
            roll_returns.append(row["trade_return"])
        else:
            taken.append(False)
        if len(roll_returns) >= ROLL_N:
            recent = np.array(roll_returns[-ROLL_N:])
            g = recent[recent > 0].sum(); l = -recent[recent < 0].sum()
            pf_roll = g / l if l > 0 else (np.inf if g > 0 else 0.0)
            if active and pf_roll < PULLOUT_PF:
                active = False
            elif not active and pf_roll >= REENTRY_PF:
                active = True
    trades["taken_after_pullout"] = taken

    all_returns = trades["trade_return"].to_numpy(np.float64)
    filt_returns = trades.loc[trades["taken_after_pullout"], "trade_return"].to_numpy(np.float64)
    print(f"\nWITHOUT pull-out: trades={len(all_returns)} return_pct={all_returns.sum()*100:.1f} pf={pf_of(all_returns):.3f}")
    print(f"WITH pull-out/re-entry: trades={len(filt_returns)} return_pct={filt_returns.sum()*100:.1f} pf={pf_of(filt_returns):.3f}")

    folds = np.array_split(np.arange(len(trades)), 3)
    windows = []
    promoted_ok = True
    for i, fidx in enumerate(folds, start=1):
        sub = trades.iloc[fidx]
        r = sub["trade_return"].to_numpy(np.float64)
        pf = pf_of(r)
        ret_pct = r.sum() * 100
        lo, hi = (sub["timestamp"].min(), sub["timestamp"].max()) if len(sub) else (None, None)
        overlap = (lo is not None) and (lo <= JUNE_END) and (hi >= JUNE_START)
        ok = len(sub) >= MIN_WINDOW_TRADES and pf >= 1.30 and ret_pct > 0
        promoted_ok = promoted_ok and ok
        windows.append({"window": i, "trades": len(sub), "return_pct": float(ret_pct), "pf": pf,
                         "start": str(lo), "end": str(hi), "june2026_overlap": bool(overlap), "clears_bar": ok})
        tag = "  <-- OVERLAPS June 2026 AI-correction window" if overlap else ""
        print(f"  window {i}: trades={len(sub)} return_pct={ret_pct:.1f} pf={pf:.3f} "
              f"dates=[{lo} -> {hi}] clears_bar={ok}{tag}")

    # explicit standalone June-2026 window report (standing rule: never folded into aggregate)
    june_mask = (trades["timestamp"] >= JUNE_START) & (trades["timestamp"] <= JUNE_END)
    june_trades = trades[june_mask]
    june_r = june_trades["trade_return"].to_numpy(np.float64)
    june_report = {
        "trades": int(len(june_trades)),
        "return_pct": float(june_r.sum() * 100) if len(june_r) else 0.0,
        "pf": pf_of(june_r) if len(june_r) else 0.0,
    }
    print(f"\nJune 2026 AI-correction window (May15-Jul1) STANDALONE: trades={june_report['trades']} "
          f"return_pct={june_report['return_pct']:.1f} pf={june_report['pf']:.3f}")

    promoted = bool(len(all_returns) >= MIN_CAL_TRADES and promoted_ok)
    print(f"\nPROMOTION CHECK (PF>=1.30 AND positive return in every window): promoted={promoted}")

    by_symbol = trades.groupby("symbol")["trade_return"].agg(["count", "sum"]).sort_values("sum")
    print("\nper-symbol holdout contribution:")
    for sym, row in by_symbol.iterrows():
        print(f"  {sym}: n={int(row['count'])} sum_return={row['sum']*100:.1f}%")

    out = {
        "thresholds": results,
        "without_pullout": {"trades": len(all_returns), "return_pct": float(all_returns.sum()*100), "pf": pf_of(all_returns)},
        "with_pullout": {"trades": len(filt_returns), "return_pct": float(filt_returns.sum()*100), "pf": pf_of(filt_returns)},
        "windows": windows,
        "june_2026_standalone": june_report,
        "promoted": promoted,
        "per_symbol": [{"symbol": s, "n": int(r["count"]), "sum_return_pct": float(r["sum"]*100)} for s, r in by_symbol.iterrows()],
    }
    (HERE / "multibot_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote multibot_results.json")


if __name__ == "__main__":
    main()
