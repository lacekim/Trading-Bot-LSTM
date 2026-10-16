#!/usr/bin/env python
"""Three algorithms (long, short, momentum) and the forest gate, run on KRAKEN hourly candles (read-only diagnostic).

This is run_three_algos.py / three_algo.py UNCHANGED (same algorithms, forests, gate, variants, nulls, CPCV, walk-forward, windows, evaluation; their md5 is recorded), pointed at
the Kraken candle folder named in a finished 14-feature Kraken run (run_meta_venue.py / run_meta_volume.py with the 14 features) instead of GMX + TradingView files.  The only
additions are: the candle reader (venue_loader.load_closes), the declared trial count, a diagnostic of how many events have a lookback whose real time span differs from the intended
number of days by more than 10% (Kraken's trade-built candles have gaps, so 480 / 720 / 960 bars are not always 20 / 30 / 40 days), and a venue section appended to SUMMARY.md.

Trial count: the running count is 111 (look 108 after the hyperparameter search, plus three comparisons for the allocation study); the four fixed-rule / model variants that carry a
choice add 4 -> 115 prior trials, the gate is the 116th; the deflated Sharpe is shown for N = 1, 10, 115, 116.

    python tools/book_study/run_three_algos_venue.py --source "$(ls -d runtime/book_seed_pairs/volume_vs_14/ctrl_20261017/*/ | tail -1)"
    python tools/book_study/verify_three_algos_venue.py "$(ls -d runtime/book_three_algo_kraken/*/ | tail -1)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_three_algos as RT  # noqa: E402
import three_algo as T  # noqa: E402
import venue_loader as VL  # noqa: E402

PRIOR_TRIALS_VENUE = 115
SPAN_TOLERANCE = 0.10


class _VenueCandles:
    """Stands in for candle_loader: the two folder arguments are the same Kraken folder; the second is ignored."""

    @staticmethod
    def load_closes(venue_dir, _unused_dir, names, timeframe="1h", min_bars=0):
        return VL.load_closes(venue_dir, names, timeframe, min_bars)


def span_diagnostic(df: pd.DataFrame, closes: dict, cutoff: str) -> dict:
    """For each lookback L (bars) the share of events (with pos >= L) whose time span ts[pos] - ts[pos - L] differs from L hours by more than SPAN_TOLERANCE, overall and for events starting
    on / after the cutoff."""
    cut = float(pd.Timestamp(cutoff).timestamp())
    out = {}
    for L in T.MOM_LOOKBACKS:
        n = bad = n_post = bad_post = 0
        for sym, g in df.groupby("symbol", sort=False):
            ts = closes[sym].index.astype("datetime64[s]").astype(np.int64)
            p = g["pos"].to_numpy(np.int64)
            t0 = g["t0"].to_numpy(np.float64)
            ok = p >= L
            span_h = (ts[p[ok]] - ts[p[ok] - L]) / 3600.0
            off = np.abs(span_h / L - 1.0) > SPAN_TOLERANCE
            n += int(ok.sum())
            bad += int(off.sum())
            post = t0[ok] >= cut
            n_post += int(post.sum())
            bad_post += int((off & post).sum())
        out[str(L)] = {"events_with_lookback": n, "events_span_off": bad, "share_off": bad / n if n else float("nan"),
                       "events_post_cutoff": n_post, "events_post_cutoff_span_off": bad_post, "share_off_post_cutoff": bad_post / n_post if n_post else float("nan")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="timestamp folder of a 14-feature Kraken run (events.csv, symbols.csv, summary_numbers.json)")
    ap.add_argument("--out", default="runtime/book_three_algo_kraken")
    ap.add_argument("--quick", action="store_true", help="smoke test only (4 groups, 10 trees)")
    ap.add_argument("--seed", type=int, default=RT.SEED)
    ap.add_argument("--prior-trials", type=int, default=PRIOR_TRIALS_VENUE)
    ap.add_argument("--candle-dir", default=None, help="default: the candle folder recorded in the source run")
    a = ap.parse_args()
    src = Path(a.source)
    src = (src if src.is_absolute() else ROOT / src).resolve()
    s_summ = json.loads((src / "summary_numbers.json").read_text(encoding="utf-8"))
    if s_summ["meta"].get("venue") != "kraken":
        raise SystemExit(f"the source run's venue is {s_summ['meta'].get('venue')!r}, not 'kraken'")
    cd = Path(a.candle_dir) if a.candle_dir else Path(s_summ["meta"]["candle_dir"])
    cd = (cd if cd.is_absolute() else ROOT / cd).resolve()
    out_root = Path(a.out)
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    before = {p.name for p in out_root.glob("*")} if out_root.exists() else set()
    RT.CL = _VenueCandles
    argv = ["run_three_algos.py", "--source", str(src), "--out", str(out_root.relative_to(ROOT)) if out_root.is_relative_to(ROOT) else str(out_root), "--seed", str(a.seed),
            "--prior-trials", str(a.prior_trials), "--candle-dirs", str(cd), str(cd)]
    if a.quick:
        argv.append("--quick")
    sys.argv = argv
    rc = RT.main()
    new = sorted(p for p in out_root.glob("*") if p.name not in before and (p / "summary_numbers.json").exists())
    if rc != 0 or not new:
        return rc or 1
    out = new[-1]
    # ---- venue additions: span diagnostic, extra file, summary section
    df = pd.read_csv(src / "events.csv")
    sym_rows = pd.read_csv(src / "symbols.csv")
    names = sorted(sym_rows["symbol"].astype(str))
    closes_all, _ = VL.load_closes(cd, names, s_summ["meta"]["timeframe"], 0)
    closes = {str(r["symbol"]): closes_all[str(r["symbol"])][closes_all[str(r["symbol"])].index <= pd.Timestamp(r["last"])] for _, r in sym_rows.iterrows()}
    spans = span_diagnostic(df, closes, s_summ["params"]["cutoff"])
    extra = {"venue": "kraken", "candle_dir": str(cd), "source_run": str(src), "prior_trials": a.prior_trials, "prior_trials_note": "111 running count + 4 variants carrying a choice; gate = 116th",
             "span_tolerance": SPAN_TOLERANCE, "span_diagnostic": spans, "script_md5": {n: __import__("run_book_study").md5_file(HERE / n) for n in ("run_three_algos_venue.py", "venue_loader.py")}}
    (out / "venue_extra.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    L = ["", "## Venue: Kraken hourly candles", "",
         f"Source events: {src}. Candles: {cd}. The study code is unchanged from the GMX / TradingView version; only the candle reader and the trial count differ (prior trials {a.prior_trials}; the gate is the {a.prior_trials + 1}th).",
         "Bars are Kraken trade-built hourly candles with gaps, so 480 / 720 / 960 bars are not always 20 / 30 / 40 days. Events whose lookback span differs from the intended number of days by more than 10%:", "",
         "| lookback (bars) | events with the lookback | span off by > 10% | share | after the cutoff | share after the cutoff |", "|---|---|---|---|---|---|"]
    for k, v in spans.items():
        L.append(f"| {k} | {v['events_with_lookback']} | {v['events_span_off']} | {v['share_off']:.4f} | {v['events_post_cutoff']} | {v['share_off_post_cutoff']:.4f} |")
    L += ["", "The DESIGN block at the top of this file is printed unchanged from the GMX / TradingView version, so its trial sentence (83 + 4 = 87 prior trials; N = 87 and 88) and its '123,768-style' event count describe that version. "
          f"This run's values are the parameter list and tables above: prior trials {a.prior_trials}, the gate the {a.prior_trials + 1}th, N = 1, 10, {a.prior_trials}, {a.prior_trials + 1}.", ""]
    L += ["The momentum lookbacks and the 200-day market state were chosen on GMX history, not on Kraken; the post-cutoff Kraken period was seen in earlier runs. One seed: the seed spread is not estimated. "
          "Kraken limits of the source run apply (survivorship in the symbol list, trade-built candles, BTC and ICX not refreshed, shorts without borrow cost).", ""]
    with open(out / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"venue additions written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
