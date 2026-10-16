"""Three algorithms (long, short, momentum) and a gate that picks one per event, on DAILY equity bars (Alpaca).

This module is three_algo.py (md5 recorded by the runner) executed with exactly three textual substitutions, each of which must occur once in that file or this module stops:
  1. `import meta_events as E`                         -> `import equity_events as E`   (the 13 daily features, side first; DIR_FEATURES = the 12 without 'side')
  2. `t_eff_div = max(1.0, (h + 1) / 24.0)`            -> `t_eff_div = max(1.0, float(h + 1))`   (bars are days: the effective sample is days / holding period in days)
  3. `math.sqrt(365)`                                  -> `math.sqrt(252)`              (annualisation of a daily Sharpe; trading days)
After that only these names are replaced below: the module text, MOM_LOOKBACKS, mom_side_from, extend_events, PRIOR_TRIALS, DEVIATIONS.  Everything else (the forests' CPCV and walk-forward
loops, gate_choice, build_variants, evaluate_three) is three_algo.py as it is.

DESIGN (fixed before any result of this module was seen; the same text is printed in SUMMARY.md):

  Events         the event table of the Alpaca daily meta run (run_meta_daily.py): CUSUM events on daily bars, entry at the next bar's close, triple-barrier
                 exit (symmetric +/-2 daily-vol barriers on the simple return, vertical barrier 5 bars after entry).  Because the barriers are symmetric, a
                 short held over the same bars exits at the same bar as a long with the opposite sign: ret_long = side * ret_side (side = sign of the
                 project's momentum score in the source run).
  Algorithms     LONG  = side +1 at every event.
                 SHORT = side -1 at every event.
                 MOM   = side +1 if the 12-1 month return C[t-21] / C[t-252] - 1 AND the 6-1 month return C[t-21] / C[t-126] - 1 (t = the signal bar, the bar
                         before entry) are both > 0, side -1 if both are < 0, otherwise no signal.  These are the project's two lookbacks (tools/run_alpaca_momentum_test.py);
                         the source run's side is the sign of their average, so on MOM events the MOM side equals the source side and MOM is the 'both agree' subset.
  Gate models    F_dir: the book's bagged forest on label y_dir = 1 when ret_long > 0 (gross), 12 features (the source 13 minus 'side'), trained on every training
                 event.  A forest for SHORT on the mirrored label would be the same model, so p(SHORT) = 1 - p(LONG).
                 F_mom: the same forest on the MOM events only, label 'bin' of the source run (the MOM side's gross return > 0), 13 features (the side feature is
                 the MOM side).
  Gate           at each event the candidates are LONG with confidence p_dir, SHORT with 1 - p_dir and, when MOM has a signal, MOM with its side and confidence
                 p_mom.  The candidate with the largest confidence is picked (ties: MOM, then LONG, then SHORT); the bet is taken only if that confidence exceeds
                 1/2, with size from Snippets 10.1 / 10.3 (step 0.1).
  Variants       long_all, short_all, mom_all (size 1 on all MOM events), mom_long_up (a fixed rule: MOM side +1 while the equal-weight index is above its own
                 200-day mean, size 1, nothing otherwise), dir_model (side by p_dir, size by its confidence), mom_meta (MOM events sized by p_mom), gate; two
                 nulls for the gate: gate_shuffled_size (the gate's sides, its sizes shuffled among all events) and gate_random_side (the gate's sizes with
                 coin-flip sides).
  Evaluation     as in meta_eval_daily: one unit of capital per candidate event, daily result = sum of size * (side * ret_long - cost * [size > 0]) over the day's
                 events divided by ALL candidate events that day, in bp; annualised Sharpe = daily Sharpe x sqrt(252); effective sample = days with events / 5;
                 CPCV N=6, k=2 (15 splits, 5 paths); a walk-forward fit on labels ending before 2024-01-01; June 2026, the sell-off and the deepest-fall windows;
                 by market state.
  Trials         5 new variants carry a choice made here (mom_all's agreement rule, mom_long_up, dir_model, mom_meta, gate); the deflated Sharpe uses
                 prior_trials = 120 (the running count of 116 after the Kraken three-algorithm study, plus 4) and shows N = 1, 10, 120 and 121 (the gate itself).
  Not claimed    mom_all and mom_long_up use rules chosen after seeing earlier history on other venues (the 200-day state came from GMX diagnostics), and the
                 universe is today's survivors, so the long side is flattered; only the deflated Sharpe tempers the former, nothing tempers the latter.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "three_algo.py"
_SUBSTITUTIONS = [
    ("import meta_events as E", "import equity_events as E"),
    ("t_eff_div = max(1.0, (h + 1) / 24.0)", "t_eff_div = max(1.0, float(h + 1))"),
    ("math.sqrt(365)", "math.sqrt(252)"),
]
_DAILY_DOC = __doc__
_src = _SRC_PATH.read_text(encoding="utf-8")
BASE_MD5 = hashlib.md5(_SRC_PATH.read_bytes()).hexdigest()
for _old, _new in _SUBSTITUTIONS:
    if _src.count(_old) != 1:
        raise SystemExit(f"three_algo.py has changed: the text {_old!r} occurs {_src.count(_old)} times, expected once; stopping")
    _src = _src.replace(_old, _new)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
exec(compile(_src, str(_SRC_PATH), "exec"), globals())          # defines everything of three_algo.py in this module's namespace
__doc__ = _DAILY_DOC

# ---------------------------------------------------------------------------------------------------------------------
# replaced names
# ---------------------------------------------------------------------------------------------------------------------
LB12, LB6, SKIP = E.LB12, E.LB6, E.SKIP                         # 252, 126, 21 (equity_events, the project's momentum score)
MOM_LOOKBACKS = (LB12, LB6)
MOM_COLUMNS = ("ret12_1", "ret6_1")
PRIOR_TRIALS = 120

DEVIATIONS = [
    "The book has no neural network. The 'which algorithm' decision is made by the book's own bagged forests (secondary models); the neural-network overseer of the intended architecture is not built here.",
    "LONG and SHORT are the two signs of one decision; a forest on the long label and one on the mirrored short label would be the same model, so one forest (F_dir) is fitted and p(SHORT) = 1 - p(LONG).",
    "MOM here is the subset of events on which the project's 12-1 and 6-1 month returns agree in sign; the source run's side is the sign of their average, so on those events MOM's side equals the source side and mom_all is the primary strategy restricted to those events. It is not an independent signal as the 20 / 30 / 40-day rule is on the hourly venues.",
    "The gate compares confidences of two different forests (F_dir on all events, F_mom on the MOM events). Both use balanced class weights, which centres their probabilities near 1/2 whatever the base rate, but the confidences are not guaranteed to be comparable; the run reports each forest and each algorithm separately so the gate can be judged against them.",
    "F_mom is trained on the MOM events only but with the source run's uniqueness and attribution weights (computed over all events, so they treat the non-MOM events as concurrent); sequential bootstrap and sample weights are otherwise as in the source run.",
    "mom_all and mom_long_up use rules chosen after seeing earlier history on other venues (the 200-day state came from GMX diagnostics). Cross-validation cannot remove that; it only removes the fitting of the forests.",
    "Long and short exits are identical by construction (symmetric barriers on the simple return); no separate short-side barrier study. The round-trip cost of 20 bp is charged on every taken bet; shorts carry no borrow cost, no locate and no short-sale restriction, and there are no capacity or liquidity limits.",
    "SURVIVORSHIP: the stock list is Alpaca's current active, tradable assets; delisted, bankrupt and acquired companies are almost certainly missing. This flatters long momentum in particular; every figure here is an optimistic bound on that count.",
    "The universe is the top 200 names by trailing dollar volume at each month-end (the project uses 1000), and the daily file has no high or low, so no Corwin-Schultz feature exists.",
    "A label needs 5 bars after entry, so events entered in the last 5 trading days of the file cannot be labelled: a reporting window that falls there has no events and is reported n/a, not filled in.",
    "Trial count: the prior of 120 counts the running total of 116 after the Kraken three-algorithm study plus the four choices made here; the order (Kraken first, Alpaca second) is a convention declared before the run.",
    "All source-run deviations apply (daily time bars, CUSUM / triple-barrier / FFD / SADF / entropy choices, a single universe screen, adjusted prices).",
]


def mom_side_from(r12, r6):
    """+1 if both trailing returns are > 0, -1 if both are < 0, else 0 (NaN anywhere gives 0)."""
    r12, r6 = (np.asarray(x, dtype=np.float64) for x in (r12, r6))
    up = (r12 > 0) & (r6 > 0)
    dn = (r12 < 0) & (r6 < 0)
    return np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)


def extend_events(df: pd.DataFrame, closes: dict) -> pd.DataFrame:
    """Adds ret12_1 = C[s-21] / C[s-252] - 1 and ret6_1 = C[s-21] / C[s-126] - 1 (s = pos - 1, the signal bar; NaN when fewer than 253 bars exist), mom_side, ret_long = side * ret_side and
    y_dir = 1[ret_long > 0].  `closes[symbol]` must be the series (closes > 0) the source run used, cut at its last recorded bar, so that df['pos'] indexes it; the signal bar's date must equal
    the event's t_sig."""
    out = df.copy()
    n = len(out)
    c12, c6 = np.full(n, np.nan), np.full(n, np.nan)
    for sym, grp in out.groupby("symbol", sort=False):
        c = closes[sym].astype(float)
        c = c[c > 0]
        cs = c.to_numpy()
        ix = grp.index.to_numpy()
        p = grp["pos"].to_numpy(np.int64)
        if p.max() >= len(cs):
            raise ValueError(f"{sym}: event position beyond the candle series")
        s = p - 1
        t_sig = c.index.values[s].astype("datetime64[s]").astype(np.int64)
        if not np.array_equal(t_sig, grp["t_sig"].to_numpy(np.int64)):
            raise ValueError(f"{sym}: the bar before the entry bar is not the event's signal bar (t_sig); the closes are not the series the source run used")
        ok = s >= LB12
        a = cs[s[ok] - SKIP]
        v12, v6 = np.full(len(p), np.nan), np.full(len(p), np.nan)
        v12[ok] = a / cs[s[ok] - LB12] - 1.0
        v6[ok] = a / cs[s[ok] - LB6] - 1.0
        c12[ix], c6[ix] = v12, v6
    out[MOM_COLUMNS[0]], out[MOM_COLUMNS[1]] = c12, c6
    out["mom_side"] = mom_side_from(out[MOM_COLUMNS[0]], out[MOM_COLUMNS[1]])
    out["ret_long"] = out["side"].to_numpy(np.float64) * out["ret_side"].to_numpy(np.float64)
    out["y_dir"] = (out["ret_long"] > 0).astype(np.int8)
    return out
