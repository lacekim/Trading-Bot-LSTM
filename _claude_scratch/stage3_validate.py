#!/usr/bin/env python3
"""Stage 3: apply the multi-agent-validation principle from the article you
sent (independent validators must agree before an output is trusted, or it
gets flagged as unreliable rather than accepted) to the calibration-selected
threshold itself -- not just to the ensemble's own 5-model vote.

WHY THIS STAGE EXISTS: stage2's threshold scan picked long_threshold=0.41
from ONE calibration window and got cal_pf=inf off of exactly 33 trades.
That number was never independently checked against a second, non-overlapping
slice of data before being trusted. This stage is that check.

PREDECLARED METHOD (before looking at the result):
- Split the calibration set in half, chronologically, by row position
  (cal rows are already in pooled chronological order from stage0, so a
  positional split IS a time split): cal_A = first half, cal_B = second half.
- Independently re-run the exact same threshold-scan procedure as stage2 on
  cal_A alone and on cal_B alone (MIN_CAL_TRADES scaled to 15, half of the
  original 30, since each half has half the data).
- A threshold is VALIDATED only if: (a) the two independent halves select
  thresholds in a similar range, AND (b) threshold_A applied out-of-sample to
  cal_B gives positive PF, AND (c) threshold_B applied out-of-sample to cal_A
  gives positive PF. This is "two independent validators must agree" applied
  to a quant signal instead of to an LLM output -- the same principle as the
  linked article, not a new invention.
- If it fails, that's reported as-is. No further threshold adjustment follows
  from this stage regardless of outcome.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_ENSEMBLE = 5
AGREE_MIN = 4
MIN_CAL_TRADES_HALF = 15


def pf_of(returns):
    if len(returns) == 0:
        return 0.0
    g = returns[returns > 0].sum(); l = -returns[returns < 0].sum()
    return float(g / l) if l > 0 else (float("inf") if g > 0 else 0.0)


def load_ensemble(direction):
    cal_probs, base_rates = [], []
    for i in range(N_ENSEMBLE):
        d = np.load(HERE / f"member_{direction}_{i}.npz")
        cal_probs.append(d["cal_proba"])
        base_rates.append(float(d["base_rate"]))
    return np.stack(cal_probs, axis=0), np.array(base_rates)


def scan_threshold(votes, mean_p, net, min_trades):
    best_thr, best_pf, best_n = None, -1.0, 0
    for thr in np.arange(0.05, 0.96, 0.01):
        mask = (votes >= AGREE_MIN) & (mean_p >= thr) & ~np.isnan(net)
        if mask.sum() < min_trades:
            continue
        sel = net[mask]
        pf = pf_of(sel)
        if sel.sum() > 0 and pf > best_pf:
            best_pf, best_thr, best_n = pf, thr, int(mask.sum())
    return best_thr, best_pf, best_n


def apply_threshold(votes, mean_p, net, thr):
    if thr is None:
        return np.array([]), 0
    mask = (votes >= AGREE_MIN) & (mean_p >= thr) & ~np.isnan(net)
    return net[mask], int(mask.sum())


def main():
    data = np.load(HERE / "data_splits.npz", allow_pickle=True)

    for direction in ["long", "short"]:
        print(f"\n===== {direction.upper()} =====")
        cal_probs, base_rates = load_ensemble(direction)
        votes = (cal_probs > base_rates[:, None]).sum(axis=0)
        mean_p = cal_probs.mean(axis=0)
        net = data[f"net_cal_{direction}"]

        n = len(net)
        half = n // 2
        idx_A = np.arange(0, half)
        idx_B = np.arange(half, n)

        votes_A, mean_p_A, net_A = votes[idx_A], mean_p[idx_A], net[idx_A]
        votes_B, mean_p_B, net_B = votes[idx_B], mean_p[idx_B], net[idx_B]

        thr_A, pf_A_in, n_A_in = scan_threshold(votes_A, mean_p_A, net_A, MIN_CAL_TRADES_HALF)
        thr_B, pf_B_in, n_B_in = scan_threshold(votes_B, mean_p_B, net_B, MIN_CAL_TRADES_HALF)

        print(f"cal_A (first half, n={half}): best_thr={thr_A} in-sample pf={pf_A_in} trades={n_A_in}")
        print(f"cal_B (second half, n={n-half}): best_thr={thr_B} in-sample pf={pf_B_in} trades={n_B_in}")

        # cross-check: threshold found on A, applied OUT-OF-SAMPLE to B, and vice versa
        ret_A_on_B, n_A_on_B = apply_threshold(votes_B, mean_p_B, net_B, thr_A)
        pf_A_on_B = pf_of(ret_A_on_B)
        ret_B_on_A, n_B_on_A = apply_threshold(votes_A, mean_p_A, net_A, thr_B)
        pf_B_on_A = pf_of(ret_B_on_A)

        print(f"threshold_A={thr_A} applied to cal_B (out-of-sample): trades={n_A_on_B} pf={pf_A_on_B}")
        print(f"threshold_B={thr_B} applied to cal_A (out-of-sample): trades={n_B_on_A} pf={pf_B_on_A}")

        agree_range = (thr_A is not None and thr_B is not None and abs(thr_A - thr_B) <= 0.10)
        oos_ok = (n_A_on_B >= 3 and pf_A_on_B > 1.0) and (n_B_on_A >= 3 and pf_B_on_A > 1.0)
        validated = agree_range and oos_ok
        print(f"VALIDATED (independent halves agree AND cross-apply profitably)?  {validated}")
        if not validated:
            reasons = []
            if thr_A is None or thr_B is None:
                reasons.append("one half never found a threshold clearing MIN_CAL_TRADES_HALF=15 with positive sum")
            elif not agree_range:
                reasons.append(f"thresholds disagree ({thr_A} vs {thr_B}, >0.10 apart)")
            if thr_A is not None and thr_B is not None and not oos_ok:
                reasons.append("cross-applied threshold failed out-of-sample (too few trades and/or PF<=1 on the other half)")
            print("  reason(s):", "; ".join(reasons) if reasons else "n/a")


if __name__ == "__main__":
    main()
