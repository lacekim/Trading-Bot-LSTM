"""
CRITIC (third stage after Executor = the test scripts, Validator = verify_engine_independent.py).
Deterministic, no LLM. OFFLINE: reads only saved result JSONs, script files and the cached ETF price file.
Writes runtime/critic_report.json. Touches no live-bot file.

WHAT IT DOES
  A. VALIDATOR LINK : engine_verification_results.json says all checks passed AND was run on the same cached
                      price file that exists now (md5 equal).
  B. SPEC LOCK      : each test script on disk has the md5 recorded when it was delivered, and its file time is
                      not later than the results file it produced (no edit after seeing results).
  C. VERDICT RECOMPUTE: every predeclared pass/fail condition is recomputed from raw saved numbers and compared
                      with (1) the stored verdict flags and (2) the verdict stated to the user.
  D. CLAIMS LEDGER  : every number quoted to the user in the last three reports (equity momentum, paper-baseline
                      TSMOM, TSH control) is checked against the saved JSON, at the precision it was quoted.
                      Claims that the saved files cannot confirm are listed as UNVERIFIABLE, never passed silently.
RULE: REJECTED if any claim MISMATCHES, any verdict is inconsistent, the validator failed or ran on different
data, or a script changed. Otherwise APPROVED, and the count of UNVERIFIABLE claims is stated.
LIMITS: it checks reported numbers against saved results. It cannot judge explanations or reasoning, claims about
external papers, or numbers that were only printed to the terminal. A ledger only covers claims written into it.

Run (project root, `javier` env):  python tools/critic_claims_check.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tools" \
    else Path(__file__).resolve().parent
RUNTIME = PROJECT_ROOT / "runtime"
TOOLS = PROJECT_ROOT / "tools"
OUT_PATH = RUNTIME / "critic_report.json"

FILES = {
    "mom": RUNTIME / "alpaca_momentum_results.json",
    "paper": RUNTIME / "alpaca_tsmom_paper_results.json",
    "tsh": RUNTIME / "alpaca_tsh_control_results.json",
    "verify": RUNTIME / "engine_verification_results.json",
}
# script md5 as DELIVERED to the user, and which results file each produced
SCRIPTS = {
    "run_alpaca_momentum_test.py": ("89b8cb43d076a03220a9b0fe53c7ffc0", "mom"),
    "run_alpaca_tsmom_paper.py": ("6a893c2dcef81a1111324a85a6f604c6", "paper"),
    "run_tsh_control_test.py": ("c44e692e8828945fdb53d4223ac2df41", "tsh"),
    "verify_engine_independent.py": ("482c125598e97d46090a448ba087eb54", "verify"),
}
ETF_CACHE = RUNTIME / "alpaca_tsmom_etfs" / "etfs.parquet"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load():
    d = {}
    for k, p in FILES.items():
        if not p.exists():
            raise SystemExit(f"missing results file for '{k}': {p}")
        d[k] = json.loads(p.read_text())
    return d


def g(d, src, *path):
    x = d[src]
    for p in path:
        x = x[p]
    return x


# ----------------------------- ledger -----------------------------
MOM = "MOM base (blend 12/6, buffer)"
M = lambda name, key: (lambda d: g(d, "mom", "overall", name, key))
TSM = "TSMOM paper baseline (5bp)"
CTL = "CONTROL same sizing, always long"
EW = "EW long-only, active ETFs"
SPY = "SPY buy&hold"
SIXTY = "60/40 (SPY/IEF)"
TS_ = "TSM (12m trend sign)"
TSH = "TSH (hist. mean sign)"
P = lambda name, key: (lambda d: g(d, "paper", "overall", name, key))
T = lambda name, key: (lambda d: g(d, "tsh", "overall", name, key))
JM = lambda name: (lambda d: g(d, "mom", "june_2026", name))
JP = lambda name: (lambda d: g(d, "paper", "june_2026", name) * 100)
JT = lambda name: (lambda d: g(d, "tsh", "june_2026", name) * 100)


def chain_cagr_ex_year(d, name, skip="2020"):
    """chain-link calendar-year returns excluding one year; day count approximated as total days minus ~252."""
    cy = g(d, "mom", "calendar_years")
    prod = 1.0
    for y, row in cy.items():
        if y == skip:
            continue
        prod *= 1 + row[name] / 100.0
    days = g(d, "mom", "overall", MOM, "days") - 252
    return (prod ** (252.0 / days) - 1) * 100


def beat_spy_years(d):
    cy = g(d, "mom", "calendar_years")
    return sum(1 for y, row in cy.items() if row[MOM] > row["SPY buy&hold"])


def class_share(cls):
    c = {"equity", "bond", "commodity", "fx"}
    return lambda d: g(d, "paper", "class_contribution_pct", cls) / sum(g(d, "paper", "class_contribution_pct", k) for k in c) * 100


def sharpe_se(d, src, name):
    yrs = g(d, src, "overall", name, "days") / 252.0
    sh = g(d, src, "overall", name, "sharpe")
    return ((1 + 0.5 * sh * sh) / yrs) ** 0.5


# (id, what was told to the user, getter or None if not in saved results, expected, tolerance)
LEDGER = [
    # ---- equity momentum report ----
    ("mom.calmar.base", "Calmar base 0.46", M(MOM, "calmar"), 0.46, 0.005),
    ("mom.calmar.spy", "Calmar SPY 0.45", M("SPY buy&hold", "calmar"), 0.45, 0.005),
    ("mom.calmar.6040", "Calmar 60/40 0.45", M("60/40 (SPY/IEF)", "calmar"), 0.45, 0.005),
    ("mom.sharpe.base", "Sharpe base 0.69", M(MOM, "sharpe"), 0.69, 0.005),
    ("mom.sharpe.spy", "Sharpe SPY 0.88", M("SPY buy&hold", "sharpe"), 0.88, 0.005),
    ("mom.sharpe.6040", "Sharpe 60/40 0.91", M("60/40 (SPY/IEF)", "sharpe"), 0.91, 0.005),
    ("mom.cagr.base", "CAGR base 23.0%", lambda d: M(MOM, "cagr")(d) * 100, 23.0, 0.05),
    ("mom.cagr.spy", "CAGR SPY 15.2%", lambda d: M("SPY buy&hold", "cagr")(d) * 100, 15.2, 0.05),
    ("mom.vol.base", "vol base 44%", lambda d: M(MOM, "vol")(d) * 100, 44.0, 0.5),
    ("mom.vol.spy", "vol SPY 18%", lambda d: M("SPY buy&hold", "vol")(d) * 100, 18.0, 0.5),
    ("mom.dd.base", "max drawdown base -50%", lambda d: M(MOM, "max_dd")(d) * 100, -50.0, 0.5),
    ("mom.dd.spy", "max drawdown SPY -34%", lambda d: M("SPY buy&hold", "max_dd")(d) * 100, -34.0, 0.5),
    ("mom.roll.spy", "rolling 5y win rate vs SPY 96.8%", lambda d: g(d, "mom", "rolling_5y", "SPY buy&hold", "win_frac") * 100, 96.8, 0.05),
    ("mom.cy.2020", "2020 base +96.8%", lambda d: g(d, "mom", "calendar_years", "2020", MOM), 96.8, 0.05),
    ("mom.cy.2020spy", "2020 SPY +18.5%", lambda d: g(d, "mom", "calendar_years", "2020", "SPY buy&hold"), 18.5, 0.05),
    ("mom.cy.beat", "beat SPY in 6 of 10 calendar years", beat_spy_years, 6, 0),
    ("mom.ex2020.base", "ex-2020 base about 16% a year", lambda d: chain_cagr_ex_year(d, MOM), 16.0, 1.0),
    ("mom.ex2020.spy", "ex-2020 SPY about 15% a year", lambda d: chain_cagr_ex_year(d, "SPY buy&hold"), 15.0, 1.0),
    ("mom.h1.calmar", "first half Calmar base 0.61", lambda d: g(d, "mom", "halves", "first half", MOM, "calmar"), 0.61, 0.005),
    ("mom.h1.calmar.spy", "first half Calmar SPY 0.53", lambda d: g(d, "mom", "halves", "first half", "SPY buy&hold", "calmar"), 0.53, 0.005),
    ("mom.h1.calmar.6040", "first half Calmar 60/40 0.67", lambda d: g(d, "mom", "halves", "first half", "60/40 (SPY/IEF)", "calmar"), 0.67, 0.005),
    ("mom.h2.calmar", "second half Calmar base 0.42", lambda d: g(d, "mom", "halves", "second half", MOM, "calmar"), 0.42, 0.005),
    ("mom.h2.calmar.spy", "second half Calmar SPY 0.51", lambda d: g(d, "mom", "halves", "second half", "SPY buy&hold", "calmar"), 0.51, 0.005),
    ("mom.h2.cagr", "second half CAGR base 17.6%", lambda d: g(d, "mom", "halves", "second half", MOM, "cagr") * 100, 17.6, 0.05),
    ("mom.h2.cagr.spy", "second half CAGR SPY 12.6%", lambda d: g(d, "mom", "halves", "second half", "SPY buy&hold", "cagr") * 100, 12.6, 0.05),
    ("mom.u500", "universe top 500 Calmar 0.64", lambda d: g(d, "mom", "sensitivity", "universe_500", "calmar"), 0.64, 0.005),
    ("mom.u2000", "universe top 2000 Calmar 0.23", lambda d: g(d, "mom", "sensitivity", "universe_2000", "calmar"), 0.23, 0.005),
    ("mom.c5", "CAGR at 5 bp 23.4%", lambda d: g(d, "mom", "sensitivity", "cost_5bp", "cagr") * 100, 23.4, 0.05),
    ("mom.c20", "CAGR at 20 bp 22.3%", lambda d: g(d, "mom", "sensitivity", "cost_20bp", "cagr") * 100, 22.3, 0.05),
    ("mom.june.base", "June 2026 base -9.08%", JM(MOM), -9.08, 0.005),
    ("mom.june.12-1", "June 12-1 only -9.85%", JM("MOM 12-1 only (buffer)"), -9.85, 0.005),
    ("mom.june.9-1", "June 9-1 only -0.73%", JM("MOM 9-1 only (buffer)"), -0.73, 0.005),
    ("mom.june.9-6", "June blend 9/6 -2.63%", JM("MOM blend 9/6 (buffer)"), -2.63, 0.005),
    ("mom.june.plain", "June plain 12-1 -6.62%", JM("PLAIN 12-1 (no buffer)"), -6.62, 0.005),
    ("mom.june.ew", "June equal-weight +4.16%", JM("EW screened universe"), 4.16, 0.005),
    ("mom.june.spy", "June SPY -0.07%", JM("SPY buy&hold"), -0.07, 0.005),
    ("mom.june.6040", "June 60/40 +0.14%", JM("60/40 (SPY/IEF)"), 0.14, 0.005),
    ("mom.top10", "top-10 share of profit 31%", lambda d: g(d, "mom", "concentration", "top10_share") * 100, 31.0, 0.5),
    ("mom.bigmoves", "6,143 symbol-days with |return| > 50%", lambda d: g(d, "mom", "big_moves"), 6143, 0),
    ("mom.nsyms", "5,039 symbols", lambda d: g(d, "mom", "n_symbols"), 5039, 0),
    # ---- paper-baseline TSMOM report ----
    ("paper.sharpe", "TSM Sharpe 0.37", P(TSM, "sharpe"), 0.37, 0.005),
    ("paper.cagr", "TSM CAGR 4.83%", lambda d: P(TSM, "cagr")(d) * 100, 4.83, 0.005),
    ("paper.vol", "TSM vol 16.2%", lambda d: P(TSM, "vol")(d) * 100, 16.2, 0.05),
    ("paper.dd", "TSM max drawdown -36.6%", lambda d: P(TSM, "max_dd")(d) * 100, -36.6, 0.05),
    ("paper.calmar", "TSM Calmar 0.13", P(TSM, "calmar"), 0.13, 0.005),
    ("paper.ctl.sharpe", "control Sharpe 0.57", P(CTL, "sharpe"), 0.57, 0.005),
    ("paper.ctl.cagr", "control CAGR 11.11%", lambda d: P(CTL, "cagr")(d) * 100, 11.11, 0.005),
    ("paper.ew.sharpe", "equal-weight Sharpe 0.65", P(EW, "sharpe"), 0.65, 0.005),
    ("paper.spy.sharpe", "SPY Sharpe 0.86", P(SPY, "sharpe"), 0.86, 0.005),
    ("paper.6040.sharpe", "60/40 Sharpe 0.88", P(SIXTY, "sharpe"), 0.88, 0.005),
    ("paper.exclass", "Sharpe ex-commodity 0.19", lambda d: g(d, "paper", "sharpe_ex_best_class"), 0.19, 0.005),
    ("paper.share.com", "commodity 51% of gross P&L", class_share("commodity"), 51.0, 0.5),
    ("paper.share.eq", "equity 22%", class_share("equity"), 22.0, 0.5),
    ("paper.share.bond", "bonds 21%", class_share("bond"), 21.0, 0.5),
    ("paper.share.fx", "FX 5%", class_share("fx"), 5.0, 0.5),
    ("paper.roll", "rolling 3y positive share 71.6%", lambda d: g(d, "paper", "rolling3y_positive_share") * 100, 71.6, 0.05),
    ("paper.fin.cagr", "financing: CAGR -2.58%", lambda d: g(d, "paper", "sensitivity", "financing_3pct", "cagr") * 100, -2.58, 0.005),
    ("paper.fin.sharpe", "financing: Sharpe -0.08", lambda d: g(d, "paper", "sensitivity", "financing_3pct", "sharpe"), -0.08, 0.005),
    ("paper.c20", "CAGR at 20 bp 3.02%", lambda d: g(d, "paper", "sensitivity", "cost_20", "cagr") * 100, 3.02, 0.005),
    ("paper.c15.sharpe", "Sharpe at 1.5 bp 0.40", lambda d: g(d, "paper", "sensitivity", "cost_1.5", "sharpe"), 0.40, 0.005),
    ("paper.c10.sharpe", "Sharpe at 10 bp 0.34", lambda d: g(d, "paper", "sensitivity", "cost_10", "sharpe"), 0.34, 0.005),
    ("paper.c20.sharpe", "Sharpe at 20 bp 0.26", lambda d: g(d, "paper", "sensitivity", "cost_20", "sharpe"), 0.26, 0.005),
    ("paper.gross", "average gross exposure 3.44x", lambda d: g(d, "paper", "avg_gross"), 3.44, 0.005),
    ("paper.net", "average net exposure +0.95x", lambda d: g(d, "paper", "avg_net"), 0.95, 0.005),
    ("paper.turns", "11.5 turns a year", lambda d: g(d, "paper", "avg_turnover") * 12, 11.5, 0.05),
    ("paper.turns.pergross", "about 3.3 turns per unit gross", lambda d: g(d, "paper", "avg_turnover") * 12 / g(d, "paper", "avg_gross"), 3.3, 0.05),
    ("paper.se", "Sharpe standard error about 0.34", lambda d: sharpe_se(d, "paper", TSM), 0.34, 0.01),
    ("paper.june.tsm", "June TSM -6.25%", JP(TSM), -6.25, 0.005),
    ("paper.june.ctl", "June control -6.63%", JP(CTL), -6.63, 0.005),
    ("paper.june.ew", "June equal-weight -4.17%", JP(EW), -4.17, 0.005),
    ("paper.june.spy", "June SPY -0.07%", JP(SPY), -0.07, 0.005),
    ("paper.june.6040", "June 60/40 +0.14%", JP(SIXTY), 0.14, 0.005),
    # ---- TSH control report ----
    ("tsh.sharpe.tsh", "TSH Sharpe 0.58", T(TSH, "sharpe"), 0.58, 0.005),
    ("tsh.cagr.tsh", "TSH CAGR 9.40%", lambda d: T(TSH, "cagr")(d) * 100, 9.40, 0.005),
    ("tsh.vol.tsh", "TSH vol 18.4%", lambda d: T(TSH, "vol")(d) * 100, 18.4, 0.05),
    ("tsh.dd.tsh", "TSH max drawdown -37.8%", lambda d: T(TSH, "max_dd")(d) * 100, -37.8, 0.05),
    ("tsh.cagr.tsm", "TSM CAGR 4.83% (TSH report)", lambda d: T(TS_, "cagr")(d) * 100, 4.83, 0.005),
    ("tsh.diffsharpe", "TSM-TSH difference Sharpe -0.23", lambda d: g(d, "tsh", "diff_sharpe"), -0.23, 0.005),
    ("tsh.monthly", "mean monthly difference -0.365%", lambda d: g(d, "tsh", "monthly_diff_mean") * 100, -0.365, 0.0005),
    ("tsh.p", "block-bootstrap p 0.736", lambda d: g(d, "tsh", "bootstrap_p"), 0.736, 0.0005),
    ("tsh.h1.diff", "first-half monthly difference -0.75%", lambda d: g(d, "tsh", "halves", "first half", "mean_monthly_diff_pct"), -0.75, 0.005),
    ("tsh.h2.diff", "second-half monthly difference -0.02%", lambda d: g(d, "tsh", "halves", "second half", "mean_monthly_diff_pct"), -0.02, 0.005),
    ("tsh.h1.sh.tsm", "first-half Sharpe TSM 0.03", lambda d: g(d, "tsh", "halves", "first half", TS_, "sharpe"), 0.03, 0.005),
    ("tsh.h1.sh.tsh", "first-half Sharpe TSH 0.47", lambda d: g(d, "tsh", "halves", "first half", TSH, "sharpe"), 0.47, 0.005),
    ("tsh.h2.sh.tsm", "second-half Sharpe TSM 0.67", lambda d: g(d, "tsh", "halves", "second half", TS_, "sharpe"), 0.67, 0.005),
    ("tsh.h2.sh.tsh", "second-half Sharpe TSH 0.73", lambda d: g(d, "tsh", "halves", "second half", TSH, "sharpe"), 0.73, 0.005),
    ("tsh.leg.tsm.long", "TSM long leg 94.2", lambda d: g(d, "tsh", "legs", "TSM", 0), 94.2, 0.05),
    ("tsh.leg.tsm.short", "TSM short leg -33.2", lambda d: g(d, "tsh", "legs", "TSM", 1), -33.2, 0.05),
    ("tsh.leg.tsh.long", "TSH long leg 111.2", lambda d: g(d, "tsh", "legs", "TSH", 0), 111.2, 0.05),
    ("tsh.leg.tsh.short", "TSH short leg -10.4", lambda d: g(d, "tsh", "legs", "TSH", 1), -10.4, 0.05),
    ("tsh.wins", "TSM beats TSH in 12 of 25 ETFs", lambda d: g(d, "tsh", "tsm_beats_tsh_etfs"), 12, 0),
    ("tsh.n", "25 ETFs", lambda d: g(d, "tsh", "n_etfs"), 25, 0),
    ("tsh.agree", "signals agree 68.9%", lambda d: g(d, "tsh", "agreement_pct"), 68.9, 0.05),
    ("tsh.june.tsh", "June TSH -5.93%", JT(TSH), -5.93, 0.005),
    ("tsh.june.tsm", "June TSM -6.25% (TSH report)", JT(TS_), -6.25, 0.005),
    ("tsh.se", "Sharpe SE about 0.33-0.34", lambda d: sharpe_se(d, "tsh", TS_), 0.335, 0.015),
    # ---- engine verification report ----
    ("verify.rebal", "112 rebalances compared", lambda d: int(g(d, "verify", "checks", "rebalance_dates_equal", "detail").split()[0]), 112, 0),
    ("verify.all", "ENGINE VERIFIED (all checks passed)", lambda d: g(d, "verify", "all_passed"), True, 0),
]

# claims quoted to the user that the saved JSON files do NOT contain (printed to the terminal only)
UNVERIFIABLE = [
    ("TSH report: 'TSH turns over 0.38x per rebalance'", "tsh JSON has no turnover"),
    ("TSH report: 'TSH long 84.7% of the time, TSM 63.5%'", "long shares not saved"),
    ("TSH report: 'annualised TSM-TSH mean difference -4.66%'", "daily-difference mean not saved"),
    ("TSH report: calendar-year table (e.g. 2021 +4.2 vs -7.7, 2022 +14.9 vs -24.9, 2019 +5.3 vs +49.1)", "calendar years not saved"),
    ("TSH report: per-class P&L (equity 13.5 vs 55.8; FX TSM higher in 5 of 6; bonds 3 of 5)", "per-class TSH P&L not saved"),
    ("TSH report: cost sensitivity for TSH and financing 1.68% / Sharpe 0.18", "TSH sensitivities not saved"),
    ("Paper report: 'TSM daily correlation with SPY 0.08'", "correlation not saved"),
    ("Paper report: calendar-year table for TSM vs control (e.g. 2022 +14.9 vs -28.3)", "calendar years not saved"),
    ("Paper report: top ETFs QQQ, DBA, GLD, UUP, LQD", "per-ETF P&L not saved"),
    ("Data checks: SLV -28.5% on 2026-01-30 and USO -25.3% on 2020-03-09", "hygiene list not saved"),
    ("Verification report: max daily diff 6.0e-05, CAGR diff 0.0056 pts, Sharpe diff 0.0002", "numeric diffs not saved (only pass/fail)"),
    ("All statements about the two papers (Deng 2025; Huang, Li, Wang, Zhou 2020) and the book repository", "external/second-hand; not checkable here"),
    ("Explanations and interpretation (why a result happened)", "reasoning is not a number"),
]


def main():
    d = load()
    report = {"sections": {}}
    reasons = []

    print("=== A. VALIDATOR LINK ===")
    ok_val = bool(g(d, "verify", "all_passed"))
    prov = g(d, "verify", "provenance")
    same = False
    if ETF_CACHE.exists():
        same = f"md5={md5(ETF_CACHE)}" in prov
    print(f"  validator all_passed={ok_val}; verified data == current cache (md5): {same}")
    if not ok_val:
        reasons.append("validator did not pass all checks")
    if not same:
        reasons.append("validator ran on different price data than the cache that exists now (or cache missing)")
    report["sections"]["validator_link"] = dict(all_passed=ok_val, same_data=same)

    print("\n=== B. SPEC LOCK (script checksums and times) ===")
    lock = {}
    for name, (want, key) in SCRIPTS.items():
        p = TOOLS / name
        if not p.exists():
            print(f"  [FAIL] {name}: missing")
            reasons.append(f"{name} missing")
            continue
        got = md5(p)
        edited_after = p.stat().st_mtime > FILES[key].stat().st_mtime + 2.0
        okk = (got == want) and not edited_after
        lock[name] = dict(md5_ok=got == want, edited_after_results=edited_after)
        print(f"  [{'PASS' if okk else 'FAIL'}] {name}: md5 {'matches' if got == want else 'DIFFERS ('+got+')'}; "
              f"{'modified AFTER its results were written' if edited_after else 'not modified after results'}")
        if not okk:
            reasons.append(f"{name}: script changed or edited after results")
    report["sections"]["spec_lock"] = lock

    print("\n=== C. VERDICT RECOMPUTE (from raw saved numbers) ===")
    rec = {}
    # paper baseline
    pv = g(d, "paper", "overall")
    r_i = pv[TSM]["sharpe"] >= 0.33
    r_ii = g(d, "paper", "sensitivity", "cost_20", "cagr") > 0
    r_iii = g(d, "paper", "rolling3y_positive_share") >= 0.60
    r_iv = g(d, "paper", "sharpe_ex_best_class") > 0
    r_v = pv[TSM]["sharpe"] > pv[CTL]["sharpe"]
    paper_re = dict(i_sharpe_ge_paper_oos=r_i, ii_cagr_pos_at_20bp=r_ii, iii_rolling3y_pos_ge_60pct=r_iii,
                    iv_pos_sharpe_ex_best_class=r_iv, v_beats_always_long_control=r_v)
    rec["paper"] = paper_re
    # tsh control
    th = g(d, "tsh", "overall")
    h = g(d, "tsh", "halves")
    tsh_re = {"a_tsm_sharpe_gt_tsh": th[TS_]["sharpe"] > th[TSH]["sharpe"],
              "b_bootstrap_p_lt_0.10": g(d, "tsh", "bootstrap_p") < 0.10,
              "c_tsm_beats_tsh_in_majority_of_etfs": g(d, "tsh", "tsm_beats_tsh_etfs") > g(d, "tsh", "n_etfs") / 2,
              "d_positive_diff_in_both_halves": h["first half"]["mean_monthly_diff_pct"] > 0 and h["second half"]["mean_monthly_diff_pct"] > 0}
    rec["tsh"] = tsh_re
    # equity momentum
    mo = g(d, "mom", "overall")
    base, spy, sx = mo[MOM], mo["SPY buy&hold"], mo["60/40 (SPY/IEF)"]
    mi = base["calmar"] > spy["calmar"] and base["calmar"] > sx["calmar"]
    mii = base["cagr"] > spy["cagr"]
    r5 = g(d, "mom", "rolling_5y")
    miii = r5["SPY buy&hold"]["win_frac"] > 0.5 and r5["60/40 (SPY/IEF)"]["win_frac"] > 0.5
    variants = [k for k in mo if k.startswith("MOM")]
    clear = sum(1 for k in variants if mo[k]["calmar"] > spy["calmar"] and mo[k]["cagr"] > spy["cagr"])
    miv = clear >= 3
    mv = g(d, "mom", "concentration", "top10_share") < 0.5
    mom_re = dict(i=mi, ii=mii, iii=miii, iv=miv, v=mv)
    rec["mom"] = mom_re
    stored = {"paper": g(d, "paper", "verdict"), "tsh": g(d, "tsh", "verdict"),
              "mom": {k: g(d, "mom", "verdict")[k] for k in ("i", "ii", "iii", "iv", "v")}}
    told = {"paper": False, "tsh": False, "mom": True}   # verdicts stated to the user: FAIL, FAIL, PASS
    stored_overall = {"paper": g(d, "paper", "overall_pass"), "tsh": g(d, "tsh", "overall_pass"), "mom": g(d, "mom", "verdict")["passed"]}
    for k in ("paper", "tsh", "mom"):
        flags_match = stored[k] == rec[k]
        overall_re = all(rec[k].values())
        okk = flags_match and overall_re == stored_overall[k] == told[k]
        print(f"  [{'PASS' if okk else 'FAIL'}] {k}: recomputed {rec[k]} -> overall {'PASS' if overall_re else 'FAIL'}; "
              f"stored flags {'match' if flags_match else 'DIFFER'}; stored overall {'PASS' if stored_overall[k] else 'FAIL'}; "
              f"told to user {'PASS' if told[k] else 'FAIL'}")
        if not okk:
            reasons.append(f"verdict inconsistency in '{k}'")
    report["sections"]["verdict_recompute"] = dict(recomputed=rec, stored=stored, told=told)

    print("\n=== D. CLAIMS LEDGER ===")
    verified, mismatched, errors = [], [], []
    for cid, text, getter, expected, tol in LEDGER:
        try:
            val = getter(d)
        except Exception as exc:  # noqa: BLE001
            errors.append((cid, text, str(exc)))
            print(f"  [ERROR   ] {cid}: {text}  ({exc})")
            continue
        if isinstance(expected, bool):
            okk = bool(val) == expected
        else:
            okk = abs(float(val) - float(expected)) <= tol + 1e-12
        (verified if okk else mismatched).append((cid, text, val, expected, tol))
        if not okk:
            print(f"  [MISMATCH] {cid}: told '{text}' (expected {expected}±{tol}); saved value {val}")
    print(f"  verified {len(verified)} / {len(LEDGER)} ledger claims; mismatched {len(mismatched)}; errors {len(errors)}")
    if mismatched:
        reasons.append(f"{len(mismatched)} claims mismatch the saved results")
    if errors:
        reasons.append(f"{len(errors)} claims could not be evaluated")
    print(f"\n  UNVERIFIABLE from saved results ({len(UNVERIFIABLE)} items; not counted for or against):")
    for text, why in UNVERIFIABLE:
        print(f"   - {text}  [{why}]")
    report["sections"]["ledger"] = dict(n_claims=len(LEDGER), verified=len(verified), mismatched=[m[:2] + (m[2],) for m in mismatched],
                                        errors=errors, unverifiable=[u[0] for u in UNVERIFIABLE])

    approved = not reasons
    print("\n=== CRITIC VERDICT ===")
    if approved:
        print(f"  APPROVED  (validator linked, scripts locked, verdicts consistent, {len(verified)} of {len(LEDGER)} ledger claims match; "
              f"{len(UNVERIFIABLE)} items are unverifiable from saved results)")
    else:
        print("  REJECTED")
        for r in reasons:
            print(f"   - {r}")
    print("  APPROVED means the reported numbers match the saved results and the checks above hold. It does not")
    print("  mean the strategy works, that explanations are right, or that anything in the UNVERIFIABLE list is correct.")
    report["approved"] = approved
    report["reasons"] = reasons
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
