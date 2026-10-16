# Equities Golden-Cross Strategy — Exit-Logic Validation & Deployment-Readiness Record

**Date:** 2026-09-27
**Scope:** `trading_bot_v4` equities golden-cross strategy — closes out the exit-logic search and the subsequent five-item validation/deployment-readiness sequence.

## 1. Locked control specification

- **Entry:** golden-cross (50-day SMA crosses above 200-day SMA), computed from trailing rolling windows only (no look-ahead).
- **Sizing:** equal-dollar, 2% of current realized capital per position.
- **Stop:** fixed −12%, checked against the day's low, filled at the exact stop price (see item 2 for the gap-aware refinement to this assumption).
- **Exit:** death-cross (50-day SMA crosses below 200-day SMA).
- **Cost:** 12bps entry + 12bps exit (24bps round trip).
- **Universe:** 60 symbols flagged `robust` and non-Crypto in `runtime/cross_asset_golden_cross_results.json` (see §6 for a disclosed correction to what this universe actually contains).
- **Starting capital:** $100,000. **History:** full available range, 1962–2026 (per symbol's own listing history), via yfinance.

## 2. Exit-logic search: closed

Ten exit-logic variants were tested against the locked control across two predeclared extended grids (TIME_STAGNATION at 40/60/100/150/200 days; PARTIAL_PROFIT-TAKE at 25/50/100/150/200%), validated against a single predeclared out-of-sample holdout plus paired block-bootstrap persistence tests. **Result: unanimous reject.** The two in-sample "candidates" (100-day stagnation, +100% partial profit) passed CAGR and winner-retention gates but failed the specific objective they were meant to improve — durable drawdown reduction — out of sample. The extended thresholds converged toward the control rather than revealing a second optimum, and bootstrap persistence did not rescue any candidate.

**Conclusion:** retain the unchanged control — golden-cross entry, equal-dollar sizing, fixed −12% stop, death-cross exit. No tested ATR, Chandelier, time-based stagnation, or partial-profit exit variant demonstrated a durable improvement under the predeclared OOS and bootstrap criteria. **No alternative exit logic is supported.**

## 3. Validation / deployment-readiness sequence

Five items, run in the order below, each additive to the locked control (one change at a time), each independently verified (syntax check, a dry-run test suite with synthetic and hand-built cases, and AST-diff confirming dry-run and production copies of new logic are byte-for-byte identical) before being run against real data.

### Item 1 — Execution-timing audit
`tools/run_execution_timing_audit_equities.py` → `runtime/execution_timing_audit_results.json`

Compared the existing SAME_CLOSE/market-on-close convention against an explicit NEXT_OPEN fill model (signal detected day *t*, filled at day *t+1*'s open), with stop/signal precedence proven deterministic under a forced same-bar coincidence.

| | n_trades | stop_exits | CAGR | Sharpe | Sortino | max_dd | avg_hold |
|---|---|---|---|---|---|---|---|
| SAME_CLOSE (control) | 1,785 | 578 | 8.64% | 0.96 | 1.13 | −24.62% | 324.6d |
| NEXT_OPEN | 1,785 | 574 | 8.64% | 0.96 | 1.13 | −24.63% | 324.0d |

**CAGR delta: +0.00pp.** The control's performance does not depend on the same-close/market-on-close execution assumption.

### Item 2 — Gap-through-stop model
`tools/run_gap_through_stop_backtest_equities.py` → `runtime/gap_through_stop_results.json`

Every prior script filled stops at exactly the stop price regardless of gap size. This item fills at the session's open when it gaps below the stop, preserving the exact-stop-price fill for an ordinary intraday touch.

| | n_trades | stop (exact) | stop (gap) | CAGR | Sharpe | Sortino | max_dd |
|---|---|---|---|---|---|---|---|
| OLD (optimistic, all prior scripts) | 1,785 | 578 | 0 | 8.64% | 0.96 | 1.13 | −24.62% |
| NEW (gap-aware) | 1,785 | 493 | 85 | 8.57% | 0.95 | 1.12 | −25.02% |

85 of 578 stop exits (14.7%) gapped through the stop; average slippage beyond the nominal stop price on those gap exits was **−2.76pp of return**. Aggregate cost: **CAGR −0.064pp, max_dd −0.393pp (deeper)**. A small, real, now-disclosed cost — not a reversal of the control decision.

### Item 3 — Cost and liquidity sensitivity
`tools/run_cost_liquidity_sensitivity_equities.py` → `runtime/cost_liquidity_sensitivity_results.json`

Five predeclared scenarios on top of the locked control: BASELINE (24bps round trip, unchanged), MODERATE_SPREAD (+10bps each way, 44bps total), WIDE_SPREAD (+25bps each way, 74bps total), LIQUIDITY_CAP (unchanged 24bps cost, but position size capped at 1% of trailing 20-day average dollar volume, lagged one day), and COMBINED_STRESS (moderate spread + the same liquidity cap together).

| Scenario | Cost (rt) | n_trades | CAGR | Sharpe | Sortino | max_dd | Capped |
|---|---|---|---|---|---|---|---|
| BASELINE | 24.0bps | 1,785 | 8.64% | 0.96 | 1.13 | −24.62% | — |
| MODERATE_SPREAD | 44.0bps | 1,785 | 8.53% | 0.95 | 1.12 | −24.64% | — |
| WIDE_SPREAD | 74.0bps | 1,786 | 8.25% | 0.92 | 1.09 | −24.67% | — |
| LIQUIDITY_CAP | 24.0bps | 1,784 | 8.18% | 0.92 | 1.07 | −23.99% | 135/1830 (7.4%), avg reduction 58.6% |
| COMBINED_STRESS | 44.0bps | 1,784 | 8.09% | 0.91 | 1.06 | −24.01% | 135/1830 (7.4%), avg reduction 58.4% |

CAGR stays positive and in a tight band (8.09%–8.64%) across every scenario. Notably, the liquidity cap — binding on only 7.4% of entries — costs more CAGR (−0.454pp) than tripling round-trip cost to 74bps (−0.382pp), while modestly improving max_dd (+0.630pp). This says the strategy's edge leans partly on being able to take full size in a fairly small set of less-liquid names — a capital-scalability fact, not a reason to change anything at this stage.

### Item 4 — Independent equity-curve reconciliation
`tools/run_independent_ledger_reconciliation_equities.py` → `runtime/independent_ledger_reconciliation_results.json`

The control's equity curve was rebuilt from scratch via a freshly-written, independently-representationed engine (explicit share counts and a cash ledger, rather than the original's notional/percent-return bookkeeping) and diffed trade-by-trade and day-by-day against the original engine's own output.

**Result: PASS.** All 1,785 trades matched exactly (max abs pnl diff $5×10⁻¹⁰, max rel diff 6.6×10⁻¹³); the full daily NAV series agreed to the same precision (max abs diff $3×10⁻⁸, 0 points outside a predeclared $10⁻⁶/10⁻⁹ tolerance); the independently-computed CAGR/Sharpe/Sortino/max_dd (8.64% / 0.96 / 1.13 / −24.62%) exactly reproduce the control's own numbers. (Two bugs were found and fixed during construction of this independent engine itself — a sizing-basis conflation and a missing entry-day mark-to-market — both caught by dry-run testing before the real run, not after; neither was a defect in the engine being audited.)

### Item 5 — Universe and concentration diagnostics
`tools/run_universe_concentration_diagnostics_equities.py` → `runtime/universe_concentration_diagnostics_results.json`

**Disclosed finding:** the universe referred to throughout this project as "the equities universe" is actually 57 Equities + 2 Commodities (GC=F gold, PA=F palladium futures) + 1 Bonds/Rates instrument (IEF, a 7–10yr Treasury ETF) = 60 symbols. This has been true of every script in this whole sequence; it is surfaced here, not changed here.

**Part A — universe robustness** (same locked strategy, four predeclared universe definitions, all drawn from the existing data source):

| Universe | n_symbols | n_trades | CAGR | Sharpe | Sortino | max_dd | Δ CAGR vs CURRENT | Δ max_dd vs CURRENT |
|---|---|---|---|---|---|---|---|---|
| CURRENT (control) | 60 | 1,785 | 8.64% | 0.96 | 1.13 | −24.62% | — | — |
| PURE_EQUITIES (3 non-equity members dropped) | 57 | 1,746 | 8.46% | 0.94 | 1.11 | −24.14% | −0.177pp | +0.487pp |
| ALL_EQUITIES_UNFILTERED (robust screen removed) | 72 | 1,989 | 9.16% | 0.92 | 1.09 | −27.14% | +0.524pp | −2.516pp |
| ALL_NONCRYPTO_UNFILTERED (broadest, adds commodities/FX/bonds) | 102 | 2,314 | 8.81% | 0.93 | 1.10 | −24.23% | +0.170pp | +0.398pp |

CAGR stays in an 8.46%–9.16% band and Sharpe in 0.92–0.96 across all four — nothing here reverses the retain-the-control conclusion. The one clear signal: removing the upstream "robust" pre-screen (ALL_EQUITIES_UNFILTERED) buys +0.52pp of CAGR at a cost of 2.5pp of deeper drawdown — **the robustness screen modestly reduces drawdown at a small CAGR cost**, consistent with what it was presumably built to do.

**Part B — concentration diagnostics** (on CURRENT's own trades, strictly descriptive per instruction — not a rationale to alter the strategy):

Top winning symbols by realized P&L: NVDA $2,747,702 (more than double the #2 name), GE $1,181,792, NFLX $1,098,687, AVGO $882,425, MSFT $591,471, LLY $544,780, WMT $436,531, AMZN $432,699, AAPL $429,144, AMD $428,382.

By sector/category: Technology $5,474,699 (30.5% of total net realized P&L — nearly triple the next-largest sector), Industrials $1,995,661 (11.1%), Financials $1,924,214 (10.7%), Communication Services $1,665,537 (9.3%), Health Care $1,575,914 (8.8%), Consumer Discretionary $1,164,585 (6.5%), Consumer Staples $1,136,447 (6.3%), Broad Index ETF $958,218 (5.3%), Commodities $629,988 (3.5%), Utilities $416,342 (2.3%), Materials $404,436 (2.2%), Energy $360,838 (2.0%), Real Estate $234,581 (1.3%), Bonds/Rates $36,281 (0.2%).

The top 3/5/10 winning symbols account for 21.0% / 27.1% / 36.6% of total positive realized P&L. A descriptive-only re-run of the locked control with CURRENT's own top-3 winners (NVDA, GE, NFLX) excluded entirely: n_trades=1,723, CAGR=7.78% (−0.860pp vs. CURRENT), Sharpe=0.92, max_dd=−23.88% (+0.746pp). **Results are materially concentrated in a few winners, especially NVDA and Technology** — reported as fact, not as a reason to change the strategy.

## 4. Survivorship-bias data-source fork (decision pending, by design)

This item is not resolved by this validation sequence and must not be conflated with the universe-subset sensitivity checks in item 5 above (those redefine which symbols within the *existing* data source are included; none of them add delisted securities or point-in-time constituents). **Survivorship bias remains unresolved under the current data source.**

The decision depends on intended use, per the following fork:

- **Private research or provisional deployment:** keep the current data source, clearly label all results as survivorship-biased, and avoid presenting the backtest as a fully investable historical result.
- **Publication, external capital, or a strong claim about historical robustness:** obtain point-in-time constituent and delisted-security data from a suitable vendor, then rerun the **locked control only** (not the rejected exit variants) and compare the result.

These two paths are never to be mixed, and the current yfinance universe is never to be presented as resolving delisted-asset bias.

## 5. Rolling walk-forward: deferred

**Rolling walk-forward validation is deferred and confirmation-only.** It remains available as a regime-confirmation and documentation exercise for future governance or publication needs, run with the control locked and no post-hoc parameter selection — never as a renewed search for a better exit rule. Given the OOS holdout, paired bootstrap, execution-timing, gap, cost, liquidity, universe, and independent-reconciliation checks have all passed, it is unlikely to change the current decision, and stays dormant unless a governance or publication requirement calls for it.

## 6. Final findings

1. The control survives timing, gap, cost, liquidity, and reasonable-universe sensitivity checks.
2. The robustness screen modestly reduces drawdown at a small CAGR cost.
3. Results are materially concentrated in a few winners, especially NVDA and Technology.
4. Survivorship bias remains unresolved under the current data source.
5. No alternative exit logic is supported.
6. Rolling walk-forward validation is deferred and confirmation-only.

## 7. Stopping point

Unless the strategy is being prepared for external use or publication, the rational stopping point is now: **freeze the control, document these limitations, and move to paper or shadow deployment** rather than continue optimizing. Any future study — a different universe construction, signal timing, or portfolio-sizing hypothesis — requires its own new predeclared objective and holdout; none of the tools built in this sequence are to be repurposed to reopen the rejected exit search.
