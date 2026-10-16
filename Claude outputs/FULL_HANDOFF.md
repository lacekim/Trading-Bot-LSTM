# Full handoff: everything that needs to happen

This covers every real bug, decision, and open task from tonight's session,
not just the Alpaca consolidation. Organized so each section can be worked
independently. File paths are all relative to the project root
(`/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera`).

---

## SECTION A — Bugs to actually fix

### A0. BLOCKING — Alpaca paper account is on 4x margin, not cash (found after this doc was first written)

Discovered while starting on A2: the live Alpaca paper account is a standard
Reg-T margin account (`multiplier: 4`), not the cash/unlevered account every
backtest this whole project has assumed. Real numbers from the account as
found: `long_market_value: $385,007` vs `equity: $95,007` (already ~4x
levered), `cash: -$290,000`, `buying_power: $0`. The zero buying power — not
only the daily-vs-4h universe mismatch (B2) — is why BTC/ETH/SOL couldn't
open new positions.

This means the 41 currently-open positions were sized under a false
assumption and don't represent what any backtest actually tested. Fixing the
sizing script (A2) is necessary but does not undo leverage already taken on
by those 41 positions.

Recommended fix, in order:
1. Dump the 41 open positions' current state (symbol, side, qty, entry,
   current price/unrealized PnL) to a CSV first — it's a real, if
   accidental, data point on what unlimited-margin 15% sizing does, worth
   keeping for the record even though it's being discarded as "the" result.
2. Reset the Alpaca paper account to a clean $100k. This is paper money, so
   nothing real is lost, and the current book isn't measuring the strategy
   as designed anyway (built on unlimited-margin sizing, not the intended
   15%-cash-constrained rule) — resetting costs little real signal.
3. Apply A2's fix.
4. Check whether the Alpaca paper account can be set cash-only / Reg-T off
   at the account/settings level. If yes, do that so this class of bug
   can't recur silently. If paper accounts always default to margin, add an
   explicit guard in the sizing code that reads `account.multiplier` and
   `account.buying_power` directly from the API before every entry and
   never sizes past what's actually available — don't assume margin is
   unlimited or trust a fixed percentage of "capital" alone.

This blocks Section B: don't migrate crypto onto an account with $0 buying
power and an unknown leverage state.

**STATUS (re-checked live): still blocked, mechanically confirmed, nothing
more to do right now.** `multiplier: 1` — the fix held. But the account is
still `equity: $95,007` vs `long_market_value: $385,008`, `cash: -$290,000`,
`buying_power: $0` — the multiplier fix only affects future margin math, it
doesn't retroactively unwind the 41 existing positions. All 41 close orders
verified genuinely queued (`status: ACCEPTED`, submitted 07:34 UTC),
waiting on market open (2026-09-14 09:30 ET). No crypto positions are caught
in this (BTC/ETH/SOL/PAXG specifically checked, none held) — no live crypto
exposure at risk overnight. Just needs the market to open; nothing to force
computationally.

### A1. Live freqtrade bot (port 8080, `GoldenCrossStrategy`) is crash-looping

Its own log has printed "Status: process died." Never diagnosed — the real
traceback goes to a nohup redirect, not anything read yet. Get it first:

```
ps aux | grep freqtrade
tail -200 <whatever log path the running process is actually writing to>
```

Do not guess a fix before seeing the real error.

**STATUS: CLOSED — not a bug.** The real traceback shows "process died" only
fires from `trade_commands.py:28` on a clean `SIGTERM`/`KeyboardInterrupt`,
not a crash. Both port-8080 and port-8081 processes show that exact line at
the identical second (2026-09-12 14:26:50) — a coordinated restart
consistent with the `CORS_origins` config edit made earlier this session,
which needs a restart to take effect. Both have run continuously and
cleanly since (current PIDs 63312/63313, 22+ hrs of unbroken heartbeats,
zero errors). No further action.

### A2. Backtest engine has a real capital-accounting bug

`tools/run_ma200_cross_strategy.py` (and anything built on it, including
`tools/run_intermarket_regime_filter_test.py`) sizes every new entry as
`capital * allocation_pct / 100` using **total** capital, with no check of
how much is already tied up in other open positions. At the live 15%
allocation this is safe (max 90% gross if all 6 assets are long at once, by
design, per the code's own comment). But it silently assumes free leverage
at higher allocations — verified directly: at 45% allocation it let the
backtest deploy notional up to 270% of capital. Confirmed via
`tools/verify_capital_constrained_sizing.py`: real capital-constrained
results are drastically worse than what the unconstrained version reports
(153 trades / $227.6M / -60.6% max DD / 134 entries rejected for insufficient
cash, vs. the unconstrained 285 trades / $5.3B / -76.3% DD / 0 rejected).

Fix: add a real available-cash check to the shared entry logic (track
`capital - sum(open position notionals)`, reject an entry if it would exceed
that) so any future allocation-sizing experiment reports something
achievable instead of implicitly assuming unlimited leverage.

### A3. Position monitor has intermittent network errors

`v4_ma200_position_monitor` throws `SSL handshake timed out` / DNS
resolution failures when refreshing live prices. Self-recovers every time
so far, not urgent — but a retry/backoff would quiet the log. Lowest
priority item in this whole document.

---

## SECTION B — Consolidate crypto + everything else onto one broker (Alpaca)

The actual goal: stop running crypto (Kraken/freqtrade) and everything else
(Alpaca) as two disconnected systems.

### B1. Blocking fact-check — do this before any code changes

```
cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv('.env')
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
client = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=True)
assets = client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.CRYPTO))
for a in assets:
    print(a.symbol, a.tradable)
"
```

Check for BTC/USD, ETH/USD, SOL/USD, TRX/USD, XMR/USD specifically. XMR is
the one most likely to be unlisted (Monero gets delisted from regulated US
venues over privacy-coin/AML concerns) — if so, it cannot move to Alpaca at
any price, same category of hard limit as GC=F/PA=F futures already being
substituted with GLD/PPLT proxies.

**STATUS: reconfirmed live, unchanged.** `BTC/USD`, `ETH/USD`, `SOL/USD`,
`PAXG/USD` all `tradable=True`. `XMR/USD` and `TRX/USD` don't appear in the
crypto asset list at all — confirmed hard platform limit, not a data gap.

### B2. Why Alpaca's universe currently only has BTC from crypto (context)

`tools/run_cross_asset_golden_cross_backtest.py` (built Alpaca's universe in
`runtime/cross_asset_golden_cross_results.json`) tests **daily** bars via
yfinance across every asset class uniformly — under that test only BTC-USD
is robust among 14 crypto symbols tried, ETH/XMR/SOL/TRX all fail. The
system actually live on Kraken (`trading_bot_v4/signals/ma200_asset_qualifier.py`,
feeding `run_ma200_cross_strategy.py` / `run_ma200_paper_trader.py` /
freqtrade) tests the same rule on **4h Kraken bars** and finds
BTC/ETH/PAXG/SOL/TRX/XMR all robust. Same rule, different timeframe/data
source, different answer — trust the 4h Kraken result, it's the one that's
actually been running live and verified. PAXG was never in the daily test at
all (not a normal Yahoo ticker).

### B3. Merge the verified crypto set into Alpaca's universe

For whichever of BTC/ETH/PAXG/SOL/TRX/XMR B1 confirms Alpaca lists:

- Add to `runtime/cross_asset_golden_cross_results.json` as robust, sourced
  from the 4h Kraken qualifier result, not the daily Yahoo backtest.
- Modify `tools/run_alpaca_paper_trader.py` so those symbols pull signal
  from Kraken 4h data (`compute_trend_state` / `current_signal` in
  `trading_bot_v4/signals/ma200_trend.py`) instead of daily `yf.download`.
  Equities/IEF/GLD/PPLT keep using daily yfinance unchanged.
- Match position sizing to the proven 15% per-asset fixed-fractional for
  migrated crypto symbols, not the 2% used for the broad equities universe.

**STATUS (B3/B4): already done, verified not just trusted from notes.**
`cross_asset_golden_cross_results.json` has BTC-USD/ETH-USD/SOL-USD/PAXG-USD
marked `robust: true, robust_source: "kraken_4h_live"`;
`run_alpaca_paper_trader.py` already has `MIGRATED_CRYPTO_YF_TO_LABEL` wired
to the Kraken 4h signal at 15% sizing.

### B4. Whatever Alpaca can't hold stays on Kraken — by necessity

Document clearly in code comments which crypto symbols stay on Kraken and
why (Alpaca doesn't list them), so this isn't re-litigated later.

### B5. Retire redundant systems, only after B3 is confirmed working for ≥1 cycle

- Kill `tools/run_cross_asset_paper_trader_loop.py` — the pre-Alpaca,
  JSON-simulated fake-fill version. Real Alpaca orders replaced it Sep 2-3;
  it's been running redundantly in parallel ever since.
- For any crypto symbol that migrates to Alpaca: stop opening new positions
  for it in `tools/run_ma200_paper_trader.py` and the live freqtrade bot
  (port 8080). Let existing open positions there close out naturally —
  don't force-close them.
- Leave the BMSB stream (BTC, PAXG in `run_ma200_paper_trader.py`) untouched
  regardless — independent signal, not part of this consolidation.

**STATUS: correctly still on hold.** The redundant process is confirmed
alive — `run_cross_asset_paper_trader_loop.py` (PID 95867, running since
Sep 2), ready to kill. But B5 is explicitly gated on "B3 confirmed working
for ≥1 cycle," and that hasn't happened yet — A0's $0 buying power has
blocked any new Alpaca crypto entry this whole time. Holding until A0
resolves at market open; will kill then, not before.

---

## SECTION C — Strategy decisions already made (context so nothing gets redone)

- **Golden cross (50MA/200MA)**: the core, verified, live system. No change
  — this is the anchor everything else is judged against.
- **Intermarket regime filter** (`GoldenCrossRegimeFilteredStrategy`, port
  8081, fed by `tools/update_regime_state.py` hourly): backtested and
  capital-constrained-verified — does not improve CAGR, does not close the
  drawdown gap to baseline at any achievable allocation (see A2's real
  numbers). It's a legitimate lower-drawdown alternative, not a free lunch.
  Currently forward-testing live alongside baseline (port 8080). **No
  code change** — let both run for months before drawing conclusions;
  forward data is the only thing left that can move this decision.
- **FVG (Fair Value Gap)**: real, tested, not fabricated placeholder logic
  (see `tools/run_fvg_strategy_test.py` / `_equities.py`). Crypto result:
  weak, inconsistent by asset/side, not robust. Equities result: **long-only
  at 2R-3R targets is a genuinely real, broad-based edge** (PF 1.17-1.40,
  not carried by a few lucky trades) — shorts have zero edge everywhere
  tested and should be dropped. This is validated and currently sitting
  idle — **the next real candidate to wire into an actual live strategy**,
  the same way golden cross went from backtest to a real freqtrade/Alpaca
  bot. Long-only, equities only, 2R-3R exit.
- **ARIMA/SARIMA/LightGBM directional forecasting**: shelved, do not revisit
  without new information. Real statistical edge exists only at 30-second
  granularity (57% directional accuracy) and cannot survive real transaction
  costs there (3bp signal vs. ~26bp round-trip fee). Re-tested honestly at
  1h/4h: edge mostly vanishes (coin-flip on most models), one narrow
  survivor (lag-only LightGBM, 54%, real but weak) that, when combined with
  golden-cross as an entry filter and tested under fixed-fractional sizing,
  turned out to be selecting trades at random with respect to quality
  (Welch t-test p=0.986 between kept/cut trade returns) — no CAGR/Sharpe
  improvement. Dead end, don't reopen.
- **Anomaly detection (Isolation Forest) as a data-quality gate**: proposed,
  not yet built. Would catch exactly the kind of bug this project has hit
  before (a stale/frozen price feed silently corrupting a signal). Worth
  building as a pre-flight check that runs before each paper-trader cycle,
  flagging any instrument whose recent price/volume behavior looks broken
  before a trade decision uses it. Lower priority than Sections A/B but a
  real, concrete next step when there's time for it.

---

## SECTION D — Already fixed, do not redo

- yfinance upgraded 0.2.57 → 1.7.0 (fixed a real Yahoo cookie/crumb auth
  break that made every price fetch fail with `AttributeError: 'str' object
  has no attribute 'name'`). If this error resurfaces anywhere, it means
  something re-pinned the old version — don't re-diagnose from scratch,
  check the pin first.
- `BRK-B` → `BRK.B` ticker mapping fixed in `run_alpaca_paper_trader.py`
  (Alpaca wants the dot form, Yahoo uses the hyphen).
- Alpaca dashboard's Open Trades panel (`tools/render_golden_cross_backtest_dashboard.py`)
  already wired to pull real positions from the Alpaca API directly, not the
  stale simulated JSON file.
- CORS (`CORS_origins`) already enabled on both freqtrade configs
  (`config.json` / `config_regime_filtered.json`) so the comparison
  dashboard can poll both bot APIs client-side.
- Forward-test infrastructure for the regime filter (separate strategy
  class, separate config, separate DB/port, hourly regime-state cache with
  a fail-safe 6-hour staleness gate) is built and running — see Section C.

---

## SECTION E — New: test whether a hard stoploss should be added (a tail-risk question, not a return question)

Confirmed across every live system: none of them place a real stoploss.
`GoldenCrossStrategy` (freqtrade, port 8080) and `GoldenCrossRegimeFilteredStrategy`
(port 8081) both set `stoploss = -0.99` in code — a deliberate no-op required
by freqtrade's config schema, not a real risk control. `trading_bot_v4
--ma200-auto` and `run_alpaca_paper_trader.py` place no stop at all. All four
exit purely on the death-cross signal (50MA crosses back below 200MA), or FLAT
for Alpaca. `ma200_position_monitor.py` watches -8%/-15%/-25% moves and sends
a Telegram alert, but never closes anything — informational only.

This was a deliberate, backtested design choice, not an oversight — the
verified profit-factor numbers this project trusts (BTC 1.85, ETH 2.61, etc.)
were measured with signal-only exits; adding a stop was never found necessary
to get those numbers, so it was never added. It's being revisited now for one
specific reason: A0 found a real capital-accounting bug that let the Alpaca
account run 4x leveraged without anyone intending it. A hard stop is an
independent safety net against exactly that class of future bug — the
question is whether one is worth its cost in PF, not whether the strategy
needs a return boost.

### E1. Reproduce the baseline honestly first

- Confirm A2's cash-constrained sizing fix is already applied before running
  anything here — testing a stoploss on top of the old unconstrained-sizing
  bug would produce meaningless numbers.
- Find the exact script/function that actually produces the BTC 1.85 / ETH
  2.61 PF numbers referenced this session (most likely
  `trading_bot_v4/signals/ma200_asset_qualifier.py`, the 4h-Kraken
  golden-cross qualifier — confirm before touching it, don't assume). Rerun
  it unchanged and record, per asset and combined: CAGR, Sharpe, profit
  factor, max drawdown, worst single-trade loss/return. This is the baseline
  every stop variant gets compared against.

### E2. Add a real hard-stop as a parameter, not a rewrite

- In the same backtest engine, add an optional check: if price moves against
  an open position by more than X% from entry, exit at that price
  immediately instead of waiting for the death-cross signal.
- Test multiple stop distances, don't pick one arbitrarily. Reuse the
  thresholds `ma200_position_monitor.py` already alerts on, for continuity:
  -8%, -15%, -25%. Add a tighter one too (e.g. -5%) to see the shape of the
  tradeoff.
- Run the same historical window and asset universe as E1 for each distance.

### E3. Compare on tail risk, not average return

- For each stop distance, report the same metrics as E1, plus how much max
  drawdown and worst-single-trade-loss improved, and how much CAGR/PF was
  given up in exchange.
- Decision rule: only worth adopting if a stop meaningfully cuts the tail
  (max DD / worst-trade-loss) for a small cost in CAGR/PF. If every distance
  just quietly bleeds PF without moving the tail-risk numbers, "keep
  signal-only" is the correct answer — that's a legitimate result, not a
  failed test.

**Results (real, complete run):**

| Stop | Worst trade | Max DD | CAGR | PF |
|------|------------|--------|------|-----|
| Baseline | -57.9% | -33.0% | 60.3% | 2.31 |
| -5% | -5.1% | -14.0% | 37.8% | 2.80 |
| -8% | -8.1% | -17.8% | 39.3% | 2.68 |
| -15% | -15.1% | -28.4% | 48.5% | 2.53 |
| -25% | -25.1% | -30.1% | 57.1% | 2.27 (≈flat) |

-5% through -15% improve PF *and* cut max drawdown simultaneously versus
baseline, at a real cost in CAGR — not the usual tighter-stop-costs-PF
tradeoff. -25% doesn't clear E3's bar (barely moves max DD, -30.1% vs
-33.0%): the portfolio's drawdown isn't driven by one catastrophic trade, so
a very loose stop does little real tail-risk work. **Decision: -8%** —
largest PF gain (2.31→2.68) and largest drawdown cut (33%→17.8%); the CAGR
given up (60.3%→39.3%) is an acceptable price given this whole item exists
because of A0's leverage scare, not a search for more return. -15% is the
fallback if more upside is wanted, but -8% is the better fit for what this
section is actually for.

### E3b. Before finalizing — one robustness check, same standard used elsewhere in this project

Every distance from -5% to -15% improving both PF and drawdown at once can
also mean a handful of the baseline's worst trades (the -57.9% single loss,
especially) are doing outsized work in these numbers. This project already
has a house standard for exactly this situation — the "PF excluding top-3
trades" bar used elsewhere (the crypto qualifier, the cross-asset backtest).
Apply it here before touching live code: recompute PF for the -8% variant
with each asset's single largest historical loss excluded. If PF still
improves broadly with that trade held out, the -8% result is real and
broad-based, not a one-trade artifact — proceed to E4. If it collapses back
toward baseline once that one trade is excluded, stop and reconsider before
wiring anything in live.

### E4. Wire -8% into live code carefully (pending E3b passing)

- Implement it as an explicit price check in the shared strategy exit logic
  used by both the Kraken and Alpaca live/paper traders — not by changing
  freqtrade's `stoploss` config value, which stays a no-op by design. A real
  stop here needs to be its own evaluated exit condition, separate from the
  framework's built-in one.
- Scope to the golden-cross streams only (BTC/ETH/XMR/SOL/TRX). Leave the
  PAXG BMSB stream out — it's a fixed 12-week hold, a different exit design,
  and a separate question if a stop there is ever wanted too.

**E4 is done — confirmed wired into `tools/run_ma200_paper_trader.py` and
`tools/run_alpaca_paper_trader.py` (`HARD_STOP_PCT = -8.0`, gated on
`is_migrated_crypto` in the Alpaca script). Both files parse clean.**

### E5. Do the same tail-risk test for the equities/bonds/commodities universe — don't just copy -8% over

The original problem this whole section responds to (A0: the Alpaca account
running with zero real risk control) applies to Alpaca's entire universe, not
just its crypto corner. E1–E3b only validated -8% against the crypto
golden-cross qualifier (4h Kraken bars) — that number has no evidence behind
it for equities, which move on a completely different volatility profile
using a completely different backtest (`tools/run_cross_asset_golden_cross_backtest.py`,
daily Yahoo bars, ~100 instruments). Copying -8% over untested would be
exactly the kind of guess this project has already been burned by more than
once. Now that Section B is bringing equities onto the same bot as crypto,
they need the same risk discipline, found the same way:

1. Rerun E1's baseline methodology against the equities/bonds/commodities
   backtest specifically (own CAGR, Sharpe, PF, max DD, worst-trade-loss).
2. Rerun E2's stop-distance sweep against that same backtest — don't assume
   the crypto thresholds (-5/-8/-15/-25%) are the right shape for equities;
   pick distances suited to equities' typically smaller daily moves (try
   something like -3%/-5%/-8%/-12% as a starting set, adjust once the actual
   volatility is in front of you).
3. Apply E3's decision rule and E3b's robustness check independently for
   this universe — the right equities distance may be different from -8%,
   and that's expected, not a problem.
4. Wire whatever passes into the non-crypto branch of
   `tools/run_alpaca_paper_trader.py` (the `else` branch that currently calls
   `_compute_signal` / `_fetch_close_series` with no stop check at all),
   using the same explicit-price-check pattern as E4, not by touching
   anything crypto-specific.

**Results (real, complete run — `tools/run_stoploss_tail_risk_test_equities.py`,
60 robust equities/bonds/commodities instruments, daily bars, 2% sizing,
A2's cash-constrained entries):**

| Stop | Worst trade | Max DD | CAGR | PF |
|------|------------|--------|------|-----|
| Baseline | -53.9% | -12.3% | 9.44% | 4.48 |
| -3% | -3.1% | -6.4% | 3.54% | 3.39 |
| -5% | -5.1% | -6.1% | 5.08% | 3.47 |
| -8% | -8.1% | -8.9% | 7.04% | 3.87 |
| -12% | -12.1% | -8.5% | 8.49% | 4.08 |

Unlike crypto, no equities stop distance improves PF — this universe has no
free lunch, every distance costs something. **Decision: -12%** — it cuts max
drawdown almost as much as -8% (-8.5% vs -8.9%, actually marginally better)
while giving up much less CAGR (8.49% vs 7.04%) and PF (4.08 vs 3.87) — it
dominates -8% on every axis except the (expected) looser worst-trade tail.
-3%/-5% cut the tail tightest but collapse CAGR to a third-to-half of
baseline for only marginally better drawdown than -12% — fails the "small
cost" bar from E3.

E3b check on -12% (excluding each asset's single worst loss, 60 of 1781
trades): PF goes up, 4.08 → 4.31 — confirmed broad-based (nearly every asset
shows the identical -12.1% stop-out), not a one-trade artifact.

**E5 is done — wired into `tools/run_alpaca_paper_trader.py`:
`EQUITIES_HARD_STOP_PCT = -12.0` for the non-crypto branch, alongside the
existing `HARD_STOP_PCT = -8.0` for migrated crypto. Held positions check
Alpaca's own `unrealized_plpc` against the right threshold before the signal
is evaluated. File parses clean.**

---

## SECTION E STATUS: CLOSED (E1–E5 complete)

Both universes now have their own independently-validated hard stop:
crypto at -8% (golden-cross streams, BTC/ETH/XMR/SOL/TRX), equities/bonds/
commodities at -12% (everything else on Alpaca). PAXG's BMSB stream remains
untouched by design (fixed 12-week hold, a separate exit design).

---

## FULL STATUS AS OF 2026-09-13 (everything re-checked live)

- **A0 — still blocked, mechanically confirmed.** Margin fix holds
  (`multiplier: 1`), but the 41 pre-fix positions' close orders are queued
  (`status: ACCEPTED`) waiting on market open (2026-09-14 09:30 ET). No
  crypto exposure at risk overnight (BTC/ETH/SOL/PAXG checked, none held).
  Nothing to force computationally — resumes automatically at open.
- **A1 — CLOSED, not a bug.** "process died" was a clean restart
  (config-reload from the `CORS_origins` edit), not a crash. Both bots
  running cleanly 22+ hrs since, zero errors.
- **A3 — untouched, still low priority**, as originally scoped.
- **B1 — reconfirmed, unchanged.** BTC/ETH/SOL/PAXG tradable on Alpaca;
  XMR/TRX confirmed absent from Alpaca's crypto asset list — hard platform
  limit stands.
- **B3/B4 — already done, verified.** The merge into
  `cross_asset_golden_cross_results.json` and the Kraken-4h signal wiring in
  `run_alpaca_paper_trader.py` are both confirmed live, not just recorded.
- **B5 — correctly still on hold.** Redundant loop (PID 95867) found alive
  and ready to kill, but stays gated on A0/B3 completing a clean cycle —
  will execute once the market opens, not before.
- **C (FVG-equities)** — unchanged, still idle, still the next real
  candidate whenever there's room to prioritize it.

Nothing is actionable before market open tomorrow. A0 and B5 resume
automatically once it does.
