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

**STATUS: RESOLVED, confirmed against a real, current, complete account
pull (2026-09-16, direct from the user — not this session's stale
2026-09-13 snapshot).** The old 41 over-levered positions are gone. The
account now holds **43 positions, all `PositionSide.LONG`, zero shorts**:
42 equities each sized ~$1,850–2,000 (matching the intended 2%-of-equity
rule) plus one BTCUSD position (~$13,717, matching `CRYPTO_ALLOCATION_PCT =
15.0` off the same equity base — the math checks: $13,717 / 0.15 ≈ $91,447
equity, and $91,447 × 0.02 ≈ $1,829, right in the middle of the observed
per-equity range). This is exactly the clean, cash-constrained, no-leverage
book A0's recommended fix (reset + reapply A2) was meant to produce, and the
numbers are internally consistent with the sizing code, not just "looks
fine." Unrealized P&L across the book is small single-digit moves either
way — nothing near the -12% equities or -8% crypto hard stops from Section
E. Only ETH/SOL/PAXG are absent from this list, which is direct, positive
confirmation of B5's current gate below (their Alpaca entries genuinely
haven't fired yet — this isn't an assumption anymore, it's what the real
book shows). Treat this 43-position pull as the live source of truth going
forward, not the retired "41 positions" figure above.

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

**STATUS: BTC's entry path now verified; ETH/SOL/PAXG still not
(2026-09-16 update, third pass).** `run_cross_asset_paper_trader_loop.py`
(PID 95867, alive since Sep 2) **is now confirmed killed** — `ps -p 95867`
returns nothing as of 2026-09-16 23:06:59 PDT. This item is done. B5 was
originally gated on "B3 confirmed working for ≥1 cycle" — that turned out
to have only been true for the equities side at first; a live check found
zero migrated-crypto positions open on Alpaca as of the 13th.

That's since changed for BTC specifically: Alpaca's BTC buy filled
**2026-09-15 08:17:54 UTC at $76,727** — real order, real fill, inside
Kraken's own $75.5k–$79k range for that window, so direction and price both
check out. One nuance the user caught and I then confirmed by reading
`tools/run_alpaca_paper_trader.py` directly (lines 168–276): this wasn't a
fresh cross-up — Kraken's BTC LONG has been open since Sep 2 per freqtrade's
own trade log — it was Alpaca starting flat post-reset, reading the shared
Kraken-4h signal (`_compute_signal_kraken_4h`), seeing the already-in-
progress LONG state, and opening to match it.

That distinction turns out not to matter here, and it's worth being
precise about why: the entry logic (`if sig["signal"] == "LONG" and not
held: <submit order>`) is a level-triggered reconciliation, not an
edge-triggered crossover detector — it has no code path that distinguishes
"the signal just flipped to LONG this cycle" from "the signal has been LONG
for weeks and I'm only now catching up." Both hit the exact same branch,
with the exact same signal source, sizing (`CRYPTO_ALLOCATION_PCT = 15.0`,
matching the live Kraken system's own sizing), and order mechanics. There
is no stricter "real crossover" test this codebase could produce that would
carry more information than what the BTC fill already showed — waiting for
one would just be waiting on a scenario the code can't tell apart from what
already happened.

**Verified for BTC: cut Kraken's redundant BTC entries** (in
`tools/run_ma200_paper_trader.py` and freqtrade) — the Alpaca path for BTC
is proven end-to-end (signal source, direction, sizing, real order, fill).
**Still not verified for ETH/SOL/PAXG** — same code path, but no fill has
come through for any of the three yet, and this project's history (A2's
allocation-dependent bug, A0's margin bug) is exactly why "the shared logic
already worked for BTC" isn't automatically extended to the others without
its own symbol-specific fill. Hold Kraken's redundant entries for ETH/SOL/
PAXG until each has its own. The CLOSE branch (`elif signal == "FLAT" and
held`) also remains completely unexercised for all four — worth noting, not
blocking, since golden-cross exits are naturally weeks-to-months out.

**Mechanics of "cutting BTC's redundant Kraken entries" — corrected
2026-09-16, second pass.** A real wrinkle surfaced doing this properly:
`run_ma200_paper_trader.py`'s active-symbol list isn't a static list to
trim — it comes from `get_qualified_assets()` in `ma200_asset_qualifier.py`,
which dynamically re-qualifies off a shared `CANDIDATE_UNIVERSE` also used
by backtests and possibly freqtrade's own qualification logic. Editing that
shared module to drop BTC would silently change more than this one paper
trader — exactly the kind of scope-creep this project has been burned by
before (A2's shared sizing bug). **Corrected, scoped approach: a small
local exclusion right at the entry check in `run_ma200_paper_trader.py`** —
skip opening new BTC positions there specifically, without touching
`ma200_asset_qualifier.py` or `CANDIDATE_UNIVERSE` — so existing open BTC
positions still get evaluated for exit every cycle exactly as before.

Correct order of operations:
1. Edit `freqtrade-develop/user_data/config.json` — remove `"BTC/USD"` from
   `pair_whitelist`.
2. Edit `tools/run_ma200_paper_trader.py` — add the local entry-check
   exclusion for BTC described above (exit path untouched).
3. Restart both processes so the edits actually load — per Section F, both
   freqtrade and `trading_bot_v4.main --ma200-auto` load pairlist/symbol
   config once at startup.

**STATUS: DONE, confirmed 2026-09-16 ~11:27PM PDT.** Both edits are in
place — verified directly: `pair_whitelist` now reads `['ETH/USD',
'XMR/USD', 'PAXG/USD', 'SOL/USD', 'TRX/USD']` (BTC gone), and
`run_ma200_paper_trader.py` line 76 defines `KRAKEN_ENTRY_EXCLUDED_ASSETS =
{"BTC"}`, wired into the qualification check at line 168. Both processes
were killed cleanly (confirmed dead via `ps -p` before relaunch, ~8s
graceful freqtrade shutdown) and relaunched detached: **new PIDs — freqtrade
(`GoldenCrossStrategy`) = 4450, `--ma200-auto` = 4451**, both started
23:27 PDT. Port 8081 (`GoldenCrossRegimeFilteredStrategy`, PID 63313)
confirmed untouched throughout.

**Proof the edit actually took effect, not just that the process
restarted:** freqtrade's own log shows the whitelist it loaded —
`2026-09-16 21:36:07` and `22:36:11` both logged `Whitelist with 6 pairs`
including BTC/USD (the earlier, wasted restart); `2026-09-16 23:27:32`
(after this restart) logs `Whitelist with 5 pairs: ['ETH/USD', 'XMR/USD',
'PAXG/USD', 'SOL/USD', 'TRX/USD']` — BTC is actually gone from what's
running, not just gone from the file on disk. This closes the earlier
"restart before edit" mistake documented below and in this doc's
2026-09-16 ~9:36PM note (kept for the record, not because it's still
true).

**One thing still worth checking before that next restart:** A1's own
STATUS note above says freqtrade runs as *two* processes — port 8080
(`GoldenCrossStrategy`) and port 8081 (`GoldenCrossRegimeFilteredStrategy`,
confirmed by the live `ps aux` pull below — a **different strategy class**,
not just a different port of the same one). So `pkill -f "freqtrade trade
--strategy GoldenCrossStrategy"` correctly targets only port 8080's process
and will NOT catch port 8081 — that risk flagged in the prior pass is now
resolved/moot, confirmed by the actual command line
(`--strategy GoldenCrossRegimeFilteredStrategy --config
user_data/config_regime_filtered.json`) rather than assumption.

`run_cross_asset_paper_trader_loop.py` (PID 95867) never placed real orders
(JSON-simulated fake fills), and is now confirmed killed as of 2026-09-16
23:06:59 PDT — this item is closed.

**B5 is now fully closed for BTC.** All three parts done: the fake-fill
loop killed, both Kraken-side systems (freqtrade + `--ma200-auto`) stopped
from opening new BTC golden-cross entries, live-verified via the running
process's own log rather than just the file on disk. One expected,
not-a-problem data point worth recording: freqtrade's existing BTC position
shows entry $77,224.60 (opened 2026-09-02) while `--ma200-auto`'s own
separate BTC position shows entry $79,009.40 — different because they're
two independent tracking systems that opened at different historical
moments on the same signal, not a new inconsistency. Both will simply hold
to their existing exits (signal-close or the -8% hard stop) with no new
BTC entries from either Kraken-side system going forward — Alpaca is now
the sole executor of new BTC golden-cross entries. **ETH/SOL/PAXG remain
gated** exactly as before — each needs its own verified Alpaca fill before
its Kraken-side entries get cut the same way.

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

**First real live firing, confirmed 2026-09-20:** freqtrade's XMR/USD
position (id 3, entry $532.49 2026-09-02) closed 2026-09-17 08:13:13 UTC
with `exit_reason = 'hard_stop_8pct'`, exit $491.12 — E4's -8% crypto
stop actually triggering live for the first time, not just backtested.
Found while verifying the Section G3 restart (Kraken side was stopped at
the time; the stop itself fired earlier, on its own, before that). Exactly
the designed behavior, not a problem.

---

## SECTION F: LIVE PROCESS RELOAD SEMANTICS + DUPLICATE-PROCESS BUG (found/fixed by the local session, 2026-09-13 evening)

After E4/E5 wired -8%/-12% into `tools/run_ma200_paper_trader.py` and
`tools/run_alpaca_paper_trader.py`, the open question was whether the
already-running live loops needed a restart to pick up the new code. This
session's bridge (`device_bash`) has no visibility into the user's actual
live process table, so an earlier answer given from here (reading
`runtime/ma200_paper_loop.sh` and `tools/run_alpaca_paper_trader_loop.py` in
isolation) concluded "no restart needed" for both sides. **That was wrong
for the Kraken side**, and the correction below came from the other
(local) Claude Code session actually running `ps aux`/`lsof` on the real
processes — not from this session.

**Corrected reload semantics:**
- **Kraken side** — the real live entry point is
  `python -m trading_bot_v4.main --ma200-auto`, which goes through
  `trading_bot_v4/signals/ma200_auto.py`. That module does
  `from tools.run_ma200_paper_trader import main as run_check_cycle` **once**
  at import time, inside one long-running process. Same reload semantics as
  freqtrade (A1): edits to `run_ma200_paper_trader.py` do **not** take effect
  until the process is restarted. `runtime/ma200_paper_loop.sh` (a bash
  `while true` wrapper that would re-launch a fresh subprocess each cycle) is
  **not actually the live mechanism** — it isn't even running — so reading it
  gave a false picture.
- **Alpaca side** — `tools/run_alpaca_paper_trader_loop.py`'s use of
  `runpy.run_path(...)` to re-execute `run_alpaca_paper_trader.py` fresh from
  disk every 4h cycle was read correctly and genuinely does not need a
  restart for code edits. But this process turned out to be a **duplicate**
  (see below), so the "no restart needed" answer was true for a process that
  shouldn't have been running at all.

**Duplicate-process bug (found and fixed by the local session, not this
one):** `python -m trading_bot_v4.main --alpaca-auto` (the current, correct
entry point) was running **alongside** the older, separate
`tools/run_alpaca_paper_trader_loop.py` (PID 54909, alive uninterrupted
since Sep 2) — two independent processes polling and capable of double-
trading the same live Alpaca account. The local session killed the stale
duplicate loop, the stale-code `--ma200-auto` process (PID 15451, running
pre-E4/E5 code), and the corresponding `--alpaca-auto` process, then
restarted `--ma200-auto` and `--alpaca-auto` cleanly (detached, `nohup ... &
disown`) as PIDs **90773** and **90774** (both started 2026-09-13 12:55:32).
A fresh module import against the running code confirmed the correct
constants are now loaded: `HARD_STOP_PCT = -8.0` (crypto, both files) and
`EQUITIES_HARD_STOP_PCT = -12.0` (Alpaca equities branch).

**Standing rule established as a result (binding on every Claude session
working on this project, not just this one):** the user directly stopped
the local session mid-action for killing/restarting his live trading
processes without asking first, stating he has said this "countless times"
before. The local session agreed this was not its call to make unilaterally.
This is now codified in the project's own memory
(`feedback_no_unilateral_process_restarts.md`, plus a line in `MEMORY.md`).
**No Claude session — including this one — runs, kills, or restarts any of
the user's live trading processes without his explicit, in-the-moment
permission.** The correct behavior when a restart or process fix looks
warranted is to report the finding and the proposed fix, then wait to be
told to act. This session's own `device_bash` bridge cannot see live
processes at all, which happens to make this moot in practice here, but the
rule applies regardless of which session or tool is involved.

**Second standing rule, added 2026-09-16, now confirmed twice in one
evening:** this file is being edited concurrently by more than one Claude
session. It has been silently reverted by the local session at least
twice tonight — first dropping the B5 qualifier-module correction, then,
within the same evening, dropping this exact paragraph the second time it
was written. Both times it was almost certainly a write from a stale
in-context copy instead of the current on-disk file. **Before writing to
`FULL_HANDOFF.md`, always re-read it fresh first — never blind-overwrite
from a copy read earlier in the session, even earlier in the same
conversation turn.** If a write reveals the file changed since last read,
merge forward onto the new content; don't clobber it back. If this
paragraph is missing again on a future read, that's not a hint to
recreate it from memory — it's a live demonstration that the rule is still
being violated; fix the underlying habit, not just the missing text.

---

## SECTION G: Cross-system visibility — freqtrade's Telegram can't see Alpaca (built 2026-09-16)

**The gap:** freqtrade's built-in Telegram `/status` only ever knows about
trades in freqtrade's own SQLite DB (`freqtrade-develop/user_data/*.sqlite`),
which only ever contains Kraken pairs. It's not filtering Alpaca out — it
has no code path to Alpaca at all. Confirmed directly: neither
`tools/run_alpaca_paper_trader.py` nor `run_alpaca_paper_trader_loop.py`
has any Telegram import or notify call. So the 43 real Alpaca positions
(Section A0) were structurally invisible from the one place (Telegram) this
project actually checks status from.

**Decision: build a combined on-demand status report, not real-time Alpaca
entry/exit alerts.** Alerts would get noisy fast (42 equities means dozens
of messages on any reset/reshuffle) and still wouldn't answer "what's open
right now" — only new events, not current state. A pulled summary solves
the actual problem (no way to spot-check the Alpaca book from Telegram at
all).

**Built:** `tools/run_combined_status_report.py`, wired as
`python -m trading_bot_v4.main --combined-status` in `trading_bot_v4/main.py`
(alongside the existing `--ma200-auto`/`--alpaca-auto` flags). Pulls
freqtrade's open trades from its SQLite DB (both strategies, ports 8080 and
8081) plus Alpaca's live positions via the API, and sends one message using
the existing `V4TelegramNotifier` class already used by the rest of the
project (`trading_bot_v4/telegram/notifier.py`) — no new Telegram
integration built from scratch. **One-shot, not persistent** — every run
starts a fresh process, reads current code/state off disk, sends, exits.
No restart is ever needed for this one; running it again always reflects
the latest code and latest state automatically.

Tested end-to-end with real delivery confirmed (not just printed — the
`✅ Telegram:` log line only fires after a successful API response). All 42
equities are listed individually per line rather than aggregated (the
user's explicit call after an initial aggregated version undersold the
detail) — verified message length (~1,726 chars) comfortably under
Telegram's 4096-char cap even fully expanded, so no truncation risk as the
book grows somewhat.

**Gap closed 2026-09-16 ~11:27PM PDT — `--ma200-auto`'s own positions are
now included, tested and confirmed live.** Added `_ma200_auto_positions()`
to `run_combined_status_report.py`, reading `runtime/ma200_paper_trades.json`
directly (imported `STATE_PATH` from `run_ma200_paper_trader.py` rather
than duplicating the path) and reporting both of its independent streams
separately: golden-cross (BTC/ETH/XMR/SOL/TRX) and BMSB (PAXG). Real test
run confirmed both new sections render correctly — `--ma200-auto
golden-cross (Kraken)`: 5 open, `--ma200-auto BMSB (Kraken)`: 1 open
(PAXG) — alongside the existing freqtrade and Alpaca sections, and the
`✅ Telegram:` log line confirmed the message actually sent, not just
printed. The report now genuinely covers all three live sources: freqtrade
(both strategies), `--ma200-auto` (both its streams), and Alpaca. No known
gaps remain.

As B5's BTC-cut (done, see above) and any future ETH/SOL/PAXG cuts land,
`--ma200-auto`'s own golden-cross entries will shrink accordingly as
positions close out naturally — this report reads live state each run, so
it reflects that automatically with no further sync needed.

### G2. Root cause of "I still only see the crypto trades" — TWO separate Telegram bots (found 2026-09-19) — ATTEMPT #1, REVERTED same day, see G3

After G was marked done above, the user reported still only ever seeing
Kraken trades in Telegram. Investigation found the real cause: this
project has been running **two separate Telegram bots** the whole time,
not one.

- **"FreqTrade559"** — token prefix `8588987238`, set directly in
  `freqtrade-develop/user_data/config.json`'s `telegram` block. This is
  freqtrade's own native integration — the one with the quick-reply
  keyboard (`/status`, etc.) the user actually watches on his phone.
- **"Test_bot"** — token prefix `8965530307`, set in `.env`'s
  `TELEGRAM_TOKEN`, flowing through `trading_bot_v4/config_v4.py`
  (`V4Config.TELEGRAM_BOT_TOKEN = LegacyConfig.TELEGRAM_BOT_TOKEN`) into
  `V4TelegramNotifier`, used by **every** `trading_bot_v4`-side send:
  `--ma200-auto`'s own trade alerts, and `--combined-status` (Section G
  above). Both bots happen to share the same numeric `chat_id`
  (`8404651515`) purely because that's the user's own Telegram user ID —
  not because they're the same bot/chat.

So `--combined-status` (Section G) was working exactly as built and
tested the whole time — it was just delivering to Test_bot's chat, which
the user doesn't watch, while he was checking freqtrade's native
`/status` in the FreqTrade559 chat (Kraken-only by design, untouched by
any of this). Not a bug in the G work itself; a routing problem one level
up.

**User's explicit final requirement (2026-09-19), verbatim intent:**
stop using Test_bot entirely, redirect everything to FreqTrade559, and
literally extend freqtrade's own native `/status` (and `/status table`)
command — not a new/separate command — so pressing the existing
quick-reply button shows the Kraken trades it already shows **plus**
the Alpaca positions (and `--ma200-auto`'s own positions) in the same
reply.

**Built, 2026-09-19 (file edits only — see caveat below):**

1. **`.env` — `TELEGRAM_TOKEN` redirected** from Test_bot's token
   (`8965530307:...`) to FreqTrade559's token (`8588987238:...`). This
   means every future `trading_bot_v4`-side send (`--ma200-auto` alerts,
   any manual `--combined-status` run, the new patch-checker's `--notify`
   alert below) now lands in the chat the user actually watches. **Open
   question for the user, not yet answered:** whether to also fully
   delete/deactivate Test_bot itself via BotFather, versus just letting
   it go quiet now that nothing sends to it.

2. **`tools/run_combined_status_report.py` — refactored, not rewritten.**
   Split the non-freqtrade portion (Alpaca + `--ma200-auto` positions)
   out into a new `build_extra_only()` function; `build_report()` now
   calls it instead of duplicating the logic. Added `--extra-only` CLI
   support to `main()` — prints just that portion and does **not** also
   send to Telegram itself (the new telegram.py code below does the
   sending; this guards against a double-send on every `/status` press).

3. **`freqtrade-develop/freqtrade/rpc/telegram.py` — extended, both
   `/status` and `/status table`.** This is vendored, third-party
   framework code the project does not own — real surgery, marked
   clearly. Added, wrapped in `# === PROJECT PATCH:
   ALPACA_STATUS_BRIDGE ... ===` comment markers (one block for the new
   function, one marker pair per call site) so the patch is easy to find
   and easy to verify intact later:
   - A new function `_alpaca_status_extra()` (plus `PROJECT_ROOT` /
     `JAVIER_PYTHON` constants) that shells out via `subprocess` to the
     project's own `javier` conda env — `python -m
     tools.run_combined_status_report --extra-only` — and returns its
     stdout. **Why a subprocess bridge, not a direct import:** freqtrade
     runs in its own isolated venv (`freqtrade-develop/.venv`) that does
     not have `alpaca-py` / `python-dotenv` / this project's own
     `tools`/`trading_bot_v4` packages installed, and shouldn't — those
     are this project's dependencies, not freqtrade's. Any failure in the
     bridge is caught and returns a short placeholder string rather than
     raising, so a bridge problem can never break freqtrade's own
     `/status` output.
     **Correction, 2026-09-20 — this premise was wrong.** The user
     pointed out (and this session confirmed via `device_list_dir`) that
     the live freqtrade process is actually launched via
     `/opt/homebrew/Caskroom/miniconda/base/envs/javier/bin/freqtrade`
     (every restart command in `handoff.md`/`handoff2.md`/`handoff3.md`
     used this path, and it's the one that's been live-verified to
     actually work — new PIDs, real log confirmation, more than once).
     `freqtrade-develop/.venv` genuinely exists on disk (confirmed: its
     own `bin/freqtrade`, its own `python3.12`, fully populated) but it
     is NOT what actually runs the live bot — it looks like a leftover
     from initial setup that nothing currently launches. In reality,
     freqtrade runs inside the same `javier` conda env as everything
     else in this project, which already has `alpaca-py` /
     `python-dotenv` / `tools`/`trading_bot_v4` installed. So if a
     `/status`-merge like this is ever revisited, it wouldn't need a
     subprocess bridge at all — a direct import of
     `tools.run_combined_status_report` into `telegram.py` would work,
     since they're already in the same environment. Doesn't change
     anything about G2 being reverted (that was about vendored-code risk,
     a separate concern), but the design would be simpler next time.
   - One call site appended at the end of `_status_msg` (after its
     existing per-trade loop) and one at the end of `_status_table`
     (after its existing per-message loop) — each just calls
     `_alpaca_status_extra()` and, if it returns something, sends one
     more Telegram message with it via the existing `self._send_msg`.
     Nothing about freqtrade's own existing Kraken-trade rendering was
     touched.

4. **New: `tools/check_telegram_patch.py` — the "did this get
   overwritten" checker the user explicitly asked for.** The real risk
   named plainly: `telegram.py` is third-party code; a future
   `freqtrade-develop` upgrade/re-pull could silently overwrite this file
   and this patch, with freqtrade continuing to run fine and `/status`
   continuing to "work" — just quietly back to Kraken-only, with no error
   anywhere to notice. This script:
   - Extracts the three marker-delimited blocks (the bridge function +
     its big comment, and each of the two call sites) from the live
     `telegram.py` and sha256-hashes each independently.
   - Compares against `tools/telegram_patch_manifest.json` (recorded via
     `--record`, done once right after the patch is verified working).
   - `python -m tools.check_telegram_patch` — prints OK / lists exactly
     which block is missing or modified, exits 1 on any problem (usable
     in a script or a cron/schedule check).
   - `python -m tools.check_telegram_patch --notify` — on failure, also
     sends a Telegram alert via the same `V4TelegramNotifier` (which, per
     item 1 above, now lands in the FreqTrade559 chat the user actually
     watches).
   - `python -m tools.check_telegram_patch --record` — re-record the
     expected hashes; run this again any time the patch itself is
     intentionally changed later.

**Caveat — not yet tested live, and this session's execution was
file-edits-only, exactly like the earlier BTC-removal handoff:** this
session's `device_bash` bridge has been unavailable this entire window
("Workspace unavailable... failed to start"), so none of steps 1-4 above
have been restarted, exercised, or confirmed against the real running
system yet. Specifically still needed, per the standing rule in Section F
(no session restarts live processes without the user's explicit,
in-the-moment go-ahead):
- Restart the port-8080 freqtrade process (`GoldenCrossStrategy`) so it
  loads the new `telegram.py` code — port 8081
  (`GoldenCrossRegimeFilteredStrategy`) uses a different, `enabled: false`
  Telegram config and is unaffected either way.
- Restart `--ma200-auto` (or just wait for its next natural cycle) so it
  picks up the new `.env` `TELEGRAM_TOKEN` value.
- Press `/status` and `/status table` for real in the FreqTrade559 chat
  and confirm the extra Alpaca/`--ma200-auto` message actually arrives
  after the existing Kraken trades, with no error and no double-send.
- Run `python -m tools.check_telegram_patch --record` once, immediately
  after that live confirmation, to establish the baseline the checker
  compares against going forward.
- Confirm whether the `javier`-env subprocess call actually succeeds from
  inside freqtrade's process (different working directory, different
  environment than a normal shell) — this is the one part of the design
  that's reasoned through carefully but not yet proven against the real
  system, unlike the BTC-removal work which was fully live-verified
  before being marked done.

**REVERTED — same day, 2026-09-19, before any of the above was ever
restarted or tested live.** The user changed his mind on the approach
right after reading the plan back: rather than editing freqtrade's own
vendored `telegram.py` (real, ongoing maintenance risk on third-party
code) to merge Alpaca into FreqTrade559's `/status`, he asked to undo
all of it and flip the architecture instead — see G3. Because none of
G2's process-restart steps had happened yet (this session's `device_bash`
stayed down the whole time), reverting was pure file edits, cleanly
undoing exactly what was added:
- `freqtrade-develop/freqtrade/rpc/telegram.py` — the `subprocess`/`Path`
  imports, the `_alpaca_status_extra()` function and its big comment
  block, and both call sites in `_status_msg`/`_status_table` were all
  removed. Verified clean: `grep -c "PROJECT PATCH\|ALPACA_STATUS_BRIDGE\|subprocess"`
  returns zero matches, file re-parses with `ast.parse` (92,810 bytes,
  essentially back to the original 92,818).
- `tools/run_combined_status_report.py` — reverted to its pre-G2 form
  (no `build_extra_only()` split, no `--extra-only` flag). Behaviorally
  identical to what Section G originally built and tested.
- `tools/check_telegram_patch.py` and `tools/telegram_patch_manifest.json`
  are now orphaned (they checked for a patch that no longer exists) —
  slated for deletion as part of G3's handoff, since `device_bash` can't
  delete remotely from this session.
- `.env`'s `TELEGRAM_TOKEN` redirect (item 1 in the "Built" list above)
  was never actually applied to the live file in the first place — the
  device bridge refused that specific write ("Writing to .env is not
  permitted via remote tools"), so there was nothing to undo there; it
  was still sitting at Test_bot's original token the whole time.

No live process was ever restarted on this attempt, so nothing about the
user's actual running system changed as a result of G2 — it was entirely
contained to file edits that have now been cleanly reverted before they
ever took effect.

### G3. New direction (2026-09-19): make Test_bot the one true bot — swap freqtrade's own token, don't touch vendored code at all

User's corrected, final direction: instead of merging Alpaca into
FreqTrade559's native `/status` (G2, reverted), swap which bot
freqtrade's own native Telegram integration is wired to. Point
freqtrade's `config.json` `telegram.token` at **Test_bot's** token
instead of FreqTrade559's. Since `.env`'s `TELEGRAM_TOKEN` already points
at Test_bot (untouched throughout G2, see above), this makes Test_bot the
single hub that receives both freqtrade's full native command interface
(the quick-reply keyboard — `/status`, `/status table`, `/daily`,
`/profit`, `/balance`, `/performance`, `/count`, `/start`, `/stop`,
`/help` — with zero code changes, since it's the same native integration
just bound to a different bot token) **and** everything `trading_bot_v4`
itself sends (`--ma200-auto` alerts, and `--combined-status` — Section
G's original combined Kraken+Alpaca report — whenever it's run).

This avoids editing vendored freqtrade code entirely — no ongoing
maintenance risk, no subprocess/venv bridge, no patch-integrity checker
needed. The tradeoff, stated plainly so it isn't rediscovered later by
surprise: freqtrade's own native `/status` button, now living in
Test_bot's chat, is still Kraken-only by itself (same as it always was)
— getting the *combined* Kraken+Alpaca view still means running
`--combined-status` separately (manually, or on a schedule), not
pressing `/status`. If that turns out to not be enough later, merging
them the way G2 attempted is still available to revisit — it's reverted,
not deleted from history (see G2 above for the exact design if so).

**Why this is safe (no new two-bot-polling conflict):** freqtrade's
native integration needs to long-poll (`getUpdates`) its bot's token to
receive button presses; only one process may safely do that per token at
a time. Nothing currently polls Test_bot's token — `V4TelegramNotifier`
only ever sends, never polls — so freqtrade starting to poll it is safe.
Once the swap takes effect, nothing will poll FreqTrade559's old token
any more either; that bot simply goes idle (still exists, just receives
nothing) unless the user separately deletes/deactivates it via BotFather
— still an open question, not yet answered either way.

**Built 2026-09-19 (file edits only, same caveat as always — this
session's `device_bash` is still down):**
1. G2 fully reverted (`telegram.py`, `run_combined_status_report.py`) —
   see G2 above for exact verification.
2. `freqtrade-develop/user_data/config.json` — `telegram.token` changed
   from FreqTrade559's token (`8588987238:...`) to Test_bot's
   (`8965530307:...`). `chat_id` unchanged (`8404651515` — this is the
   user's own Telegram user ID, not bot-specific, so it's correct
   regardless of which bot token is used). Port 8081's
   `config_regime_filtered.json` deliberately left untouched — its
   Telegram block is `enabled: false` already (never sends), no reason
   to touch it.
3. `.env`'s `TELEGRAM_TOKEN` — **correction, 2026-09-20, verified live by
   the VS Code session:** this is actually already `8965530307:...`
   (Test_bot's token) — NOT the original FreqTrade559 token this doc
   assumed above. That assumption ("no change needed, still Test_bot's
   original token") turns out to have been wrong; what's true is that
   `.env` already matches Test_bot's token by the time this was checked,
   so there's nothing to fix — it's actually more consistent than
   planned (both freqtrade's `config.json` and `.env`'s `TELEGRAM_TOKEN`
   now point at the same bot). Flagged here rather than silently
   corrected in place, per this doc's own standing rule about not
   quietly overwriting a prior claim.

**Live status as of 2026-09-20 (verified by the VS Code session, not
this one — this session's `device_bash` is still down):**
- Both `telegram.py`'s revert and `config.json`'s token swap are
  confirmed in place on disk, exactly as built.
- **The port-8080 freqtrade process (`GoldenCrossStrategy`) is currently
  STOPPED** — the user had the VS Code session stop it while verifying,
  separately from anything in `handoff3.md`. So as of right now, nothing
  is polling either FreqTrade559's or Test_bot's token for the native
  command interface, and — more importantly — **the live golden-cross
  strategy engine for ETH/XMR/PAXG/SOL/TRX on Kraken is not running at
  all**: no entry/exit signal checks, no position management, while
  stopped.
- `tools/check_telegram_patch.py` / `tools/telegram_patch_manifest.json`
  are still present (not yet deleted) — confirmed by this session via
  `device_list_dir`.

**RESTARTED AND LIVE-VERIFIED, 2026-09-20 (by the VS Code session, with
the user's explicit go-ahead first — standing rule honored):**
- New PID **23539**, clean startup, no errors/tracebacks/conflicts in the
  log. `Whitelist with 5 pairs: ['ETH/USD', 'XMR/USD', 'PAXG/USD',
  'SOL/USD', 'TRX/USD']` confirms BTC still correctly excluded.
- **Telegram authenticated successfully against Test_bot's token** —
  `telegram.ext.Application - INFO - Application started` with no auth
  error, plus a clean worker heartbeat right after
  (`state='RUNNING'`) — this is the actual proof the token swap works
  live, not just that the process didn't crash.
- `rpc.telegram is listening for following commands` log line confirms
  the full native command set is live on this token: `status`, `profit`,
  `balance`, `start`, `stop`, `forceexit`/`forcebuy`, `performance`,
  `daily`/`weekly`/`monthly`, `count`, `whitelist`/`blacklist`, `help`,
  and more — this is G3's actual goal (Test_bot gets the full button
  interface), confirmed structurally at the RPC layer, not just assumed.
- **Real open trades found on restart:** BTC/USD (id 1, $77,224.60),
  ETH/USD (id 2, $2,390.08), SOL/USD (id 5, $99.35), TRX/USD (id 7,
  $0.335689). **XMR/USD (id 3) is no longer open** — checked directly in
  `tradesv3.sqlite`: it closed 2026-09-17 08:13:13 UTC, `exit_reason =
  'hard_stop_8pct'`, entry $532.49 → exit $491.12. **This is the first
  real live firing of Section E4's -8% hard stop** (previously only
  backtested/wired, never actually triggered live before now) — a
  genuinely new data point, not a problem: the stop did exactly what it
  was built to do, on its own, while freqtrade happened to be stopped
  for unrelated reasons. Worth its own line in Section E's status.
- **Still not done:** `tools/check_telegram_patch.py` and
  `tools/telegram_patch_manifest.json` were not deleted during this
  restart pass — still orphaned, harmless, low priority cleanup.
- `--ma200-auto`'s own running status still unconfirmed as of this
  checkpoint.

**User's live check in Telegram, 2026-09-20 — one real gap confirmed,
not a bug, exactly the documented tradeoff:** pressing `/status` in
Test_bot does **not** show Alpaca equities — confirmed directly by the
user ("as you can see its not expanding to equites"). This is expected
and was called out explicitly when G3 was built: the token swap gives
Test_bot the native command interface, but `/status` itself is still
freqtrade's own Kraken-only view — it was never wired to show Alpaca or
equities. Getting that requires either running `--combined-status`
separately (already built, already lands in Test_bot via `.env`), or
redoing something like G2's merge — which, per the correction above, can
now be done as a plain direct import (no subprocess bridge needed, same
`javier` conda env) rather than the more complex design G2 first tried.

### G4. The merge, done right — direct import, pointed at Test_bot (2026-09-20)

The user pushed back correctly: his original 2026-09-19 ask ("when I hit
`/status` it will not only show the freqtrade stuff but it will also show
the alpaca stuff") was never actually dropped when he asked to swap bots
— "make it so it shows everything on the test_bot" meant the same merge,
just re-targeted. G3 only did the token swap and treated the pre-existing
separate `--combined-status` report as good enough; that was reading the
request narrower than it was given, not something the user signaled as
optional. Corrected here rather than left as a gap.

**Built 2026-09-20 (file edits only, same standing caveat — this
session's `device_bash` is still down):**
- `freqtrade-develop/freqtrade/rpc/telegram.py` — the
  `ALPACA_STATUS_BRIDGE` patch re-applied, same three marker-wrapped
  blocks as G2 (one function, two call sites in `_status_msg` and
  `_status_table`), but **simplified per the G2 correction above:**
  `_alpaca_status_extra()` now does a plain direct import of
  `tools.run_combined_status_report.build_extra_only()` (inserting the
  project root onto `sys.path` if needed) instead of shelling out via
  `subprocess` to a separate interpreter — safe because freqtrade
  actually runs in the same `javier` conda env. No `subprocess`/`Path`
  module-level imports needed this time; `sys`/`Path` are imported
  locally inside the function. Verified: `ast.parse` clean, all 6 marker
  lines present (`grep -n "PROJECT PATCH"` → lines 64, 119, 914, 918,
  990, 994).
- `tools/run_combined_status_report.py` — `build_extra_only()` and the
  `--extra-only` CLI flag re-added (same shape as G2's version — this
  file's revert/re-add round-trip was pure churn, not a design change).
- `tools/check_telegram_patch.py` — unchanged, still works as-is (it
  checks by marker-delimited block content, agnostic of what's inside);
  `tools/telegram_patch_manifest.json` re-recorded against the new
  (direct-import) patch content via `--record`.

**Still needed — restart + real test, handed off (see `handoff4.md`),
same standing rule as always (no restart without the user's go-ahead):**
- Restart the port-8080 freqtrade process (currently PID 23539, running
  the pre-G4 code) so it loads this patch.
- Confirm the direct import actually works from inside freqtrade's
  process at runtime — reasoned through carefully (same env, `sys.path`
  handled) but not yet proven live, same category of unverified-until-
  tested as G2's subprocess version was before it got reverted.
- Press `/status` and `/status table` for real in Test_bot and confirm
  the extra Alpaca/`--ma200-auto` message actually arrives after the
  existing Kraken trades.
- Note: `check_telegram_patch.py`/`telegram_patch_manifest.json` are
  back in active use as of this rebuild — nothing to delete this round.

### G5. One-command consolidated restart script (built 2026-09-20, in
response to "how about you make all this run with 1 command also CLEAN
IT UP")

After several rounds of copy-pasting restart snippets across
`handoff.md`–`handoff4.md` (each with its own incrementing log filename
— `restart.log` through `restart5.log`/`freqtrade_live_trade_restart4.log`
etc.) to fix Test_bot going non-responsive (root cause: freqtrade/port
8080 had been stopped since G3 and never restarted — unrelated to the
separate `--alpaca-auto` restart the user had done), the user asked for
this to be one command instead of a growing pile of handoff docs.

Built `tools/restart_bots.sh` — single script that:
- Stops (if running) and restarts, in order: freqtrade port 8080
  (`GoldenCrossStrategy`), `--ma200-auto`, `--alpaca-auto`.
- **Deliberately never touches port 8081**
  (`GoldenCrossRegimeFilteredStrategy`) — separate forward-test process,
  `telegram.enabled: false`, out of scope for all of this restart work.
- Logs to fixed filenames under a new `logs/` dir at the project root
  (`logs/freqtrade_8080.log`, `logs/ma200_auto.log`,
  `logs/alpaca_auto.log`), each run appending a
  `=== restart <UTC timestamp> ===` separator instead of creating a new
  incrementing filename every time — directly fixes the log-sprawl part
  of "clean it up."
- Ends with a live `ps aux` check across all four strategy patterns, a
  tail of freqtrade's fresh log, and a grep for
  `conflict|traceback|unauthorized`.

Usage (one command, from the project root): `bash tools/restart_bots.sh`

**CONFIRMED RUN LIVE 2026-09-20/21:** user ran it directly. New PIDs:
freqtrade 68884, `--ma200-auto` 68889, `--alpaca-auto` 68894. Clean
startup, no errors, Telegram `Application started`, whitelist correct (5
pairs, no BTC), open trades found (BTC/ETH/SOL/TRX — XMR closed via the
-8% stop). Port 8081 (PID 63313) correctly left untouched. This restart
also covered G4's still-needed restart step.

**Not done as part of "clean it up"** (needs either `device_bash` coming
back or the local/VS Code session, since this session still has no
shell access): deleting the now-superseded `handoff.md`–`handoff4.md`
files and the old `/private/tmp/freqtrade_live_trade_restart*.log` /
`ma200_auto*.log` stragglers. Their content is already folded into this
doc, so they're safe to delete once someone with shell access wants to.

### G6. Real, non-hacky live web dashboard (built 2026-09-21) — replaces
the "Golden Cross" DOM-injection hack in freqUI's index.html

**Context, from the user directly:** the "Golden Cross" nav tab / inline
panel the user's screenshots showed (a "Market Snapshot" table of 72
equities/17 commodities/5 bonds sitting on freqUI's real Dashboard page)
was built by the local VS Code Claude Code session, not this session,
and the user does not want more of that approach — "the goldencross
thing was a nightmare, it was trying to make it separate, it never made
the live trading for assets other then the kraken." The screenshots were
the user showing what he ultimately wants his web UI to look like, not a
bug report on that specific implementation.

**What that hack actually was, found by reading the real files (this
session has no shell access, so this was traced entirely via
`device_stage_files`/`device_list_dir`):**
- `freqtrade-develop/freqtrade/rpc/api_server/ui/installed/index.html`
  (the *compiled* freqUI dist, not source — freqUI ships as a pre-built
  Vue SPA here, no source to add a real route to) has a hand-written
  `<script>` block injected into it. It uses a `MutationObserver` +
  200ms poll to find freqUI's own nav list in the live DOM, clones a nav
  item to add a fake "Golden Cross" tab (opens a full-screen iframe
  overlay), and *separately* appends a second iframe directly to
  `document.body` (sibling to Vue's `#app`, never inside Vue-managed
  DOM) that's shown/hidden via a `setInterval` checking whether the text
  "Bot comparison" appears on screen — i.e. whether the real Dashboard
  route is currently rendered. Both iframes point at
  `http://localhost:8765/golden_cross_backtest_dashboard.html`, a
  completely separate static backtest report (equities/commodities/
  bonds/forex trend state — **not live positions, nothing there is
  actually traded**), served by `tools/serve_dashboard.py` on its own
  port.
- The injected script's own comments admit an earlier version of this
  same hack *did* insert its panel inside Vue's own layout container
  and "desynced Vue's DOM bookkeeping," breaking the real Trade page
  (Multi Pane + Chart came back blank) — patched around by never
  touching Vue's DOM, but still a fragile hand-edit sitting on top of a
  compiled app, not a real feature.
- Separately (unrelated to the hack): `tools/serve_dashboard.py` already
  had the *right* idea in its own comments — it explicitly enables CORS
  so "the real freqUI page... fetches the live paper-trade/snapshot JSON
  from here client-side to merge into its own Dashboard." That plan was
  never finished; the DOM-injection hack was built instead of a real
  client-side fetch+render.

**What this session built instead (file edits only, standing caveat —
`device_bash` still down this whole build):**
- `tools/serve_dashboard.py` — added a real `/api/live_trades` JSON
  endpoint. Computes fresh on every request (no cache to go stale) by
  importing the exact same, already-tested functions
  `tools/run_combined_status_report.py` uses for Test_bot's `/status`
  merge (G4): `FREQTRADE_DBS`/`_freqtrade_open_trades` (Kraken, both
  strategies' own SQLite DBs), `_ma200_auto_positions()`, and
  `_alpaca_positions()`. Any failure still returns valid JSON
  (`{"ok": false, "error": ...}`) rather than breaking the page. Static
  file serving (the old `cross_asset_dashboard.html` etc.) and the
  UTF-8/CORS handling are untouched from before.
- `runtime/live_dashboard.html` — new, self-contained page (own
  CSS/JS, no build step, no framework). Two sections, clearly separated:
  1. **"Live positions — real, right now"** — three cards (Kraken via
     freqtrade, `--ma200-auto`, Alpaca), fetched from `/api/live_trades`
     on load and every 15s, rendered as real tables with live PnL%.
  2. **"Market snapshot — backtest / signal state (not live
     positions)"** — the existing, already-working
     `golden_cross_backtest_dashboard.html` embedded via a plain
     `<iframe>` — a completely normal embed on a page this session
     owns, not a DOM-injection hack into someone else's compiled app,
     so it carries none of the Vue-DOM-corruption risk the old approach
     had. Explicitly labeled that nothing above the divider trades
     commodities/bonds/forex live.
- freqUI itself was **not touched** — the injected script in its
  `index.html` is still there, doing what it always did. Not removed
  yet; the user hasn't said whether he wants it gone now that a real
  replacement exists, so it's left alone pending that decision (see
  below).

**CONFIRMED LIVE 2026-09-21:** first attempt showed a "disconnected"
banner — root cause was just that the user's first restart of
`serve_dashboard.py` launched the new code, but `print()` output is
block-buffered (not line-buffered) when redirected to a log file, so
the confirming startup lines hadn't been flushed yet; there was no
actual crash, `pgrep` confirmed the process (PID 75050) was up the
whole time. Not the bug it looked like. After a real restart, the user
confirmed all three live cards populate correctly with real, matching
data: Kraken (freqtrade) 4 open (BTC/ETH/SOL/TRX, all showing live
unrealized %), `--ma200-auto` 5 golden-cross + 1 BMSB (BTC/ETH/XMR/SOL/
TRX + PAXG), Alpaca 43 open positions (AAPL through XOM, plus BTCUSD as
the migrated-crypto position) — the market-snapshot iframe underneath
rendered correctly the whole time (never depended on the API). This
page is now the real, working "one true dashboard."

**Still open:**
- Whether the user wants the old "Golden Cross" DOM-injection hack in
  freqUI's `index.html` removed now that this real page exists — not
  removed proactively, asked, not yet answered.

### G7. What the user actually meant: Alpaca + --ma200-auto as native
rows inside the REAL freqUI, not a separate page (built 2026-09-21)

**Correction, from the user directly, after G6:** "i want this to
appear as it did in the freqtrades layout. Any live trades are tracked
on a live view as the screen shot showed and ever[y] other menu." G6's
`live_dashboard.html` was a reasonable read of "everything, done
properly" but was still its own separate page — not what was actually
asked. The real ask: the genuine freqUI app (Trade tab, its live trade
list/multi-pane, Dashboard, Chart, Logs — unmodified UI, unmodified
styling) should show Alpaca's and `--ma200-auto`'s positions as native
rows, the same way it already shows Kraken trades.

**Built (file edits only, standing caveat — `device_bash` still down):**
- `tools/run_combined_status_report.py`:
  - `_ma200_auto_positions()`'s `_with_pnl()` and `_alpaca_positions()`
    now also return `current_price`/`avg_entry_price`/`qty` (additive —
    all already available from each source, just not extracted before).
  - New `build_synthetic_open_trades(start_id=-900001)`: reshapes both
    sources into freqtrade's own `OpenTradeSchema` field names (exactly
    what `/api/v1/status` — and therefore freqUI's own Trade page —
    already expects). Every synthetic `trade_id` is a large negative
    number, counting down from -900001, so it can never collide with a
    real freqtrade trade_id (freqtrade's own are positive/
    autoincrementing) — any frontend action on one of these rows
    (force-exit, etc.) hits freqtrade's real lookup with an ID that will
    never be found, a harmless "not found," never a real action on a
    real position. `open_date`/`open_fill_date` are approximate ("now,"
    each call) since neither source gives a real entry timestamp
    without extra calls this bridge doesn't make — the one field that's
    approximate rather than live; entry price, current price, and PnL
    are all real.
- `freqtrade-develop/freqtrade/rpc/api_server/api_v1.py` — new
  `ALPACA_STATUS_BRIDGE` patch block (same marker convention as the
  Telegram bridge in `telegram.py`, same direct-import pattern, same
  swallow-and-log-on-failure safety). `/status` now returns
  `real_trades + _synthetic_open_trades_extra()`. Patched at this one
  endpoint because freqUI's shared bot-data store (`ftbotwrapper.js`)
  polls `/status` for the open-trades list and that store's data reaches
  every page that shows trades — no frontend changes needed at all.

**Then the user pushed further — "do it" — on the chart gap that first
build left open (candlestick chart not working for Alpaca-only symbols).
Built the same day, still 2026-09-21:**
- `tools/run_combined_status_report.py` — new
  `fetch_alpaca_or_crypto_candles(base_symbol, timeframe, limit)`: real
  OHLCV bars from Alpaca's own market-data API. Tries stock bars first
  (covers equities: AAPL, MSFT, ...), falls back to crypto bars (covers
  BTC — deliberately dropped from the strategy's Kraken whitelist per
  B5, so freqtrade has nothing analyzed for it right now either — and
  anything else Kraken-native but outside the active whitelist). Returns
  an empty DataFrame, never raises, if neither source has data.
- `freqtrade-develop/freqtrade/rpc/api_server/api_v1.py` — new
  `ALPACA_CANDLE_BRIDGE` patch block on `/pair_candles` (both the GET
  and POST/filtered variants — this is the endpoint freqUI's own
  candlestick chart calls for whatever pair is selected). Tries
  freqtrade's real answer first; only when that comes back with zero
  candles does it fetch from the function above and reshape the result
  through freqtrade's own `RPC._convert_dataframe_to_dict` static method
  so the response is identical in shape to a normal analyzed-dataframe
  response — freqUI's chart code needs no changes to render it. If the
  bridge also comes back empty, the original (empty) result is returned
  unchanged — never worse than before either patch existed.
- Import paths (`alpaca.data.historical.StockHistoricalDataClient` /
  `CryptoHistoricalDataClient`, `alpaca.data.requests.StockBarsRequest` /
  `CryptoBarsRequest`, `alpaca.data.timeframe.TimeFrame`/
  `TimeFrameUnit`) match the modern alpaca-py package layout, consistent
  with how `_alpaca_positions()` already imports
  `alpaca.trading.client.TradingClient` elsewhere in this same file —
  NOT independently verified against the exact installed alpaca-py
  version from this session (no shell access); every call site is
  wrapped in try/except specifically so a version mismatch degrades to
  "no chart" rather than breaking anything.

**Not yet done — a natural next piece, not attempted this round:**
extending `/api/v1/balance` (and possibly `/count`) the same way, so
Dashboard's account-total/balance numbers also reflect Alpaca's real
equity, not just the Trade tab's row list and its chart. `Balances` is a
more complex nested schema than `OpenTradeSchema`; skipped for now
rather than ship a rushed patch without being able to test it live from
this session.

**Still needed — user/local-session action, this session has no
shell:** freqtrade (port 8080) needs a restart to load both new patches
— `bash tools/restart_bots.sh` covers it (same one command as G5,
restarts freqtrade + `--ma200-auto` + `--alpaca-auto` together; port
8081 untouched as always). After restarting: open freqUI's own Trade
tab (`127.0.0.1:8080/trade`), confirm the Alpaca/`--ma200-auto` rows
appear alongside the real Kraken trades, AND click one of the
Alpaca-only rows (e.g. an equity) to confirm its candlestick chart now
actually draws. NOT yet confirmed live as of this writing — this is the
second time this session has written a G7-equivalent update to this
doc's G-section only to find it missing on the next fresh read (the
local/VS Code session appears to be saving its own full-file rewrites
over this one concurrently) — if a fresh read of this doc is still
missing G7 when this gets picked up again, that's why, not a sign the
work itself didn't happen; re-check the actual code files
(`tools/run_combined_status_report.py`,
`freqtrade-develop/freqtrade/rpc/api_server/api_v1.py`) for the
ALPACA_STATUS_BRIDGE/ALPACA_CANDLE_BRIDGE markers directly if this
section's presence is ever in doubt.

### G8. G7 debugged live, then the real remaining gap found and fixed:
the Multi Pane sidebar (built 2026-09-21, same day as G7, cloud session,
`device_bash` still down all session — everything below is file edits
via `device_stage_files`/`device_commit_files` only)

**G7's own bug, found from the user's log paste:** both `parents[3]`
occurrences in `api_v1.py` (`_synthetic_open_trades_extra()` and
`_bridge_pair_candles()`) were off by one directory level —
`api_v1.py` sits one level deeper than `telegram.py`, whose working
`parents[3]` was copied without adjusting for that. Fixed to
`parents[4]` in both places. **Confirmed fully working** live, via a
credential-safe script the user ran himself (reads `config.json`
locally, never pasted a secret here): `/status` returns all 53 trades
(4 real Kraken + 6 `--ma200-auto` + 43 Alpaca), correctly shaped.

**Then two real, separate bugs found in freqUI itself (not this
project's bridge code), debugged live via DevTools with the user
driving the browser:**
1. An expired JWT whose silent refresh-and-retry failed
   (`AxiosError: timeout of 20000ms exceeded` after a "Refreshing
   token..." log line) left the Open Trades table frozen on stale data
   — the failure path only logs to console, never recovers. **Fixed**
   by a full logout/login (a hard refresh alone reuses the same broken
   token from localStorage, so that alone does not fix it).
2. After that, the Open Trades table still visually showed only 1 row.
   `document.querySelectorAll('.open-trades tbody tr').length` in the
   console returned 53 — Vue had rendered everything correctly; the
   panel's fixed/small height in the vue-grid-layout dashboard was
   clipping it. **Fixed** by scrolling within the panel / resizing it.
   Both confirmed resolved live: all 53 rows, including the full Alpaca
   list, visible in the real Open Trades table.

**MA overlay lines (ma50/ma200) added to the Alpaca-side candlestick
chart**, matching `GoldenCrossStrategy`'s own `plot_config` (the same
colors/names it already draws for real Kraken pairs) — the user's own
choice from a clarifying question, over entry markers or nothing.
`tools/run_combined_status_report.py`'s
`fetch_alpaca_or_crypto_candles()` now fetches `limit + 210` candles
(`MA_WARMUP = 210`, matching the strategy's own `startup_candle_count`)
instead of exactly `limit`, computes `ma50`/`ma200` as plain rolling
means via a new `_with_ma_overlay()` helper (identical output to
`ta.SMA` for a simple average), then trims the warmup padding back off
so the response still has exactly `limit` rows. **Confirmed computed
correctly server-side** via the live `/pair_candles` response for
AAPL/USD: its `all_columns` field shows
`["date","open","high","low","close","volume","ma50","ma200"]`. The
lines still weren't drawing, though — traced to freqUI's own "Plot
Configurator" (gear icon next to the chart's "default" template
dropdown) having an empty `main_plot` indicator profile ("No data")
for every pair, not just Alpaca-derived ones — a freqUI display-config
gap, unrelated to this bridge. The user was mid-way through clicking
"From strategy" then "Save" in that dialog when the conversation moved
on — **not yet confirmed whether that made the lines appear; re-check
this first if MA lines still aren't showing.**

**The actual remaining issue, per the user's own explicit correction
("you are not taking care of the actual issue"):** the Trade page's
"Multi Pane" left-sidebar pair list (`PairSummary.vue`, compiled into
`ui/installed/assets/PairSummary-*.js`) showed real Kraken pairs only
(PAXG/USD, XMR/USD, BTC/USD, ETH/USD, SOL/USD, TRX/USD) — never any
Alpaca/`--ma200-auto` pair — even after G7 made the Open Trades table
show all 53 correctly. Reading that component confirmed why: it builds
its rows with `pairlist.forEach(pair => trades.filter(t => t.pair ===
pair))` — it iterates the *whitelist*, not the trades, so a pair
missing from the whitelist can never appear there no matter what
`/status` returns.

That whitelist reaches the browser over the websocket
(`/api/v1/message/ws`, message type `whitelist`), server-side in
`freqtrade/rpc/api_server/api_ws.py`'s `_process_consumer_request()` —
confirmed by reading it directly: on `RPCRequestType.WHITELIST` it
calls `rpc._ws_request_whitelist()` (in `rpc.py`, a **plain read** of
`self._freqtrade.active_pair_whitelist`) and wraps the result in a
`WSWhitelistMessage`. `active_pair_whitelist` is *not* just a display
list — confirmed in `freqtradebot.py`: it's reset every loop iteration
(`self.active_pair_whitelist = self._refresh_active_whitelist(trades)`,
~line 262) and immediately used to decide what real OHLCV to fetch and
analyze (`self.dataprovider.refresh(...)`, `self.strategy.analyze(...)`,
lines ~265-275) — i.e. it's the live bot's own real trading-decision
state, not something safe to inject fake pairs into directly (that
would make the live engine try to fetch Kraken data for "AAPL/USD"
every loop and fail repeatedly). This is why this needed to be handled
differently from G7's bridges and why the user was asked first — his
answer was to investigate further rather than pick blind between
"leave it Kraken-only" and "risky DOM injection."

**Fixed the safe way, mirroring G7's own pattern exactly:** new
`ALPACA_WHITELIST_BRIDGE` patch block added to `api_ws.py`'s
`RPCRequestType.WHITELIST` branch — appends the same pair names
`ALPACA_STATUS_BRIDGE` already surfaces (via the shared
`tools.run_combined_status_report.build_synthetic_open_trades()`) onto
the *local* `whitelist` variable, after `rpc._ws_request_whitelist()`
is called and before it's wrapped in the response — never writing back
to `self._freqtrade.active_pair_whitelist` or `rpc` itself. The live
bot loop reads its own attribute directly in-process and never goes
through this websocket handler, so it is completely unaffected — this
is a display-layer-only change, architecturally identical to
`ALPACA_STATUS_BRIDGE`/`ALPACA_CANDLE_BRIDGE`, just one layer further
in (websocket response, not REST response). Also confirmed from
`PairSummary.vue`'s own code that this is sufficient on its own: its
click handler just does `activeBot.selectedPair = pair` (a plain
string), so an appended pair row is fully interactive (selectable,
loads its own chart) with no other frontend change needed. File edit
only; syntax-checked (`ast.parse`); verified via a fresh
`device_list_dir` immediately before commit that no concurrent edit
had touched `api_ws.py` since it was staged; committed with the
mtime guard.

**Still needed — user/local-session action, this session has no
shell:** restart (`bash tools/restart_bots.sh`) to load this patch,
then reload freqUI's Trade page and confirm the Multi Pane sidebar now
lists the Alpaca/`--ma200-auto` pairs alongside the Kraken ones, and
that clicking one loads its chart correctly. **CONFIRMED working live**
— user's own screenshot after restart shows the sidebar populated with
Alpaca equities (XOM, WFC, VZ, VTI, ...) alongside Kraken pairs, MA
lines drawing on the chart too (the Plot Configurator "From strategy" →
"Save" step from earlier in G8 also confirmed working).

**Follow-up from reading that live XOM chart together with the user:**
apparent death cross on the chart didn't match the position still being
held. Root cause, confirmed from code + the account's own live log
(`logs/alpaca_auto.log`, exact restart cycle `2026-09-21T21:50Z` says
`XOM: LONG, holding`): the chart's ma50/ma200 are 4h Alpaca bars
(matching the chart's own timeframe); the real exit decision
(`tools/run_alpaca_paper_trader.py`, `--alpaca-auto`, checked every 4h)
uses DAILY Yahoo closes instead — different, slower series, so a 4h
crossover doesn't mean the real daily-bar signal has flipped.

**User asked to switch the live equities signal to 4h Alpaca bars
globally; agreed instead to backtest 4h first** (the 60-symbol
universe's robustness/-12% hard stop were only ever validated on daily
bars — flagged before touching anything, same bar every other change
here has cleared). Built
`tools/run_cross_asset_golden_cross_backtest_4h_alpaca.py` (same
PF-excl-top-3 methodology as the daily version, sourced from
`StockHistoricalDataClient` at `TimeFrame(4, Hour)`). **User ran it — first pass crashed** (`TypeError`
formatting a `None` daily PF-excl3 for a symbol whose daily value was
itself infinite/null in the source JSON) after printing 26/60 rows.
Fixed the crash (an `isinstance` guard before formatting — briefly two
independent fixes in flight, this cloud session's and the local/VS Code
session's, both converging on the same root cause with no conflict) and
added a same-universe portfolio-level simulation (reusing
`run_cross_asset_golden_cross_backtest.py`'s own event-driven sim) so
the comparison is portfolio-vs-portfolio, not just per-symbol pass/fail
counts.

For a fair comparison point (raw daily backtest history goes back to
1962 for some tickers, so its own "portfolio" number is not a fair
baseline): computed directly from the already-existing daily trades in
`cross_asset_golden_cross_results.json`, restricted to this same
60-symbol universe AND to the last 6 years (matching the 4h script's
own lookback) — **daily, 6y, this universe only: $100k → $166,686
(+66.7%), max drawdown -2.0%, 204 trades.**

**RUN COMPLETE (full 60-symbol run, 2026-09-21). VERDICT: NO — 4h does
not hold up, live signal stays on daily bars.** Only 6/60 symbols (10%)
robust on daily are still robust on 4h: LLY (PF_x3 1.91), GC=F/GLD
(1.55), BRK-B (1.54), SPG (1.04), CAT (1.04), BAC (1.03) — the other 54
fail, many badly (XOM 0.32, MSFT 0.36, NVDA 0.21, AAPL 0.58, SPY 0.87,
QQQ 0.62). Portfolio-level, same universe/window/sizing as the daily
number above: **4h — $100k → $122,132 (+22.1%), max drawdown -16.4%,
1,087 trades** (pooled PF 1.29). Worse than daily on every axis: ~1/3
the return, ~8x the drawdown, ~5x the trade count (a 200-bar 4h MA
covers ~5 months vs ~10 for the daily version — a genuinely faster,
noisier rule, not a re-sampling of the same one, exactly the caveat this
doc flagged before the test ran). **Verdict given to and accepted by the
user: `run_alpaca_paper_trader.py` stays on daily bars for equities.
No live signal change.**

**Separately, the ETH-USD/SOL-USD insufficient-cash bug (found via
`logs/alpaca_auto.log`, confirmed recurring on every cycle since at
least 2026-09-20T07:17Z) — user agreed to the proposed fix.** Added
`ALPACA_CASH_CHECK` patch block to `run_alpaca_paper_trader.py`: tracks
a running cash estimate through the loop (`available_cash`, seeded from
`account.cash`, decremented after each order this run actually submits,
`CASH_SAFETY_BUFFER = $50`), skips an entry BEFORE attempting it once
sizing would exceed that estimate (prints + calls a new local `notify()`
helper, same Telegram pattern as `run_ma200_paper_trader.py` -- this
script had zero Telegram integration before this patch). File edit
only; syntax-checked; verified no concurrent edit via `device_list_dir`
before commit; committed with the mtime guard. **CONFIRMED restarted
live by the user (2026-09-21) — `bash tools/restart_bots.sh` run.** Not
yet independently re-confirmed from a fresh `logs/alpaca_auto.log` cycle
that the `SKIP ...` message actually replaced `OPEN FAILED ...
insufficient balance` for ETH-USD/SOL-USD (that cycle hadn't come around
yet as of this writing) — worth a spot-check next time the log is
pulled, but the user has done everything required on his end here.

### G9. Chart now matches the real signal (equities: daily); G4's
Telegram merge spot-checked as working (2026-09-21, cloud session,
`device_bash` still down, file edits only)

**User's explicit instruction, given the G8 verdict above: make the
chart match the real signal -- change its timeframe from 4h to daily
for equities.** Edited `tools/run_combined_status_report.py`'s
`fetch_alpaca_or_crypto_candles()`: previously it fetched whatever
timeframe freqUI's own request asked for (always "4h", since
`GoldenCrossStrategy` is a 4h Kraken strategy and freqUI requests
candles at the active strategy's timeframe for every pair, Alpaca-
bridged ones included) for BOTH the stock-bars path (equities/
commodities/bonds, including the GLD/PPLT futures proxies and IEF) and
the crypto-bars fallback path (migrated BTC/ETH/SOL/PAXG). Split this
into a new `_tf_params()` helper called separately per path: the stock
path now always requests `TimeFrame(1, Day)` regardless of what
timeframe freqUI asked for (a new `STOCK_CHART_TIMEFRAME = "1d"`
constant, `EQUITIES_CHART_DAILY` marker comment); the crypto path is
deliberately left on whatever timeframe was actually requested (still
4h) because migrated crypto's own live signal genuinely is 4h Kraken
bars -- that's its validated timeframe (Section B2/B3: daily bars only
found BTC robust for crypto, 4h Kraken found BTC/ETH/PAXG/SOL/TRX/XMR
all robust, the opposite finding from equities) -- forcing daily there
would just reintroduce the same chart/signal mismatch in reverse.
Syntax-checked (`ast.parse`); verified via a fresh `device_list_dir`
immediately before commit that no concurrent edit had touched the file
since it was staged; committed with the mtime guard.

**This exact fix was then clobbered back to its pre-fix state by a
concurrent edit before ever being tested live** -- caught by re-reading
the on-device file directly (not assumed), redone from scratch. While
redoing it, found and fixed a second, real bug in the original attempt:
`fetch_alpaca_or_crypto_candles()` only ever returned the OHLCV
DataFrame, so its caller (`api_v1.py`'s `_bridge_pair_candles`) kept
reporting the response's `timeframe`/`timeframe_ms` using the
ORIGINALLY REQUESTED "4h" even for the now-daily equities data --
mismatched metadata alongside now-correct data. Fixed by changing the
function's return type to `(df, resolved_timeframe)` ("1d" for the
stock-bars hit, whatever was requested for the crypto-bars hit) and
updating `_bridge_pair_candles` to pass `resolved_timeframe` into
`RPC._convert_dataframe_to_dict` instead of the original `timeframe`.
Both files re-verified on disk after this second commit.

**CONFIRMED WORKING LIVE, 2026-09-22** -- user's own screenshot of
XOM/USD after restart: chart's date range now spans Oct 2021 -> Sep
2023 (~2 years) for the same candle-count request that previously only
spanned ~4 months of 4h bars, and the ma50/ma200 lines are visibly
smoother/slower, consistent with real daily 50/200 SMAs rather than the
old faster 4h-based ones. (The chart header still reads "GoldenCrossStrategy
| 4h" -- that label reflects the bot's globally configured strategy
timeframe, unrelated to this per-pair data-source fix, and was never
going to change without editing freqUI's own compiled JS, which is out
of scope here -- purely cosmetic, not a sign anything is still wrong.)

**G4's Telegram `/status` merge (written 2026-09-20, restart/test
status previously unconfirmed) -- spot-checked working.** User pasted a
`/status`-style listing this session showing three sections together:
`--ma200-auto golden-cross (Kraken): 5 open` (BTC/ETH/XMR/SOL/TRX),
`--ma200-auto BMSB (Kraken): 1 open` (PAXG), and `Alpaca: 43 open` (all
equities + migrated BTCUSD) -- i.e. exactly the merged, single-command
view G4 was built to produce (Alpaca positions folded into the same
native command Test_bot already had, rather than needing
`--combined-status` run as a separate step). Taken as sufficient
confirmation that G4 is working as designed; not independently
re-verified against the raw Telegram transcript by this session.

### G10. User said "both of them": DOM-injection hack removed, G7's
`/api/v1/balance` gap closed (2026-09-22, cloud session, `device_bash`
still down all session — file edits only via
`device_stage_files`/`device_commit_files`)

**1. Old "Golden Cross" DOM-injection hack removed from freqUI's
`index.html`** (the open question from G6, now answered). Deleted the
entire injected `<script>` block — the `MutationObserver`-based fake
nav tab plus the `document.body`-level iframe panel toggled via
`setInterval` — from
`freqtrade-develop/freqtrade/rpc/api_server/ui/installed/index.html`;
file dropped from 8402 to 1921 bytes, now just the stock compiled
freqUI shell with no injected script. `tools/serve_dashboard.py` and
`golden_cross_backtest_dashboard.html` were not touched, only the
injection in `index.html` was in scope. Verified no concurrent edit via
`device_list_dir` immediately before commit, committed with the mtime
guard, **confirmed via a post-commit re-stage + re-read that the change
actually persisted on disk.** Needs `bash tools/restart_bots.sh` before
the fake nav tab actually disappears from the browser.

**2. G7's "not yet done" `/api/v1/balance` extension — now done.** New
`alpaca_total_equity()` helper added to
`tools/run_combined_status_report.py` (right after
`_alpaca_positions()`): calls Alpaca's own
`TradingClient(...).get_account().equity` directly — deliberately NOT
summed from `_alpaca_positions()`'s `market_value` column, since that
would omit any uninvested cash sitting in the account and undercount
real equity (same accounting `run_alpaca_paper_trader.py` itself
already uses for order sizing). Returns `None` if Alpaca isn't
configured or the call fails, so the caller can leave freqtrade's real
balance result untouched rather than guess.

New `ALPACA_BALANCE_BRIDGE` patch block in
`freqtrade-develop/freqtrade/rpc/api_server/api_v1.py`'s `/balance`
endpoint: calls that helper and, when it returns a number, folds it
into the EXISTING `Balances` response fields rather than adding a new
key (`response_model=Balances` means FastAPI silently drops any key
outside that Pydantic schema) — adds the equity into
`total`/`total_bot`/`value`/`value_bot` and appends one synthetic
`"ALPACA"` row to the `currencies` list (using the response's own
`stake`/fiat ratio, same pattern as every other currency row). Wrapped
in try/except with a `logger.warning` on failure, mirroring every other
bridge patch in this file (ALPACA_STATUS_BRIDGE, ALPACA_CANDLE_BRIDGE)
— a failure here degrades to "balance just shows the real
freqtrade-only total, same as before this patch," never a broken
Dashboard.

Scope note: deliberately covers Alpaca only, not `--ma200-auto`.
`--ma200-auto`'s positions (bridged into `/status` via
`build_synthetic_open_trades`, G7) use a placeholder
`amount=1.0`/`open_rate=entry_price`, not a real dollar position size,
so summing them into a real dollar balance total would be wrong;
Alpaca's own `market_value`/`equity` figures are real dollars
throughout.

Both files syntax-checked (`ast.parse`); each re-verified via a fresh
`device_list_dir` immediately before its commit that no concurrent edit
had touched it since being staged; both committed with mtime guards;
each then re-staged and re-read after commit to confirm the change
actually persisted — this file itself (`run_combined_status_report.py`)
got clobbered once mid-edit today before the helper landed for good,
caught the same way, redone from the current on-device state rather
than assumed.

**CONFIRMED LIVE, 2026-09-22.** User restarted (`bash tools/restart_bots.sh`
— same command as G5/G7/G8/G9), logged out/in, hard-refreshed. Nav bar
confirmed clean — the fake "Golden Cross" tab is gone, just Trade /
Dashboard / Chart / Logs. Alpaca equities (XOM, WFC, VZ, ...) also
confirmed back in the Trade tab's Multi Pane sidebar with live P&L.

**One real wrinkle worth recording, not a bug in either patch:** on the
FIRST restart after this change, the Trade tab's sidebar briefly showed
the equities, then lost them on a subsequent refresh (only the 6
`--ma200-auto` crypto pairs remained) — with zero exceptions logged
anywhere, because `_alpaca_positions()` returns `[]` silently on any
failure, by design, making a real miss indistinguishable from "Alpaca
genuinely has zero positions." Added a `ALPACA_DIAGNOSTIC_LOGGING`
patch to `_alpaca_positions()` (prints to stderr, captured in
`logs/freqtrade_8080.log`, whether it's a missing-env-var miss or the
Alpaca API call itself failing) so this is diagnosable on sight if it
recurs. On the very next restart it did NOT recur — no diagnostic
prints, no errors, equities held steady — so this looks like a one-off
hiccup (most likely Alpaca's API needing a moment right at process
startup) rather than a real defect in today's two patches, neither of
which touches `_alpaca_positions()` at all.

**Balance-bridge verified directly against the raw API response, not
just eyeballed in the UI** — this freqUI build has no separate
Balance/pie-chart panel on the Dashboard page (only a "Balance" column
inside the "Bot comparison" table), so a new `tools/
check_balance_endpoint.py` script was added: logs into the real
`/api/v1/balance` endpoint using `config.json`'s own api_server
credentials and prints the raw JSON. User ran it: `total` $202,211.96,
`total_bot` $201,222.70 (matches the Dashboard's Bot-comparison Balance
figure, small drift from live price movement between checks), and a
real `ALPACA` currency row present with `balance: 96102.51`, matching
Alpaca's own account equity. `value`/`value_bot` show 0 only because
this config has no `fiat_display_currency` set — expected, not a bug.

Two new one-off diagnostic scripts left in `tools/` for future
reuse if either bridge acts up again: `check_alpaca_env.py` (are the
keys visible + does the Alpaca API answer, run from anywhere) and
`check_balance_endpoint.py` (hits the real `/balance` endpoint directly,
bypassing freqUI entirely).

### G11. The real cause of the "equities pop up then vanish" flicker:
the plain REST `/whitelist` endpoint was never patched (2026-09-22,
cloud session, `device_bash` still down, file edit only)

**The G10 blip recurred** — user reported the Trade tab sidebar losing
the Alpaca equities again (back to just the 6 `--ma200-auto` crypto
pairs), on the SAME long-running process from the last restart, no new
restart in between. That ruled out "one-off hiccup at process startup"
as the explanation.

**Checked the `ALPACA_DIAGNOSTIC_LOGGING` print added in G10 — it never
fired, not once, across the whole episode.** That means `_alpaca_
positions()` itself was succeeding every single time it was called,
which in turn means `build_synthetic_open_trades()` (which calls it
directly, no caching) was always returning the full combined trade
list, equities included. So the server-side data was never the problem
— the bug had to be somewhere in what freqUI's frontend does with two
different sources of the pair list.

**Found it by re-reading `api_ws.py`'s `ALPACA_WHITELIST_BRIDGE` (G8)
against `api_v1.py` side by side:** the websocket `WHITELIST` handler
(`api_ws.py`) got the synthetic-pairs union appended back on
2026-09-21. The plain REST `GET /whitelist` endpoint (`api_v1.py`,
`def whitelist(rpc): return rpc._rpc_whitelist()`) never did — it was
simply missed, since G8's note only mentions PairSummary.vue being fed
by the websocket message, not that freqUI might also poll the REST
endpoint directly. Whenever that unpatched REST response landed after
the augmented websocket one, it would overwrite the frontend's pair
list back down to the bare set — exactly matching "pops up, then a
few seconds later it's gone," and leaving zero server-side trace either
way, since a plain unmodified REST response isn't an error.

**Fixed the same way as every other bridge in this file:** `/whitelist`
now calls `rpc._rpc_whitelist()` as before, then — wrapped in the same
try/except-and-log pattern as G7/G8/G10 — appends the same
`build_synthetic_open_trades()`-derived pair union onto
`result["whitelist"]` (updating `result["length"]` to match) before
returning. Any failure here just leaves the real, un-augmented
whitelist in place, same safety guarantee as every other patch here.
Syntax-checked, verified no concurrent edit via `device_list_dir`
immediately before commit, committed with the mtime guard, re-staged
and re-read after commit to confirm it persisted.

**NOT yet restarted/confirmed live** — needs `bash tools/restart_bots.sh`,
then watch the Trade tab's sidebar for at least 20-30 seconds after a
hard refresh (not just an instant glance) to confirm the equities now
hold steady instead of flickering. If it still recurs after this,
the next place to look is freqUI's own compiled frontend JS (which
store the whitelist gets read from, and whether it merges or replaces
on each source), since both server-side pair-list endpoints will then
be confirmed doing the right thing.

### G12. Equities charts went from "stale data" to "no data at all" —
a `NameError` regression self-introduced while fixing G9, plus a repeat
of the "fix never actually landed on disk" failure mode (2026-09-22,
cloud session, `device_bash` still down all session, file edits only
via `device_stage_files`/`device_commit_files`)

**Symptom, reported after G9's stale-data fix was restarted:** "btc/usd
works but nothing from alpaca displays" — every equities chart
(XOM/WFC/VZ/...) went from showing 3-year-stale data (G9's bug) to
showing nothing at all. Migrated-crypto charts (BTC/USD, which uses the
same function's crypto-bars fallback path) kept working.

**Root cause: a line was accidentally deleted while writing G9's fix.**
The `old_string`/`new_string` edit that removed `limit=fetch_limit`
from `StockBarsRequest(...)` (G9) also, by mistake, dropped the
preceding `stock_tf, stock_start = _tf_params(STOCK_CHART_TIMEFRAME)`
assignment — while the `StockBarsRequest(...)` call two lines below it
still referenced `stock_tf`/`stock_start`. Every stock-path call in
`fetch_alpaca_or_crypto_candles()` therefore raised `NameError: name
'stock_tf' is not defined` on every single invocation, 100% of the
time, for every equities pair. This was invisible anywhere in the logs
because the surrounding `except Exception as exc: print(...)` catches
and swallows it (by design, so one bad symbol doesn't take down the
whole bridge) and returns an empty DataFrame — indistinguishable,
server-side, from "Alpaca genuinely has no data for this symbol."

**Made worse by a second, separate failure: an earlier `flush=True` +
"EMPTY dataframe" diagnostic-print fix never actually landed on disk.**
Re-reading the live file to investigate turned up the OLD prints
(un-flushed, no EMPTY-case print) even though the commit that was
supposed to add them had reported success — the same "fix reported as
written, but the file on disk doesn't have it" failure this doc already
warns about (see G9's "clobbered back to its pre-fix state" note
above). Whether this specific instance was a concurrent edit or simply
a commit that was never re-verified isn't established either way — but
the fix for it is the same regardless, and is now enforced explicitly:
**never again trust a `device_commit_files` "written" response alone —
always re-stage and re-read (or md5sum) the file immediately after
committing, before telling the user anything is fixed.**

**Diagnosis tool added:** `tools/check_vz_candle_bridge.py` — a small
standalone script that calls `fetch_alpaca_or_crypto_candles("VZ",
"4h", 500)` directly (the exact function/args the chart bridge itself
uses) and prints `resolved_timeframe`, row count, and either the tail
of the DataFrame or `EMPTY DATAFRAME`. Running it produced the
definitive diagnostic: `stock bars failed for VZ: name 'stock_tf' is
not defined` / `rows: 0` / `EMPTY DATAFRAME` — confirming the exact
line and exact bug. Left in `tools/` for reuse if this bridge acts up
again, same pattern as `check_alpaca_env.py`/`check_balance_endpoint.py`.

**Fix, applied to `tools/run_combined_status_report.py`:** restored
the dropped `stock_tf, stock_start = _tf_params(STOCK_CHART_TIMEFRAME)`
line (with a "BUGFIX 2026-09-22 #2" comment explaining why it's there),
and re-added `flush=True` to both the stock-path and crypto-path
exception-handler `print()` calls, plus new prints for the silent-
empty-without-exception case ("stock/crypto bars EMPTY for `<symbol>`
(start=..., no limit cap)") so a future empty result is diagnosable
from the log alone, without needing a standalone script.

**Verification, deliberately more thorough than usual given the G9
repeat-failure above:** syntax-checked (`ast.parse`); local md5sum
recorded (`c2d0b9585c9bb0e6a05abf126c3ddd01`); confirmed via
`device_list_dir` immediately before commit that the on-device file's
mtime hadn't changed since it was last staged (no concurrent edit);
committed with the mtime guard; **then immediately re-staged the file
from the device and re-computed its md5sum, confirming it matched the
local value exactly** — closing the specific verification gap that let
the earlier flush=True fix silently fail to land.

**CONFIRMED LIVE, 2026-09-22.** User restarted (`bash
tools/restart_bots.sh`), re-ran `check_vz_candle_bridge.py`:
`resolved_timeframe: 1d`, `rows: 500`, last row dated 2026-09-22
(today), close ≈ $46.84 — matching VZ's real current market price
(previously ~30% off, from G9's now-fixed stale-data bug). `NameError`
gone.

**Side effect worth recording: this also answers the question that
started the whole investigation.** With real current data, VZ/USD's
ma50 (47.86) is currently above its ma200 (46.19) — a golden-cross
configuration, not the death cross it appeared to show under the old
~3-year-stale data. The death-cross appearance earlier this session was
purely a data-staleness artifact, not the real market state.

**XOM/WFC independently re-verified, same day.** New
`tools/check_equities_candle_bridge.py` (same pattern as
`check_vz_candle_bridge.py`, one script covering both symbols) —
user ran it: both back `resolved_timeframe: 1d`, `rows: 500`, last row
dated 2026-09-22 (today). XOM: close $158.48, ma50 $158.45 > ma200
$147.32 (golden cross). WFC: close $83.675 (down ~3.4% intraday from
the prior close of $86.54, a real move, not a data artifact — nothing
in this fix touches volatility/gaps), ma50 $87.10 > ma200 $85.15
(golden cross, narrower gap than XOM). Same stock-bars code path as
VZ, same fix, same result — confirms the fix isn't VZ-specific.

**Still not independently re-verified this session:** BTC/USD's
crypto-bars fallback path (touched by this same edit, but only to add
a `flush=True`/EMPTY print — no logic change there, so lower risk, but
not spot-checked live either).

### G13. BTC/USD's chart bridge was quietly returning a completely
different security's data — a real stock ticker "BTC" — not a stale
or empty result this time, an outright wrong-instrument bug found
while closing out G12's last open item (2026-09-22, cloud session,
`device_bash` still down, file edit only)

**Spot-checking G12's one remaining open item (BTC/USD) surfaced a
third, previously-latent bug, unrelated to G9/G12's NameError/staleness
fixes.** `check_crypto_candle_bridge.py` (new, same pattern as the
VZ/equities checkers) called `fetch_alpaca_or_crypto_candles("BTC",
"4h", 500)` and got back `resolved_timeframe: 1d`, 500 rows, prices in
the $34-38 range — obviously not bitcoin (real BTC price that day was
~$86k). The stock-bars path had silently succeeded for the bare symbol
`"BTC"`: Alpaca's stock data API returned real, non-empty, non-error
daily bars for some equity that also trades under ticker "BTC" — an
identity this session did not conclusively pin down (web search turned
up no clear NYSE/NASDAQ match; whatever it is, Alpaca's stock feed
resolves it), but the price scale alone was proof enough it wasn't
crypto.

**Root cause: the function's own design assumed "stock lookup finds no
such ticker" would always be true for migrated crypto's short symbols
(BTC/ETH/SOL/PAXG) and used that as an implicit crypto detector** —
explicitly stated in this file's own `EQUITIES_CHART_DAILY` comment
block ("reached via the crypto-bars fallback path below when the stock
lookup finds no such ticker"). That assumption held for however long
this bridge has existed, until "BTC" specifically turned out to
collide with a real, Alpaca-tradable stock ticker. Every call for
BTC/USD's chart has likely been showing this wrong instrument's data
since the candle bridge was first built (2026-09-21, Section G8/G9) —
this is not a NameError-style outage, no exception anywhere, so nothing
about it would ever have shown up in a log.

**Fix, in `tools/run_combined_status_report.py`:** stopped relying on
"stock lookup fails" as the crypto detector. This file already had a
module-level `MIGRATED_CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD",
"PAXGUSD"}` constant (used elsewhere, in `_format_combined_status`, to
split Alpaca positions into crypto vs. equities for the Telegram
report) but `fetch_alpaca_or_crypto_candles` never consulted it. Added
`is_known_migrated_crypto = f"{base_symbol}USD" in
MIGRATED_CRYPTO_SYMBOLS` right after the API-key check, and wrapped the
entire stock-bars try/except block in `if not is_known_migrated_crypto:
... else: print(...)` — so BTC/ETH/SOL/PAXG now skip the stock path
entirely and go straight to crypto bars, with a diagnostic print
recording that the skip happened (in case this ever needs re-diagnosing
without re-reading the source). Equities and anything else not in that
set are unaffected — same stock-then-crypto order as before.

Syntax-checked (`ast.parse`); confirmed via `device_list_dir`
immediately before commit that the file's mtime hadn't changed since
last staged; committed with the mtime guard; re-staged and md5sum-
verified after commit (`8b8d27b6b612578b8404538d96b985f1`) that it
landed — same verification discipline as G12.

**CONFIRMED WORKING, 2026-09-22** — user re-ran
`check_crypto_candle_bridge.py`: printed the new skip-diagnostic line,
then `resolved_timeframe: 4h`, `rows: 500`, close ≈ $86,406 (today,
matching real BTC market price), ma50 $79,887 > ma200 $78,651 — a
golden cross for BTC too, on genuinely its own crypto data this time.

**Restarted (`bash tools/restart_bots.sh`) and re-checked directly in
freqUI itself, via the built-in browser, 2026-09-22.** VZ/USD's chart
(picked as the equities spot-check, since equities have zero Kraken
data and always require the bridge) came back correct: real dates
through today, "Long entries: 0 / Long exit: 0" (the tell-tale sign of
the bridge's synthetic response — real strategy analysis always
reports real signal counts), and nothing in `logs/freqtrade_8080.log`
— consistent with a clean success (this bridge only prints on an EMPTY
result or an exception, never on success, so silence here is the
correct/expected outcome, not a gap).

**BTC/USD's live chart, however, turned out NOT to exercise this fix at
all — an important correction to the optimistic assumption above.**
Opening BTC/USD's chart in freqUI showed correct data too (~$87,945,
visible golden cross), but `logs/freqtrade_8080.log` had zero bridge
activity for it, and the chart's own "Long entries: 3 / Long exit: 2"
line is real-strategy-only (the bridge always reports 0/0, as VZ's
check above just demonstrated). Root cause: BTC/USD still has an open
trade (`Trade(id=1)`, opened 2026-09-02) — freqtrade keeps a pair with
an open position in its live analysis set even after it's dropped from
the entry whitelist (it still needs real data to evaluate that
position's exit), so `_rpc_analysed_dataframe` returns real, non-empty
Kraken data for BTC/USD directly and the request never reaches
`_bridge_pair_candles`'s fallback at all. **This means the G13 fix is
real, and correctly verified at the function level
(`check_crypto_candle_bridge.py`, called directly, hitting the actual
buggy/fixed code) — but it has never actually been exercised by
BTC/USD's live chart, before or after the fix, because that chart has
had another, unrelated, correct data source the whole time.** The bug
this fix closes would only start mattering for BTC/USD specifically if
its open trade ever closes and freqtrade drops it from the active
analysis set, forcing a fallback to the bridge — worth remembering if
that trade closes later and BTC/USD's chart is ever spot-checked again.
ETH/SOL/PAXG (the other three names in `MIGRATED_CRYPTO_SYMBOLS`) are
current Kraken whitelist pairs and always have real freqtrade data
regardless of trade status, so in practice this fix is really only
insurance for BTC specifically, for whenever it's not holding a
position.

**Worth a broader gut-check later, not done this session:** whether any
other bare symbol this bridge might someday be asked for (equities
ticker collisions with crypto-style short symbols, or vice versa)
could hit the same class of silent-wrong-instrument bug — the fix here
only closes it for the four symbols already in `MIGRATED_CRYPTO_SYMBOLS`,
which is everything currently in scope, but the underlying "trust
whichever data source responds first" pattern is still how the stock
path and the crypto path relate to each other for any symbol *not* in
that set.

---

## FULL STATUS AS OF 2026-09-16 (everything re-checked live; this
header itself was stale against the body for days — see the date-drift
note directly below before trusting any date in this doc at a glance)

**Note on dates in this document:** this doc has repeatedly cited a
date one-plus days off from the user's actual local date (this
session's own clock said the 17th when the user's local date was the
16th; earlier passes had similar drift). Every date-stamped claim
above was corrected against the user's own explicit, in-the-moment
confirmation, not this session's system clock — trust those, and treat
any bare "today" reference this session generates on its own as
suspect until the user confirms it.

- **A0 — RESOLVED, confirmed 2026-09-16 against a real account pull.** The
  old 41 over-levered positions are gone. Current book: 43 positions, all
  LONG, zero shorts — 42 equities at ~2%-of-equity sizing, one BTCUSD at
  ~15%-of-equity sizing, both matching the intended cash-constrained rule
  exactly. See A0's full note above for the math check.
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
- **B5 — two separate things, do not conflate (correcting this doc's own
  2026-09-13 note, which wrongly merged them):**
  1. The Section F duplicate-process bug (`run_alpaca_paper_trader_loop.py`
     PID 54909 racing `--alpaca-auto`) is **resolved** — that was a
     previously-unknown bug found live, unrelated to B5's actual plan, fixed
     by the local session under the user's direction.
  2. **B5's own plan — fully done as of 2026-09-16 ~11:27PM PDT.** Killing
     `run_cross_asset_paper_trader_loop.py` (PID 95867) — done, confirmed
     dead. Cutting Kraken-side BTC entries — done: both edits verified in
     the code, both processes restarted (new PIDs: freqtrade 4450,
     `--ma200-auto` 4451), and — critically — confirmed via freqtrade's own
     log that the *running* whitelist actually dropped BTC (`Whitelist with
     5 pairs`, no BTC/USD), not just the file on disk. BTC's Alpaca entry
     (filled 2026-09-15 08:17:54 UTC at $76,727) was the evidence that
     justified this; Alpaca is now the sole executor of new BTC golden-cross
     entries. ETH/SOL/PAXG remain gated exactly as before — each needs its
     own verified Alpaca fill before the same cut happens for it.
- **C (FVG-equities)** — unchanged, still idle, still the next real
  candidate whenever there's room to prioritize it.
- **G (Telegram visibility) — `--combined-status` itself fully built and
  tested (no known gaps there). G2 (2026-09-19) found the real root
  cause of "I only see crypto trades" (two separate bots, Test_bot vs.
  FreqTrade559 — see G2) and built a fix (merge Alpaca into FreqTrade559's
  native `/status`) — reverted the same day before ever being tested
  live, once the user decided against editing freqtrade's vendored code.
  **G3 (Test_bot becomes the one bot) is now LIVE, confirmed 2026-09-20:**
  freqtrade restarted (new PID 23539), Telegram authenticated cleanly
  against Test_bot's token, full native command set confirmed listening
  (`/status`, `/daily`, `/profit`, `/balance`, `/performance`, `/count`,
  `/start`, `/stop`, force-entry/exit, and more). **One confirmed,
  expected gap, not a bug:** the user checked `/status` in Test_bot
  directly and confirmed it does not show Alpaca equities — exactly the
  documented tradeoff (the token swap gives Test_bot the keyboard, it
  never wired Alpaca into `/status` itself; that still needs
  `--combined-status` run separately, or a G2-style merge redone simpler
  now that freqtrade is confirmed to run in the same `javier` conda env,
  not an isolated venv as first assumed). **G4 (2026-09-20) closes that
  gap:** the merge is rebuilt — simplified to a direct import instead of
  G2's subprocess bridge — and re-applied to `telegram.py`. File edits
  done, NOT YET restarted or tested live (freqtrade is still running the
  pre-G4 code, PID 23539). See G4 for exact verification and
  `handoff4.md` for the remaining restart + test steps.** **G5
  (2026-09-20): restart is one command, `bash tools/restart_bots.sh` —
  CONFIRMED RUN LIVE.** **G6 (2026-09-21): a separate live-dashboard
  page — CONFIRMED LIVE, then superseded by G7.** **G7 (2026-09-21,
  current, most important): Alpaca + `--ma200-auto` patched directly
  into freqtrade's real REST API (`/status` for the trade list,
  `/pair_candles` for the chart) so they appear as native rows in the
  REAL freqUI — no separate page. Both patches are file-edits-only,
  committed, syntax-checked, but NOT YET restarted/confirmed live — run
  `bash tools/restart_bots.sh` then check `127.0.0.1:8080/trade`. See G7
  for the full detail, including the chart-data source (Alpaca's own
  market-data API) and its one unverified risk (exact alpaca-py import
  paths not confirmed against the installed version from this
  session).**

Sections A/B/E are closed or confirmed live (E: see the new 2026-09-20
note on the first real -8% stop firing, XMR); C (FVG-equities) remains
idle by choice, not blocked; the pre-existing, deliberate gates on
ETH/SOL/PAXG stand (each needs its own Alpaca fill before its Kraken
entries get cut). G3 is live and working as designed; whether to close
its one remaining gap (Alpaca not in `/status`) is now the one open
question. Live re-check, 2026-09-20 (freqtrade restart): PID 23539, open
Kraken trades BTC/ETH/SOL/TRX (XMR closed via the -8% hard stop
2026-09-17), whitelist correctly 5 pairs with no BTC. Older re-check
(2026-09-16 ~11:27PM PDT): `--ma200-auto` PID 4451, port 8081 unchanged
at PID 63313, PID 95867 confirmed dead, 43 Alpaca positions unchanged —
`--ma200-auto`'s current running status not reconfirmed since.
