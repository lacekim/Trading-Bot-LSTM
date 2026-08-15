# Session summary — 2026-08-13/14

Two separate trading bot projects were investigated tonight: the **freqtrade
bot** (`/Volumes/Extreme_SSD/freqtrade-develop`) and **trading_bot_v4**
(`/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera`, this repo).
Both turned out to have the same class of problem — backtest/validation
numbers that looked good but didn't reflect real, honestly-measured edge —
for different underlying reasons. This is a full account of what was found,
what was fixed, and what's still open.

---

## Part 1 — freqtrade bot (OnnxEthusdtStrategy, Kraken spot)

**Starting point:** a backtest of the combined long+short signal reported
+40.64% return, 66.7% win rate, Sharpe 2.08 over 2025-01-09→2026-01-09.

**What we found:** the ONNX models were trained with a simple 80/20
proportional chronological split per pair. Because most pairs (BNB, LINK,
ALGO, etc.) only had ~6 months of Kraken history available, their 80% train
window entirely overlapped the backtest evaluation window — the "backtest"
was mostly grading the model on data it had memorized. Cross-referencing
actual trade dates against each pair's train/test split: **83 of 90 trades
(92%) opened before their pair's split date**, accounting for **99.2% of the
reported profit**. The 7 genuinely out-of-sample trades made +$322 combined
— statistically indistinguishable from nothing.

**The fix:** spliced in deep historical OHLCV (a local Kraken bulk-CSV dump
going back to 2013-2019 depending on pair, found at
`/Volumes/Extreme_SSD/Algorithmic-Trading-with-Deep-Learning/data/Kraken_OHLCVT`)
with the already-downloaded live data, then retrained all 18 usable pairs
(BNB and DOGE were excluded — their only available history postdates the
backtest window entirely, so they can't be trained-then-tested cleanly) using
a **fixed calendar cutoff** (train ends 2024-11-15, validate 2024-11-15→
2025-01-09, backtest only after 2025-01-09) instead of a proportional split.

**The honest result: -13.98% over the same window**, vs the market's
-35.27% (so it did beat buy-and-hold, but it lost money outright). Held-out
AUCs came back at 0.50-0.67 — real but weak. Root cause of the loss, once
the number could be trusted: the strategy is **long-only** (spot exchange,
Kraken has no margin/futures support in freqtrade — `TradingMode.SPOT` is
the only mode `_supported_trading_mode_margin_pairs` allows) in a year the
market fell 35%, combined with an EMA200 filter that gates entry *regime*
but not entry *timing* (doesn't stop you buying the top of a dead-cat
bounce), combined with weak model skill. 31 of 40 trades in that window got
stopped out via the ATR-based stop, 3.2% win rate on that bucket.

**Also tested and rejected as fixes** (both made things worse, not better):
- Adding `CooldownPeriod`/`StoplossGuard` protections to stop revenge-trading
  after losses: -36.05% and -39.49% respectively (blocked winning re-entries
  along with losing ones).
- Making `custom_stoploss` profit-aware (tighten as unrealized profit grows,
  instead of pure ATR): first attempt collapsed returns to +9.65% (a floor
  parameter dominated and clamped nearly every trade to a flat 2% stop); a
  corrected version still only reached +17.87%, confirming the wide ATR band
  is load-bearing for letting winners survive to the ROI target, not sloppiness.

**Where this stands:** the freqtrade project's long-only, single-strategy
edge is real but weak and currently unprofitable net of the long-only
handicap. Going long+short would require a different venue (Kraken has no
margin support in freqtrade) — see the sibling-project angle in Part 2.

---

## Part 2 — trading_bot_v4 (GMX perpetuals, long+short)

### 2a. The bug: single shared models, not per-asset

The "original" LONG model (`models/lstm_ada_model.h5`) was trained **only on
ADA** and applied unchanged to all ~120 GMX perpetual assets it had never
seen a row of. The bearish/SHORT model was similarly **one model pooled
across every symbol**, not per-asset. Neither matched the intended design
(confirmed by the user: every asset should get its own model, mirroring the
freqtrade project's working `model_{PAIR}.onnx` pattern).

Fully mapped every consumption site (`production_backtest.py`, the live
scheduler `research/scheduler.py` — a **separate** model-loading path from
the backtest, both needed fixing — plus ~12 research/comparison tools left
deliberately out of scope: `ranking_engine.py`, `walk_forward.py`,
`trade_filter_research.py`, `asset_selection_engine.py`,
`model_comparison.py`, `smc_shadow_backtest.py`, `comparison_engine.py`,
`backtest_engine.py`, `paper_model_performance.py`, `paper_smc_filter.py`,
`smc_model_paper.py`, `predictor.py`, `daily_research.py`).

**The fix:** new `trading_bot_v4/ml/per_asset_trainer.py` trains one
independent model per GMX symbol per direction (18 base features, no SMC
pipeline needed, sourced straight from `load_gmx_ohlc`), mirroring
`bearish_trainer.py`'s already-sound 70/10/10/10 chronological split +
3-window walk-forward promotion gate, but per-symbol instead of pooled. New
`PerAssetModelCache` in `utils/model_cache.py` resolves models lazily per
`(direction, symbol)`. `production_backtest.py` and `research/scheduler.py`
both rewired to use it. Fully verified: trained 240 models (120 assets × 2
directions) with zero errors, mechanically working end-to-end.

### 2b. Two more bugs found and fixed along the way

- **Market-cap risk gate blocking everything regardless of model quality**:
  `Config.MARKET_CAP_RISK_ENABLED`'s snapshot (`data/MARKET_CAP/market_cap_snapshots.csv`)
  had gone stale past its 48h freshness gate (was 69.9h old), rejecting all
  120 assets with `"market-cap snapshot is stale"`. This is a live cache
  that needs periodic refreshing (`collect_market_caps()`, a public CoinGecko
  call) — not a code bug, just needs to run periodically (normally the live
  scheduler's job; it went stale because the scheduler wasn't running
  continuously). Refreshed once tonight; **will go stale again** without the
  scheduler running regularly.
- **A timestamp-labeling bug in `per_asset_trainer.py`'s own walk-forward
  reporting** (`df.columns[0]` was evaluated against the pre-reset frame,
  mislabeling the `Open` price column as `"timestamp"`) — caught via a smoke
  test before the full sweep ran, fixed, did not affect any trained model
  weights (the mislabeled column was never in `Config.FEATURE_COLUMNS`),
  only affected the human-readable window date ranges in reports.

### 2c. Honest promotion results (horizon = 1, i.e. predict the very next candle)

Full 120-asset × 2-direction sweep: **0/240 promoted**, on either side.
Every symbol trained cleanly; AUCs were real but modest (0.5-0.75); none
cleared the gate's precision/profit-factor/stability bar. This is not a
pipeline failure — it's the same finding as the freqtrade project, now
confirmed per-asset instead of in aggregate: a single-candle-ahead 1%
threshold is mostly noise, and no amount of "give every asset its own
model" fixes that if the target itself is too noisy to predict.

Backtest was re-run with real qualification (no `--ignore-qualification`)
after the market-cap fix: `risk_allowed_long/short_assets` went from 0→48/47
(confirming that bug was real and is now fixed), but `eligible_assets`
stayed 0 — SHORT because 0 promoted, LONG because the pre-existing daily
GO/WATCH/NO-GO list (`reports/v4_daily_go_status.csv`) currently shows 0 GO
/ 117 WATCH / 3 NO-GO. That list needs its own separate validated-ranking
pipeline (`ranking_engine.py`/`walk_forward.py`, out of scope tonight) *and*
real accumulated live-trading days under the new models — neither of which
a script can fast-forward.

### 2d. Horizon experiment (predict N candles ahead instead of 1)

Hypothesis: a single-candle-ahead target is mostly noise; giving the signal
more time to resolve (predict the next 12 hours instead of the next 1 hour)
should raise precision. Added a `horizon` parameter throughout
(`per_asset_trainer.py`, `model_cache.py` — non-default horizons get their
own `models/{direction}_h{N}/` namespace so they never collide with the
production horizon=1 models).

**10-asset pilot at horizon=12**: still 0/20 promoted, but several precision
numbers jumped a lot (some to 0.6-1.0) — mixed with obvious small-sample
artifacts (10-12 signal counts hitting "perfect" precision by chance).
Inconclusive from 10 assets.

**Full 120-asset sweep at horizon=12**: **4/240 promoted** — long/MEW,
short/ANIME, short/ORDI, short/SATS — a real, non-zero result, unlike
horizon=1. Applying the same small-sample scrutiny used all night: ANIME
(223 signals, consistent 51-74% precision across all 3 windows, no PF
outliers) was genuinely convincing; MEW (94 signals) mostly solid with one
outlier window; ORDI and SATS both had extreme profit-factor numbers (391.9
and 45.4/25.4) in thin windows that are almost certainly small-sample
artifacts, not real edge.

**The critical finding — and the actual open problem:** none of the 4 held
up when run through the real execution engine
(`production_backtest.py --ignore-qualification`, horizon=12, real ATR
stops/targets/costs). ANIME **-79.4%**, ORDI **-64.2%**, SATS **-8.0%**, MEW
**0 trades** (long-side signal generation still uses the pre-existing
`policy_for()` threshold system, not the per-asset model's own calibrated
threshold — a wiring gap, separate issue). Root cause: the promotion
walk-forward's P&L calculation (`_walk_forward_rows`, inherited from
`bearish_trainer.py`'s original design) is a simplified proxy — "enter now,
hold exactly N candles, take the raw return, no stops, no early exit" — completely
disconnected from what `simulate_production_symbol` (the real engine)
actually does: a **0.75×ATR stop** and, critically, **`maximum_hold_candles=8`**
— shorter than the 12-candle horizon the model was even being scored on.
Positions are being force-exited or stopped out before the predicted move
has time to happen.

**This means every promotion decision made by this validation methodology —
the original pooled bearish model's, and both per-asset horizons' — has
never actually measured what the real trading engine would do.** That's a
deeper, more important problem than "is horizon=1 or horizon=12 better."

---

## Future goals / where to pick this up

1. **(Highest priority) Fix the calibration/promotion methodology to match
   real execution.** `_walk_forward_rows`/`_select_threshold` in
   `per_asset_trainer.py` (and the equivalent logic in `bearish_trainer.py`)
   need to score candidates using the same exit rules
   `simulate_production_symbol` actually uses (ATR stop/target,
   `max_hold_candles`, real fees/slippage) instead of a naive
   hold-to-horizon proxy. Until this is fixed, no promotion count from
   tonight (0 at horizon=1, 4 at horizon=12) should be treated as validated,
   including the ones that "looked" statistically clean like ANIME.
2. **Fix the MEW-style wiring gap**: LONG signal generation
   (`predict_original_baseline_signals` / `policy_for()`) doesn't consult
   the per-asset model's own calibrated promotion threshold at all — it uses
   a separate, mostly-flat default threshold system. Worth deciding whether
   LONG should gate on promotion the same way SHORT already does.
2b. **After the calibration fix, if it holds up, systematically explore
   horizon** rather than assuming 12 is the right value — the pilot showed a
   real precision/AUC tradeoff (higher horizon = higher precision, lower
   ranking power) that's worth mapping properly (e.g. 4, 6, 8, 12, 24) once
   the scoring itself can be trusted.
3. **Keep the market-cap snapshot fresh** — it will go stale again (48h
   gate) without the live scheduler running continuously; either keep the
   scheduler running or add an explicit refresh step to any manual backtest
   workflow.
4. **The daily GO list is separately stale** in a different sense — it
   reflects live paper-trading track record from the *old* single-shared
   models. It can't be fast-forwarded; it needs the ranking/walk-forward
   tools rewired to per-asset models (explicitly deferred tonight) *and*
   real elapsed trading days under the new models before anything can
   realistically show GO.
5. **freqtrade project**: the long-only structural handicap is probably the
   single biggest lever if that project continues — either accept it's
   long-only and only tradeable in bull regimes, or move to a venue with
   margin/futures support if a genuine long+short freqtrade bot is wanted
   (Kraken doesn't support it; Binance/Bybit do in freqtrade).
6. **Both projects would benefit from the same discipline going forward**:
   any time a "backtest" or "promotion" number looks good, check (a) whether
   the evaluation window was truly unseen by training, and (b) whether the
   number was computed by the same rules the live/execution engine actually
   uses. Both bugs found tonight were exactly this class of mistake, just in
   different places (data leakage vs. scoring-methodology mismatch).
