# Session summary — 2026-08-13/15

Two separate trading bot projects were investigated across this session: the **freqtrade
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

## Part 3 — closing the loop (2026-08-15): calibration fix, threshold wiring, and the real answer

Picked back up on the single highest-priority item from Part 2: the
calibration/promotion methodology never actually measured what the real
engine would do. Fixed it properly, then followed the chain all the way
through to a final, trustworthy answer.

### 3a. Fixed the calibration methodology to match real execution

`_walk_forward_rows`/`_select_threshold` in `per_asset_trainer.py` now build
a `simulate_production_symbol`-compatible signals frame for every candidate
threshold (new `_build_signals_frame`/`_simulate_threshold` helpers) and
score it with the **real** engine — real ATR stop/target, real
`max_hold_candles`, real fees/slippage — instead of the old "hold exactly N
candles, take the raw return" proxy. Threshold selection now picks by real
profit_factor/return_pct (≥10 real trades on the calibration slice);
promotion requires ≥3 real trades per holdout window with profit_factor≥1.30
and positive return in every window — same conceptual bar as before, now
honest. `PerAssetTrainingResult`'s fields changed from classification
precision/recall to real `holdout_trades`/`return_pct`/`profit_factor`/
`win_rate_pct` (AUC kept as a diagnostic only, no longer gates promotion).

**Verified directly against the 4 old "promotions"**: re-trained MEW, ANIME,
ORDI, SATS individually under the new methodology — every one came back
non-promoted with real numbers matching (and explaining) the earlier
targeted-backtest failures: MEW -13.94%/PF 0.36, ANIME -24.56%/PF 0.69, ORDI
+3.71%/PF 1.05 (closest, still fails), SATS -27.80%/PF 0.59. The fix
correctly rejects every one of the old false positives.

**Re-ran the full 120-asset × 2-direction horizon=12 sweep** under the
corrected methodology (`--force-retrain`, since all 120 symbols already had
stale entries from the flawed run): **promoted count dropped from 4/240 to
1/240** (short/WLD: 24 real trades, +13.35% return, profit factor 2.12, win
rate 58.3%, positive in all 3 walk-forward windows — window 1's "infinite"
PF is a 3-trade artifact, but windows 2-3 are a real, substantive sample).
No errors across the sweep. Also much faster than expected (~23s/model vs
~70-90s before), making a full resweep a ~1.5-2 hour job instead of 4-5.

### 3b. Fixed the LONG threshold wiring gap

Root cause: `predict_original_baseline_signals`/`_predict_original_model_signals`
always derived `model_direction` via `direction_for_probability()`
(`persistent_policy.py`), which uses `policy_for(symbol).threshold` — a flat
global default (`Config.MIN_SIGNAL_THRESHOLD` = 0.70) for any
non-persistent-policy symbol. The `threshold` column added afterward was
display-only metadata; it never fed back into the direction decision. So
even a correctly-promoted LONG model would never actually change live/backtest
behavior. The SHORT side never had this problem (`predict_per_asset_short_signals`
already took an explicit threshold param).

Fix: both functions gained an optional `threshold` param that, when passed,
directly drives the LONG/HOLD decision (LONG-only, no SHORT interpretation
— matches how the model was actually calibrated). `research/scheduler.py`
needed no changes — it already threads `per_asset_cache` through and picks
up the fix automatically. `daily_research.py`'s callers (never pass
`per_asset_cache`) are unaffected — `threshold=None` preserves the exact old
behavior. **Verified concretely**: same BTC model, `threshold=None` → 692 of
78,150 rows flagged LONG (the flat 0.70 default); explicit `threshold=0.3` →
75,828 of 78,150 — proving the parameter was previously silently ignored and
now genuinely drives the decision.

### 3c. The final, complete answer: WLD through the real pipeline

With both fixes in place, ran the one promoted symbol (short/WLD) through
`--backtest-production` with **real** qualification (no
`--ignore-qualification`) to check the last remaining link: does a
correctly-promoted, correctly-wired signal actually clear risk/liquidity
gating too? `qualified_short=True` confirmed the promotion and threshold
wiring are both working. But `short_risk_allowed=False`:
**"insufficient GMX liquidity/open interest for LARGE"**. WLD's ~$1.23B
market cap puts it in the LARGE tier, which requires $100k liquidity /
$25k open interest on GMX. Actual GMX state for WLD: $70,284 available
liquidity (would clear MID's $50k floor, misses LARGE's $100k) and **$1,234
total open interest** — roughly 20x short of the $25k LARGE floor. WLD is a
large-cap token by market cap, but its GMX perpetual market is nearly
dormant. Not a bug — the risk system correctly caught a real venue-liquidity
constraint no amount of model-quality work can fix.

**Final honest state of the entire per-asset investigation: 0 of 240
trained models produce an actually-tradeable signal**, once calibration
economics, threshold wiring, and real GMX liquidity are all correctly
accounted for together. This is a complete, validated answer — not a dead
end from a bug, but a genuine finding about what's currently tradeable on
this venue with this feature/target design.

### 3d. Liquidity check: is WLD a one-off, or systemic?

Ran `evaluate_asset_risk` for all 120 GMX symbols directly (no training
needed, just the already-cached market-cap/GMX-state data) to see whether
WLD's liquidity failure was a one-off or the norm.

**Liquidity is not a universal blocker**: 48/120 symbols clear the LONG
liquidity floor (47/120 for SHORT), including solid majors — BTC, ETH, SOL,
ADA, AVAX, LINK, XRP, DOGE, NEAR, ARB, and more. A meaningful ~40% of the
universe is genuinely tradeable on GMX today.

**But every symbol that ever got promoted tonight — WLD, MEW, ANIME, ORDI,
SATS — is liquidity-blocked. All five, zero overlap** with the 48 tradeable
symbols. That's a real pattern worth taking seriously, not a coincidence to
shrug off: thin, illiquid markets are less efficiently priced, which is
exactly where a technical-indicator model is more likely to find *apparent*
statistical edge — precisely because there's less arbitrage keeping prices
honest. Those are also, by definition, the markets GMX doesn't have the
depth to actually let you trade. If this pattern holds, the current
feature/model setup may be systematically finding "signal" in exactly the
corners of the market that can never be traded at real size, which would
make further horizon/model tuning across the full 120-symbol universe a
trap — chasing edge that can never be realized.

**Next step**: restrict training to the ~48 liquidity-cleared symbols from
the start (a symbol allowlist before the sweep, not a methodology rewrite)
to directly test whether the model has any real edge on assets that could
actually be traded — the only version of this question that actually
matters going forward.

### 3e. Done — and the answer is no, not yet

Added `liquid_symbols()` (`risk/market_cap_tiers.py`) and a `--liquid-only`
flag on `--train-per-asset`, computed separately per direction (the liquid
set differs slightly between LONG's and SHORT's available-liquidity
columns). Ran it against the already-trained horizon=12 sweep — no
retraining needed, this just filters the existing corrected-methodology
results down to the 48 LONG / 47 SHORT liquid symbols.

**Result: 0 of 95 liquid symbol/direction combinations promoted.** Real
economics were mostly negative across the entire tradeable universe — BTC
long -4.78%/PF 0.69, ETH long -4.11%/PF 0.79, SOL long -7.89%/PF 0.65, INJ
short -82.81%/PF 0.76, and so on. A handful showed small positive numbers
(SEI short +11.05%/PF 1.34, FIL short +7.70%/PF 6.57 on only 8 trades) but
none cleared the full 3-window bar.

**This is the final, complete, honest answer for the whole per-asset
investigation across both sessions**: once every validation bug is fixed —
train/test leakage (the original single-model bug), calibration methodology
matching real execution, threshold wiring, and now venue liquidity — the
current 18-feature, single-candle-threshold (or 12-candle horizon) design
has no demonstrated edge on any GMX asset that could actually be traded.
Not a bug anywhere left to find; a genuine result about this feature/target
design's lack of edge on the tradeable universe as it exists today. Any
future work here should treat this as the honest baseline and change
something real (features, target definition, a fundamentally different
signal source) rather than re-deriving the same answer through more
validation-methodology fixes.

---

## Future goals / where to pick this up

**Done since Part 2** (all fixed, verified, committed, pushed):
1. ~~Fix the calibration/promotion methodology~~ — done, see 3a. Every
   promotion decision made by this system is now honest.
2. ~~Fix the MEW-style LONG threshold wiring gap~~ — done, see 3b.
3. ~~Check whether GMX liquidity is a one-off or systemic constraint~~ —
   done, see 3d/3e. It's real but not universal (48/120 symbols are liquid);
   restricting to that set and checking for edge there gave a clean 0/95
   promoted. The full per-asset investigation now has a complete, honest
   answer: no demonstrated edge on the tradeable universe with this design.

**Still open:**
4. **Change something real, not just fix more validation plumbing.**
   Everything findable through "is this measuring the truth correctly" has
   been found and fixed across both sessions. The honest result is that this
   feature/target design doesn't have edge. Next steps here mean actually
   different features, a different target definition, a different horizon
   *and* re-checking against liquid symbols only from the start, or a
   different signal source entirely — not another round of methodology
   auditing.
5. **Keep the market-cap snapshot fresh** — 48h gate, needs the live
   scheduler running continuously or a manual refresh step in any backtest
   workflow (`collect_market_caps()`).
6. **The daily GO list is separately stale** — reflects live paper-trading
   track record from the *old* single-shared models, and needs the
   ranking/walk-forward tools (`ranking_engine.py`/`walk_forward.py`,
   explicitly out of scope both sessions) rewired to per-asset models *and*
   real elapsed trading days before anything can realistically show GO.
7. **freqtrade project**: the long-only structural handicap is probably the
   single biggest lever if that project continues — either accept it's
   long-only and only tradeable in bull regimes, or move to a venue with
   margin/futures support if a genuine long+short freqtrade bot is wanted
   (Kraken doesn't support it; Binance/Bybit do in freqtrade).
8. **Both projects would benefit from the same discipline going forward**:
   any time a "backtest" or "promotion" number looks good, check (a) whether
   the evaluation window was truly unseen by training, (b) whether the
   number was computed by the same rules the live/execution engine actually
   uses, and (c) — the newest lesson — whether the venue can actually
   support the size/liquidity the strategy assumes. All three classes of
   mistake showed up across this session.
