# Consolidation handoff: one broker (Alpaca), all asset classes

Goal: stop running crypto (Kraken/freqtrade) and everything else (Alpaca) as two
separate, disconnected systems. One golden-cross strategy, one broker, one place
to look, wherever Alpaca can actually hold the asset.

Do these in order — later steps depend on the result of step 0.

## 0. Blocking fact-check (do this first, everything else depends on it)

Find out which crypto pairs Alpaca's paper account actually lists as tradable.
We need this before touching any code, because it determines what's even possible.

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

Check specifically for BTC/USD, ETH/USD, SOL/USD, TRX/USD, XMR/USD. XMR is the
one most likely to be missing (Monero gets delisted from regulated US venues
over privacy-coin/AML concerns) -- if it's not listed, it cannot move to Alpaca
at any price, full stop, same category of hard limit as GC=F/PA=F futures
already being substituted with GLD/PPLT.

## 1. Why the two systems currently disagree (context, not a task)

`tools/run_cross_asset_golden_cross_backtest.py` (built Alpaca's current
universe in `runtime/cross_asset_golden_cross_results.json`) tests **daily**
bars via yfinance, across every asset class uniformly. Under that test, only
BTC-USD is robust among 14 crypto symbols tried -- ETH/XMR/SOL/TRX all fail.

The system actually live on Kraken (`trading_bot_v4/signals/ma200_asset_qualifier.py`,
feeding `tools/run_ma200_cross_strategy.py`, `tools/run_ma200_paper_trader.py`,
and the freqtrade bots) tests the same golden-cross rule on **4h Kraken bars**
and finds BTC/ETH/PAXG/SOL/TRX/XMR all robust.

Same rule, same assets, different timeframe/data source, different answer.
The 4h Kraken result is the one that's actually been running live and
verified -- trust that one, not the daily Yahoo one, for which crypto assets
have a real edge. PAXG was never in the daily cross-asset test at all (not a
normal Yahoo ticker), so it has no daily-timeframe conclusion either way.

## 2. Merge the verified crypto set into the Alpaca universe

For whichever of BTC/ETH/PAXG/SOL/TRX/XMR step 0 confirms Alpaca lists:

- Add them to `runtime/cross_asset_golden_cross_results.json` (or a new
  merged universe file) as robust, sourced from the **4h Kraken qualifier**
  result, not the daily Yahoo backtest.
- Modify `tools/run_alpaca_paper_trader.py` so those specific symbols pull
  their signal from Kraken 4h data (reuse `compute_trend_state` /
  `current_signal` from `trading_bot_v4/signals/ma200_trend.py`) instead of
  the daily `yf.download` path the rest of the universe uses. Equities/IEF/
  GLD/PPLT keep using daily yfinance exactly as they do now -- only the
  crypto legs change data source.
- Match position sizing to whatever the golden-cross system already proved
  out (15% per-asset, fixed-fractional) for the migrated crypto symbols,
  not the 2% used for the broad equities universe -- don't silently change
  the sizing that was actually verified.

## 3. Whatever Alpaca can't hold stays on Kraken -- by necessity, not oversight

Any of ETH/PAXG/SOL/TRX/XMR that step 0 shows as unlisted on Alpaca keep
trading exactly where they are now (`tools/run_ma200_paper_trader.py` +
the live freqtrade bot on Kraken). Don't try to force them onto Alpaca.
Document in the code/comments which ones and why, so this doesn't get
re-litigated later.

## 4. Retire the redundant systems once step 2 is confirmed working

Only after the consolidated Alpaca bot is confirmed holding the right
positions for at least one full cycle:

- Kill `tools/run_cross_asset_paper_trader_loop.py` (the pre-Alpaca,
  JSON-simulated fake-fill version) -- real Alpaca orders replaced it back
  on Sep 2-3, it's been running redundantly in parallel ever since.
- For any crypto symbol that successfully migrated to Alpaca in step 2,
  stop opening new positions for it in `tools/run_ma200_paper_trader.py`
  and on the live freqtrade bot (`GoldenCrossStrategy`, port 8080) -- let
  existing open positions there close out naturally, don't force-close them.
- Leave the BMSB stream (`BTC`, `PAXG` in `run_ma200_paper_trader.py`) alone
  regardless -- it's an independent signal, not part of this consolidation.

## 5. Separate, still-open bug: the live freqtrade bot crash-loops

Port 8080 (`GoldenCrossStrategy`) has printed "Status: process died" -- a
real crash, not cosmetic. Never diagnosed because the crash traceback goes
to a nohup log outside any folder this session can reach. Get the actual
error:

```
tail -200 /tmp/freqtrade_live_trade_restart.log
```
(or whatever file the currently-running instance is actually logging to --
check with `ps aux | grep freqtrade` for the exact launch command/redirect
target if that path is stale). Paste the real traceback before assuming a fix.

## 6. Minor hardening, not urgent

`v4_ma200_position_monitor` throws intermittent `SSL handshake timed out` /
DNS resolution errors when refreshing live prices. It skips and recovers on
its own every time so far -- not breaking anything -- but a retry/backoff
would quiet the log. Low priority, do last if at all.

## 7. Leave alone, no action needed

- The intermarket regime-filter forward-test bot (port 8081,
  `GoldenCrossRegimeFilteredStrategy`) and its hourly regime-state updater --
  keep running as-is. Verdict from backtesting: baseline (unfiltered) is the
  real system to trust; the regime filter is a legitimate lower-drawdown
  alternative but not a CAGR improvement, and no position-sizing trick closes
  that gap under real capital constraints. Forward data is the only thing
  left to change that conclusion -- give it months, not another backtest pass.
- `runtime/bot_comparison_dashboard.html` -- fine as-is for comparing the two
  freqtrade bots; revisit only if/when Alpaca fully absorbs the crypto side
  and a Kraken-vs-Kraken comparison stops being the relevant question.
