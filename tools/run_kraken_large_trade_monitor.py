"""Large-trade ("whale") monitor for the crypto leg, built on data the
project already has access to -- Kraken's own PUBLIC trades endpoint --
rather than a third-party Whale Alert subscription. This is the crypto-
side half of the user's own request ("golden cross plus whale alert, buy
and sale"); the equities-side half (Alpaca trade prints) is a separate,
larger piece of work because equities trade prints don't carry an
explicit buyer/seller-initiated flag the way Kraken's do, and classifying
aggressor side there needs a tick-rule/quote-comparison algorithm this
script does not attempt.

READ-ONLY, and NOT wired into any trading decision yet. This is
deliberately the first step, not the whole feature: it produces the
large-trade signal and reports it descriptively so it can be looked at
against real data BEFORE deciding how (or whether) to combine it with the
existing 4h golden-cross signal -- confirmation filter, exit-side early
warning, or independent overlay are all still open, same as every other
diagnostic in this project has been built and looked at before being
wired into anything live.

WHAT "LARGE TRADE" MEANS HERE, PRECISELY, PREDECLARED BEFORE RUNNING
AGAINST REAL DATA:
  - Pulls the most recent TRADES_PER_FETCH individual trades (Kraken's
    public /0/public/Trades endpoint, no API key needed -- this is public
    market data, the same real trades Kraken shows anyone) for each of
    the 6 assets already covered by the live Kraken/freqtrade golden-
    cross system (see tools/run_ma200_cross_strategy.py's own ASSETS --
    imported directly here, not redefined, so the pair codes can never
    drift from what the live system already uses: BTC, ETH, XMR, PAXG,
    SOL, TRX).
  - "Large" is defined RELATIVE to each asset's own recent trade-size
    distribution, not a fixed dollar amount -- a $50k trade is unremarkable
    for BTC and enormous for PAXG. A trade is "large" if its notional
    value (price * volume, in USD) is at or above the
    LARGE_TRADE_PERCENTILE-th percentile of the SAME fetched batch's own
    notional sizes, computed separately per asset.
  - Each trade already carries Kraken's own buy/sell tag (the actual
    resting side the trade executed against) -- used as-is, no inference
    needed (this is the one advantage crypto has over equities here).
  - Aggregated into ACCUMULATION / DISTRIBUTION / NEUTRAL per asset via
    NET_IMBALANCE_THRESHOLD_PCT (see constant below) applied to the large
    trades' own buy-vs-sell notional split.

DISCLOSED LIMITATIONS:
  - TRADES_PER_FETCH is a fixed TRADE COUNT (Kraken's API is paginated by
    count, not by calendar time), so the actual calendar span this
    represents differs by asset -- a much more actively-traded pair like
    BTC covers a far shorter time window for the same trade count than a
    quieter one like PAXG. Not apples-to-apples across assets; disclosed,
    not fixed here.
  - This reads Kraken's SPOT market trades. It says nothing about
    aggregate on-chain wallet movements (the actual Whale Alert concept)
    -- it is a large-ORDER detector on the same exchange the live system
    already trades on, not a blockchain-transfer detector. That is a
    real, disclosed difference in what "whale" means here versus the
    on-chain framing discussed earlier.
  - No historical backtest yet -- Kraken's public REST API only exposes
    RECENT trades, not a deep historical trade-tape archive, so testing
    this signal against past golden-cross trades would need a different,
    larger data-engineering effort (capturing trades going forward, or
    sourcing a paid historical tick archive) -- out of scope for this
    first pass, which is a live/forward-looking monitor only.

Requires no credentials -- this is public market data. Run it locally:

    python tools/run_kraken_large_trade_monitor.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests

from tools.run_ma200_cross_strategy import ASSETS

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
OUT_PATH = RUNTIME_DIR / "kraken_large_trade_monitor_results.json"

KRAKEN_TRADES_URL = "https://api.kraken.com/0/public/Trades"
TRADES_PER_FETCH = 1000            # Kraken's own max per call; a trade-COUNT window, not a time window
LARGE_TRADE_PERCENTILE = 99        # top 1% of THIS batch's own notional sizes, per asset
NET_IMBALANCE_THRESHOLD_PCT = 60.0 # >=60% of large notional on one side to call ACCUMULATION/DISTRIBUTION
REQUEST_TIMEOUT_SECONDS = 15


def fetch_recent_trades(pair: str) -> list[dict]:
    """Thin, isolated Kraken-API glue -- ALL network/JSON-shape handling
    lives here so everything else in this module is testable with plain
    dicts and no network access. Returns a list of
    {"price": float, "volume": float, "time": float, "side": "b"|"s"}."""
    resp = requests.get(KRAKEN_TRADES_URL, params={"pair": pair, "count": TRADES_PER_FETCH},
                        timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken API error for {pair}: {data['error']}")
    result = data.get("result", {})
    # result has one key per requested pair (Kraken's own normalized pair name,
    # which can differ slightly in casing/format from the request param) plus "last"
    trade_lists = [v for k, v in result.items() if k != "last"]
    if not trade_lists:
        return []
    raw = trade_lists[0]
    out = []
    for entry in raw:
        # Kraken's schema: [price, volume, time, buy/sell, market/limit, misc, trade_id]
        price, volume, ts, side = entry[0], entry[1], entry[2], entry[3]
        out.append({"price": float(price), "volume": float(volume), "time": float(ts), "side": side})
    return out


def classify_large_trades(trades: list[dict], percentile: float = LARGE_TRADE_PERCENTILE) -> dict:
    """Pure logic, no network: given a batch of trades (as returned by
    fetch_recent_trades, or an equivalent hand-built list in tests),
    compute each trade's notional value, find this batch's own
    percentile-based large-trade threshold, and split large trades by
    side."""
    if not trades:
        return {"n_trades": 0, "threshold_notional": None, "large_trades": []}
    notionals = np.array([t["price"] * t["volume"] for t in trades])
    threshold = float(np.percentile(notionals, percentile))
    large = [
        {**t, "notional": float(t["price"] * t["volume"])}
        for t in trades
        if t["price"] * t["volume"] >= threshold
    ]
    return {"n_trades": len(trades), "threshold_notional": threshold, "large_trades": large}


def summarize_large_trades(classified: dict,
                            net_imbalance_threshold_pct: float = NET_IMBALANCE_THRESHOLD_PCT) -> dict:
    """Aggregates classify_large_trades' output into a single
    ACCUMULATION/DISTRIBUTION/NEUTRAL verdict plus the underlying counts,
    so the verdict's arithmetic is always inspectable rather than a black
    box."""
    large = classified.get("large_trades", [])
    buy_notional = sum(t["notional"] for t in large if t["side"] == "b")
    sell_notional = sum(t["notional"] for t in large if t["side"] == "s")
    total_notional = buy_notional + sell_notional
    if total_notional <= 0:
        pct_buy = float("nan")
        verdict = "NEUTRAL"
    else:
        pct_buy = buy_notional / total_notional * 100
        if pct_buy >= net_imbalance_threshold_pct:
            verdict = "ACCUMULATION"
        elif (100 - pct_buy) >= net_imbalance_threshold_pct:
            verdict = "DISTRIBUTION"
        else:
            verdict = "NEUTRAL"
    return {
        "n_large_trades": len(large),
        "n_large_buys": sum(1 for t in large if t["side"] == "b"),
        "n_large_sells": sum(1 for t in large if t["side"] == "s"),
        "buy_notional_usd": buy_notional,
        "sell_notional_usd": sell_notional,
        "net_notional_usd": buy_notional - sell_notional,
        "pct_of_large_notional_that_is_buy": pct_buy,
        "verdict": verdict,
    }


def main() -> None:
    print("=" * 100)
    print("KRAKEN LARGE-TRADE MONITOR -- crypto leg, read-only, live/forward-looking only")
    print("(public market data, no credentials needed; see this script's own docstring for what")
    print(" 'large' means here and what this does NOT do -- no on-chain data, no historical backtest)")
    print("=" * 100)

    results = {}
    for label, (pair, _path) in ASSETS.items():
        try:
            trades = fetch_recent_trades(pair)
        except Exception as exc:
            print(f"  {label:<6} ({pair}): FETCH FAILED -- {exc}")
            results[label] = {"pair": pair, "error": str(exc)}
            continue
        classified = classify_large_trades(trades)
        summary = summarize_large_trades(classified)
        results[label] = {"pair": pair, "fetched_at": datetime.now(timezone.utc).isoformat(),
                           **classified, **summary}
        # large_trades list is verbose -- keep it in the JSON output but not the console print
        del results[label]["large_trades"]

        print(f"  {label:<6} ({pair}): n_trades={summary['n_large_trades']:>3} of {classified['n_trades']:<4}  "
              f"buys={summary['n_large_buys']:>3} (${summary['buy_notional_usd']:>12,.0f})  "
              f"sells={summary['n_large_sells']:>3} (${summary['sell_notional_usd']:>12,.0f})  "
              f"verdict={summary['verdict']}")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "large_trade_percentile": LARGE_TRADE_PERCENTILE,
        "net_imbalance_threshold_pct": NET_IMBALANCE_THRESHOLD_PCT,
        "trades_per_fetch": TRADES_PER_FETCH,
        "results": results,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    print("\nThis is a snapshot, not a trading decision -- nothing in the live golden-cross system")
    print("reads this output yet. Run it a few times across different market conditions before")
    print("deciding how (or whether) to combine it with entries/exits.")


if __name__ == "__main__":
    main()
