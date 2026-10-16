"""Renders cross_asset_golden_cross_results.json (see
run_cross_asset_golden_cross_backtest.py) into a local HTML dashboard
matching the REAL FreqUI Dashboard tab -- design tokens (colors, font)
pulled directly via getComputedStyle against the actual logged-in running
app over CDP, not approximated from screenshots or docs. Earlier versions
of this page guessed a teal navbar + monospace terminal look from an
incompletely-painted headless DOM dump; the real authenticated app is
dark/near-black with a cyan accent (#00c3e6) and a plain system sans-serif
font -- confirmed live, not assumed.

Real FreqUI Dashboard widgets: Bot comparison table, Daily Profit chart,
Open Trades table, Cumulative Profit chart, Closed Trades table, Profit
Distribution histogram, Trades Log ticker strip. This maps our golden-cross
backtest data into each of those exactly, using asset classes as the "bots"
being compared (Equities/Commodities/Bonds-Rates/Forex/Crypto) since that's
the real grouping this backtest has -- and stays honest where the mapping
doesn't cleanly fit (no open positions in a historical backtest, capital
is pooled not siloed per class, so "Open Profit"/"Balance" show as n/a
rather than a fabricated number).
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
DATA_PATH = RUNTIME_DIR / "cross_asset_golden_cross_results.json"
TRADE_PANE_PATH = RUNTIME_DIR / "trade_pane_data.json"
LOG_PATH = RUNTIME_DIR / "golden_cross_backtest.log"
SNAPSHOT_PATH = RUNTIME_DIR / "cross_asset_snapshot.csv"
PAPER_TRADES_PATH = RUNTIME_DIR / "cross_asset_paper_trades.json"
OUT_PATH = RUNTIME_DIR / "golden_cross_backtest_dashboard.html"

CLASS_ORDER = ["Equities", "Commodities", "Bonds/Rates", "Forex", "Crypto"]
NUMERIC_SNAPSHOT_FIELDS = [
    "price", "ma50", "ma200", "dist_from_200ma_pct", "ret_1m_pct",
    "ret_3m_pct", "ret_6m_pct", "ret_1y_pct", "pct_off_52w_high",
]


def _load_snapshot() -> list[dict]:
    if not SNAPSHOT_PATH.exists():
        return []
    rows = []
    with SNAPSHOT_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in NUMERIC_SNAPSHOT_FIELDS:
                v = row.get(field, "")
                try:
                    fv = float(v)
                    row[field] = None if math.isnan(fv) else fv
                except (TypeError, ValueError):
                    row[field] = None
            rows.append(row)
    return rows


# Inverse of run_alpaca_paper_trader.py's own symbol maps -- real Alpaca
# positions come back keyed by Alpaca's own ticker convention (BRK.B,
# BTCUSD, GLD/PPLT proxies), not the Yahoo/yfinance symbols the rest of
# this project uses (BRK-B, BTC-USD, GC=F/PA=F). Needed to look asset_class
# back up from the robust-universe results, which are keyed by yf symbol.
_ALPACA_TO_YF = {"BRK.B": "BRK-B", "BTCUSD": "BTC-USD", "GLD": "GC=F", "PPLT": "PA=F"}


def _load_open_positions(results: list[dict]) -> list[dict]:
    """Real currently-open positions pulled live from Alpaca's paper
    trading account (tools/run_alpaca_paper_trader_loop.py) -- this is the
    live counterpart to the (necessarily historical, always-closed) __TRADES__
    backtest list, spanning every asset class Alpaca actually trades for us
    (equities, IEF, BTC/USD, GLD/PPLT commodity proxies) -- not just crypto,
    which is all freqtrade's own Trade page can ever show since Kraken
    doesn't list stocks/commodities/forex. Falls back to the older
    simulated-JSON paper trader (cross_asset_paper_trades.json) only if no
    Alpaca credentials are configured, so this still works before that
    migration for anyone re-running the renderer standalone."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    asset_class_by_symbol = {r["symbol"]: r["asset_class"] for r in results}

    if api_key and secret_key:
        from alpaca.trading.client import TradingClient

        client = TradingClient(api_key, secret_key, paper=True)
        positions = []
        for p in client.get_all_positions():
            yf_symbol = _ALPACA_TO_YF.get(p.symbol, p.symbol)
            entry_price = float(p.avg_entry_price)
            current_price = float(p.current_price)
            profit_pct = float(p.unrealized_plpc) * 100
            positions.append({
                "asset_class": asset_class_by_symbol.get(yf_symbol, "Equities"),
                "symbol": yf_symbol,
                "amount": float(p.qty),
                "stake_amount": float(p.cost_basis),
                "open_rate": entry_price,
                "current_rate": current_price,
                "profit_pct": profit_pct,
                "profit_abs": float(p.unrealized_pl),
                "open_date": "",  # Alpaca positions don't carry an open timestamp
            })
        return positions

    if not PAPER_TRADES_PATH.exists():
        return []
    paper = json.loads(PAPER_TRADES_PATH.read_text(encoding="utf-8"))
    snapshot = _load_snapshot()
    price_by_symbol = {row["symbol"]: row.get("price") for row in snapshot}
    positions = []
    for symbol, pos in paper.get("positions", {}).items():
        entry_price = pos["entry_price"]
        current_price = price_by_symbol.get(symbol)
        profit_pct = (
            (current_price - entry_price) / entry_price * 100
            if current_price is not None and entry_price
            else None
        )
        allocated = pos["allocated_amount"]
        positions.append({
            "asset_class": pos["asset_class"],
            "symbol": symbol,
            "amount": allocated / entry_price if entry_price else 0,
            "stake_amount": allocated,
            "open_rate": entry_price,
            "current_rate": current_price,
            "profit_pct": profit_pct,
            "profit_abs": allocated * profit_pct / 100 if profit_pct is not None else None,
            "open_date": pos["entry_time"],
        })
    positions.sort(key=lambda p: p["open_date"], reverse=True)
    return positions


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    trade_pane = json.loads(TRADE_PANE_PATH.read_text(encoding="utf-8")) if TRADE_PANE_PATH.exists() else {"assets": {}}
    log_text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else "(no captured log for this run)"
    snapshot = _load_snapshot()
    results = data["results"]
    open_positions = _load_open_positions(results)
    portfolio = data["portfolio"]
    trades = data["trades"]
    total = len(results)
    robust = [r for r in results if r["robust"]]

    bot_rows = []
    for ac in CLASS_ORDER:
        ac_trades = [t for t in trades if t["asset_class"] == ac]
        ac_open = [p for p in open_positions if p["asset_class"] == ac]
        if not ac_trades and not ac_open:
            continue
        wins = sum(1 for t in ac_trades if t["net_ret_pct"] > 0)
        losses = len(ac_trades) - wins
        closed_profit_abs = sum(t["profit_abs"] for t in ac_trades)
        closed_profit_pct = sum(t["net_ret_pct"] for t in ac_trades) / len(ac_trades) if ac_trades else 0
        open_profit_abs = sum(p["profit_abs"] for p in ac_open if p["profit_abs"] is not None)
        bot_rows.append({
            "bot": ac, "trades": len(ac_trades), "wins": wins, "losses": losses,
            "closed_profit_pct": closed_profit_pct, "closed_profit_abs": closed_profit_abs,
            "open_count": len(ac_open), "open_profit_abs": open_profit_abs,
        })
    total_wins = sum(b["wins"] for b in bot_rows)
    total_losses = sum(b["losses"] for b in bot_rows)
    total_trades_n = sum(b["trades"] for b in bot_rows)
    summary_row = {
        "bot": "Summary", "trades": total_trades_n, "wins": total_wins, "losses": total_losses,
        "closed_profit_pct": sum(t["net_ret_pct"] for t in trades) / len(trades) if trades else 0,
        "closed_profit_abs": sum(t["profit_abs"] for t in trades),
        "open_count": len(open_positions),
        "open_profit_abs": sum(p["profit_abs"] for p in open_positions if p["profit_abs"] is not None),
    }

    html = HTML_TEMPLATE.replace("__RESULTS__", json.dumps(results)) \
        .replace("__TRADES__", json.dumps(trades)) \
        .replace("__OPEN_POSITIONS__", json.dumps(open_positions)) \
        .replace("__EQUITY_CURVE__", json.dumps(portfolio["equity_curve"])) \
        .replace("__BOT_ROWS__", json.dumps(bot_rows)) \
        .replace("__SUMMARY_ROW__", json.dumps(summary_row)) \
        .replace("__TRADE_PANE__", json.dumps(trade_pane["assets"])) \
        .replace("__LOG_TEXT__", json.dumps(log_text)) \
        .replace("__SNAPSHOT__", json.dumps(snapshot)) \
        .replace("__CLASS_ORDER__", json.dumps(CLASS_ORDER)) \
        .replace("__GENERATED_AT__", data["generated_at"][:16].replace("T", " ") + " UTC") \
        .replace("__COST__", f'{data["cost_bps_round_trip"]:.0f}') \
        .replace("__ALLOC_PCT__", f'{data["per_trade_allocation_pct"]:.0f}') \
        .replace("__MIN_TRADES__", str(data["min_trades"])) \
        .replace("__TOTAL__", str(total)) \
        .replace("__ROBUST__", str(len(robust))) \
        .replace("__START_CAP__", f'{portfolio["starting_capital"]:,.0f}') \
        .replace("__FINAL_CAP__", f'{portfolio["final_capital"]:,.0f}') \
        .replace("__TOTAL_RET__", f'{portfolio["total_return_pct"]:+,.0f}') \
        .replace("__MAX_DD__", f'{portfolio["max_drawdown_pct"]:.1f}') \
        .replace("__TOTAL_TRADES_LABEL__", f'{len(trades):,} trades')

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


HTML_TEMPLATE = r"""<title>Golden Cross / All Markets</title>
<style>
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  /* Tokens pulled directly from the real, running FreqUI's computed styles
     (getComputedStyle via CDP against the actual logged-in app) -- not
     approximated from screenshots. Earlier versions of this page guessed
     a teal navbar + monospace terminal look from an incompletely-painted
     headless dump; the authenticated app is dark/near-black with a cyan
     accent and a plain system sans-serif font, confirmed live. */
  :root {
    --bg: oklch(21.8% 0.008 223.9);
    --bg-elevated: oklch(27.5% 0.011 216.9);
    --border: oklch(27.5% 0.011 216.9);
    --text: oklch(92.5% 0.005 214.3);
    --text-muted: oklch(72.3% 0.014 214.4);
    --text-highlighted: #fff;
    --primary: #00c3e6;
    --primary-dim: #0089a1;
    --positive: oklch(44.8% 0.119 151.328);
    --positive-text: #d1fae5;
    --negative: oklch(57% 0.245 27.33);
    --negative-text: #fee2e2;
    --sans: -apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; line-height: 1.5; }

  .navbar { background: var(--bg); border-bottom: 1px solid var(--border); color: var(--text); display: flex; align-items: center; gap: 20px; padding: 0 20px; height: 46px; }
  .navbar .brand { font-weight: 700; font-size: 15px; display: flex; align-items: center; gap: 8px; }
  .navbar .brand .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--primary); }
  .navbar nav { display: flex; gap: 18px; font-size: 13px; }
  .navbar nav span { cursor: pointer; padding-bottom: 14px; border-bottom: 2px solid transparent; user-select: none; color: var(--text-muted); }
  .navbar nav span:hover { color: var(--text-highlighted); }
  .navbar nav span.active { font-weight: 600; color: var(--text-highlighted); border-bottom: 2px solid var(--primary); }
  .navbar .spacer { flex: 1; }
  .navbar .status { font-size: 12px; display: flex; align-items: center; gap: 8px; color: var(--text-muted); }
  .navbar .status .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--positive); }
  .navbar .avatar { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px; padding: 3px 8px; font-weight: 700; font-size: 11px; }

  .wrap { max-width: 1500px; margin: 0 auto; padding: 16px 16px 60px; }
  .subtitle { font-size: 11px; color: var(--text-muted); margin-bottom: 14px; }

  .profit-banner { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px 24px; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; }
  .profit-banner .label { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 5px; }
  .profit-banner .figures { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text-highlighted); }
  .profit-banner .figures .arrow { color: var(--text-muted); font-weight: 400; margin: 0 8px; }
  .profit-banner .pct { font-size: 24px; font-weight: 800; font-variant-numeric: tabular-nums; }
  .profit-banner .pct.pos { color: #34d399; }
  .profit-banner .pct.neg { color: #f87171; }
  .profit-banner .dd { font-size: 12px; color: #f87171; }

  .grid { display: grid; grid-template-columns: 1.7fr 1fr; gap: 14px; align-items: start; }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
  .col { display: flex; flex-direction: column; gap: 14px; }

  .panel { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .panel h3 { margin: 0; padding: 6px 8px; font-size: 12.5px; font-weight: 500; color: var(--text); border-bottom: 1px solid var(--border); background: var(--bg-elevated); text-align: center; }
  .panel .body { padding: 6px 0; }
  .panel canvas { display: block; width: 100%; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  thead th { text-align: right; font-weight: 600; font-size: 11.5px; color: var(--text-highlighted); padding: 7px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; cursor: pointer; }
  thead th:first-child, thead th:nth-child(2) { text-align: left; }
  thead th.sorted { color: var(--primary); }
  tbody td { text-align: right; padding: 6px 12px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; white-space: nowrap; }
  tbody td:first-child, tbody td:nth-child(2) { text-align: left; }
  tbody tr:hover td { background: var(--bg-elevated); }
  .pill { display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; min-width: 64px; justify-content: center; }
  .pill.pos { background: var(--positive); color: var(--positive-text); }
  .pill.neg { background: var(--negative); color: var(--negative-text); }
  .pill.na { color: var(--text-muted); border: 1px solid var(--border); background: transparent; }
  .pill.trans { color: var(--text-muted); border: 1px solid var(--border); background: transparent; }

  .snapshot-controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; padding: 10px 12px; border-bottom: 1px solid var(--border); }
  .snapshot-controls .chip { font-size: 11px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-elevated); color: var(--text-muted); cursor: pointer; user-select: none; }
  .snapshot-controls .chip:hover { color: var(--text); }
  .snapshot-controls .chip.active { background: color-mix(in srgb, var(--primary) 18%, transparent); border-color: var(--primary); color: var(--primary); }
  .snapshot-controls #snap-search { font-size: 12px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 999px; width: 160px; margin-left: auto; }
  .snap-class-header { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--primary); padding: 10px 12px 4px; }
  .snap-class-header .count { color: var(--text-muted); font-weight: 400; text-transform: none; letter-spacing: normal; }
  .ret-cell { border-radius: 4px; }

  .table-scroll { max-height: 320px; overflow: auto; }
  .empty-state { padding: 30px; text-align: center; color: var(--text-muted); font-size: 12px; }
  td.empty-state { padding: 24px; }

  .info-i { font-size: 12px; color: var(--text-muted); cursor: help; margin-left: 4px; }

  .pagination { display: flex; justify-content: center; align-items: center; gap: 4px; padding: 10px 12px; }
  .pagination button { background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-muted); border-radius: 6px; padding: 4px 9px; font-size: 12px; cursor: pointer; }
  .pagination button:hover { color: var(--text); }
  .pagination button.active { background: var(--primary); border-color: var(--primary); color: var(--bg); font-weight: 700; }
  .pagination button:disabled { opacity: 0.4; cursor: default; }

  .chart-toolbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  .seg { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .seg-btn { background: var(--bg); border: none; border-right: 1px solid var(--border); color: var(--text-muted); font-size: 11px; padding: 4px 10px; cursor: pointer; }
  .seg-btn:last-child { border-right: none; }
  .seg-btn.active { background: var(--primary); color: var(--bg); font-weight: 700; }
  .chart-legend-inline { margin-left: auto; font-size: 11px; color: var(--text-muted); display: flex; align-items: center; }
  .chart-legend-inline .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }

  .analysis { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--primary); border-radius: 8px; padding: 16px 20px; margin: 14px 0; }
  .analysis h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--primary); margin: 0 0 10px; }
  .analysis p { margin: 0 0 10px; max-width: 82ch; color: var(--text); }
  .analysis p:last-child { margin-bottom: 0; }
  .analysis .flag { color: #f87171; font-weight: 700; }
  .analysis .good { color: #34d399; font-weight: 700; }
  .analysis ul { margin: 0 0 10px; padding-left: 18px; }
  .analysis li { margin-bottom: 5px; }

  .trade-layout { display: grid; grid-template-columns: 220px 1fr; gap: 14px; align-items: start; }
  @media (max-width: 860px) { .trade-layout { grid-template-columns: 1fr; } }
  .sidebar { padding-bottom: 8px; }
  .pair-list { max-height: 480px; overflow-y: auto; }
  .pair-item { padding: 9px 14px; font-size: 12.5px; cursor: pointer; border-left: 3px solid transparent; color: var(--text); display: flex; justify-content: space-between; }
  .pair-item:hover { background: var(--bg-elevated); }
  .pair-item.active { background: var(--bg-elevated); border-left-color: var(--primary); color: var(--primary); font-weight: 600; }
  .pair-item .n { color: var(--text-muted); font-weight: 400; }
  .trade-main { display: flex; flex-direction: column; gap: 14px; }
  .chart-legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px; color: var(--text-muted); padding: 0 14px 8px; }
  .chart-legend span.sw { display: inline-block; width: 10px; height: 2.5px; margin-right: 5px; vertical-align: 2px; }
  .chart-legend span.tri { color: #34d399; }
  .chart-legend span.dia { color: #f87171; }
  .logs-pre { margin: 0; padding: 14px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11.5px; color: #9fe6b0; white-space: pre-wrap; max-height: 640px; overflow-y: auto; line-height: 1.6; }

  .stat-strip { display: grid; grid-template-columns: repeat(7, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); margin-bottom: 14px; border-radius: 8px; overflow: hidden; }
  .stat { background: var(--bg); padding: 10px 12px; }
  .stat .label { font-size: 10.5px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .stat .value { font-size: 18px; font-weight: 700; color: var(--primary); font-variant-numeric: tabular-nums; }
  .stat .value .of { font-size: 12px; color: var(--text-muted); font-weight: 400; }

  .ticker { display: flex; gap: 4px; overflow-x: auto; padding: 10px 14px; }
  .ticker .tag { font-size: 11px; font-weight: 600; padding: 3px 6px; border-radius: 4px; white-space: nowrap; }
  .ticker .tag.pos { background: var(--positive); color: var(--positive-text); }
  .ticker .tag.neg { background: var(--negative); color: var(--negative-text); }

  #search { font-size: 12px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 6px; margin: 8px 12px; width: calc(100% - 24px); }
  #search:focus { outline: none; border-color: var(--primary); }
</style>

<div class="navbar">
  <div class="brand"><span class="dot"></span>Freqtrade UI</div>
  <nav>
    <span data-tab="trade">Trade</span><span data-tab="dashboard" class="active">Dashboard</span><span data-tab="chart">Chart</span><span data-tab="logs">Logs</span>
  </nav>
  <div class="spacer"></div>
  <div class="status"><span class="dot"></span>golden_cross_all_markets &middot; Online</div>
  <div class="avatar">FT</div>
</div>

<div class="wrap">
  <div class="subtitle">generated __GENERATED_AT__ &middot; __COST__bps round-trip cost &middot; __ALLOC_PCT__% capital per trade &middot; min __MIN_TRADES__ trades to score robustness &middot; historical backtest, not live</div>

<div id="tab-dashboard" class="tab-page">

  <div class="profit-banner">
    <div>
      <div class="label">PORTFOLIO RESULT</div>
      <div class="figures">$__START_CAP__<span class="arrow">&rarr;</span>$__FINAL_CAP__</div>
    </div>
    <div class="pct" id="total-ret-pct">__TOTAL_RET__%</div>
    <div class="dd">max drawdown __MAX_DD__%</div>
  </div>

  <div class="grid">
    <div class="col">
      <div class="panel">
        <h3>Bot comparison</h3>
        <div class="body">
          <table>
            <thead><tr><th></th><th>Bot Name</th><th>Trades</th><th>Open Profit</th><th>Closed Profit</th><th>Balance</th><th>W/L</th></tr></thead>
            <tbody id="bot-comparison-body"></tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <h3>Open Trades <span class="info-i" title="Real currently-open positions from the live cross-asset paper trader (tools/run_cross_asset_paper_trader_loop.py) -- spans every asset class it trades, not just the crypto pairs freqtrade's own Trade page can show">&#9432;</span></h3>
        <div class="table-scroll">
          <table>
            <thead><tr>
              <th>Bot</th><th>ID</th><th>Pair</th><th>Amount</th><th>Stake amount</th>
              <th>Open rate</th><th>Current rate</th><th>Current profit %</th><th>Open date</th>
            </tr></thead>
            <tbody id="open-trades-body"><tr><td colspan="9" class="empty-state">No Trades to show.</td></tr></tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <h3>Market Snapshot <span style="color:var(--muted);font-weight:400;">(live trend state, same 50/200MA read as the backtest)</span></h3>
        <div class="snapshot-controls">
          <button class="chip active" data-snap-filter="ALL">All</button>
          <button class="chip" data-snap-filter="BULL">Bull</button>
          <button class="chip" data-snap-filter="BEAR">Bear</button>
          <button class="chip" data-snap-filter="TRANSITION">Transition</button>
          <input id="snap-search" type="text" placeholder="filter ticker&hellip;" autocomplete="off" />
        </div>
        <div id="snapshot-sections"></div>
      </div>

      <div class="panel">
        <h3>Closed Trades <span style="color:var(--muted);font-weight:400;">(__TOTAL_TRADES_LABEL__)</span></h3>
        <input id="search" type="text" placeholder="filter symbol&hellip;" autocomplete="off" />
        <div class="table-scroll">
          <table>
            <thead><tr>
              <th data-key="asset_class">Bot</th><th data-key="id">ID</th><th data-key="symbol">Pair</th><th data-key="amount">Amount</th>
              <th data-key="stake_amount">Total stake amount</th><th data-key="open_rate">Open rate</th>
              <th data-key="close_rate">Close rate</th><th data-key="net_ret_pct">Profit %</th>
              <th data-key="entry_time">Open date</th><th data-key="exit_time">Close date</th><th>Close Reason</th>
            </tr></thead>
            <tbody id="closed-trades-body"></tbody>
          </table>
        </div>
        <div class="pagination" id="closed-trades-pagination"></div>
      </div>
    </div>

    <div class="col">
      <div class="panel">
        <h3>Profit over time combined</h3>
        <div class="chart-toolbar">
          <div class="seg" id="daily-period-toggle">
            <button class="seg-btn active" data-period="days">Days</button>
            <button class="seg-btn" data-period="weeks">Weeks</button>
            <button class="seg-btn" data-period="months">Months</button>
          </div>
          <div class="chart-legend-inline"><span class="sw" style="background:#f87171"></span>Absolute profit<span class="sw" style="background:#4b5563;margin-left:10px;"></span>Trade Count</div>
          <div class="seg" id="daily-mode-toggle">
            <button class="seg-btn active" data-mode="abs">Abs $</button>
            <button class="seg-btn" data-mode="rel">Rel %</button>
          </div>
        </div>
        <div class="body"><canvas id="daily-canvas" style="height:170px"></canvas></div>
      </div>
      <div class="panel">
        <h3>Cumulative Profit</h3>
        <div class="body"><canvas id="cumulative-canvas" style="height:230px"></canvas></div>
      </div>
      <div class="panel">
        <h3>Wallet History</h3>
        <div class="body"><div class="empty-state">No historic wallet data available.<br><span style="opacity:0.7">This is a backtest, not a live/dry-run bot instance -- freqtrade only records real wallet-balance history for running bots.</span></div></div>
      </div>
      <div class="panel">
        <h3>Profit Distribution</h3>
        <div class="body"><canvas id="dist-canvas" style="height:170px"></canvas></div>
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:14px;">
    <h3>Trades Log <span style="color:var(--muted);font-weight:400;">(most recent 200, chronological)</span></h3>
    <div class="ticker" id="trades-log"></div>
  </div>

  <div class="analysis">
    <h2>What's actually going on here</h2>
    <p><strong>__ROBUST__ of __TOTAL__</strong> instruments cleared the same bar used on crypto all session
    (profit factor &ge; 1.0 excluding the 3 biggest winning trades) &mdash; but that's almost entirely equities,
    and needs two real caveats before you trust it.</p>
    <ul>
      <li><span class="flag">Small samples.</span> Many "robust" stocks cleared the bar on as few as 6&ndash;15
      trades total. Crypto's numbers elsewhere in this project came from 30&ndash;70+ trades per asset on 4h
      bars over a decade-plus; most equities here run a third of that sample on daily bars.</li>
      <li><span class="flag">Survivorship bias.</span> This list is built from companies large and successful
      <em>today</em> &mdash; it structurally cannot include anything that got delisted or went to zero.</li>
    </ul>
    <p><span class="good">Forex: 0 of 8 robust, clean</span> &mdash; a real negative result, not an artifact.
    Crypto only kept <strong>BTC-USD</strong>; ETH/XMR/SOL/TRX are robust in the main 4h Kraken system elsewhere
    in this project but failed here, almost certainly because Yahoo's daily crypto history is much shorter
    (6&ndash;16 trades here vs. 29&ndash;59 on the real system), not a real disagreement.</p>
  </div>

  <div class="stat-strip">
    <div class="stat"><div class="label">Tested</div><div class="value">__TOTAL__</div></div>
    <div class="stat"><div class="label">Robust</div><div class="value">__ROBUST__</div></div>
  </div>
</div><!-- /tab-dashboard -->
</div><!-- /wrap -->

<div id="tab-trade" class="tab-page wrap" style="display:none;">
  <div class="trade-layout">
    <div class="panel sidebar">
      <h3>Multi Pane</h3>
      <div class="pair-list" id="pair-list"></div>
    </div>
    <div class="trade-main">
      <div class="panel">
        <h3 id="trade-chart-title">Chart</h3>
        <div class="chart-legend" id="chart-legend"></div>
        <div class="body">
          <canvas id="candle-canvas" style="height:340px"></canvas>
          <canvas id="volume-canvas" style="height:70px"></canvas>
        </div>
      </div>
      <div class="panel">
        <h3>Open Trades</h3>
        <div class="body"><div class="empty-state">Currently no open trades.</div></div>
      </div>
      <div class="panel">
        <h3 id="trade-closed-title">Closed Trades</h3>
        <div class="table-scroll">
          <table>
            <thead><tr><th>ID</th><th>Pair</th><th>Open rate</th><th>Close rate</th><th>Profit %</th><th>Open date</th><th>Close date</th></tr></thead>
            <tbody id="trade-closed-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div><!-- /tab-trade -->

<div id="tab-chart" class="tab-page wrap" style="display:none;">
  <div class="panel">
    <h3>Instruments &mdash; all 116, golden cross robustness</h3>
    <input id="chart-search" type="text" placeholder="filter symbol&hellip;" autocomplete="off" />
    <div class="table-scroll" style="max-height:640px;">
      <table>
        <thead><tr>
          <th data-ckey="symbol">Symbol</th><th data-ckey="asset_class">Class</th><th data-ckey="robust">Status</th>
          <th data-ckey="n_trades">N</th><th data-ckey="win_rate">Win Rate</th><th data-ckey="pf">PF</th>
          <th data-ckey="pf_excl3">PF excl3</th><th data-ckey="sum_excl3">Sum excl3</th>
        </tr></thead>
        <tbody id="chart-tab-body"></tbody>
      </table>
    </div>
  </div>
</div><!-- /tab-chart -->

<div id="tab-logs" class="tab-page wrap" style="display:none;">
  <div class="panel">
    <h3>Logs &mdash; real captured stdout from the last backtest run</h3>
    <pre id="logs-content" class="logs-pre"></pre>
  </div>
</div><!-- /tab-logs -->

<script>
  const RESULTS = __RESULTS__;
  const TRADES = __TRADES__;
  const OPEN_POSITIONS = __OPEN_POSITIONS__;
  const EQUITY_CURVE = __EQUITY_CURVE__;
  const BOT_ROWS = __BOT_ROWS__;
  const SUMMARY_ROW = __SUMMARY_ROW__;
  const SNAPSHOT = __SNAPSHOT__;
  const SNAPSHOT_CLASS_ORDER = __CLASS_ORDER__;
  const BULL = "#34d399", BEAR = "#f87171", GRID = "#3a4150", MUTED = "#8a93a3", LINE = "#e5e9ec", HIST = "#00c3e6";

  const totalRetEl = document.getElementById("total-ret-pct");
  totalRetEl.classList.add(totalRetEl.textContent.trim().startsWith("-") ? "neg" : "pos");

  // ==================== Market Snapshot panel ====================
  (function () {
    const SNAP_COLS = [
      { key: "symbol", label: "Symbol" }, { key: "trend", label: "Trend" },
      { key: "price", label: "Price" }, { key: "dist_from_200ma_pct", label: "vs 200MA" },
      { key: "ret_1m_pct", label: "1M" }, { key: "ret_3m_pct", label: "3M" },
      { key: "ret_6m_pct", label: "6M" }, { key: "ret_1y_pct", label: "1Y" },
      { key: "pct_off_52w_high", label: "Off 52wHi" },
    ];
    const snapState = { filter: "ALL", search: "", sort: {} };

    function fmtPrice(v) {
      if (v === null || v === undefined) return "—";
      const abs = Math.abs(v);
      const dp = abs >= 100 ? 2 : abs >= 1 ? 3 : 5;
      return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
    }
    function fmtSnapPct(v) { return v === null || v === undefined ? "—" : (v >= 0 ? "+" : "") + v.toFixed(1) + "%"; }
    function heat(v) {
      if (v === null || v === undefined) return "";
      const mag = Math.min(Math.abs(v) / 40, 1);
      const alpha = (mag * 0.30).toFixed(2);
      return v >= 0 ? `rgba(52,211,153,${alpha})` : `rgba(248,113,113,${alpha})`;
    }
    function trendPillClass(t) { return t === "BULL" ? "pos" : t === "BEAR" ? "neg" : "trans"; }
    function trendGlyph(t) { return t === "BULL" ? "▲" : t === "BEAR" ? "▼" : "―"; }

    const container = document.getElementById("snapshot-sections");
    SNAPSHOT_CLASS_ORDER.forEach(cls => {
      const rows = SNAPSHOT.filter(r => r.asset_class === cls);
      if (!rows.length) return;
      const header = document.createElement("div");
      header.className = "snap-class-header";
      header.innerHTML = `${cls} <span class="count">(${rows.length})</span>`;
      container.appendChild(header);

      const scroll = document.createElement("div");
      scroll.className = "table-scroll";
      scroll.style.maxHeight = "240px";
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const trh = document.createElement("tr");
      SNAP_COLS.forEach(c => {
        const th = document.createElement("th");
        th.textContent = c.label;
        th.dataset.snapKey = c.key;
        th.addEventListener("click", () => sortSnap(cls, c.key));
        trh.appendChild(th);
      });
      thead.appendChild(trh);
      const tbody = document.createElement("tbody");
      table.appendChild(thead);
      table.appendChild(tbody);
      scroll.appendChild(table);
      container.appendChild(scroll);

      snapState.sort[cls] = { key: "ret_3m_pct", dir: -1 };
      renderSnapTable(cls, rows);
    });

    function renderSnapTable(cls, rows) {
      const scrollDiv = [...container.querySelectorAll(".snap-class-header")]
        .find(h => h.textContent.startsWith(cls)).nextElementSibling;
      const tbody = scrollDiv.querySelector("tbody");
      const { key, dir } = snapState.sort[cls];
      const sorted = [...rows].sort((a, b) => {
        let av = a[key], bv = b[key];
        if (key === "symbol" || key === "trend") { av = av || ""; bv = bv || ""; return dir * String(av).localeCompare(String(bv)); }
        av = av === null || av === undefined ? -Infinity : av;
        bv = bv === null || bv === undefined ? -Infinity : bv;
        return dir * (av - bv);
      });
      scrollDiv.querySelectorAll("th").forEach(th => {
        th.classList.toggle("sorted", th.dataset.snapKey === key);
      });
      tbody.innerHTML = "";
      let visible = 0;
      sorted.forEach(r => {
        const matchesFilter = snapState.filter === "ALL" || r.trend === snapState.filter;
        const matchesSearch = !snapState.search || r.symbol.toLowerCase().includes(snapState.search);
        if (!matchesFilter || !matchesSearch) return;
        visible++;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${r.symbol}</td>
          <td><span class="pill ${trendPillClass(r.trend)}">${trendGlyph(r.trend)} ${r.trend}</span></td>
          <td>${fmtPrice(r.price)}</td>
          <td>${fmtSnapPct(r.dist_from_200ma_pct)}</td>
          <td class="ret-cell" style="background:${heat(r.ret_1m_pct)}">${fmtSnapPct(r.ret_1m_pct)}</td>
          <td class="ret-cell" style="background:${heat(r.ret_3m_pct)}">${fmtSnapPct(r.ret_3m_pct)}</td>
          <td class="ret-cell" style="background:${heat(r.ret_6m_pct)}">${fmtSnapPct(r.ret_6m_pct)}</td>
          <td class="ret-cell" style="background:${heat(r.ret_1y_pct)}">${fmtSnapPct(r.ret_1y_pct)}</td>
          <td>${fmtSnapPct(r.pct_off_52w_high)}</td>
        `;
        tbody.appendChild(tr);
      });
      const headerEl = [...container.querySelectorAll(".snap-class-header")].find(h => h.textContent.startsWith(cls));
      headerEl.style.display = visible === 0 ? "none" : "";
      scrollDiv.style.display = visible === 0 ? "none" : "";
    }

    function sortSnap(cls, key) {
      const cur = snapState.sort[cls];
      snapState.sort[cls] = cur.key === key ? { key, dir: -cur.dir } : { key, dir: key === "symbol" || key === "trend" ? 1 : -1 };
      renderSnapTable(cls, SNAPSHOT.filter(r => r.asset_class === cls));
    }

    function renderAllSnap() {
      SNAPSHOT_CLASS_ORDER.forEach(cls => {
        const rows = SNAPSHOT.filter(r => r.asset_class === cls);
        if (rows.length) renderSnapTable(cls, rows);
      });
    }

    document.querySelectorAll("[data-snap-filter]").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("[data-snap-filter]").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        snapState.filter = chip.dataset.snapFilter;
        renderAllSnap();
      });
    });
    document.getElementById("snap-search").addEventListener("input", e => {
      snapState.search = e.target.value.trim().toLowerCase();
      renderAllSnap();
    });
  })();

  // ---- Bot comparison ----
  function fmtPct(v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "%"; }
  function fmtAbs(v) {
    const a = Math.abs(v);
    const s = a >= 1e6 ? (a/1e6).toFixed(2)+"M" : a >= 1e3 ? (a/1e3).toFixed(1)+"k" : a.toFixed(0);
    return (v >= 0 ? "" : "-") + s;
  }
  function pill(pctValue, absValue) {
    const cls = pctValue >= 0 ? "pos" : "neg";
    const arrow = pctValue >= 0 ? "▲" : "▼";
    const absPart = absValue === undefined ? "" : ` (${fmtAbs(absValue)})`;
    return `<span class="pill ${cls}">${arrow} ${fmtPct(pctValue)}${absPart}</span>`;
  }
  function dollarPill(absValue) {
    const cls = absValue >= 0 ? "pos" : "neg";
    const arrow = absValue >= 0 ? "▲" : "▼";
    return `<span class="pill ${cls}">${arrow} ${fmtAbs(absValue)}</span>`;
  }
  const botBody = document.getElementById("bot-comparison-body");
  [...BOT_ROWS, SUMMARY_ROW].forEach((b, i) => {
    const tr = document.createElement("tr");
    const openCell = b.open_count ? dollarPill(b.open_profit_abs) : `<span class="pill na">n/a</span>`;
    const tradesCell = `${b.trades} / ${b.open_count || 0}`;
    if (b.bot === "Summary") {
      tr.style.fontWeight = "700";
      tr.innerHTML = `
        <td></td><td>${b.bot}</td>
        <td>${tradesCell}</td>
        <td>${openCell}</td>
        <td>${pill(b.closed_profit_pct, b.closed_profit_abs)}</td>
        <td><span class="pill na">n/a</span></td>
        <td><span style="color:${BULL}">${b.wins}</span> / <span style="color:${BEAR}">${b.losses}</span></td>
      `;
    } else {
      tr.innerHTML = `
        <td><input type="checkbox" checked disabled /></td>
        <td>${b.bot}</td>
        <td>${tradesCell}</td>
        <td>${openCell}</td>
        <td>${pill(b.closed_profit_pct, b.closed_profit_abs)}</td>
        <td><span class="pill na">n/a</span></td>
        <td><span style="color:${BULL}">${b.wins}</span> / <span style="color:${BEAR}">${b.losses}</span></td>
      `;
    }
    botBody.appendChild(tr);
  });

  // ---- Open Trades table (live positions from the cross-asset paper trader) ----
  const openBody = document.getElementById("open-trades-body");
  if (OPEN_POSITIONS.length) {
    openBody.innerHTML = "";
    const frag = document.createDocumentFragment();
    OPEN_POSITIONS.forEach((p, i) => {
      const tr = document.createElement("tr");
      const curRate = p.current_rate === null || p.current_rate === undefined ? "—" : (p.current_rate < 10 ? p.current_rate.toFixed(4) : p.current_rate.toFixed(2));
      const profitCell = p.profit_pct === null || p.profit_pct === undefined ? `<span class="pill na">n/a</span>` : pill(p.profit_pct, p.profit_abs);
      tr.innerHTML = `
        <td>${p.asset_class}</td>
        <td>${i + 1}</td>
        <td>${p.symbol}</td>
        <td>${p.amount < 1 ? p.amount.toFixed(4) : p.amount.toFixed(2)}</td>
        <td>${fmtAbs(p.stake_amount)}</td>
        <td>${p.open_rate < 10 ? p.open_rate.toFixed(4) : p.open_rate.toFixed(2)}</td>
        <td>${curRate}</td>
        <td>${profitCell}</td>
        <td>${p.open_date ? p.open_date.slice(0,10) : "&mdash;"}</td>
      `;
      frag.appendChild(tr);
    });
    openBody.appendChild(frag);
  }

  // ---- Closed trades table ----
  const closedBody = document.getElementById("closed-trades-body");
  const PAGE_SIZE = 50;
  const tableState = { sort: { key: "exit_time", dir: -1 }, search: "", page: 1 };
  function renderClosedTrades() {
    const { key, dir } = tableState.sort;
    let rows = TRADES;
    if (tableState.search) rows = rows.filter(t => t.symbol.toLowerCase().includes(tableState.search));
    rows = [...rows].sort((a, b) => {
      let av = a[key], bv = b[key];
      if (typeof av === "string") return dir * av.localeCompare(bv);
      return dir * (av - bv);
    });
    document.querySelectorAll("th[data-key]").forEach(th => th.classList.toggle("sorted", th.dataset.key === key));

    const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    tableState.page = Math.min(tableState.page, totalPages);
    const pageRows = rows.slice((tableState.page - 1) * PAGE_SIZE, tableState.page * PAGE_SIZE);

    closedBody.innerHTML = "";
    const frag = document.createDocumentFragment();
    pageRows.forEach(t => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${t.asset_class}</td>
        <td>${t.id}</td>
        <td>${t.symbol}</td>
        <td>${t.amount < 1 ? t.amount.toFixed(4) : t.amount.toFixed(2)}</td>
        <td>${fmtAbs(t.stake_amount)}</td>
        <td>${t.open_rate < 10 ? t.open_rate.toFixed(4) : t.open_rate.toFixed(2)}</td>
        <td>${t.close_rate < 10 ? t.close_rate.toFixed(4) : t.close_rate.toFixed(2)}</td>
        <td>${pill(t.net_ret_pct, t.profit_abs)}</td>
        <td>${t.entry_time.slice(0,10)}</td>
        <td>${t.exit_time.slice(0,10)}</td>
        <td>death_cross</td>
      `;
      frag.appendChild(tr);
    });
    closedBody.appendChild(frag);

    const pag = document.getElementById("closed-trades-pagination");
    pag.innerHTML = "";
    const mkBtn = (label, page, opts) => {
      opts = opts || {};
      const b = document.createElement("button");
      b.textContent = label;
      if (opts.active) b.classList.add("active");
      if (opts.disabled) b.disabled = true;
      b.addEventListener("click", () => { tableState.page = page; renderClosedTrades(); });
      return b;
    };
    pag.appendChild(mkBtn("«", 1, { disabled: tableState.page === 1 }));
    pag.appendChild(mkBtn("‹", Math.max(1, tableState.page - 1), { disabled: tableState.page === 1 }));
    const startPage = Math.max(1, tableState.page - 2);
    const endPage = Math.min(totalPages, startPage + 4);
    for (let p = startPage; p <= endPage; p++) pag.appendChild(mkBtn(String(p), p, { active: p === tableState.page }));
    pag.appendChild(mkBtn("›", Math.min(totalPages, tableState.page + 1), { disabled: tableState.page === totalPages }));
    pag.appendChild(mkBtn("»", totalPages, { disabled: tableState.page === totalPages }));
  }
  document.querySelectorAll("th[data-key]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      tableState.sort = tableState.sort.key === key ? { key, dir: -tableState.sort.dir } : { key, dir: key === "symbol" || key === "asset_class" ? 1 : -1 };
      tableState.page = 1;
      renderClosedTrades();
    });
  });
  document.getElementById("search").addEventListener("input", e => {
    tableState.search = e.target.value.trim().toLowerCase();
    tableState.page = 1;
    renderClosedTrades();
  });
  renderClosedTrades();

  // ---- Trades log ticker (most recent 200) ----
  const tickerEl = document.getElementById("trades-log");
  TRADES.slice(-200).forEach(t => {
    const span = document.createElement("span");
    span.className = "tag " + (t.net_ret_pct >= 0 ? "pos" : "neg");
    span.textContent = fmtPct(t.net_ret_pct);
    span.title = `${t.symbol} ${t.exit_time.slice(0,10)}`;
    tickerEl.appendChild(span);
  });

  // ---- Cumulative profit chart ----
  function setupCanvas(id, cssH) {
    const canvas = document.getElementById(id);
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 700;
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { ctx, cssW, cssH };
  }

  function drawCumulative() {
    const { ctx, cssW, cssH } = setupCanvas("cumulative-canvas", 230);
    const padL = 60, padR = 12, padT = 10, padB = 20;
    const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
    const points = EQUITY_CURVE.map(p => ({ t: new Date(p.time).getTime(), v: p.equity - EQUITY_CURVE[0].equity }));
    const tMin = points[0].t, tMax = points[points.length - 1].t;
    const yMin = Math.min(0, ...points.map(p => p.v)), yMax = Math.max(0, ...points.map(p => p.v));
    const pad = (yMax - yMin) * 0.06 || 1;
    const yLo = yMin - pad, yHi = yMax + pad;
    const xToPx = t => padL + ((t - tMin) / (tMax - tMin || 1)) * plotW;
    const yToPx = v => padT + (1 - (v - yLo) / (yHi - yLo || 1)) * plotH;

    ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.font = "10px ui-monospace, monospace";
    for (let i = 0; i <= 4; i++) {
      const v = yLo + (yHi - yLo) * i / 4;
      const y = yToPx(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cssW - padR, y); ctx.stroke();
      ctx.fillText(fmtAbs(v), 4, y + 3);
    }
    const firstYear = new Date(tMin).getFullYear(), lastYear = new Date(tMax).getFullYear();
    const yearStep = Math.max(1, Math.round((lastYear - firstYear || 1) / 7));
    for (let y = firstYear; y <= lastYear; y += yearStep) {
      const x = xToPx(new Date(Date.UTC(y, 0, 1)).getTime());
      if (x < padL || x > cssW - padR) continue;
      ctx.fillText(String(y), x - 12, cssH - 5);
    }

    ctx.strokeStyle = LINE; ctx.lineWidth = 1.4;
    ctx.beginPath();
    points.forEach((p, i) => { const x = xToPx(p.t), y = yToPx(p.v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
  }

  const dailyProfitState = { period: "days", mode: "abs" };

  function _bucketKey(dateStr, period) {
    const d = new Date(dateStr);
    if (period === "days") return dateStr;
    if (period === "months") return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
    // weeks: ISO-ish week bucket -- Monday of that week
    const day = d.getUTCDay() || 7;
    const monday = new Date(d);
    monday.setUTCDate(d.getUTCDate() - day + 1);
    return monday.toISOString().slice(0, 10);
  }

  function drawDailyProfit() {
    const { ctx, cssW, cssH } = setupCanvas("daily-canvas", 170);
    const padL = 60, padR = 12, padT = 10, padB = 20;
    const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
    const { period, mode } = dailyProfitState;

    const buckets = new Map(); // key -> {sum, count, rets: []}
    TRADES.forEach(t => {
      const key = _bucketKey(t.exit_time.slice(0, 10), period);
      if (!buckets.has(key)) buckets.set(key, { sum: 0, count: 0, retSum: 0 });
      const b = buckets.get(key);
      b.sum += t.profit_abs; b.count += 1; b.retSum += t.net_ret_pct;
    });
    const points = [...buckets.entries()]
      .map(([k, b]) => ({ t: new Date(k).getTime(), v: mode === "abs" ? b.sum : b.retSum / b.count, count: b.count }))
      .sort((a, b) => a.t - b.t);
    if (!points.length) return;
    const tMin = points[0].t, tMax = points[points.length - 1].t;
    const maxAbs = Math.max(...points.map(d => Math.abs(d.v)), mode === "abs" ? 1 : 0.1);
    const xToPx = t => padL + ((t - tMin) / (tMax - tMin || 1)) * plotW;
    const yZero = padT + plotH / 2;
    const fmtY = mode === "abs" ? fmtAbs : (v => (v >= 0 ? "+" : "") + v.toFixed(1) + "%");

    ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.font = "10px ui-monospace, monospace";
    ctx.beginPath(); ctx.moveTo(padL, yZero); ctx.lineTo(cssW - padR, yZero); ctx.stroke();
    ctx.fillText(fmtY(maxAbs), 4, padT + 8);
    ctx.fillText("0", 4, yZero + 3);
    ctx.fillText(fmtY(-maxAbs), 4, cssH - padB);

    const barW = Math.max(1.2, plotW / points.length * 0.7);
    points.forEach(d => {
      const x = xToPx(d.t);
      const h = (Math.abs(d.v) / maxAbs) * (plotH / 2);
      ctx.fillStyle = d.v >= 0 ? BULL : BEAR;
      ctx.fillRect(x, d.v >= 0 ? yZero - h : yZero, barW, h);
    });
  }

  document.querySelectorAll("#daily-period-toggle .seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#daily-period-toggle .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      dailyProfitState.period = btn.dataset.period;
      drawDailyProfit();
    });
  });
  document.querySelectorAll("#daily-mode-toggle .seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#daily-mode-toggle .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      dailyProfitState.mode = btn.dataset.mode;
      drawDailyProfit();
    });
  });

  function drawDistribution() {
    const { ctx, cssW, cssH } = setupCanvas("dist-canvas", 170);
    const padL = 40, padR = 12, padT = 10, padB = 20;
    const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
    const rets = TRADES.map(t => t.net_ret_pct);
    const lo = Math.max(-50, Math.min(...rets)), hi = Math.min(50, Math.max(...rets));
    const bins = 40;
    const binW = (hi - lo) / bins;
    const counts = new Array(bins).fill(0);
    rets.forEach(r => {
      const clamped = Math.max(lo, Math.min(hi - 1e-9, r));
      const idx = Math.floor((clamped - lo) / binW);
      counts[Math.max(0, Math.min(bins - 1, idx))]++;
    });
    const maxCount = Math.max(...counts, 1);
    ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.font = "10px ui-monospace, monospace";
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, cssH - padB); ctx.lineTo(cssW - padR, cssH - padB); ctx.stroke();
    ctx.fillText("0", 4, cssH - padB);
    ctx.fillText(String(maxCount), 4, padT + 8);
    const barW = plotW / bins;
    counts.forEach((c, i) => {
      const h = (c / maxCount) * plotH;
      ctx.fillStyle = HIST;
      ctx.fillRect(padL + i * barW, cssH - padB - h, Math.max(1, barW - 1), h);
    });
    ctx.fillText(lo.toFixed(0) + "%", padL, cssH - 4);
    ctx.fillText(hi.toFixed(0) + "%", cssW - padR - 24, cssH - 4);
  }

  drawCumulative();
  drawDailyProfit();
  drawDistribution();
  window.addEventListener("resize", () => { drawCumulative(); drawDailyProfit(); drawDistribution(); if (document.getElementById("tab-trade").style.display !== "none") drawCandles(); });

  // ==================== Tab navigation ====================
  const TAB_IDS = ["dashboard", "trade", "chart", "logs"];
  function showTab(name) {
    TAB_IDS.forEach(id => {
      document.getElementById("tab-" + id).style.display = id === name ? "" : "none";
    });
    document.querySelectorAll(".navbar nav span").forEach(s => s.classList.toggle("active", s.dataset.tab === name));
    if (name === "trade") drawCandles();
  }
  document.querySelectorAll(".navbar nav span").forEach(s => {
    s.addEventListener("click", () => showTab(s.dataset.tab));
  });

  // ==================== Trade tab: real candlestick pane ====================
  const TRADE_PANE = __TRADE_PANE__;
  const PAIR_LABELS = Object.keys(TRADE_PANE);
  let selectedPair = PAIR_LABELS[0];

  const pairListEl = document.getElementById("pair-list");
  PAIR_LABELS.forEach(label => {
    const div = document.createElement("div");
    div.className = "pair-item" + (label === selectedPair ? " active" : "");
    div.dataset.pair = label;
    div.innerHTML = `<span>${TRADE_PANE[label].pair}</span><span class="n">${TRADE_PANE[label].n_trades}</span>`;
    div.addEventListener("click", () => {
      selectedPair = label;
      document.querySelectorAll(".pair-item").forEach(p => p.classList.toggle("active", p.dataset.pair === label));
      drawCandles();
      renderTradeClosedTable();
    });
    pairListEl.appendChild(div);
  });

  function drawCandles() {
    const asset = TRADE_PANE[selectedPair];
    if (!asset) return;
    document.getElementById("trade-chart-title").textContent = `Chart — ${asset.pair} | 4h`;
    document.getElementById("chart-legend").innerHTML =
      `<span><span class="sw" style="background:${BULL}"></span>Candles</span>` +
      `<span><span class="sw" style="background:#00c3e6"></span>ma50</span>` +
      `<span><span class="sw" style="background:#f5a524"></span>ma200</span>` +
      `<span class="tri">&#9650; Entry</span><span class="dia">&#9670; Exit</span>`;

    const candles = asset.candles;
    const { ctx, cssW, cssH } = setupCanvas("candle-canvas", 340);
    ctx.clearRect(0, 0, cssW, cssH);
    const PADL = 60, PADR = 12, PADT = 10, PADB = 10;
    const plotW = cssW - PADL - PADR, plotH = cssH - PADT - PADB;
    const n = candles.length;
    const slot = plotW / n;

    let yMin = Infinity, yMax = -Infinity;
    candles.forEach(c => { yMin = Math.min(yMin, c.low); yMax = Math.max(yMax, c.high); });
    const pad = (yMax - yMin) * 0.05 || 1;
    yMin -= pad; yMax += pad;
    const yToPx = v => PADT + (1 - (v - yMin) / (yMax - yMin)) * plotH;
    const xForIdx = i => PADL + i * slot + slot / 2;

    ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.font = "10px ui-monospace, monospace";
    for (let i = 0; i <= 4; i++) {
      const v = yMin + (yMax - yMin) * i / 4;
      const y = yToPx(v);
      ctx.beginPath(); ctx.moveTo(PADL, y); ctx.lineTo(cssW - PADR, y); ctx.stroke();
      ctx.fillText(v < 10 ? v.toFixed(4) : v.toFixed(2), 2, y + 3);
    }

    candles.forEach((c, i) => {
      const x = xForIdx(i);
      const up = c.close >= c.open;
      ctx.strokeStyle = ctx.fillStyle = up ? BULL : BEAR;
      ctx.beginPath(); ctx.moveTo(x, yToPx(c.high)); ctx.lineTo(x, yToPx(c.low)); ctx.stroke();
      const bodyTop = yToPx(Math.max(c.open, c.close)), bodyBot = yToPx(Math.min(c.open, c.close));
      ctx.fillRect(x - slot * 0.32, bodyTop, slot * 0.64, Math.max(1, bodyBot - bodyTop));
    });

    ["ma50", "ma200"].forEach(key => {
      ctx.strokeStyle = key === "ma50" ? "#00c3e6" : "#f5a524";
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      let started = false;
      candles.forEach((c, i) => {
        if (c[key] === null) return;
        const x = xForIdx(i), y = yToPx(c[key]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    const firstTime = new Date(candles[0].time).getTime(), lastTime = new Date(candles[n - 1].time).getTime();
    asset.trades.forEach(t => {
      const et = new Date(t.entry_time).getTime(), xt = new Date(t.exit_time).getTime();
      if (et >= firstTime && et <= lastTime) {
        const idx = Math.round((et - firstTime) / (lastTime - firstTime) * (n - 1));
        const x = xForIdx(idx), y = yToPx(t.entry_price);
        ctx.fillStyle = BULL;
        ctx.beginPath(); ctx.moveTo(x, y + 8); ctx.lineTo(x - 5, y + 16); ctx.lineTo(x + 5, y + 16); ctx.closePath(); ctx.fill();
      }
      if (xt >= firstTime && xt <= lastTime) {
        const idx = Math.round((xt - firstTime) / (lastTime - firstTime) * (n - 1));
        const x = xForIdx(idx), y = yToPx(t.exit_price);
        ctx.fillStyle = BEAR;
        ctx.beginPath(); ctx.moveTo(x, y - 12); ctx.lineTo(x - 5, y - 6); ctx.lineTo(x, y); ctx.lineTo(x + 5, y - 6); ctx.closePath(); ctx.fill();
      }
    });

    // volume subplot
    const vol = setupCanvas("volume-canvas", 70);
    vol.ctx.clearRect(0, 0, vol.cssW, vol.cssH);
    const maxVol = Math.max(...candles.map(c => c.volume), 1);
    candles.forEach((c, i) => {
      const x = PADL + i * slot;
      const h = (c.volume / maxVol) * (vol.cssH - 8);
      vol.ctx.fillStyle = MUTED;
      vol.ctx.fillRect(x, vol.cssH - h, Math.max(1, slot * 0.8), h);
    });
  }

  function renderTradeClosedTable() {
    const asset = TRADE_PANE[selectedPair];
    document.getElementById("trade-closed-title").innerHTML = `Closed Trades <span style="color:var(--muted);font-weight:400;">(${asset.trades.length} real historical trades)</span>`;
    const body = document.getElementById("trade-closed-body");
    body.innerHTML = "";
    [...asset.trades].reverse().forEach((t, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${asset.trades.length - i}</td>
        <td>${asset.pair}</td>
        <td>${t.entry_price < 10 ? t.entry_price.toFixed(4) : t.entry_price.toFixed(2)}</td>
        <td>${t.exit_price < 10 ? t.exit_price.toFixed(4) : t.exit_price.toFixed(2)}</td>
        <td>${pill(t.net_ret_pct)}</td>
        <td>${t.entry_time.slice(0,10)}</td>
        <td>${t.exit_time.slice(0,10)}</td>
      `;
      body.appendChild(tr);
    });
  }
  renderTradeClosedTable();

  // ==================== Chart tab: instruments table ====================
  const chartState = { sort: { key: "sum_excl3", dir: -1 }, search: "" };
  function renderChartTab() {
    const { key, dir } = chartState.sort;
    let rows = RESULTS;
    if (chartState.search) rows = rows.filter(r => r.symbol.toLowerCase().includes(chartState.search));
    rows = [...rows].sort((a, b) => {
      let av = a[key], bv = b[key];
      if (key === "symbol" || key === "asset_class") return dir * String(av).localeCompare(String(bv));
      if (key === "robust") return dir * ((a.robust === b.robust) ? 0 : (a.robust ? 1 : -1));
      av = av === null || av === undefined ? -Infinity : av;
      bv = bv === null || bv === undefined ? -Infinity : bv;
      return dir * (av - bv);
    });
    document.querySelectorAll("th[data-ckey]").forEach(th => th.classList.toggle("sorted", th.dataset.ckey === key));
    const body = document.getElementById("chart-tab-body");
    body.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.symbol}</td><td>${r.asset_class}</td>
        <td><span class="pill ${r.robust ? 'pos' : 'neg'}">${r.robust ? 'ROBUST' : 'no'}</span></td>
        <td>${r.n_trades}</td><td>${r.win_rate.toFixed(1)}%</td>
        <td>${r.pf === null ? "&infin;" : r.pf.toFixed(2)}</td>
        <td>${r.pf_excl3 === null ? "&infin;" : r.pf_excl3.toFixed(2)}</td>
        <td>${fmtPct(r.sum_excl3)}</td>
      `;
      body.appendChild(tr);
    });
  }
  document.querySelectorAll("th[data-ckey]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.ckey;
      chartState.sort = chartState.sort.key === key ? { key, dir: -chartState.sort.dir } : { key, dir: key === "symbol" || key === "asset_class" ? 1 : -1 };
      renderChartTab();
    });
  });
  document.getElementById("chart-search").addEventListener("input", e => { chartState.search = e.target.value.trim().toLowerCase(); renderChartTab(); });
  renderChartTab();

  // ==================== Logs tab ====================
  document.getElementById("logs-content").textContent = __LOG_TEXT__;
</script>
"""

if __name__ == "__main__":
    main()
