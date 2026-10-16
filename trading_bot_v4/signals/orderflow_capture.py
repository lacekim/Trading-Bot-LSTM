"""Captures real order-book context around real golden-cross/death-cross
events as they happen, so a genuine joint dataset (cross event x order flow)
accumulates over time.

This does NOT feed back into any trading decision -- the golden-cross rule
stays exactly as backtested and verified. It only logs what the order book
looked like at the moment of each real cross event, so that once enough
events have accumulated, it becomes possible to honestly test whether
order-flow imbalance around a cross event says anything about how that trade
plays out (there wasn't enough joint history to test this as of 2026-09-01 --
the WS order-book collector only started 2026-08-31, and the most recent real
cross events predate that). Until there's a real sample to test, this is
pure data collection, not a signal.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot_v4.utils.logger import build_logger

logger = build_logger("v4_orderflow_capture")

WS_CSV_PATH = Path(__file__).resolve().parents[2] / "market_data_collectors" / "data" / "kraken_orderbook_ws_snapshots.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "runtime" / "orderflow_cross_events.jsonl"
LOOKBACK_MINUTES = 15
# Sized against BTC's real observed rate (~69 updates/sec during active resync
# churn) so the 15-minute window isn't silently truncated -- verified via
# actual_window_seconds in the captured record, not assumed.
MAX_GREP_LINES = 80_000

_LABEL_TO_WS_SYMBOL = {
    "BTC": "BTC/USD", "ETH": "ETH/USD", "XMR": "XMR/USD",
    "PAXG": "PAXG/USD", "SOL": "SOL/USD", "TRX": "TRX/USD",
}


def _recent_snapshots(ws_symbol: str) -> list[dict]:
    if not WS_CSV_PATH.exists():
        return []
    try:
        grep = subprocess.run(
            ["grep", f",{ws_symbol},", str(WS_CSV_PATH)],
            stdout=subprocess.PIPE, text=True, timeout=30,
        )
    except Exception as exc:
        logger.error(f"grep on order-book CSV failed: {exc}")
        return []
    lines = grep.stdout.splitlines()[-MAX_GREP_LINES:]
    if not lines:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
    rows = []
    for row in csv.reader(lines):
        try:
            ts = datetime.fromisoformat(row[0])
            if ts < cutoff:
                continue
            rows.append({
                "snapshot_ts": row[0], "best_bid": float(row[2]), "best_bid_size": float(row[3]),
                "best_ask": float(row[4]), "best_ask_size": float(row[5]),
                "spread_bps": float(row[7]), "imbalance": float(row[8]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def capture_cross_event_context(label: str, event_type: str, event_price: float, event_time: str) -> None:
    """event_type: 'golden_cross_up' or 'death_cross'. Appends one JSONL
    record -- either real order-flow context if the WS collector had enough
    recent history for this symbol, or an explicit null with a reason if not
    (never fabricates a summary from insufficient data)."""
    ws_symbol = _LABEL_TO_WS_SYMBOL.get(label)
    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label, "event_type": event_type,
        "event_price": event_price, "event_time": event_time,
        "orderflow": None,
    }
    if ws_symbol is None:
        record["orderflow_unavailable_reason"] = f"no WS symbol mapping for {label}"
    else:
        rows = _recent_snapshots(ws_symbol)
        if not rows:
            record["orderflow_unavailable_reason"] = f"no order-book snapshots in the last {LOOKBACK_MINUTES}m for {ws_symbol}"
        else:
            imbalances = [r["imbalance"] for r in rows]
            spreads = [r["spread_bps"] for r in rows]
            # Report the ACTUAL time span the retained rows cover, not the
            # configured target -- MAX_GREP_LINES caps how far back the tail
            # reaches, and on high-update-rate symbols (BTC in particular,
            # inflated further by resync churn) that cap can be hit well
            # before LOOKBACK_MINUTES, so the true window is shorter than
            # requested. Claiming the configured value regardless would be
            # exactly the kind of misleading-but-technically-labeled result
            # this project has repeatedly had to catch and correct.
            first_ts = datetime.fromisoformat(rows[0]["snapshot_ts"])
            last_ts = datetime.fromisoformat(rows[-1]["snapshot_ts"])
            actual_window_seconds = (last_ts - first_ts).total_seconds()
            record["orderflow"] = {
                "n_snapshots": len(rows),
                "requested_window_minutes": LOOKBACK_MINUTES,
                "actual_window_seconds": actual_window_seconds,
                "capped_by_max_lines": len(rows) >= MAX_GREP_LINES,
                "mean_imbalance": sum(imbalances) / len(imbalances),
                "last_imbalance": rows[-1]["imbalance"],
                "mean_spread_bps": sum(spreads) / len(spreads),
                "last_best_bid": rows[-1]["best_bid"],
                "last_best_ask": rows[-1]["best_ask"],
            }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"orderflow context captured for {label} {event_type}: "
                f"{'real data' if record['orderflow'] else record.get('orderflow_unavailable_reason')}")
