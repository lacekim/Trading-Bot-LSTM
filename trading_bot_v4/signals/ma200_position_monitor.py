"""Fast, separate safety monitor for open MA200/golden-cross paper positions.

The golden-cross signal itself is bar-close-based and cannot update faster
than every 4h (that's what was backtested and verified) -- but that's no
reason to be blind to what happens to an open position in between checks.
This polls current price every CHECK_INTERVAL_SECONDS (much faster than the
4h signal cadence, mirroring the real bot's PositionMonitor pattern) and
alerts on a real adverse move since entry. It never closes a position or
changes the signal logic -- informational only, so the tested exit rule
stays exactly as verified.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.signals.ma200_asset_qualifier import PAIR_OVERRIDE
from trading_bot_v4.signals.seasonal_risk import seasonal_context_note
from trading_bot_v4.telegram.notifier import V4TelegramNotifier

logger = build_logger("v4_ma200_position_monitor")

STATE_PATH = Path(__file__).resolve().parents[2] / "runtime" / "ma200_paper_trades.json"
ALERT_STATE_PATH = Path(__file__).resolve().parents[2] / "runtime" / "ma200_position_monitor_alerts.json"
CHECK_INTERVAL_SECONDS = 120  # much faster than the 4h signal cadence, gentle on Kraken's public API
ADVERSE_THRESHOLDS_PCT = [-8.0, -15.0, -25.0]  # alert once per tier per open position


RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [1, 2, 4]  # short -- this monitor is informational-only,
# a slow retry loop shouldn't delay the next position's check


def fetch_current_price(pair: str) -> float | None:
    """SSL handshake timeouts / DNS blips against Kraken's public API are
    real but self-recovering (observed intermittently, always cleared by
    the next poll) -- retrying a couple of times before logging quiets the
    log for exactly the transient case, while still logging loudly if a
    failure persists across all attempts (a real, not transient, problem)."""
    url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
    last_exc: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            key = list(data["result"].keys())[0]
            return float(data["result"][key]["c"][0])
        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    logger.error(f"price fetch failed for {pair} after {RETRY_ATTEMPTS} attempts: {last_exc}")
    return None


def load_positions() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8")).get("positions", {})


def load_alert_state() -> dict:
    if ALERT_STATE_PATH.exists():
        return json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_alert_state(state: dict) -> None:
    ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def notify(text: str) -> None:
    if not getattr(Config, "TELEGRAM_ENABLED", False) or not getattr(Config, "TELEGRAM_BOT_TOKEN", None):
        logger.info(f"[telegram disabled] {text}")
        return
    try:
        V4TelegramNotifier().send_message(text)
    except Exception as exc:
        logger.error(f"telegram send failed: {exc}")


def run_check() -> None:
    positions = load_positions()
    alert_state = load_alert_state()

    for label in list(alert_state.keys()):
        if label not in positions:
            del alert_state[label]  # position closed since -- reset for next time it opens

    for label, pos in positions.items():
        pair = PAIR_OVERRIDE.get(label, f"{label}USD")
        price = fetch_current_price(pair)
        if price is None:
            continue
        entry_price = pos["entry_price"]
        move_pct = (price / entry_price - 1) * 100

        already_alerted = alert_state.get(label, [])
        for threshold in ADVERSE_THRESHOLDS_PCT:
            if move_pct <= threshold and threshold not in already_alerted:
                text = (
                    f"🚨 {label} down {move_pct:+.1f}% intra-bar since paper entry (${entry_price:,.4f})\n"
                    f"Current: ${price:,.4f}\n"
                    f"Signal itself won't update until the next 4h close -- this is a safety alert only, "
                    f"not an exit (informational, no action taken)."
                )
                season_note = seasonal_context_note(datetime.now(timezone.utc), move_pct)
                if season_note:
                    text += f"\n\n{season_note}"
                notify(text)
                already_alerted.append(threshold)
        alert_state[label] = already_alerted

    save_alert_state(alert_state)
