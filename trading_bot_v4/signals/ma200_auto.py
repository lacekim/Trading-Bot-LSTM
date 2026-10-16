"""Persistent 200MA paper-trading process -- runs continuously in one Python
process until interrupted, matching the architecture of the existing
research/scheduler.py --auto scheduler (a real long-running process with its
own internal scheduling and signal handling) instead of an external shell
script re-invoking the CLI repeatedly.
"""
from __future__ import annotations

import signal
import threading
import time
from datetime import datetime, timedelta, timezone

from trading_bot_v4.utils.logger import build_logger
from tools.run_ma200_paper_trader import main as run_check_cycle
from trading_bot_v4.signals.ma200_position_monitor import (
    run_check as run_position_monitor_check,
    CHECK_INTERVAL_SECONDS as POSITION_MONITOR_INTERVAL_SECONDS,
)

logger = build_logger("v4_ma200_auto")

CHECK_INTERVAL_HOURS = 4
BUFFER_SECONDS = 90  # give Kraken's API a moment to have the new candle ready

_stop_requested = False


def _handle_signal(signum, _frame) -> None:
    global _stop_requested
    logger.info(f"received signal {signum}, will stop after the current cycle")
    _stop_requested = True


def _position_monitor_loop() -> None:
    """Runs on its own, much faster cadence than the 4h signal loop --
    watches open positions for a real adverse intra-bar move and alerts,
    without ever touching the signal or closing anything itself."""
    logger.info(f"position monitor started, checking every {POSITION_MONITOR_INTERVAL_SECONDS}s")
    while not _stop_requested:
        try:
            run_position_monitor_check()
        except Exception as exc:
            logger.error(f"position monitor check failed: {exc}")
        slept = 0.0
        while slept < POSITION_MONITOR_INTERVAL_SECONDS and not _stop_requested:
            interval = min(5.0, POSITION_MONITOR_INTERVAL_SECONDS - slept)
            time.sleep(interval)
            slept += interval
    logger.info("position monitor stopped")


def _seconds_until_next_boundary(now: datetime) -> float:
    hours_since_epoch_cycle = now.hour % CHECK_INTERVAL_HOURS
    seconds_into_cycle = hours_since_epoch_cycle * 3600 + now.minute * 60 + now.second
    return (CHECK_INTERVAL_HOURS * 3600 - seconds_into_cycle) + BUFFER_SECONDS


def run() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    logger.info("V5 MA200 paper-trading process started (dry-run only, no real orders)")

    monitor_thread = threading.Thread(target=_position_monitor_loop, daemon=True)
    monitor_thread.start()

    while not _stop_requested:
        try:
            run_check_cycle()
        except Exception as exc:
            logger.error(f"check cycle failed: {exc}")

        if _stop_requested:
            break

        sleep_seconds = _seconds_until_next_boundary(datetime.now(timezone.utc))
        next_check = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
        logger.info(f"sleeping {sleep_seconds:.0f}s until next check at {next_check.isoformat()}")

        slept = 0.0
        while slept < sleep_seconds and not _stop_requested:
            interval = min(5.0, sleep_seconds - slept)
            time.sleep(interval)
            slept += interval

    logger.info("V5 MA200 paper-trading process stopped cleanly")


if __name__ == "__main__":
    run()
