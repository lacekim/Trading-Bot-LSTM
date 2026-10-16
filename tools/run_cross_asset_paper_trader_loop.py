"""Persistent loop wrapper for run_cross_asset_paper_trader.py -- these are
daily bars (unlike the crypto system's 4h), so a new golden/death cross can
only appear once a real trading day closes. Checks every CHECK_INTERVAL_HOURS
rather than continuously polling, gentle on Yahoo Finance and pointless to
run more often than the underlying data can change."""
from __future__ import annotations

import runpy
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CHECK_INTERVAL_HOURS = 4
LOG_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_paper_trader_loop.log"

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    log("cross-asset paper trader loop starting")
    while not _stop:
        try:
            runpy.run_path(str(Path(__file__).parent / "run_cross_asset_paper_trader.py"), run_name="__main__")
        except SystemExit:
            pass
        except Exception:
            log("run failed:\n" + traceback.format_exc())
        if _stop:
            break
        log(f"sleeping {CHECK_INTERVAL_HOURS}h until next check")
        for _ in range(CHECK_INTERVAL_HOURS * 60 * 60):
            if _stop:
                break
            time.sleep(1)
    log("stopped")


if __name__ == "__main__":
    main()
