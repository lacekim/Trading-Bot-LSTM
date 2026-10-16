"""Persistent loop wrapper for run_alpaca_paper_trader.py -- same
CHECK_INTERVAL_HOURS reasoning as run_cross_asset_paper_trader_loop.py:
these are daily bars, a new golden/death cross can only appear once a real
trading day closes, so there's nothing to gain from polling more often."""
from __future__ import annotations

import runpy
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CHECK_INTERVAL_HOURS = 4
LOG_PATH = Path(__file__).resolve().parents[1] / "runtime" / "alpaca_paper_trader_loop.log"

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
    log("alpaca paper trader loop starting")
    while not _stop:
        try:
            runpy.run_path(str(Path(__file__).parent / "run_alpaca_paper_trader.py"), run_name="__main__")
        except SystemExit as e:
            log(f"run exited: {e}")
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
