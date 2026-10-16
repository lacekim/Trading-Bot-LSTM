"""Persistent loop wrapper for update_regime_state.py. Regime inputs are
daily bars, so hourly is more than sufficient and gentle on Yahoo Finance --
matches the cadence reasoning in run_cross_asset_paper_trader_loop.py."""
from __future__ import annotations

import runpy
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CHECK_INTERVAL_HOURS = 1
LOG_PATH = Path(__file__).resolve().parents[1] / "runtime" / "regime_state_update_loop.log"

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
    log("regime state update loop starting")
    while not _stop:
        try:
            runpy.run_path(str(Path(__file__).parent / "update_regime_state.py"), run_name="__main__")
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
