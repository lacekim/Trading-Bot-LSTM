"""Automatic paper-only scheduler for V4 research tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
import time
import traceback

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.execution.smc_model_paper import run_smc_model_paper_trading
from trading_bot_v4.research.daily_research import _safe_top_validated_symbols, _update_smc_features, run_daily_research


SCHEDULER_LOG_PATH = Path("logs/v4_scheduler.log")
DAILY_RESEARCH_HOUR = 0
DAILY_RESEARCH_MINUTE = 5


@dataclass(frozen=True)
class SchedulerState:
    next_hourly_update: datetime
    next_daily_research: datetime


def _scheduler_args(**kwargs: Any) -> Any:
    defaults = {
        "timeframe": Config.TIMEFRAME,
        "all_assets": False,
        "validated_whitelist": False,
        "symbol": Config.GMX_SYMBOL,
        "top_validated": 10,
        "capital": 100000.0,
    }
    defaults.update(kwargs)
    return type("Args", (), defaults)()


def _log(message: str) -> None:
    SCHEDULER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with SCHEDULER_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def _run_guarded(name: str, task: Callable[[], Any]) -> Any | None:
    _log(f"START {name}")
    try:
        result = task()
    except Exception as exc:  # pragma: no cover - long-running scheduler resilience
        _log(f"ERROR {name}: {exc}")
        _log(traceback.format_exc().rstrip())
        return None
    _log(f"DONE {name}")
    return result


def _next_hour_boundary(now: datetime) -> datetime:
    return (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))


def _next_daily_time(now: datetime) -> datetime:
    candidate = now.replace(
        hour=DAILY_RESEARCH_HOUR,
        minute=DAILY_RESEARCH_MINUTE,
        second=0,
        microsecond=0,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _run_hourly_update(timeframe: str, top_validated: int) -> None:
    handler = V4DataHandler()

    def refresh_market_data() -> bool:
        return handler.refresh_gmx_cache(force=True)

    refresh_result = _run_guarded("hourly.refresh_market_data", refresh_market_data)
    if refresh_result is False:
        _log("WARNING hourly.refresh_market_data returned False; continuing with cached data")

    def update_smc_features() -> int:
        symbols = _safe_top_validated_symbols(timeframe, top_validated)
        outputs = _update_smc_features(symbols, timeframe)
        return len(outputs)

    _run_guarded("hourly.update_smc_features", update_smc_features)

    def update_all_asset_paper_signals() -> dict[str, Any]:
        return run_smc_model_paper_trading(_scheduler_args(timeframe=timeframe, all_assets=True))

    _run_guarded("hourly.update_smc_model_paper_signals", update_all_asset_paper_signals)

    def update_validated_whitelist_paper_signals() -> dict[str, Any]:
        return run_smc_model_paper_trading(_scheduler_args(timeframe=timeframe, validated_whitelist=True))

    _run_guarded("hourly.update_validated_whitelist_paper_signals", update_validated_whitelist_paper_signals)


def _run_daily_research(timeframe: str, top_validated: int) -> None:
    _run_guarded(
        "daily.run_daily_research",
        lambda: run_daily_research(_scheduler_args(timeframe=timeframe, top_validated=top_validated)),
    )


def run_auto_scheduler(args: Any) -> None:
    """Run the paper-only V4 scheduler until interrupted."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    top_validated = int(getattr(args, "top_validated", 10) or 10)
    now = datetime.now()
    state = SchedulerState(
        next_hourly_update=now,
        next_daily_research=_next_daily_time(now),
    )

    print("Scheduler started.")
    print(f"Next hourly update: {state.next_hourly_update.isoformat(timespec='seconds')}")
    print(f"Next daily research: {state.next_daily_research.isoformat(timespec='seconds')}")
    print("No live trading.")
    _log("Scheduler started")
    _log(f"Next hourly update: {state.next_hourly_update.isoformat(timespec='seconds')}")
    _log(f"Next daily research: {state.next_daily_research.isoformat(timespec='seconds')}")
    _log("No live trading")

    next_hourly = state.next_hourly_update
    next_daily = state.next_daily_research
    while True:
        now = datetime.now()
        if now >= next_hourly:
            _run_hourly_update(timeframe, top_validated)
            next_hourly = _next_hour_boundary(datetime.now())
            _log(f"Next hourly update: {next_hourly.isoformat(timespec='seconds')}")

        now = datetime.now()
        if now >= next_daily:
            _run_daily_research(timeframe, top_validated)
            next_daily = _next_daily_time(datetime.now())
            _log(f"Next daily research: {next_daily.isoformat(timespec='seconds')}")

        time.sleep(30)
