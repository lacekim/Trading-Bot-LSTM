"""Automatic paper-only scheduler for V4 research tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
import json
import subprocess
import sys
import time
import traceback

import pandas as pd

from trading_bot import load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.smc_model_paper import run_smc_model_paper_trading
from trading_bot_v4.execution.order_manager import OrderManager, PaperCycleSummary
from trading_bot_v4.execution.web3_readonly import check_web3_readiness
from trading_bot_v4.ml.smc_trainer import SMC_MODEL_PATH, SMC_SCALER_PATH
from trading_bot_v4.research.daily_research import DAILY_GO_STATUS_PATH, _update_smc_features, run_daily_research
from trading_bot_v4.utils.model_cache import ModelScalerCache


SCHEDULER_LOG_PATH = Path("logs/v4_scheduler.log")
RELOAD_MODEL_REQUEST_PATH = Path("logs/v4_reload_model.request")
DAILY_RESEARCH_HOUR = 5
DAILY_RESEARCH_MINUTE = 0


@dataclass(frozen=True)
class SchedulerState:
    next_hourly_update: datetime
    next_daily_research: datetime


@dataclass
class SchedulerModelBundle:
    original_model: Any
    original_scaler: Any
    smc_model: Any
    smc_scaler: Any
    original_model_mtime: float
    original_scaler_mtime: float
    smc_model_mtime: float
    smc_scaler_mtime: float


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


def request_model_reload() -> Path:
    """Request the running scheduler to reload model artifacts on its next cycle."""
    RELOAD_MODEL_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELOAD_MODEL_REQUEST_PATH.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    return RELOAD_MODEL_REQUEST_PATH


def _mtime(path: Path) -> float:
    return float(path.stat().st_mtime) if path.exists() else 0.0


def _load_scheduler_models(reason: str) -> SchedulerModelBundle:
    _log(f"Loading scheduler models: {reason}")
    original_cache = ModelScalerCache()
    original_model, original_scaler = original_cache.load()
    smc_cache = ModelScalerCache(model_path=SMC_MODEL_PATH, scaler_path=SMC_SCALER_PATH)
    smc_model, smc_scaler = smc_cache.load()
    _log("Scheduler models loaded")
    return SchedulerModelBundle(
        original_model=original_model,
        original_scaler=original_scaler,
        smc_model=smc_model,
        smc_scaler=smc_scaler,
        original_model_mtime=_mtime(Path(original_cache.model_path)),
        original_scaler_mtime=_mtime(Path(original_cache.scaler_path)),
        smc_model_mtime=_mtime(Path(SMC_MODEL_PATH)),
        smc_scaler_mtime=_mtime(Path(SMC_SCALER_PATH)),
    )


def _reload_reasons(bundle: SchedulerModelBundle) -> list[str]:
    reasons: list[str] = []
    original_model_path = Path(Config.MODEL_DIR / Config.MODEL_NAME)
    original_scaler_path = Path(Config.MODEL_DIR / Config.SCALER_NAME)
    checks = [
        ("original model timestamp changed", original_model_path, bundle.original_model_mtime),
        ("original scaler timestamp changed", original_scaler_path, bundle.original_scaler_mtime),
        ("SMC model timestamp changed", Path(SMC_MODEL_PATH), bundle.smc_model_mtime),
        ("SMC scaler timestamp changed", Path(SMC_SCALER_PATH), bundle.smc_scaler_mtime),
    ]
    for reason, path, previous_mtime in checks:
        current_mtime = _mtime(path)
        if current_mtime and previous_mtime and current_mtime != previous_mtime:
            reasons.append(reason)
    if RELOAD_MODEL_REQUEST_PATH.exists():
        reasons.append(f"reload requested via {RELOAD_MODEL_REQUEST_PATH}")
    return reasons


def _maybe_reload_models(bundle: SchedulerModelBundle) -> SchedulerModelBundle:
    reasons = _reload_reasons(bundle)
    if not reasons:
        return bundle
    reason_text = "; ".join(reasons)
    _log(f"Reloading scheduler models: {reason_text}")
    if RELOAD_MODEL_REQUEST_PATH.exists():
        try:
            RELOAD_MODEL_REQUEST_PATH.unlink()
        except OSError as exc:
            _log(f"WARNING failed to remove reload request: {exc}")
    return _load_scheduler_models(reason_text)


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


def _hourly_refresh_symbols(timeframe: str) -> list[str]:
    del timeframe  # The daily status is produced for the scheduler's active timeframe.
    if not DAILY_GO_STATUS_PATH.exists():
        return []
    try:
        status = pd.read_csv(DAILY_GO_STATUS_PATH)
    except Exception as exc:
        _log(f"WARNING failed to load daily GO list: {exc}")
        return []
    if not {"symbol", "decision"}.issubset(status.columns):
        _log(f"WARNING daily GO list has invalid columns: {DAILY_GO_STATUS_PATH}")
        return []
    selected = status.loc[
        status["decision"].astype(str).str.upper().eq("GO"),
        "symbol",
    ]
    return list(dict.fromkeys(selected.astype(str).str.upper().tolist()))


def _refresh_live_market_data(symbols: list[str], timeframe: str) -> bool:
    """Refresh active GMX symbols from source using a temporary symbol-scoped config."""
    if getattr(Config, "DATA_SOURCE", "").upper() != "GMX":
        return True
    if not getattr(Config, "GMX_AUTO_REFRESH_ENABLED", True):
        _log("hourly.live_market_data disabled by GMX_AUTO_REFRESH_ENABLED")
        return False

    update_script = Path(getattr(Config, "GMX_UPDATE_SCRIPT"))
    update_config = Path(getattr(Config, "GMX_UPDATE_CONFIG"))
    if not update_script.exists():
        _log(f"WARNING hourly.live_market_data update script not found: {update_script}")
        return False
    if not update_config.exists():
        _log(f"WARNING hourly.live_market_data update config not found: {update_config}")
        return False

    active_config_path = SCHEDULER_LOG_PATH.parent / "v4_hourly_gmx_config.json"
    try:
        with update_config.open("r", encoding="utf-8") as handle:
            active_config = json.load(handle)
    except Exception as exc:
        _log(f"WARNING hourly.live_market_data failed to read base config: {exc}")
        return False

    active_config["gmx_token_symbol"] = "ALL"
    active_config["gmx_token_symbols"] = symbols
    active_config["gmx_primary_symbol"] = symbols[0] if symbols else active_config.get("gmx_primary_symbol", "BTC")
    active_config_path.parent.mkdir(parents=True, exist_ok=True)
    active_config_path.write_text(json.dumps(active_config, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(update_script),
        "--config",
        str(active_config_path.resolve()),
        "--period",
        timeframe,
        "--chain",
        getattr(Config, "GMX_UPDATE_CHAIN", "arbitrum"),
    ]
    _log(f"hourly.live_market_data refreshing source symbols: {', '.join(symbols)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=update_script.parent,
            text=True,
            capture_output=True,
            timeout=int(getattr(Config, "GMX_UPDATE_TIMEOUT_SECONDS", 900)),
        )
    except Exception as exc:
        _log(f"WARNING hourly.live_market_data failed; using cached OHLC fallback: {exc}")
        return False

    if result.stdout:
        _log(result.stdout.strip())
    if result.stderr:
        _log(result.stderr.strip())
    if result.returncode != 0:
        _log(f"WARNING hourly.live_market_data exited {result.returncode}; using cached OHLC fallback")
        return False
    return True


def _refresh_active_market_data(symbols: list[str], timeframe: str) -> int:
    live_refreshed = _refresh_live_market_data(symbols, timeframe)
    if not live_refreshed:
        _log("hourly.refresh_active_market_data using cached OHLC backup")

    refreshed = 0
    for symbol in symbols:
        data = load_gmx_ohlc(symbol, timeframe)
        if data is None or data.empty:
            raise ValueError(f"No cached OHLC rows available for {symbol} {timeframe}")
        refreshed += 1
    return refreshed


def _format_paper_summary(summary: PaperCycleSummary) -> str:
    return (
        "Hourly Summary | "
        f"Assets scanned: {summary.assets_scanned} | Signals generated: {summary.signals_generated} | "
        f"Signals rejected: {summary.signals_rejected} | Paper orders opened: {summary.orders_opened} | "
        f"Paper orders closed: {summary.orders_closed} | Open positions: {summary.open_positions} | "
        f"Equity: ${summary.equity:,.2f} | Realized P&L: ${summary.realized_pnl:,.2f} | "
        f"Unrealized P&L: ${summary.unrealized_pnl:,.2f} | Fees: ${summary.fees:,.2f}"
    )


def _run_hourly_update(timeframe: str, models: SchedulerModelBundle, orders: OrderManager) -> None:
    symbols = _hourly_refresh_symbols(timeframe)
    if not symbols:
        _log("Hourly paper cycle skipped: today's qualified GO list is empty")
        _log(_format_paper_summary(orders.sync()))
        return
    _log(f"Hourly qualified assets: {', '.join(symbols)}")

    _run_guarded(
        "hourly.refresh_active_market_data",
        lambda: _refresh_active_market_data(symbols, timeframe),
    )

    def update_smc_features() -> int:
        outputs = _update_smc_features(symbols, timeframe)
        return len(outputs)

    _run_guarded("hourly.update_smc_features", update_smc_features)

    def update_active_paper_signals() -> dict[str, Any]:
        return run_smc_model_paper_trading(
            _scheduler_args(timeframe=timeframe, symbols=symbols, model=models.smc_model, scaler=models.smc_scaler)
        )

    signal_result = _run_guarded("hourly.update_active_smc_model_paper_signals", update_active_paper_signals)
    if signal_result is None:
        _log(_format_paper_summary(orders.sync()))
        return
    signals_path = Path(signal_result["signals_path"])
    signals = pd.read_csv(signals_path) if signals_path.exists() else pd.DataFrame()
    summary = _run_guarded("hourly.paper_execution", lambda: orders.process_signals(signals))
    if summary is not None:
        _log(_format_paper_summary(summary))
        print(_format_paper_summary(summary))


def _run_daily_research(timeframe: str, top_validated: int, models: SchedulerModelBundle) -> None:
    _run_guarded(
        "daily.run_daily_research",
        lambda: run_daily_research(
            _scheduler_args(
                timeframe=timeframe,
                top_validated=top_validated,
                model=models.original_model,
                scaler=models.original_scaler,
                smc_model=models.smc_model,
                smc_scaler=models.smc_scaler,
            )
        ),
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
    displayed_next_hourly = _next_hour_boundary(now)
    models = _load_scheduler_models("scheduler startup")
    orders = OrderManager()
    hourly_symbols = _hourly_refresh_symbols(timeframe)

    print("V5 Scheduler started.")
    print(f"Execution mode: {Config.EXECUTION_MODE}")
    print("Model loaded.")
    print("Scaler loaded.")
    print(f"Today's qualified assets: {', '.join(hourly_symbols) if hourly_symbols else 'none'}")
    print("Daily research refresh: all assets")
    print(f"Next hourly update: {displayed_next_hourly.isoformat(timespec='seconds')}")
    print(f"Next daily research: {state.next_daily_research.isoformat(timespec='seconds')}")
    print("Live signing: disabled.")
    web3_status = check_web3_readiness()
    print(f"Web3 read-only: {'connected' if web3_status.connected else web3_status.error}")
    _log("Scheduler started")
    _log(f"Today's qualified assets: {', '.join(hourly_symbols) if hourly_symbols else 'none'}")
    _log("Daily research refresh: all assets")
    _log(f"Next hourly update: {displayed_next_hourly.isoformat(timespec='seconds')}")
    _log(f"Next daily research: {state.next_daily_research.isoformat(timespec='seconds')}")
    _log("Live signing disabled")
    _log(f"Web3 read-only status: {web3_status.to_dict()}")

    next_hourly = state.next_hourly_update
    next_daily = state.next_daily_research
    try:
        while True:
            now = datetime.now()
            if now >= next_hourly:
                models = _maybe_reload_models(models)
                _run_hourly_update(timeframe, models, orders)
                next_hourly = _next_hour_boundary(datetime.now())
                _log(f"Next hourly update: {next_hourly.isoformat(timespec='seconds')}")

            now = datetime.now()
            if now >= next_daily:
                models = _maybe_reload_models(models)
                _run_daily_research(timeframe, top_validated, models)
                next_daily = _next_daily_time(datetime.now())
                _log(f"Next daily research: {next_daily.isoformat(timespec='seconds')}")

            time.sleep(30)
    except KeyboardInterrupt:
        _log("Scheduler stopped by user.")
        print("Scheduler stopped.")
    finally:
        orders.close()
