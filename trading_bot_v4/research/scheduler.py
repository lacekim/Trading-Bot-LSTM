"""Automatic paper-only scheduler for V4 research tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import signal

import pandas as pd

from trading_bot import load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.smc_model_paper import run_smc_model_paper_trading
from trading_bot_v4.execution.bearish_model_paper import (
    combine_directional_signals, load_bearish_calibration, predict_bearish_signals,
)
from trading_bot_v4.execution.paper_model_comparison import run_original_baseline_paper_signals
from trading_bot_v4.execution.order_manager import OrderManager, PaperCycleSummary
from trading_bot_v4.execution.web3_readonly import check_web3_readiness
from trading_bot_v4.execution.position_monitor import PositionMonitor
from trading_bot_v4.execution.qualified_market_monitor import QualifiedMarketMonitor
from trading_bot_v4.features.gmx_feature_collection import collect_hourly_gmx_features
from trading_bot_v4.risk.market_cap_tiers import collect_market_caps
from trading_bot_v4.execution.shutdown import ShutdownCoordinator
from trading_bot_v4.shutdown_controller import ShutdownController, ShutdownMode, graceful_signal_handler
from trading_bot_v4.runtime_control import ControlServer, InstanceLock
from trading_bot_v4.telegram.control_listener import TelegramControlListener
from trading_bot_v4.ml.smc_trainer import SMC_MODEL_PATH, SMC_SCALER_PATH
from trading_bot_v4.ml.smc_trainer import train_smc_model
from trading_bot_v4.ml.bearish_trainer import (
    BEARISH_CALIBRATION_PATH, BEARISH_MODEL_PATH, BEARISH_SCALER_PATH,
    BEARISH_VALIDATION_PATH, BEARISH_WALK_FORWARD_PATH, train_bearish_model,
)
from trading_bot_v4.ml.trainer import train_v4_model
from trading_bot_v4.features.smc_feature_builder import build_all_assets_smc_training_data
from trading_bot_v4.backtesting.walk_forward import (
    WALK_FORWARD_REPORT_PATH, WALK_FORWARD_SUMMARY_PATH, run_walk_forward_smc_validation,
)
from trading_bot_v4.core.data_handler import V4DataHandler
from trading_bot_v4.research.daily_research import (
    DAILY_DASHBOARD_PATH, DAILY_GO_STATUS_PATH, DAILY_QUALIFICATION_AUDIT_PATH,
    _update_smc_features, run_daily_research,
)
from trading_bot_v4.utils.model_cache import ModelScalerCache
from trading_bot_v4.utils.artifact_lineage import write_model_manifest


SCHEDULER_LOG_PATH = Path("logs/v4_scheduler.log")
RELOAD_MODEL_REQUEST_PATH = Path("logs/v4_reload_model.request")
DAILY_RESEARCH_HOUR = 5
DAILY_RESEARCH_MINUTE = 0
TELEGRAM_EXECUTION_MODES = {"PAPER", "WEB3_READ_ONLY", "LIVE_SMALL", "LIVE"}


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
    bearish_model: Any | None = None
    bearish_scaler: Any | None = None
    bearish_model_mtime: float = 0.0
    bearish_scaler_mtime: float = 0.0


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


def _start_telegram_listener(controller: ShutdownController, transport=None,
                             output: Callable[[str], None] = print) -> TelegramControlListener:
    listener = TelegramControlListener(
        Config.TELEGRAM_BOT_TOKEN,
        Config.TELEGRAM_ALLOWED_CHAT_IDS,
        controller,
        transport=transport,
        on_error=lambda message: _log(f"ERROR {message}"),
        on_event=lambda message: _log(message),
    )
    if Config.EXECUTION_MODE not in TELEGRAM_EXECUTION_MODES:
        message = f"Telegram: disabled in {Config.EXECUTION_MODE} mode"
        output(message); _log(message)
        return listener
    if not Config.TELEGRAM_ENABLED:
        message = "Telegram: disabled — TELEGRAM_ENABLED is false"
        output(message); _log(message)
        return listener
    if not Config.TELEGRAM_BOT_TOKEN:
        message = "Telegram: unavailable — TELEGRAM_BOT_TOKEN is missing"
        output(message); _log(message)
        return listener
    try:
        listener.start()
    except Exception as exc:
        output(f"Telegram startup failed: {exc}")
        _log(f"ERROR Telegram startup failed: {exc}")
        return listener
    output("Telegram: enabled")
    output("Telegram listener started.")
    output(f"Authorized Telegram chat IDs: {len(Config.TELEGRAM_ALLOWED_CHAT_IDS)}")
    _log("Telegram: enabled")
    _log("Telegram listener started")
    _log(f"Authorized Telegram chat IDs: {len(Config.TELEGRAM_ALLOWED_CHAT_IDS)}")
    listener.send_startup_message()
    return listener


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
    bearish_model = bearish_scaler = None
    if BEARISH_MODEL_PATH.exists() and BEARISH_SCALER_PATH.exists():
        try:
            calibration = load_bearish_calibration()
            if calibration.get("promoted", False):
                bearish_model, bearish_scaler = ModelScalerCache(
                    model_path=BEARISH_MODEL_PATH, scaler_path=BEARISH_SCALER_PATH
                ).load()
                _log("Validated bearish model loaded")
            else:
                _log("Bearish model artifacts exist but validation promotion is false")
        except Exception as exc:
            _log(f"WARNING bearish model unavailable: {exc}")
    _log("Scheduler models loaded")
    manifest_paths = [
        original_cache.model_path, original_cache.scaler_path,
        SMC_MODEL_PATH, SMC_SCALER_PATH,
    ]
    if BEARISH_MODEL_PATH.exists() and BEARISH_SCALER_PATH.exists() and BEARISH_CALIBRATION_PATH.exists():
        manifest_paths.extend([BEARISH_MODEL_PATH, BEARISH_SCALER_PATH, BEARISH_CALIBRATION_PATH])
    manifest = write_model_manifest(manifest_paths, reason)
    _log(f"Model lineage manifest: {manifest}")
    return SchedulerModelBundle(
        original_model=original_model,
        original_scaler=original_scaler,
        smc_model=smc_model,
        smc_scaler=smc_scaler,
        original_model_mtime=_mtime(Path(original_cache.model_path)),
        original_scaler_mtime=_mtime(Path(original_cache.scaler_path)),
        smc_model_mtime=_mtime(Path(SMC_MODEL_PATH)),
        smc_scaler_mtime=_mtime(Path(SMC_SCALER_PATH)),
        bearish_model=bearish_model,
        bearish_scaler=bearish_scaler,
        bearish_model_mtime=_mtime(BEARISH_MODEL_PATH),
        bearish_scaler_mtime=_mtime(BEARISH_SCALER_PATH),
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
        ("bearish model timestamp changed", BEARISH_MODEL_PATH, bundle.bearish_model_mtime),
        ("bearish scaler timestamp changed", BEARISH_SCALER_PATH, bundle.bearish_scaler_mtime),
    ]
    for reason, path, previous_mtime in checks:
        current_mtime = _mtime(path)
        if current_mtime and current_mtime != previous_mtime:
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


def _hourly_analysis_symbols(timeframe: str) -> list[str]:
    """Return GO plus WATCH assets; WATCH is shadow-only and cannot open baseline positions."""
    del timeframe
    if not DAILY_GO_STATUS_PATH.exists():
        return []
    try:
        status = pd.read_csv(DAILY_GO_STATUS_PATH)
    except Exception as exc:
        _log(f"WARNING failed to load daily analysis list: {exc}")
        return []
    if not {"symbol", "decision"}.issubset(status.columns):
        return []
    selected = status.loc[
        status["decision"].astype(str).str.upper().isin({"GO", "WATCH"}), "symbol"
    ]
    return list(dict.fromkeys(selected.astype(str).str.upper().tolist()))


def _hourly_watch_symbols(timeframe: str) -> list[str]:
    """Return the current WATCH universe for runtime status reporting."""
    del timeframe
    if not DAILY_GO_STATUS_PATH.exists():
        return []
    try:
        status = pd.read_csv(DAILY_GO_STATUS_PATH)
    except Exception as exc:
        _log(f"WARNING failed to load daily WATCH list: {exc}")
        return []
    if not {"symbol", "decision"}.issubset(status.columns):
        return []
    selected = status.loc[
        status["decision"].astype(str).str.upper().eq("WATCH"), "symbol"
    ]
    return list(dict.fromkeys(selected.astype(str).str.upper().tolist()))


def _qualified_short_symbols() -> list[str]:
    """Return bearish promotions plus bidirectional forward-paper symbols."""
    try:
        calibration = load_bearish_calibration()
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return []
    if not calibration.get("promoted", False):
        return []
    return sorted({
        str(symbol).upper() for symbol in calibration.get("promoted_symbols", {})
    })


def _analysis_symbols_with_positions(analysis_symbols: list[str], short_symbols: set[str],
                                     positions: list[dict[str, Any]]) -> list[str]:
    """Ensure open positions remain in hourly analysis after daily demotion."""
    open_symbols = sorted({
        str(position["symbol"]).upper() for position in positions if position.get("symbol")
    })
    return list(dict.fromkeys([*analysis_symbols, *sorted(short_symbols), *open_symbols]))


def _active_directional_signals(signals: pd.DataFrame, long_symbols: list[str],
                                short_symbols: set[str],
                                open_symbols: set[str] | None = None) -> pd.DataFrame:
    """Keep qualified entries plus signals needed to manage existing positions."""
    if signals.empty:
        return signals
    open_symbols = {str(value).upper() for value in (open_symbols or set())}
    symbol = signals["symbol"].astype(str).str.upper()
    direction = signals["model_direction"].astype(str).str.upper()
    directional_qualification = (direction.eq("LONG") & symbol.isin(long_symbols)) | (
        direction.eq("SHORT") & symbol.isin(short_symbols)
    )
    entry_eligible = directional_qualification
    allowed = entry_eligible | direction.eq("HOLD") | symbol.isin(open_symbols)
    active = signals.loc[allowed].copy()
    active["_entry_eligible"] = entry_eligible.loc[allowed].to_numpy()
    return active


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
        f" | Deduplicated: {summary.deduplicated_signals} | Rejections: {summary.rejection_reasons}"
        f" | Risk: {summary.risk_status}"
    )


def _run_hourly_update(timeframe: str, models: SchedulerModelBundle, orders: OrderManager,
                       controller: ShutdownController, telegram: TelegramControlListener | None = None) -> None:
    cycle_started = time.monotonic()
    positions_before = orders.position_snapshots()
    challenger_positions_before = orders.challenger_position_snapshots()
    _run_guarded("hourly.collect_gmx_market_features", collect_hourly_gmx_features)
    symbols = _hourly_refresh_symbols(timeframe)
    analysis_symbols = _hourly_analysis_symbols(timeframe)
    short_symbols = set(_qualified_short_symbols())
    analysis_symbols = _analysis_symbols_with_positions(
        analysis_symbols, short_symbols, [*positions_before, *challenger_positions_before],
    )
    if not analysis_symbols:
        _log("Hourly paper cycle skipped: today's GO/WATCH analysis list is empty")
        _log(_format_paper_summary(orders.sync()))
        return
    _log(f"Hourly qualified LONG assets: {', '.join(symbols) if symbols else 'none'}")
    _log(f"Hourly qualified SHORT assets: {', '.join(sorted(short_symbols)) if short_symbols else 'none'}")
    _log(f"Hourly GO/WATCH analysis assets: {', '.join(analysis_symbols)}")

    refresh_result = _run_guarded(
        "hourly.refresh_active_market_data",
        lambda: _refresh_active_market_data(analysis_symbols, timeframe),
    )

    def update_smc_features() -> int:
        outputs = _update_smc_features(analysis_symbols, timeframe)
        return len(outputs)

    _run_guarded("hourly.update_smc_features", update_smc_features)

    def update_active_paper_signals() -> dict[str, Any]:
        return run_original_baseline_paper_signals(
            analysis_symbols, timeframe, models.original_model, models.original_scaler
        )

    signal_result = _run_guarded("hourly.update_original_baseline_paper_signals", update_active_paper_signals)
    if signal_result is None:
        _log(_format_paper_summary(orders.sync()))
        return
    signals_path = Path(signal_result["signals_path"])
    long_signals = pd.read_csv(signals_path) if signals_path.exists() else pd.DataFrame()
    short_signals = pd.DataFrame()
    if models.bearish_model is not None and models.bearish_scaler is not None:
        def update_bearish_signals() -> pd.DataFrame:
            thresholds = {
                str(key).upper(): float(value)
                for key, value in load_bearish_calibration().get("promoted_symbols", {}).items()
            }
            return pd.concat([
                predict_bearish_signals(models.bearish_model, models.bearish_scaler, symbol, timeframe, thresholds[symbol])
                for symbol in analysis_symbols if symbol in thresholds
            ], ignore_index=True) if any(symbol in thresholds for symbol in analysis_symbols) else pd.DataFrame()
        bearish_result = _run_guarded("hourly.update_bearish_model_signals", update_bearish_signals)
        if bearish_result is not None:
            short_signals = bearish_result
    signals = combine_directional_signals(long_signals, short_signals)
    _run_guarded("hourly.shadow_challenger", lambda: orders.process_challenger_signals(signals))
    orders.set_new_entries(controller.entries_allowed())
    if Config.EXECUTION_MODE != "PAPER":
        _log(f"hourly.paper_execution skipped in {Config.EXECUTION_MODE} mode")
        return
    baseline_open_symbols = {
        str(position["symbol"]).upper() for position in positions_before if position.get("symbol")
    }
    active_signals = _active_directional_signals(
        signals, symbols, short_symbols, baseline_open_symbols,
    )
    summary = _run_guarded("hourly.paper_execution", lambda: orders.process_signals(active_signals))
    if summary is not None:
        positions_after = orders.position_snapshots()
        controller.update_runtime_snapshot(
            open_positions=summary.open_positions, equity=summary.equity,
            realized_pnl=summary.realized_pnl, unrealized_pnl=summary.unrealized_pnl,
            fees=summary.fees, positions=positions_after,
            entries_allowed=controller.entries_allowed(),
        )
        if telegram and telegram.is_running:
            telegram.notify_position_cycle(positions_before, positions_after)
        _log(_format_paper_summary(summary))
        latest_signals = signals.sort_values("timestamp").groupby("symbol").tail(1) if not signals.empty else signals
        latest_timestamp = str(latest_signals["timestamp"].max()) if not latest_signals.empty else "none"
        directions = latest_signals["model_direction"].value_counts().to_dict() if not latest_signals.empty else {}
        telegram_status = controller.runtime_snapshot().get("telegram_status", "unknown")
        _log(
            "Hourly Cycle Health | "
            f"LONG: {','.join(symbols) if symbols else 'none'} | "
            f"SHORT: {','.join(sorted(short_symbols)) if short_symbols else 'none'} | "
            f"Analyzed: {','.join(analysis_symbols)} | Refreshed: {int(refresh_result or 0)}/{len(analysis_symbols)} | "
            f"Latest candle: {latest_timestamp} | Predictions generated: {len(latest_signals)} | Directions: {directions} | "
            f"Telegram: {telegram_status} | Duration: {time.monotonic() - cycle_started:.2f}s"
        )
        print(_format_paper_summary(summary))


def _run_daily_research(timeframe: str, top_validated: int, models: SchedulerModelBundle) -> Any | None:
    return _run_guarded(
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


def _daily_artifact_paths() -> list[Path]:
    original_cache = ModelScalerCache()
    return [
        Path(original_cache.model_path), Path(original_cache.scaler_path),
        SMC_MODEL_PATH, SMC_SCALER_PATH,
        BEARISH_MODEL_PATH, BEARISH_SCALER_PATH, BEARISH_CALIBRATION_PATH,
        BEARISH_VALIDATION_PATH, BEARISH_WALK_FORWARD_PATH,
        WALK_FORWARD_SUMMARY_PATH, WALK_FORWARD_REPORT_PATH,
        DAILY_GO_STATUS_PATH, DAILY_QUALIFICATION_AUDIT_PATH, DAILY_DASHBOARD_PATH,
    ]


def _run_complete_daily_refresh(timeframe: str, top_validated: int) -> tuple[Any, SchedulerModelBundle]:
    """Retrain and revalidate every model before publishing daily qualification."""
    paths = _daily_artifact_paths()
    with tempfile.TemporaryDirectory(prefix="v5-daily-artifacts-") as directory:
        backup_root = Path(directory)
        existing: dict[Path, Path] = {}
        missing: set[Path] = set()
        for index, path in enumerate(paths):
            if path.exists():
                backup = backup_root / f"{index}_{path.name}"
                shutil.copy2(path, backup)
                existing[path] = backup
            else:
                missing.add(path)
        try:
            _log("Daily full refresh 1/7: refresh all GMX market data")
            if not V4DataHandler().refresh_gmx_cache(force=True):
                raise RuntimeError("daily all-asset GMX refresh failed")
            _log("Daily full refresh 2/7: rebuild all-asset SMC training data")
            build_all_assets_smc_training_data(timeframe)
            _log("Daily full refresh 3/7: retrain original LONG model")
            train_v4_model(timeframe=timeframe, send_telegram=False)
            _log("Daily full refresh 4/7: retrain SMC LONG model")
            train_smc_model(timeframe)
            _log("Daily full refresh 5/7: retrain and calibrate bearish model")
            train_bearish_model(timeframe)
            _log("Daily full refresh 6/7: regenerate all-asset LONG walk-forward validation")
            run_walk_forward_smc_validation(_scheduler_args(timeframe=timeframe, all_assets=True))
            refreshed_models = _load_scheduler_models("complete daily retraining")
            _log("Daily full refresh 7/7: regenerate rankings, constrained performance, and qualification")
            daily_result = _run_daily_research(timeframe, top_validated, refreshed_models)
            if daily_result is None:
                raise RuntimeError("daily qualification pipeline failed")
            return daily_result, refreshed_models
        except Exception:
            _log("ERROR complete daily refresh failed; restoring previous validated artifacts")
            for path, backup in existing.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)
            for path in missing:
                path.unlink(missing_ok=True)
            raise


def run_auto_scheduler(args: Any) -> None:
    """Run the paper-only V4 scheduler until interrupted."""
    allowed_modes = {"RESEARCH", "PAPER", "WEB3_READ_ONLY", "LIVE_SMALL", "LIVE"}
    if Config.EXECUTION_MODE not in allowed_modes:
        raise ValueError(f"Invalid EXECUTION_MODE: {Config.EXECUTION_MODE}")
    if Config.EXECUTION_MODE in {"LIVE_SMALL", "LIVE"}:
        raise RuntimeError("Live signing is disabled: no confirmed GMX live execution adapter is configured")
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    top_validated = int(getattr(args, "top_validated", 10) or 10)
    controller = ShutdownController()
    instance_lock = InstanceLock(); instance_lock.acquire()
    control_server = ControlServer(controller); control_server.start()
    telegram = _start_telegram_listener(controller)
    request_graceful = graceful_signal_handler(controller, _log)
    previous_sigint = signal.signal(signal.SIGINT, request_graceful)
    previous_sigterm = signal.signal(signal.SIGTERM, request_graceful)

    now = datetime.now()
    state = SchedulerState(
        next_hourly_update=now,
        next_daily_research=_next_daily_time(now),
    )
    displayed_next_hourly = _next_hour_boundary(now)
    orders = None
    position_monitor = None
    qualified_market_monitor = None
    try:
        models = _load_scheduler_models("scheduler startup")
        if Config.MARKET_CAP_RISK_ENABLED:
            _run_guarded("startup.refresh_market_caps", collect_market_caps)
        orders = OrderManager()
        hourly_symbols = _hourly_refresh_symbols(timeframe)
        hourly_watch_symbols = _hourly_watch_symbols(timeframe)
        hourly_short_symbols = _qualified_short_symbols()
        _log("Restoring persistent state...")
        restored = orders.restore_and_reconcile(Config.EXECUTION_MODE, hourly_symbols)
        _log(f"Previous shutdown clean: {restored['previous_shutdown_clean']}")
        _log(f"Paper balance restored: ${restored['balance']:.2f}")
        _log(f"Open positions restored: {restored['open_positions']}")
        _log(f"Pending orders restored: {restored['pending_orders']}")
        if Config.EXECUTION_MODE == "PAPER":
            position_monitor = PositionMonitor(controller, telegram, log=_log)
            position_monitor.start()
            _log(
                f"Position monitor started: interval={Config.POSITION_MONITOR_INTERVAL_SECONDS}s "
                f"open_positions={restored['open_positions']}"
            )
            qualified_market_monitor = QualifiedMarketMonitor(
                controller,
                lambda: (_hourly_refresh_symbols(timeframe), _qualified_short_symbols()),
                telegram,
                log=_log,
            )
            qualified_market_monitor.start()
            _log(
                f"Qualified market monitor started: interval={Config.QUALIFIED_MARKET_INTERVAL_SECONDS}s "
                f"LONG={','.join(hourly_symbols) if hourly_symbols else 'none'} "
                f"SHORT={','.join(hourly_short_symbols) if hourly_short_symbols else 'none'}"
            )
        controller.configure_entry_guard(True, Config.EXECUTION_MODE in {"PAPER", "LIVE_SMALL", "LIVE"})
        controller.enable_new_entries(); orders.set_new_entries(controller.entries_allowed())
    except Exception:
        _log("ERROR scheduler startup failed; releasing runtime resources")
        if position_monitor is not None:
            position_monitor.stop()
        if qualified_market_monitor is not None:
            qualified_market_monitor.stop()
        telegram.stop(); control_server.stop()
        if orders is not None:
            orders.close()
        instance_lock.release()
        signal.signal(signal.SIGINT, previous_sigint); signal.signal(signal.SIGTERM, previous_sigterm)
        raise

    print("V5 Scheduler started.")
    print(f"Execution mode: {Config.EXECUTION_MODE}")
    print(f"New entries: {'enabled' if orders.new_entries_allowed() else 'BLOCKED (run --resume-paper to recover)'}")
    print("Model loaded.")
    print("Scaler loaded.")
    print(f"Qualified LONG: {', '.join(hourly_symbols) if hourly_symbols else 'none'}")
    print(f"Qualified SHORT: {', '.join(hourly_short_symbols) if hourly_short_symbols else 'none'}")
    print(f"WATCH: {', '.join(hourly_watch_symbols) if hourly_watch_symbols else 'none'}")
    print(f"Qualified market monitoring: every {Config.QUALIFIED_MARKET_INTERVAL_SECONDS} seconds")
    print("Daily research refresh: all assets")
    print(f"Next hourly update: {displayed_next_hourly.isoformat(timespec='seconds')}")
    print(f"Next daily research: {state.next_daily_research.isoformat(timespec='seconds')}")
    print("Live signing: disabled.")
    web3_status = check_web3_readiness()
    print(f"Web3 read-only: {'connected' if web3_status.connected else web3_status.error}")
    _log("Scheduler started")
    _log(f"Qualified LONG: {', '.join(hourly_symbols) if hourly_symbols else 'none'}")
    _log(f"Qualified SHORT: {', '.join(hourly_short_symbols) if hourly_short_symbols else 'none'}")
    _log(f"WATCH: {', '.join(hourly_watch_symbols) if hourly_watch_symbols else 'none'}")
    _log("Daily research refresh: all assets")
    _log(f"Next hourly update: {displayed_next_hourly.isoformat(timespec='seconds')}")
    _log(f"Next daily research: {state.next_daily_research.isoformat(timespec='seconds')}")
    _log("Live signing disabled")
    _log(f"Web3 read-only status: {web3_status.to_dict()}")

    next_hourly = state.next_hourly_update
    next_daily = state.next_daily_research
    monitor_watchdog_alerted = False
    qualified_monitor_watchdog_alerted = False
    monitor_started_at = time.monotonic()
    initial_summary = orders.sync()
    controller.update_runtime_snapshot(
        previous_shutdown_clean=restored["previous_shutdown_clean"],
        pending_orders=restored["pending_orders"], execution_mode=Config.EXECUTION_MODE, scheduler_running=True,
        entries_allowed=controller.entries_allowed(), qualified_assets=hourly_symbols,
        qualified_long_assets=hourly_symbols, qualified_short_assets=hourly_short_symbols,
        watch_assets=hourly_watch_symbols,
        open_positions=initial_summary.open_positions, equity=initial_summary.equity,
        realized_pnl=initial_summary.realized_pnl, unrealized_pnl=initial_summary.unrealized_pnl,
        fees=initial_summary.fees, positions=orders.position_snapshots(),
        next_hourly_update=next_hourly.isoformat(timespec="seconds"),
        next_daily_research=next_daily.isoformat(timespec="seconds"),
        web3_read_only_status="connected" if web3_status.connected else web3_status.error,
        live_signing_status="disabled",
    )
    try:
        while not controller.is_shutdown_requested():
            if position_monitor is not None and time.monotonic() - monitor_started_at > Config.POSITION_MONITOR_WATCHDOG_SECONDS:
                if not position_monitor.is_healthy() and not monitor_watchdog_alerted:
                    monitor_watchdog_alerted = True
                    issue = position_monitor.health_issue()
                    _log(f"ERROR position monitor unhealthy: {issue}")
                    controller.update_runtime_snapshot(position_monitor_status=issue)
                    if telegram.is_running:
                        telegram.broadcast(
                            "🚨 <b>POSITION MONITOR UNHEALTHY</b>\n"
                            f"Reason: <code>{issue}</code>\n"
                            "The scheduler is running but protective monitoring needs attention."
                        )
                elif position_monitor.is_healthy():
                    monitor_watchdog_alerted = False
            if (qualified_market_monitor is not None
                    and time.monotonic() - monitor_started_at > Config.QUALIFIED_MARKET_WATCHDOG_SECONDS):
                if not qualified_market_monitor.is_healthy() and not qualified_monitor_watchdog_alerted:
                    qualified_monitor_watchdog_alerted = True
                    issue = qualified_market_monitor.health_issue()
                    _log(f"ERROR qualified market monitor unhealthy: {issue}")
                    controller.update_runtime_snapshot(qualified_market_status=issue)
                    if telegram.is_running:
                        telegram.broadcast(
                            "🚨 <b>QUALIFIED MARKET MONITOR UNHEALTHY</b>\n"
                            f"Reason: <code>{issue}</code>\n"
                            "Live entry-price observation needs attention."
                        )
                elif qualified_market_monitor.is_healthy():
                    qualified_monitor_watchdog_alerted = False
            now = datetime.now()
            if now >= next_hourly:
                with controller.active_cycle() as admitted:
                    if admitted:
                        models = _maybe_reload_models(models)
                        _run_hourly_update(timeframe, models, orders, controller, telegram)
                next_hourly = _next_hour_boundary(datetime.now())
                controller.update_runtime_snapshot(next_hourly_update=next_hourly.isoformat(timespec="seconds"))
                _log(f"Next hourly update: {next_hourly.isoformat(timespec='seconds')}")

            now = datetime.now()
            if now >= next_daily:
                with controller.active_cycle() as admitted:
                    if admitted:
                        if Config.MARKET_CAP_RISK_ENABLED:
                            _run_guarded("daily.refresh_market_caps", collect_market_caps)
                        if telegram.is_running:
                            telegram.broadcast(
                                "🔄 <b>DAILY FULL REFRESH STARTED</b>\n"
                                "Refreshing data, retraining all models, and rerunning all validation."
                            )
                        complete_result = _run_guarded(
                            "daily.complete_retrain_revalidate",
                            lambda: _run_complete_daily_refresh(timeframe, top_validated),
                        )
                        if complete_result is not None:
                            daily_result, models = complete_result
                            updated_symbols = _hourly_refresh_symbols(timeframe)
                            updated_watch_symbols = _hourly_watch_symbols(timeframe)
                            updated_short_symbols = _qualified_short_symbols()
                            orders.update_whitelist(updated_symbols)
                            controller.update_runtime_snapshot(
                                qualified_assets=updated_symbols,
                                qualified_long_assets=updated_symbols,
                                qualified_short_assets=updated_short_symbols,
                                watch_assets=updated_watch_symbols,
                            )
                            _log(f"Persisted qualified LONG list: {', '.join(updated_symbols) if updated_symbols else 'none'}")
                            _log(f"Persisted qualified SHORT list: {', '.join(updated_short_symbols) if updated_short_symbols else 'none'}")
                            _log(f"Persisted WATCH list: {', '.join(updated_watch_symbols) if updated_watch_symbols else 'none'}")
                            if telegram.is_running:
                                telegram.broadcast(
                                    "✅ <b>DAILY FULL REFRESH COMPLETE</b>\n"
                                    f"LONG: <code>{', '.join(updated_symbols) if updated_symbols else 'none'}</code>\n"
                                    f"SHORT: <code>{', '.join(updated_short_symbols) if updated_short_symbols else 'none'}</code>\n"
                                    f"WATCH: <code>{', '.join(updated_watch_symbols) if updated_watch_symbols else 'none'}</code>"
                                )
                        elif telegram.is_running:
                            telegram.broadcast(
                                "🚨 <b>DAILY FULL REFRESH FAILED</b>\n"
                                "Previous validated model artifacts were restored. No new qualification was published."
                            )
                next_daily = _next_daily_time(datetime.now())
                controller.update_runtime_snapshot(next_daily_research=next_daily.isoformat(timespec="seconds"))
                _log(f"Next daily research: {next_daily.isoformat(timespec='seconds')}")

            time.sleep(1)
    finally:
        controller.update_runtime_snapshot(scheduler_running=False, entries_allowed=False)
        controller.block_new_entries(); orders.set_new_entries(False)
        controller.wait_for_active_cycles()
        if position_monitor is not None:
            _log("Stopping position monitor...")
            position_monitor.stop()
        if qualified_market_monitor is not None:
            _log("Stopping qualified market monitor...")
            qualified_market_monitor.stop()
        mode = controller.get_requested_mode()
        if mode is ShutdownMode.NONE:
            mode = ShutdownMode.GRACEFUL
            controller.request_shutdown(mode, "scheduler-finally")
        report = ShutdownCoordinator(orders, alert=lambda message: _log(f"ALERT {message}"),
                                     controller=controller, log=_log).execute(mode)
        print(f"Scheduler stopped using {report.mode.name}; account flat: {report.account_flat}")
        _log("Committing database..."); orders.save_state()
        _log("Stopping Telegram listener..."); telegram.stop()
        _log("Stopping scheduler and local control server..."); control_server.stop()
        orders.close(); instance_lock.release()
        _log(f"{report.mode.name} shutdown complete.")
        signal.signal(signal.SIGINT, previous_sigint); signal.signal(signal.SIGTERM, previous_sigterm)
