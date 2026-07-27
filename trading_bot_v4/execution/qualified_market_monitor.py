"""Continuous observation for qualified assets without intrahour entry inference."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.execution.position_monitor import GMXTickerPriceProvider
from trading_bot_v4.shutdown_controller import ShutdownController


class QualifiedMarketMonitor:
    def __init__(self, controller: ShutdownController,
                 qualified_symbols: Callable[[], tuple[list[str], list[str]]],
                 notifier: Any | None = None, provider: Any | None = None, db_path=None,
                 interval_seconds: int | None = None, log=lambda _message: None):
        self.controller = controller
        self.qualified_symbols = qualified_symbols
        self.notifier = notifier
        self.provider = provider or GMXTickerPriceProvider()
        self.db_path = db_path or Config.PAPER_DB_PATH
        self.interval = int(interval_seconds or Config.QUALIFIED_MARKET_INTERVAL_SECONDS)
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failures = 0
        self._alerted = False
        self._spread_alerted = False
        self._last_heartbeat_monotonic = 0.0

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="v5-qualified-market-monitor", daemon=True)
        self._thread.start()

    def is_healthy(self) -> bool:
        return self.is_running and bool(self._last_heartbeat_monotonic) and (
            time.monotonic() - self._last_heartbeat_monotonic
            <= Config.QUALIFIED_MARKET_WATCHDOG_SECONDS
        )

    def health_issue(self) -> str:
        if not self.is_running:
            return "thread_stopped"
        if not self._last_heartbeat_monotonic:
            return "no_heartbeat"
        if time.monotonic() - self._last_heartbeat_monotonic > Config.QUALIFIED_MARKET_WATCHDOG_SECONDS:
            return "heartbeat_stale"
        return ""

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(5, Config.POSITION_MONITOR_REQUEST_TIMEOUT_SECONDS + 2))

    def _alert(self, message: str) -> None:
        if self.notifier and getattr(self.notifier, "is_running", True):
            self.notifier.broadcast(message)

    def run_once(self, manager: OrderManager) -> dict[str, Any]:
        begin_cycle = getattr(self.provider, "begin_cycle", None)
        if callable(begin_cycle):
            begin_cycle()
        long_symbols, short_symbols = self.qualified_symbols()
        symbols = sorted(set(long_symbols) | set(short_symbols))
        prepare_cycle = getattr(self.provider, "prepare_cycle", None)
        if symbols and callable(prepare_cycle):
            prepare_cycle()
        errors, wide = [], []
        checked = 0
        for symbol in symbols:
            started = time.monotonic()
            try:
                quote = self.provider.fetch_pair(symbol)
                latency_ms = (time.monotonic() - started) * 1000.0
                if quote.age_seconds > Config.QUALIFIED_MARKET_SNAPSHOT_MAX_AGE_SECONDS:
                    raise TimeoutError(
                        f"GMX ticker is stale by {quote.age_seconds:.1f}s "
                        f"(limit {Config.QUALIFIED_MARKET_SNAPSHOT_MAX_AGE_SECONDS}s)"
                    )
                midpoint = (quote.min_price + quote.max_price) / 2.0
                spread_bps = (quote.max_price - quote.min_price) / midpoint * 10000.0
                status = "wide_spread" if spread_bps > Config.QUALIFIED_MARKET_MAX_SPREAD_BPS else "healthy"
                manager.record_qualified_market_snapshot(
                    symbol, quote.min_price, quote.max_price, quote.observed_at, spread_bps,
                    latency_ms, quote.age_seconds, status,
                )
                checked += 1
                if status == "wide_spread":
                    wide.append(f"{symbol} {spread_bps:.1f}bps")
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
                manager.record_qualified_market_snapshot(
                    symbol, 0.0, 0.0, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    0.0, (time.monotonic() - started) * 1000.0, 0.0, "failed", str(exc),
                )
        if errors:
            self._failures += 1
            message = "; ".join(errors)
            self.log(f"WARNING qualified market feed failure #{self._failures}: {message}")
            if self._failures >= Config.POSITION_MONITOR_FAILURE_THRESHOLD and not self._alerted:
                self._alerted = True
                self._alert(f"🚨 <b>QUALIFIED MARKET FEED DEGRADED</b>\n<code>{message}</code>")
        else:
            if self._failures:
                self.log("Qualified market feed recovered")
            self._failures = 0
            self._alerted = False
        if wide:
            self.log(f"WARNING qualified market wide spread: {'; '.join(wide)}")
            if not self._spread_alerted:
                self._spread_alerted = True
                self._alert(f"⚠️ <b>QUALIFIED MARKET WIDE SPREAD</b>\n<code>{'; '.join(wide)}</code>")
        else:
            self._spread_alerted = False
        self.controller.update_runtime_snapshot(
            qualified_market_status="degraded" if errors else "healthy",
            qualified_market_symbols=checked,
            qualified_market_failures=self._failures,
            qualified_market_last_error="; ".join(errors),
            qualified_market_last_update=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._last_heartbeat_monotonic = time.monotonic()
        return {"symbols": symbols, "checked": checked, "errors": errors, "wide_spreads": wide}

    def _run(self) -> None:
        manager = OrderManager(self.db_path)
        try:
            while not self._stop.is_set():
                try:
                    self.run_once(manager)
                except Exception as exc:
                    self._failures += 1
                    self._last_heartbeat_monotonic = time.monotonic()
                    self.log(f"ERROR qualified market monitor cycle failed: {exc}")
                    self.controller.update_runtime_snapshot(
                        qualified_market_status="failed",
                        qualified_market_failures=self._failures,
                        qualified_market_last_error=str(exc),
                        qualified_market_last_update=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    )
                    if self._failures >= Config.POSITION_MONITOR_FAILURE_THRESHOLD and not self._alerted:
                        self._alerted = True
                        self._alert("🚨 <b>QUALIFIED MARKET MONITOR FAILED</b>\nManual review is required.")
                self._stop.wait(self.interval)
        finally:
            manager.close()
