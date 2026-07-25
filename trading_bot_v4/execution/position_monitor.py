"""Live protective monitoring for persistent paper positions.

Entry/model decisions remain hourly. This service only marks open positions and
executes stored stop-loss or take-profit protection using current GMX oracle
tickers, with GMX one-minute candles as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time
from typing import Any, Protocol

import requests

from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.shutdown_controller import ShutdownController


class AlertSink(Protocol):
    def broadcast(self, text: str) -> None: ...


@dataclass(frozen=True)
class MarketPrice:
    symbol: str
    price: float
    observed_at: str
    age_seconds: float
    source: str = "GMX_TICKER"
    fallback_reason: str = ""


@dataclass(frozen=True)
class MarketPricePair:
    symbol: str
    min_price: float
    max_price: float
    observed_at: str
    age_seconds: float
    source: str = "GMX_TICKER"


class GMXMinutePriceProvider:
    SYMBOL_ALIASES = {"APE_DEPRECATED": "APE", "XAUT.V2": "XAUT", "WBTC.B": "BTC"}

    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        self.base_url = (base_url or Config.GMX_PRICE_API_BASE).rstrip("/")
        self.timeout = int(timeout or Config.POSITION_MONITOR_REQUEST_TIMEOUT_SECONDS)

    def fetch(self, symbol: str, direction: str | None = None) -> MarketPrice:
        del direction
        requested = str(symbol).upper()
        api_symbol = self.SYMBOL_ALIASES.get(requested, requested)
        response = requests.get(
            f"{self.base_url}/prices/candles",
            params={"tokenSymbol": api_symbol, "period": "1m", "limit": 3},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        candles = payload.get("candles", payload) if isinstance(payload, dict) else payload
        if not isinstance(candles, list) or not candles:
            raise ValueError(f"GMX returned no one-minute prices for {requested}")

        parsed: list[tuple[float, float]] = []
        for candle in candles:
            if isinstance(candle, (list, tuple)) and len(candle) >= 5:
                parsed.append((float(candle[0]), float(candle[4])))
            elif isinstance(candle, dict):
                timestamp = candle.get("timestamp", candle.get("time", candle.get("openTime")))
                close = candle.get("close", candle.get("closePrice"))
                if timestamp is not None and close is not None:
                    parsed.append((float(timestamp), float(close)))
        if not parsed:
            raise ValueError(f"GMX returned malformed one-minute prices for {requested}")
        raw_timestamp, price = max(parsed, key=lambda item: item[0])
        if raw_timestamp > 10_000_000_000:
            raw_timestamp /= 1000.0
        observed = datetime.fromtimestamp(raw_timestamp, tz=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - observed).total_seconds())
        if price <= 0:
            raise ValueError(f"GMX returned an invalid price for {requested}")
        if age > Config.POSITION_MONITOR_MAX_PRICE_AGE_SECONDS:
            raise TimeoutError(f"GMX one-minute price for {requested} is stale by {age:.0f}s")
        return MarketPrice(requested, price, observed.isoformat(timespec="seconds"), age, "GMX_1M")


class GMXTickerPriceProvider:
    """Read executable-side GMX oracle prices, falling back to 1m closes."""

    SYMBOL_ALIASES = GMXMinutePriceProvider.SYMBOL_ALIASES

    def __init__(self, base_url: str | None = None, timeout: int | None = None,
                 fallback: Any | None = None):
        self.base_url = (base_url or Config.GMX_PRICE_API_BASE).rstrip("/")
        self.timeout = int(timeout or Config.POSITION_MONITOR_REQUEST_TIMEOUT_SECONDS)
        self.fallback = fallback or GMXMinutePriceProvider(self.base_url, self.timeout)
        self._token_decimals: dict[str, int] = {}
        self._ticker_cache: list[dict[str, Any]] = []
        self._ticker_cache_monotonic = 0.0

    def _load_token_decimals(self) -> None:
        if self._token_decimals:
            return
        response = requests.get(f"{self.base_url}/tokens", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens", payload) if isinstance(payload, dict) else payload
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("GMX returned no token metadata")
        for token in tokens:
            if isinstance(token, dict) and token.get("symbol") is not None and token.get("decimals") is not None:
                self._token_decimals[str(token["symbol"]).upper()] = int(token["decimals"])

    def _load_tickers(self) -> list[dict[str, Any]]:
        # One response contains all symbols. Reuse it while a monitor cycle checks
        # multiple positions instead of issuing one request per open position.
        if self._ticker_cache and time.monotonic() - self._ticker_cache_monotonic < 1.0:
            return self._ticker_cache
        response = requests.get(f"{self.base_url}/prices/tickers", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        tickers = payload.get("tickers", payload) if isinstance(payload, dict) else payload
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("GMX returned no current oracle tickers")
        self._ticker_cache = tickers
        self._ticker_cache_monotonic = time.monotonic()
        return tickers

    def _fetch_ticker(self, symbol: str, direction: str) -> MarketPrice:
        requested = str(symbol).upper()
        api_symbol = self.SYMBOL_ALIASES.get(requested, requested).upper()
        self._load_token_decimals()
        ticker = next(
            (item for item in self._load_tickers()
             if isinstance(item, dict) and str(item.get("tokenSymbol", "")).upper() == api_symbol),
            None,
        )
        if ticker is None:
            raise ValueError(f"GMX returned no current oracle ticker for {requested}")
        decimals = self._token_decimals.get(api_symbol)
        if decimals is None:
            raise ValueError(f"GMX returned no token decimals for {requested}")

        side = "minPrice" if str(direction).upper() == "LONG" else "maxPrice"
        raw_price = int(ticker[side])
        price = raw_price / (10 ** (30 - decimals))
        raw_timestamp = float(ticker.get("updatedAt", ticker.get("timestamp")))
        if raw_timestamp > 10_000_000_000:
            raw_timestamp /= 1000.0
        observed = datetime.fromtimestamp(raw_timestamp, tz=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - observed).total_seconds())
        if price <= 0:
            raise ValueError(f"GMX returned an invalid current oracle price for {requested}")
        if age > Config.POSITION_MONITOR_MAX_PRICE_AGE_SECONDS:
            raise TimeoutError(f"GMX current oracle price for {requested} is stale by {age:.0f}s")
        return MarketPrice(requested, price, observed.isoformat(timespec="seconds"), age,
                           f"GMX_TICKER_{'MIN' if side == 'minPrice' else 'MAX'}")

    def fetch_pair(self, symbol: str) -> MarketPricePair:
        requested = str(symbol).upper()
        api_symbol = self.SYMBOL_ALIASES.get(requested, requested).upper()
        self._load_token_decimals()
        ticker = next(
            (item for item in self._load_tickers()
             if isinstance(item, dict) and str(item.get("tokenSymbol", "")).upper() == api_symbol), None,
        )
        if ticker is None:
            raise ValueError(f"GMX returned no current oracle ticker for {requested}")
        decimals = self._token_decimals.get(api_symbol)
        if decimals is None:
            raise ValueError(f"GMX returned no token decimals for {requested}")
        divisor = 10 ** (30 - decimals)
        min_price, max_price = int(ticker["minPrice"]) / divisor, int(ticker["maxPrice"]) / divisor
        raw_timestamp = float(ticker.get("updatedAt", ticker.get("timestamp")))
        if raw_timestamp > 10_000_000_000:
            raw_timestamp /= 1000.0
        observed = datetime.fromtimestamp(raw_timestamp, tz=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - observed).total_seconds())
        if min_price <= 0 or max_price <= 0 or max_price < min_price:
            raise ValueError(f"GMX returned an invalid price pair for {requested}")
        if age > Config.POSITION_MONITOR_MAX_PRICE_AGE_SECONDS:
            raise TimeoutError(f"GMX current oracle price for {requested} is stale by {age:.0f}s")
        return MarketPricePair(requested, min_price, max_price, observed.isoformat(timespec="seconds"), age)

    def fetch(self, symbol: str, direction: str | None = None) -> MarketPrice:
        direction = str(direction or "LONG").upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError(f"invalid position direction for {symbol}: {direction}")
        try:
            return self._fetch_ticker(symbol, direction)
        except Exception as exc:
            try:
                quote = self.fallback.fetch(symbol, direction)
            except TypeError:
                quote = self.fallback.fetch(symbol)
            return MarketPrice(
                quote.symbol, quote.price, quote.observed_at, quote.age_seconds,
                "GMX_1M_FALLBACK", str(exc),
            )


class PositionMonitor:
    def __init__(self, controller: ShutdownController, notifier: AlertSink | None = None,
                 provider: Any | None = None, db_path=None, interval_seconds: int | None = None,
                 failure_threshold: int | None = None, log=lambda _message: None):
        self.controller = controller
        self.notifier = notifier
        self.provider = provider or GMXTickerPriceProvider()
        self.db_path = db_path or Config.PAPER_DB_PATH
        self.interval = int(interval_seconds or Config.POSITION_MONITOR_INTERVAL_SECONDS)
        self.failure_threshold = int(failure_threshold or Config.POSITION_MONITOR_FAILURE_THRESHOLD)
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_heartbeat_monotonic = 0.0
        self._consecutive_failures = 0
        self._failure_alerted = False

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def is_healthy(self) -> bool:
        maximum_age = max(10.0, float(Config.POSITION_MONITOR_WATCHDOG_SECONDS))
        return self.is_running and (time.monotonic() - self._last_heartbeat_monotonic) <= maximum_age

    def health_issue(self) -> str:
        if not self.is_running:
            return "thread_stopped"
        if not self._last_heartbeat_monotonic:
            return "no_heartbeat"
        if time.monotonic() - self._last_heartbeat_monotonic > Config.POSITION_MONITOR_WATCHDOG_SECONDS:
            return "heartbeat_stale"
        return ""

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="v5-position-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(5, Config.POSITION_MONITOR_REQUEST_TIMEOUT_SECONDS + 2))

    def _alert(self, text: str) -> None:
        if self.notifier and getattr(self.notifier, "is_running", True):
            self.notifier.broadcast(text)

    def run_once(self, manager: OrderManager) -> list[dict[str, Any]]:
        positions = manager.position_snapshots()
        exits: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        for position in positions:
            symbol = str(position["symbol"])
            try:
                try:
                    quote = self.provider.fetch(symbol, str(position["direction"]))
                except TypeError:
                    # Compatibility for simple injected providers used by callers/tests.
                    quote = self.provider.fetch(symbol)
                if quote.fallback_reason:
                    warnings.append(f"{symbol}: ticker unavailable; using 1m fallback ({quote.fallback_reason})")
                event = manager.monitor_market_price(symbol, quote.price, quote.observed_at, quote.source)
                if event:
                    exits.append(event)
                    icon = "🛑" if event["reason"] == "stop_loss" else "🏁"
                    self._alert(
                        f"{icon} <b>IMMEDIATE PAPER EXIT</b>\n"
                        "━━━━━━━━━━━━━━━━━━\n"
                        f"🪙 <b>{event['symbol']}</b> · <code>{event['direction']}</code>\n"
                        f"Reason: <b>{event['reason'].replace('_', ' ').upper()}</b>\n"
                        f"Observed: <code>${event['observed_price']:.8g}</code>\n"
                        f"Accounting price: <code>${event['accounting_price']:.8g}</code>\n"
                        f"Feed time: <code>{event['observed_at']}</code>"
                    )
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

        feed_issues = errors + warnings
        if feed_issues:
            self._consecutive_failures += 1
            error_text = "; ".join(feed_issues)
            manager.record_monitor_heartbeat("degraded", len(positions) - len(errors), error_text)
            self.controller.update_runtime_snapshot(
                position_monitor_status="degraded",
                position_monitor_failures=self._consecutive_failures,
                position_monitor_last_error=error_text,
            )
            self.log(f"WARNING position monitor failure #{self._consecutive_failures}: {error_text}")
            if self._consecutive_failures >= self.failure_threshold and not self._failure_alerted:
                self._failure_alerted = True
                self._alert(
                    "🚨 <b>POSITION MONITOR DEGRADED</b>\n"
                    f"The primary live price feed was degraded for {self._consecutive_failures} consecutive checks.\n"
                    "Open positions remain persisted. Manual review is required."
                )
        else:
            if self._consecutive_failures:
                self.log("Position monitor price feed recovered")
            self._consecutive_failures = 0
            self._failure_alerted = False
            manager.record_monitor_heartbeat("healthy", len(positions))
            self.controller.update_runtime_snapshot(
                position_monitor_status="healthy",
                position_monitor_failures=0,
                position_monitor_last_error="",
            )
        summary = manager.sync()
        self.controller.update_runtime_snapshot(
            open_positions=summary.open_positions,
            equity=summary.equity,
            realized_pnl=summary.realized_pnl,
            unrealized_pnl=summary.unrealized_pnl,
            fees=summary.fees,
            positions=manager.position_snapshots(),
            position_monitor_last_heartbeat=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._last_heartbeat_monotonic = time.monotonic()
        return exits

    def _run(self) -> None:
        manager = OrderManager(self.db_path)
        try:
            while not self._stop.is_set():
                try:
                    self.run_once(manager)
                except Exception as exc:
                    self._consecutive_failures += 1
                    self._last_heartbeat_monotonic = time.monotonic()
                    try:
                        manager.record_monitor_heartbeat("failed", 0, str(exc))
                    except Exception:
                        pass
                    self.controller.update_runtime_snapshot(
                        position_monitor_status="failed",
                        position_monitor_failures=self._consecutive_failures,
                        position_monitor_last_error=str(exc),
                    )
                    self.log(f"ERROR position monitor cycle failed: {exc}")
                    if self._consecutive_failures >= self.failure_threshold and not self._failure_alerted:
                        self._failure_alerted = True
                        self._alert("🚨 <b>POSITION MONITOR FAILED</b>\nManual intervention may be required.")
                self._stop.wait(self.interval)
        finally:
            try:
                manager.record_monitor_heartbeat("stopped", 0)
            finally:
                manager.close()
