"""V5 command handling built on the project's original Telegram transport."""

from __future__ import annotations

import threading
import random
import re
from html import escape
from typing import Callable, Protocol

import requests

from trading_bot_v4.shutdown_controller import ShutdownController, ShutdownMode
from trading_bot_v4.telegram.notifier import V4TelegramNotifier


class TelegramTransport(Protocol):
    def verify_connection(self) -> dict: ...
    def get_updates_checked(self, offset: int, timeout: int = 10) -> list[dict]: ...
    def send_to(self, chat_id: str, text: str) -> None: ...


class TelegramControlListener:
    def __init__(self, token: str, authorized_chat_ids: set[str], controller: ShutdownController,
                 send: Callable[[str, str], None] | None = None,
                 transport: TelegramTransport | None = None,
                 on_error: Callable[[str], None] | None = None,
                 on_event: Callable[[str], None] | None = None):
        self.token = token
        self.authorized = {str(value) for value in authorized_chat_ids if str(value)}
        self.controller = controller
        self._send_override = send
        self.transport = transport
        self.on_error = on_error or (lambda _message: None)
        self.on_event = on_event or (lambda _message: None)
        self._pending_confirmation: dict[str, ShutdownMode] = {}
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._consecutive_failures = 0

    def _safe_error(self, exc: Exception) -> str:
        message = str(exc)
        if self.token:
            message = message.replace(self.token, "<redacted>")
        return re.sub(r"https://api\.telegram\.org/bot[^/\s]+", "https://api.telegram.org/bot<redacted>", message)

    def _set_health(self, status: str, error: str | None = None) -> None:
        self.controller.update_runtime_snapshot(
            telegram_status=status,
            telegram_consecutive_failures=self._consecutive_failures,
            telegram_last_error=error or "",
        )

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._started.is_set())

    def start(self) -> None:
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is missing")
        if not self.authorized:
            raise ValueError("TELEGRAM_ALLOWED_CHAT_IDS and TELEGRAM_CHAT_ID are missing")
        if self.transport is None:
            # Reuse the old project's notifier, but point it at the normalized
            # token rather than maintaining a second Telegram implementation.
            transport = V4TelegramNotifier()
            transport.token = self.token
            transport.base_url = f"https://api.telegram.org/bot{self.token}"
            self.transport = transport
        self.transport.verify_connection()
        self._set_health("connected")
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="v5-telegram-control", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=2) or not self._thread.is_alive():
            raise RuntimeError("Telegram polling thread did not start")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=17)
        self._started.clear()
        self._set_health("stopped")

    def _send(self, chat_id: str, text: str) -> None:
        if self._send_override:
            self._send_override(chat_id, text)
        else:
            self.transport.send_to(chat_id, text)

    def broadcast(self, text: str) -> None:
        for chat_id in sorted(self.authorized):
            try:
                self._send(chat_id, text)
            except Exception as exc:
                self.on_error(f"Telegram send error for authorized chat {chat_id}: {self._safe_error(exc)}")

    def send_startup_message(self) -> None:
        self.broadcast(
            "🤖 <b>V5 PAPER BOT ONLINE</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "✅ <b>Operating Automatically</b>\n"
            "📊 Research and hourly management active\n"
            "🔒 Live signing disabled\n\n"
            "📱 <b>Quick Commands</b>\n"
            "/status — Complete bot status\n"
            "/balance — Paper account balance\n"
            "/positions — View open positions\n"
            "/help — All controls"
        )

    def notify_position_cycle(self, before: list[dict], after: list[dict]) -> None:
        before_by_symbol = {position["symbol"]: position for position in before}
        after_by_symbol = {position["symbol"]: position for position in after}
        for symbol in sorted(after_by_symbol.keys() - before_by_symbol.keys()):
            position = after_by_symbol[symbol]
            direction_emoji = "🟢" if position["direction"] == "LONG" else "🔴"
            self.broadcast(
                f"{direction_emoji} <b>NEW PAPER POSITION</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"🪙 <b>{escape(symbol)}</b> · <code>{escape(position['direction'])}</code>\n"
                f"🎯 Entry: <code>${position['entry_price']:.8g}</code>\n"
                f"💵 Size: <code>${position['notional']:.2f}</code>\n"
                f"🛑 Stop: <code>${position['stop_price']:.8g}</code>\n"
                f"🏁 Target: <code>${position['target_price']:.8g}</code>"
            )
        for symbol in sorted(before_by_symbol.keys() - after_by_symbol.keys()):
            self.broadcast(
                "✅ <b>PAPER POSITION CLOSED</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"🪙 <b>{escape(symbol)}</b>\n"
                "📒 Final result recorded in the paper ledger"
            )
        # Preserve the original bot behavior: report every still-open position
        # after its hourly management cycle, not only on entry/exit.
        if after:
            self.broadcast(self._positions_text({"positions": after}))

    @staticmethod
    def _status_text(snapshot: dict) -> str:
        def value(key, default="unknown"): return snapshot.get(key, default)
        running = bool(value("scheduler_running", False))
        entries = bool(value("entries_allowed", False))
        telegram = str(value("telegram_status", "unknown"))
        qualified_long = ", ".join(value("qualified_long_assets", value("qualified_assets", []))) or "none"
        qualified_short = ", ".join(value("qualified_short_assets", [])) or "none"
        watch_assets = ", ".join(value("watch_assets", [])) or "none"
        return "\n".join([
            "📊 <b>V5 STATUS REPORT</b>",
            "━━━━━━━━━━━━━━━━━━",
            f"⚙️ Mode: <code>{escape(str(value('execution_mode')))}</code>",
            f"{'🟢' if running else '🔴'} Scheduler: <b>{'RUNNING' if running else 'STOPPED'}</b>",
            f"{'✅' if entries else '⏸️'} New entries: <b>{'ENABLED' if entries else 'PAUSED'}</b>",
            f"📈 LONG: <code>{escape(qualified_long)}</code>",
            f"📉 SHORT: <code>{escape(qualified_short)}</code>",
            f"👀 WATCH: <code>{escape(watch_assets)}</code>",
            f"📡 Qualified feed: <code>{escape(str(value('qualified_market_status', 'starting')))}</code>",
            f"📍 Open positions: <b>{value('open_positions', 0)}</b>",
            "",
            "💰 <b>Paper Portfolio</b>",
            f"Equity: <code>${float(value('equity', 0)):.2f}</code>",
            f"Realized P&amp;L: <code>${float(value('realized_pnl', 0)):+.2f}</code>",
            f"Unrealized P&amp;L: <code>${float(value('unrealized_pnl', 0)):+.2f}</code>",
            f"Fees: <code>${float(value('fees', 0)):.2f}</code>",
            "",
            "⏰ <b>Schedule</b>",
            f"Hourly: <code>{escape(str(value('next_hourly_update')))}</code>",
            f"Daily: <code>{escape(str(value('next_daily_research')))}</code>",
            "",
            "🛡️ <b>Services</b>",
            f"Telegram: <b>{escape(telegram.upper())}</b>",
            f"Position monitor: <b>{escape(str(value('position_monitor_status', 'starting')).upper())}</b>",
            f"Monitor heartbeat: <code>{escape(str(value('position_monitor_last_heartbeat', 'pending')))}</code>",
            f"Web3 read-only: <code>{escape(str(value('web3_read_only_status')))}</code>",
            f"Live signing: <b>{escape(str(value('live_signing_status', 'disabled')).upper())}</b>",
        ])

    @staticmethod
    def _positions_text(snapshot: dict) -> str:
        positions = snapshot.get("positions", [])
        if not positions:
            return "📭 <b>NO OPEN POSITIONS</b>\nThe paper account is currently flat."
        lines = ["📍 <b>OPEN POSITIONS</b>", "━━━━━━━━━━━━━━━━━━"]
        for position in positions:
            pnl = float(position["unrealized_pnl"])
            emoji = "🟢" if position["direction"] == "LONG" else "🔴"
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            lines.extend([
                f"{emoji} <b>{escape(str(position['symbol']))} {escape(str(position['direction']))}</b>",
                f"Entry: <code>${position['entry_price']:.8g}</code> → Current: <code>${position['current_price']:.8g}</code>",
                f"{pnl_emoji} Unrealized P&amp;L: <code>${pnl:+.2f}</code>",
                f"🛑 SL: <code>${position['stop_price']:.8g}</code> · 🏁 TP: <code>${position['target_price']:.8g}</code>",
                "",
            ])
        return "\n".join(lines)

    @staticmethod
    def _balance_text(snapshot: dict) -> str:
        return "\n".join([
            "💰 <b>PAPER ACCOUNT</b>",
            "━━━━━━━━━━━━━━━━━━",
            f"🏦 Equity: <code>${float(snapshot.get('equity', 0)):.2f}</code>",
            f"✅ Realized P&amp;L: <code>${float(snapshot.get('realized_pnl', 0)):+.2f}</code>",
            f"📊 Unrealized P&amp;L: <code>${float(snapshot.get('unrealized_pnl', 0)):+.2f}</code>",
            f"🧾 Fees: <code>${float(snapshot.get('fees', 0)):.2f}</code>",
        ])

    @staticmethod
    def _help_text() -> str:
        return "\n".join([
            "🤖 <b>V5 COMMAND CENTER</b>",
            "━━━━━━━━━━━━━━━━━━",
            "📊 <b>Monitoring</b>",
            "/status — Complete status report",
            "/balance — Paper account balance",
            "/positions — All positions",
            "",
            "🎛️ <b>Entry Controls</b>",
            "/pause_entries — Pause new entries",
            "/resume_entries — Safely resume entries",
            "",
            "🛑 <b>Shutdown Controls</b>",
            "/shutdown_graceful — Preserve positions and stop",
            "/shutdown_close — Close positions and stop",
            "/shutdown_emergency — Emergency flatten",
            "",
            "/help — Show this command center",
        ])

    def handle_message(self, chat_id: str, text: str) -> bool:
        chat_id, text = str(chat_id), text.strip()
        if chat_id not in self.authorized:
            self.on_event(f"Telegram command rejected from unauthorized chat ID {chat_id}")
            return False
        self.on_event(f"Telegram command received from authorized chat ID {chat_id}: {text.split()[0] if text else '<empty>'}")
        confirmations = {"CONFIRM CLOSE": ShutdownMode.CLOSE_POSITIONS,
                         "CONFIRM EMERGENCY": ShutdownMode.EMERGENCY}
        if text in confirmations and self._pending_confirmation.get(chat_id) is confirmations[text]:
            mode = self._pending_confirmation.pop(chat_id)
            self.controller.request_shutdown(mode, f"telegram:{chat_id}")
            self._send(chat_id, f"✅ <b>{mode.name} SHUTDOWN ACCEPTED</b>\nThe shared shutdown controller is now handling the request.")
            return True
        if text == "/shutdown_close":
            self._pending_confirmation[chat_id] = ShutdownMode.CLOSE_POSITIONS
            self._send(chat_id, "⚠️ <b>CONFIRM CLOSE-POSITIONS SHUTDOWN</b>\n\nThis will close every paper position.\nReply exactly:\n<code>CONFIRM CLOSE</code>")
        elif text == "/shutdown_emergency":
            self._pending_confirmation[chat_id] = ShutdownMode.EMERGENCY
            self._send(chat_id, "🚨 <b>CONFIRM EMERGENCY SHUTDOWN</b>\n\nThis will immediately attempt to flatten the account.\nReply exactly:\n<code>CONFIRM EMERGENCY</code>")
        elif text == "/shutdown_graceful":
            self.controller.request_shutdown(ShutdownMode.GRACEFUL, f"telegram:{chat_id}")
            self._send(chat_id, "🛡️ <b>GRACEFUL SHUTDOWN ACCEPTED</b>\nOpen positions and protective state will be preserved.")
        elif text == "/pause_entries":
            self.controller.block_new_entries(); self._send(chat_id, "⏸️ <b>NEW ENTRIES PAUSED</b>\nOpen-position management remains active.")
        elif text == "/resume_entries":
            before = self.controller.entries_allowed()
            self.controller.enable_new_entries()
            after = self.controller.entries_allowed()
            self._send(chat_id, "▶️ <b>NEW ENTRIES RESUMED</b>\nAll safety gates passed." if after else "🚫 <b>RESUME BLOCKED</b>\nShutdown, reconciliation, or execution-mode safety prevents new entries.")
        elif text == "/status":
            self._send(chat_id, self._status_text(self.controller.runtime_snapshot()))
        elif text == "/positions":
            self._send(chat_id, self._positions_text(self.controller.runtime_snapshot()))
        elif text == "/balance":
            self._send(chat_id, self._balance_text(self.controller.runtime_snapshot()))
        elif text == "/help":
            self._send(chat_id, self._help_text())
        else:
            return False
        return True

    def _poll(self) -> None:
        self._started.set()
        while not self._stop.is_set():
            try:
                for update in self.transport.get_updates_checked(self._offset, timeout=10):
                    self._offset = max(self._offset, int(update["update_id"]) + 1)
                    message = update.get("message", {})
                    self.handle_message(str(message.get("chat", {}).get("id", "")), str(message.get("text", "")))
                if self._consecutive_failures:
                    self.on_event("Telegram polling recovered")
                self._consecutive_failures = 0
                self._set_health("connected")
            except Exception as exc:
                self._consecutive_failures += 1
                safe_error = self._safe_error(exc)
                self._set_health("degraded", safe_error)
                if isinstance(exc, (requests.Timeout, requests.ConnectionError)) and self._consecutive_failures < 3:
                    self.on_event(f"Telegram polling transient failure: {safe_error}")
                else:
                    self.on_error(f"Telegram polling failure #{self._consecutive_failures}: {safe_error}")
                delay = min(60.0, 2.0 ** min(self._consecutive_failures, 5)) + random.uniform(0.0, 1.0)
                self._stop.wait(delay)
