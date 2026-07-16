"""V5 command handling built on the project's original Telegram transport."""

from __future__ import annotations

import threading
from typing import Callable, Protocol

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
                self.on_error(f"Telegram send error for authorized chat {chat_id}: {exc}")

    def send_startup_message(self) -> None:
        self.broadcast(
            "✅ Bot Operating Automatically\n\n"
            "📱 Available Commands:\n"
            "/status - View Status\n"
            "/balance - View Balance\n"
            "/position - View Current Position\n"
            "/help - Help"
        )

    def notify_position_cycle(self, before: list[dict], after: list[dict]) -> None:
        before_by_symbol = {position["symbol"]: position for position in before}
        after_by_symbol = {position["symbol"]: position for position in after}
        for symbol in sorted(after_by_symbol.keys() - before_by_symbol.keys()):
            position = after_by_symbol[symbol]
            self.broadcast(
                f"🟢 Paper position opened\n{symbol} {position['direction']}\n"
                f"Entry: ${position['entry_price']:.8g}\nSize: ${position['notional']:.2f}\n"
                f"Stop: ${position['stop_price']:.8g}\nTarget: ${position['target_price']:.8g}"
            )
        for symbol in sorted(before_by_symbol.keys() - after_by_symbol.keys()):
            self.broadcast(f"✅ Paper position closed\n{symbol}")
        # Preserve the original bot behavior: report every still-open position
        # after its hourly management cycle, not only on entry/exit.
        if after:
            self.broadcast(self._positions_text({"positions": after}))

    @staticmethod
    def _status_text(snapshot: dict) -> str:
        def value(key, default="unknown"): return snapshot.get(key, default)
        return "\n".join([
            "V5 Status",
            f"Execution mode: {value('execution_mode')}",
            f"Scheduler running: {value('scheduler_running', False)}",
            f"New entries: {'enabled' if value('entries_allowed', False) else 'disabled'}",
            f"Qualified assets: {', '.join(value('qualified_assets', [])) or 'none'}",
            f"Open positions: {value('open_positions', 0)}",
            f"Equity: ${float(value('equity', 0)):.2f}",
            f"Realized P&L: ${float(value('realized_pnl', 0)):.2f}",
            f"Unrealized P&L: ${float(value('unrealized_pnl', 0)):.2f}",
            f"Fees: ${float(value('fees', 0)):.2f}",
            f"Next hourly update: {value('next_hourly_update')}",
            f"Next daily research: {value('next_daily_research')}",
            f"Web3 read-only: {value('web3_read_only_status')}",
            f"Live signing: {value('live_signing_status', 'disabled')}",
        ])

    @staticmethod
    def _positions_text(snapshot: dict) -> str:
        positions = snapshot.get("positions", [])
        if not positions:
            return "Open positions: none"
        lines = ["Open positions:"]
        for position in positions:
            lines.append(
                f"{position['symbol']} {position['direction']} | entry={position['entry_price']:.8g} "
                f"current={position['current_price']:.8g} | P&L=${position['unrealized_pnl']:.2f} "
                f"SL={position['stop_price']:.8g} TP={position['target_price']:.8g}"
            )
        return "\n".join(lines)

    @staticmethod
    def _balance_text(snapshot: dict) -> str:
        return "\n".join([
            "💰 Paper Balance",
            f"Equity: ${float(snapshot.get('equity', 0)):.2f}",
            f"Realized P&L: ${float(snapshot.get('realized_pnl', 0)):.2f}",
            f"Unrealized P&L: ${float(snapshot.get('unrealized_pnl', 0)):.2f}",
            f"Fees: ${float(snapshot.get('fees', 0)):.2f}",
        ])

    @staticmethod
    def _help_text() -> str:
        return "\n".join([
            "📱 Available Commands:",
            "/status - View Status",
            "/balance - View Balance",
            "/position - View Current Position",
            "/positions - View All Positions",
            "/pause_entries - Pause New Entries",
            "/resume_entries - Resume New Entries",
            "/shutdown_graceful - Preserve Positions and Stop",
            "/shutdown_close - Close Positions and Stop",
            "/shutdown_emergency - Emergency Shutdown",
            "/help - Help",
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
            self._send(chat_id, f"{mode.name} shutdown accepted.")
            return True
        if text == "/shutdown_close":
            self._pending_confirmation[chat_id] = ShutdownMode.CLOSE_POSITIONS
            self._send(chat_id, "Confirm closing all positions and shutting down:\nCONFIRM CLOSE")
        elif text == "/shutdown_emergency":
            self._pending_confirmation[chat_id] = ShutdownMode.EMERGENCY
            self._send(chat_id, "Confirm emergency shutdown:\nCONFIRM EMERGENCY")
        elif text == "/shutdown_graceful":
            self.controller.request_shutdown(ShutdownMode.GRACEFUL, f"telegram:{chat_id}")
            self._send(chat_id, "GRACEFUL shutdown accepted.")
        elif text == "/pause_entries":
            self.controller.block_new_entries(); self._send(chat_id, "New entries paused.")
        elif text == "/resume_entries":
            before = self.controller.entries_allowed()
            self.controller.enable_new_entries()
            after = self.controller.entries_allowed()
            self._send(chat_id, "New entries resumed." if after else "Cannot resume entries while shutdown/reconciliation is active.")
        elif text == "/status":
            self._send(chat_id, self._status_text(self.controller.runtime_snapshot()))
        elif text in {"/position", "/positions"}:
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
            except Exception as exc:
                self.on_error(f"Telegram polling error: {exc}")
                self._stop.wait(2)
