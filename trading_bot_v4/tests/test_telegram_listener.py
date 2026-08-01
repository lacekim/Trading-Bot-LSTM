import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_bot_v4.research.scheduler import _start_telegram_listener
from trading_bot_v4.shutdown_controller import ShutdownController, ShutdownMode
from trading_bot_v4.telegram.control_listener import TelegramControlListener
from trading_bot_v4.telegram.notifier import V4TelegramNotifier


class FakeTransport:
    def __init__(self):
        self.verified = False
        self.polling = threading.Event()
        self.release = threading.Event()
        self.sent = []

    def verify_connection(self):
        self.verified = True
        return {"username": "test_bot"}

    def get_updates_checked(self, offset, timeout=10):
        self.polling.set()
        self.release.wait(0.02)
        return []

    def send_to(self, chat_id, text):
        self.sent.append((chat_id, text))


class FailOnceTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.send_attempts = 0

    def send_to(self, chat_id, text):
        self.send_attempts += 1
        if self.send_attempts == 1:
            raise ConnectionError("temporary Telegram outage")
        super().send_to(chat_id, text)


class UpdateTransport(FakeTransport):
    def __init__(self, updates):
        super().__init__()
        self.updates = updates

    def get_updates_checked(self, offset, timeout=10):
        self.polling.set()
        self.release.wait(0.02)
        return [item for item in self.updates if int(item["update_id"]) >= offset]


class TelegramListenerTests(unittest.TestCase):
    def configured_controller(self):
        controller = ShutdownController()
        controller.configure_entry_guard(True, True)
        controller.enable_new_entries()
        controller.update_runtime_snapshot(
            execution_mode="PAPER", scheduler_running=True, entries_allowed=True,
            qualified_assets=["ENA", "LDO"], qualified_long_assets=["ENA", "LDO"],
            qualified_short_assets=["CHZ"], watch_assets=["EIGEN", "PUMP"],
            open_positions=0, equity=10000,
            realized_pnl=0, unrealized_pnl=0, fees=0,
            next_hourly_update="next-hour", next_daily_research="next-day",
            web3_read_only_status="not configured", live_signing_status="disabled", positions=[],
        )
        return controller

    def test_status_includes_watch_assets_separately(self):
        text = TelegramControlListener._status_text(self.configured_controller().runtime_snapshot())
        self.assertIn("WATCH", text)
        self.assertIn("EIGEN, PUMP", text)

    def test_status_distinguishes_watchlists_and_shows_complete_position_exits(self):
        controller = self.configured_controller()
        controller.update_runtime_snapshot(
            open_positions=1,
            positions=[{
                "symbol": "VVV", "direction": "LONG", "entry_price": 11.61,
                "current_price": 11.85, "unrealized_pnl": 195.75,
                "stop_price": 11.52, "target_price": 15.08,
                "notional": 8000.0, "leverage": 5.0,
            }],
        )
        text = TelegramControlListener._status_text(controller.runtime_snapshot())
        self.assertIn("Qualified LONG", text)
        self.assertIn("Qualified SHORT", text)
        self.assertNotIn("Qualified LONG watchlist", text)
        self.assertNotIn("Qualified SHORT watchlist", text)
        self.assertIn("ACTIVE POSITION DETAILS", text)
        self.assertIn("VVV LONG", text)
        self.assertIn("Stop exit", text)
        self.assertIn("Target exit", text)
        self.assertIn("Notional", text)
        self.assertIn("live monitor + hourly model exits", text)

    def test_listener_thread_actually_starts_and_stops(self):
        transport = FakeTransport()
        listener = TelegramControlListener("token", {"123"}, self.configured_controller(), transport=transport)
        listener.start()
        self.assertTrue(transport.verified)
        self.assertTrue(transport.polling.wait(1))
        self.assertTrue(listener.is_running)
        transport.release.set(); listener.stop()
        self.assertFalse(listener.is_running)

    def test_failed_command_reply_is_queued_and_retried(self):
        transport = FailOnceTransport()
        listener = TelegramControlListener(
            "token", {"123"}, self.configured_controller(), transport=transport
        )
        listener.start()
        self.assertTrue(listener.handle_message("123", "/status"))
        deadline = time.monotonic() + 5
        while not transport.sent and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(transport.sent), 1)
        self.assertIn("V5 STATUS REPORT", transport.sent[0][1])
        transport.release.set()
        listener.stop()

    def test_shutdown_update_before_start_is_ignored_and_offset_persisted(self):
        stale_shutdown = {
            "update_id": 77,
            "message": {
                "date": int(time.time()) - 60,
                "chat": {"id": 123},
                "text": "/shutdown_graceful",
            },
        }
        with TemporaryDirectory() as directory:
            offset_path = Path(directory) / "telegram.offset"
            controller = self.configured_controller()
            transport = UpdateTransport([stale_shutdown])
            listener = TelegramControlListener(
                "token", {"123"}, controller, transport=transport, offset_path=offset_path
            )
            listener.start()
            deadline = time.monotonic() + 2
            while (not offset_path.exists() or offset_path.read_text() != "78") and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(controller.is_shutdown_requested())
            self.assertEqual(offset_path.read_text(), "78")
            transport.release.set()
            listener.stop()

            restarted = TelegramControlListener(
                "token", {"123"}, controller,
                transport=UpdateTransport([stale_shutdown]), offset_path=offset_path,
            )
            self.assertEqual(restarted._offset, 78)

    def test_auto_paper_start_helper_starts_listener(self):
        transport, output = FakeTransport(), []
        with patch("trading_bot_v4.research.scheduler.Config.EXECUTION_MODE", "PAPER"), \
             patch("trading_bot_v4.research.scheduler.Config.TELEGRAM_ENABLED", True), \
             patch("trading_bot_v4.research.scheduler.Config.TELEGRAM_BOT_TOKEN", "token"), \
             patch("trading_bot_v4.research.scheduler.Config.TELEGRAM_ALLOWED_CHAT_IDS", {"123"}):
            listener = _start_telegram_listener(self.configured_controller(), transport, output.append)
        self.assertTrue(listener.is_running)
        self.assertIn("Telegram listener started.", output)
        self.assertTrue(any("Operating Automatically" in text for _chat, text in transport.sent))
        transport.release.set(); listener.stop()

    def test_missing_token_and_disabled_have_clear_status(self):
        for enabled, token, expected in [
            (True, "", "Telegram: unavailable — TELEGRAM_BOT_TOKEN is missing"),
            (False, "token", "Telegram: disabled — TELEGRAM_ENABLED is false"),
        ]:
            output = []
            with patch("trading_bot_v4.research.scheduler.Config.EXECUTION_MODE", "PAPER"), \
                 patch("trading_bot_v4.research.scheduler.Config.TELEGRAM_ENABLED", enabled), \
                 patch("trading_bot_v4.research.scheduler.Config.TELEGRAM_BOT_TOKEN", token), \
                 patch("trading_bot_v4.research.scheduler.Config.TELEGRAM_ALLOWED_CHAT_IDS", {"123"}):
                listener = _start_telegram_listener(self.configured_controller(), FakeTransport(), output.append)
            self.assertIn(expected, output)
            self.assertFalse(listener.is_running)

    def test_status_authorization_pause_resume_and_graceful(self):
        controller, sent = self.configured_controller(), []
        listener = TelegramControlListener("token", {"123"}, controller, send=lambda c, t: sent.append((c, t)))
        self.assertTrue(listener.handle_message("123", "/status"))
        self.assertIn("Mode: <code>PAPER</code>", sent[-1][1])
        self.assertFalse(listener.handle_message("999", "/status"))
        listener.handle_message("123", "/pause_entries")
        self.assertFalse(controller.entries_allowed())
        listener.handle_message("123", "/resume_entries")
        self.assertTrue(controller.entries_allowed())
        listener.handle_message("123", "/shutdown_graceful")
        self.assertEqual(controller.get_requested_mode(), ShutdownMode.GRACEFUL)

    def test_legacy_balance_position_help_commands_are_restored(self):
        controller, sent = self.configured_controller(), []
        listener = TelegramControlListener("token", {"123"}, controller, send=lambda c, t: sent.append(t))
        for command, expected in [("/balance", "PAPER ACCOUNT"), ("/positions", "NO OPEN POSITIONS"),
                                  ("/help", "/balance — Paper account balance")]:
            self.assertTrue(listener.handle_message("123", command))
            self.assertIn(expected, sent[-1])

    def test_automatic_position_open_update_and_close_notifications(self):
        sent = []
        listener = TelegramControlListener("token", {"123"}, self.configured_controller(),
                                           send=lambda _c, text: sent.append(text))
        position = {"symbol": "ENA", "direction": "LONG", "entry_price": 0.25,
                    "current_price": 0.26, "unrealized_pnl": 4.0, "stop_price": 0.24,
                    "target_price": 0.28, "notional": 500.0}
        listener.notify_position_cycle([], [position])
        self.assertTrue(any("NEW PAPER POSITION" in text for text in sent))
        self.assertTrue(any("OPEN POSITIONS" in text for text in sent))
        sent.clear(); listener.notify_position_cycle([position], [])
        self.assertTrue(any("PAPER POSITION CLOSED" in text for text in sent))

    def test_both_destructive_modes_require_confirmation(self):
        for command, wrong, confirmation, expected in [
            ("/shutdown_close", "confirm close", "CONFIRM CLOSE", ShutdownMode.CLOSE_POSITIONS),
            ("/shutdown_emergency", "confirm emergency", "CONFIRM EMERGENCY", ShutdownMode.EMERGENCY),
        ]:
            controller = self.configured_controller()
            listener = TelegramControlListener("token", {"123"}, controller, send=lambda _c, _t: None)
            listener.handle_message("123", command)
            listener.handle_message("123", wrong)
            self.assertFalse(controller.is_shutdown_requested())
            listener.handle_message("123", confirmation)
            self.assertEqual(controller.get_requested_mode(), expected)

    def test_telegram_errors_redact_bot_token_and_api_url(self):
        token = "123456:super-secret-token"
        listener = TelegramControlListener(token, {"123"}, self.configured_controller(), send=lambda _c, _t: None)
        safe = listener._safe_error(Exception(f"failure at https://api.telegram.org/bot{token}/getUpdates"))
        self.assertNotIn(token, safe)
        self.assertIn("bot<redacted>", safe)

    def test_notifier_falls_back_to_plain_text_when_telegram_rejects_html(self):
        class Response:
            def __init__(self, status, description=""):
                self.status_code = status
                self._description = description
                self.text = description

            def json(self):
                return {"description": self._description}

        notifier = V4TelegramNotifier()
        notifier.token = "token"
        notifier.base_url = "https://api.telegram.org/bottoken"
        with patch(
            "trading_bot_v4.telegram.notifier.requests.post",
            side_effect=[Response(400, "Bad Request: can't parse entities"), Response(200)],
        ) as post:
            notifier.send_to("123", "<b>Status</b> <code>bad <value></code>")
        self.assertEqual(post.call_count, 2)
        fallback_payload = post.call_args_list[1].kwargs["json"]
        self.assertNotIn("parse_mode", fallback_payload)
        self.assertNotIn("<b>", fallback_payload["text"])


if __name__ == "__main__": unittest.main()
