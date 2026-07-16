import tempfile
import unittest
from pathlib import Path

from trading_bot_v4.runtime_control import ControlServer, InstanceLock, send_shutdown_request
from trading_bot_v4.shutdown_controller import ShutdownController, ShutdownMode, graceful_signal_handler
from trading_bot_v4.telegram.control_listener import TelegramControlListener


class ShutdownControllerTests(unittest.TestCase):
    def test_sigint_requests_graceful_only(self):
        controller = ShutdownController()
        graceful_signal_handler(controller)(2, None)
        self.assertEqual(controller.get_requested_mode(), ShutdownMode.GRACEFUL)

    def test_priority_and_idempotence(self):
        controller = ShutdownController()
        self.assertTrue(controller.request_shutdown("graceful", "test"))
        self.assertTrue(controller.request_shutdown("close-positions", "test"))
        self.assertFalse(controller.request_shutdown("graceful", "test"))
        self.assertTrue(controller.request_shutdown("emergency", "test"))
        self.assertEqual(controller.get_requested_mode(), ShutdownMode.EMERGENCY)
        self.assertFalse(controller.entries_allowed())

    def test_cycle_refused_after_shutdown(self):
        controller = ShutdownController(); controller.request_shutdown("graceful", "test")
        with controller.active_cycle() as admitted:
            self.assertFalse(admitted)

    def test_unix_socket_request(self):
        Path("runtime").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runtime") as temp:
            path = Path(temp) / "bot.sock"
            controller = ShutdownController(); server = ControlServer(controller, path); server.start()
            try:
                response = send_shutdown_request("close-positions", path)
                self.assertTrue(response["ok"])
                self.assertEqual(controller.get_requested_mode(), ShutdownMode.CLOSE_POSITIONS)
            finally: server.stop()

    def test_second_instance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            lock_path, pid_path = Path(temp) / "bot.lock", Path(temp) / "bot.pid"
            first, second = InstanceLock(lock_path, pid_path), InstanceLock(lock_path, pid_path)
            first.acquire()
            try:
                with self.assertRaises(RuntimeError): second.acquire()
            finally: first.release()


class TelegramControlTests(unittest.TestCase):
    def setUp(self):
        self.controller = ShutdownController(); self.messages = []
        self.listener = TelegramControlListener("token", {"123"}, self.controller,
                                                lambda chat, text: self.messages.append((chat, text)))

    def test_unauthorized_user_cannot_control_bot(self):
        self.assertFalse(self.listener.handle_message("999", "/shutdown_graceful"))
        self.assertFalse(self.controller.is_shutdown_requested())

    def test_destructive_shutdown_requires_exact_confirmation(self):
        self.listener.handle_message("123", "/shutdown_close")
        self.assertFalse(self.controller.is_shutdown_requested())
        self.listener.handle_message("123", "confirm close")
        self.assertFalse(self.controller.is_shutdown_requested())
        self.listener.handle_message("123", "CONFIRM CLOSE")
        self.assertEqual(self.controller.get_requested_mode(), ShutdownMode.CLOSE_POSITIONS)


if __name__ == "__main__": unittest.main()
