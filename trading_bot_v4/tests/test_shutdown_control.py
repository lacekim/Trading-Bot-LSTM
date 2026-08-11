import errno
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        # Connects to server.path (not the requested path) because on filesystems that
        # don't support AF_UNIX (e.g. exFAT), start() transparently rebinds to a fallback
        # path -- see test_socket_bind_falls_back_for_documented_errnos below.
        Path("runtime").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="runtime") as temp:
            path = Path(temp) / "bot.sock"
            controller = ShutdownController(); server = ControlServer(controller, path); server.start()
            try:
                response = send_shutdown_request("close-positions", server.path)
                self.assertTrue(response["ok"])
                self.assertEqual(controller.get_requested_mode(), ShutdownMode.CLOSE_POSITIONS)
            finally: server.stop()

    def test_socket_bind_falls_back_for_documented_errnos(self):
        real_bind = socket.socket.bind
        calls: list = []

        def flaky_bind(sock_self, address):
            calls.append(address)
            if len(calls) == 1:
                raise OSError(errno.ENOTSUP, "Operation not supported")
            return real_bind(sock_self, address)

        Path("runtime").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            primary = Path("runtime") / "unreachable_primary.sock"
            fallback = Path(temp) / "fallback.sock"
            with patch.object(socket.socket, "bind", flaky_bind), \
                 patch.dict(os.environ, {"V4_SOCKET_PATH": str(fallback)}):
                controller = ShutdownController()
                server = ControlServer(controller, primary)
                try:
                    server.start()
                    self.assertEqual(server.path, fallback)
                    self.assertTrue(fallback.exists())
                    response = send_shutdown_request("graceful", server.path)
                    self.assertTrue(response["ok"])
                finally:
                    server.stop()
        self.assertEqual(len(calls), 2)

    def test_socket_bind_reraises_for_unrelated_errno(self):
        calls: list = []

        def failing_bind(sock_self, address):
            calls.append(address)
            raise OSError(errno.EACCES, "Permission denied")

        Path("runtime").mkdir(exist_ok=True)
        with patch.object(socket.socket, "bind", failing_bind):
            controller = ShutdownController()
            server = ControlServer(controller, Path("runtime") / "wont_bind.sock")
            with self.assertRaises(OSError):
                server.start()
            self.assertIsNone(server._thread)
        # Must not have attempted a fallback retry for an errno outside the
        # documented exFAT/network-mount set -- only the original bind call.
        self.assertEqual(len(calls), 1)

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
