"""Single-instance lock and Unix-domain-socket control plane."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import errno
import socket
import threading

from trading_bot_v4.shutdown_controller import ShutdownController


RUNTIME_DIR = Path(os.getenv("V4_RUNTIME_DIR", "runtime"))
SOCKET_PATH = RUNTIME_DIR / "trading_bot_v4.sock"
PID_PATH = RUNTIME_DIR / "trading_bot_v4.pid"
LOCK_PATH = RUNTIME_DIR / "trading_bot_v4.lock"


class InstanceLock:
    def __init__(self, path: Path = LOCK_PATH, pid_path: Path = PID_PATH):
        self.path, self.pid_path, self._handle = path, pid_path, None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close(); self._handle = None
            raise RuntimeError("another trading bot instance is already running") from exc
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            self.pid_path.unlink(missing_ok=True)
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close(); self._handle = None


class ControlServer:
    def __init__(self, controller: ShutdownController, path: Path = SOCKET_PATH):
        self.controller, self.path = controller, path
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._socket.bind(str(self.path)); os.chmod(self.path, 0o600); self._socket.listen(4)
        except OSError as exc:
            # Some filesystems (exFAT, network mounts) do not support AF_UNIX sockets.
            # Fall back to a socket in /tmp or to an explicit V4_SOCKET_PATH if provided.
            if exc.errno in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL):
                fallback = Path(os.getenv("V4_SOCKET_PATH", f"/tmp/trading_bot_v4_{os.getuid()}.sock"))
                try:
                    # clean up and retry on fallback path
                    fallback.parent.mkdir(parents=True, exist_ok=True)
                    fallback.unlink(missing_ok=True)
                    self.path = fallback
                    self._socket.bind(str(self.path)); os.chmod(self.path, 0o600); self._socket.listen(4)
                except Exception:
                    self._socket.close()
                    raise
            else:
                self._socket.close()
                raise
        self._socket.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, name="v5-control-socket", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try: client, _ = self._socket.accept()
            except socket.timeout: continue
            except OSError: break
            with client:
                try:
                    message = json.loads(client.recv(4096).decode())
                    if message.get("command") != "shutdown": raise ValueError("unsupported command")
                    changed = self.controller.request_shutdown(message["mode"], "ipc", message.get("reason"))
                    response = {"ok": True, "accepted": changed, "mode": self.controller.get_requested_mode().name}
                except Exception as exc: response = {"ok": False, "error": str(exc)}
                client.sendall(json.dumps(response).encode())

    def stop(self) -> None:
        self._stop.set()
        if self._socket: self._socket.close()
        if self._thread: self._thread.join(timeout=2)
        self.path.unlink(missing_ok=True)


def send_shutdown_request(mode: str, path: Path = SOCKET_PATH) -> dict:
    message = json.dumps({"command": "shutdown", "mode": mode}).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3); client.connect(str(path)); client.sendall(message)
        return json.loads(client.recv(4096).decode())
