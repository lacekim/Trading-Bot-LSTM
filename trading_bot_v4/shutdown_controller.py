"""Thread-safe process-wide shutdown state and cycle coordination."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
import threading
import time
from typing import Iterator


class ShutdownMode(IntEnum):
    NONE = 0
    GRACEFUL = 1
    CLOSE_POSITIONS = 2
    EMERGENCY = 3

    @classmethod
    def parse(cls, value: "ShutdownMode | str") -> "ShutdownMode":
        if isinstance(value, cls):
            return value
        return cls[str(value).strip().upper().replace("-", "_")]


@dataclass(frozen=True)
class ShutdownStatus:
    requested_mode: ShutdownMode
    requested_at: str | None
    requested_by: str | None
    reason: str | None
    shutdown_in_progress: bool
    entries_allowed: bool
    active_cycle_count: int
    completed: bool
    manual_intervention_required: bool
    error: str | None


class ShutdownController:
    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._mode = ShutdownMode.NONE
        self._requested_at: str | None = None
        self._requested_by: str | None = None
        self._reason: str | None = None
        self._in_progress = False
        self._entries_allowed = False
        self._active_cycles = 0
        self._completed = False
        self._manual = False
        self._error: str | None = None
        self._status_snapshot: dict = {}
        self._reconciliation_succeeded = False
        self._execution_permits_entries = False

    def request_shutdown(self, mode: ShutdownMode | str, requested_by: str, reason: str | None = None) -> bool:
        requested = ShutdownMode.parse(mode)
        if requested is ShutdownMode.NONE:
            return False
        with self._condition:
            changed = requested > self._mode
            if changed:
                self._mode = requested
                self._requested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._requested_by = requested_by
                self._reason = reason
            self._entries_allowed = False
            self._condition.notify_all()
            return changed

    def is_shutdown_requested(self) -> bool:
        with self._condition:
            return self._mode is not ShutdownMode.NONE

    def get_requested_mode(self) -> ShutdownMode:
        with self._condition:
            return self._mode

    def block_new_entries(self) -> None:
        with self._condition:
            self._entries_allowed = False

    def enable_new_entries(self) -> None:
        with self._condition:
            if (self._mode is ShutdownMode.NONE and not self._manual
                    and self._reconciliation_succeeded and self._execution_permits_entries):
                self._entries_allowed = True

    def configure_entry_guard(self, reconciliation_succeeded: bool, execution_permits_entries: bool) -> None:
        with self._condition:
            self._reconciliation_succeeded = reconciliation_succeeded
            self._execution_permits_entries = execution_permits_entries

    def entries_allowed(self) -> bool:
        with self._condition:
            return self._entries_allowed

    @contextmanager
    def active_cycle(self) -> Iterator[bool]:
        with self._condition:
            if self._mode is not ShutdownMode.NONE or self._active_cycles:
                admitted = False
            else:
                self._active_cycles += 1
                admitted = True
        if not admitted:
            yield False
            return
        try:
            yield True
        finally:
            with self._condition:
                self._active_cycles -= 1
                self._condition.notify_all()

    def wait_for_active_cycles(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._active_cycles:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def mark_shutdown_started(self) -> None:
        with self._condition:
            self._in_progress = True

    def mark_shutdown_complete(self) -> None:
        with self._condition:
            self._in_progress = False
            self._completed = True

    def mark_manual_intervention_required(self, error: str) -> None:
        with self._condition:
            self._manual = True
            self._error = error
            self._in_progress = False

    def update_runtime_snapshot(self, **values) -> None:
        with self._condition:
            self._status_snapshot.update(values)

    def runtime_snapshot(self) -> dict:
        with self._condition:
            return dict(self._status_snapshot)

    def status(self) -> ShutdownStatus:
        with self._condition:
            return ShutdownStatus(self._mode, self._requested_at, self._requested_by, self._reason,
                                  self._in_progress, self._entries_allowed, self._active_cycles,
                                  self._completed, self._manual, self._error)


def graceful_signal_handler(controller: ShutdownController, log=lambda _message: None):
    """Build a signal-safe callback that only records a GRACEFUL request."""
    def handler(signum, _frame):
        log(f"Signal {signum} received; requesting GRACEFUL shutdown")
        controller.request_shutdown(ShutdownMode.GRACEFUL, f"signal:{signum}")
    return handler
