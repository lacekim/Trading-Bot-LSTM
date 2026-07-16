"""Explicit, auditable shutdown policies for paper and future live execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable

from trading_bot_v4.execution.order_manager import OrderManager


class ShutdownMode(str, Enum):
    GRACEFUL = "GRACEFUL"
    CLOSE_POSITIONS = "CLOSE_POSITIONS"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class ShutdownReport:
    mode: ShutdownMode
    entries_blocked: bool
    pending_orders_cancelled: int
    positions_closed: int
    account_flat: bool
    manual_intervention_required: bool


class ShutdownCoordinator:
    def __init__(self, orders: OrderManager, alert: Callable[[str], None] | None = None):
        self.orders = orders
        self.alert = alert or (lambda _message: None)

    def execute(self, mode: ShutdownMode | str) -> ShutdownReport:
        selected = ShutdownMode(str(mode).upper())
        self.orders.set_new_entries(False)
        cancelled = closed = 0

        if selected is ShutdownMode.GRACEFUL:
            # Positions, stops and targets remain in persistent storage.
            self.orders.save_state()
        elif selected is ShutdownMode.CLOSE_POSITIONS:
            cancelled = self.orders.cancel_pending_entries()
            closed = self.orders.close_all_positions("shutdown_close_positions")
            self.orders.save_state()
        else:
            self.alert("EMERGENCY shutdown started: blocking entries and flattening the paper account")
            cancelled = self.orders.cancel_pending_entries()
            # Paper fills are synchronous. The loop mirrors the contract that a
            # future live adapter must keep reconciling until flat.
            for _ in range(3):
                if self.orders.is_flat():
                    break
                closed += self.orders.close_all_positions("emergency_market_close")
                self.orders.save_state()
                if not self.orders.is_flat():
                    time.sleep(0.1)
            if not self.orders.is_flat():
                self.alert("EMERGENCY shutdown requires manual intervention: account is not flat")

        flat = self.orders.is_flat()
        return ShutdownReport(selected, True, cancelled, closed, flat, not flat and selected is not ShutdownMode.GRACEFUL)
