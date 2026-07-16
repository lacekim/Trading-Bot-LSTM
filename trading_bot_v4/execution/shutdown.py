"""Explicit, auditable shutdown policies for paper and future live execution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.shutdown_controller import ShutdownController, ShutdownMode


@dataclass(frozen=True)
class ShutdownReport:
    mode: ShutdownMode
    entries_blocked: bool
    pending_orders_cancelled: int
    positions_closed: int
    account_flat: bool
    manual_intervention_required: bool


class ShutdownCoordinator:
    def __init__(self, orders: OrderManager, alert: Callable[[str], None] | None = None,
                 controller: ShutdownController | None = None, log: Callable[[str], None] | None = None):
        self.orders = orders
        self.alert = alert or (lambda _message: None)
        self.controller = controller
        self.log = log or (lambda _message: None)

    def execute(self, mode: ShutdownMode | str) -> ShutdownReport:
        selected = ShutdownMode.parse(mode)
        if selected is ShutdownMode.NONE:
            raise ValueError("NONE is not a shutdown policy")
        if self.controller:
            self.controller.mark_shutdown_started()
        self.log(f"Shutdown request received: {selected.name}")
        self.log("Blocking new entries...")
        self.orders.set_new_entries(False)
        self.orders.record_shutdown(selected.name, completed=False)
        cancelled = closed = 0

        if selected is ShutdownMode.GRACEFUL:
            self.log("Preserving open positions and protective orders...")
            # Positions, stops and targets remain in persistent storage.
            self.orders.save_state()
        elif selected is ShutdownMode.CLOSE_POSITIONS:
            self.log("Cancelling pending entry orders...")
            cancelled = self.orders.cancel_pending_entries()
            self.log(f"Open positions found: {self.orders.open_position_count()}")
            self.log("Submitting reduce-only paper closes...")
            try:
                closed = self.orders.close_all_positions("shutdown_close_positions")
            except Exception as exc:
                self.log(f"Close-positions error: {exc}")
                self.alert(f"Manual intervention required during close shutdown: {exc}")
            self.orders.save_state()
        else:
            self.log("EMERGENCY shutdown requested.")
            self.alert("EMERGENCY shutdown started: blocking entries and flattening the paper account")
            cancelled = self.orders.cancel_pending_entries()
            # Paper fills are synchronous. The loop mirrors the contract that a
            # future live adapter must keep reconciling until flat.
            for _ in range(3):
                if self.orders.is_flat():
                    break
                try:
                    closed += self.orders.close_all_positions("emergency_market_close")
                except Exception as exc:
                    self.log(f"Emergency close retry failed: {exc}")
                    self.alert(f"Emergency close retry failed: {exc}")
                self.orders.save_state()
                if not self.orders.is_flat():
                    time.sleep(0.1)
            if not self.orders.is_flat():
                self.alert("EMERGENCY shutdown requires manual intervention: account is not flat")

        flat = self.orders.is_flat()
        manual = not flat and selected is not ShutdownMode.GRACEFUL
        completed = selected is ShutdownMode.GRACEFUL or flat
        self.orders.record_shutdown(selected.name, completed=completed, error="positions remain open" if manual else None)
        self.orders.save_state()
        if self.controller:
            if manual:
                self.controller.mark_manual_intervention_required("positions remain open")
            else:
                self.controller.mark_shutdown_complete()
        self.log(f"Account flat: {flat}")
        self.log(f"Manual intervention required: {manual}")
        return ShutdownReport(selected, True, cancelled, closed, flat, manual)
