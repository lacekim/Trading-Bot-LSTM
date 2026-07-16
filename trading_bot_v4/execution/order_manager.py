"""Persistent paper execution used by the V5 forward-walk scheduler.

No method in this module signs or broadcasts a transaction.  SQLite is used so
restarts cannot erase orders, positions, fills, or account history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot_v4.config_v4 import V4Config as Config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PaperCycleSummary:
    assets_scanned: int
    signals_generated: int
    signals_rejected: int
    orders_opened: int
    orders_closed: int
    open_positions: int
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float


class OrderManager:
    """Simulate fills and manage one persistent position per symbol."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or Config.PAPER_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS account (
          id INTEGER PRIMARY KEY CHECK(id=1), starting_balance REAL NOT NULL,
          cash REAL NOT NULL, realized_pnl REAL NOT NULL DEFAULT 0,
          fees REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS signals (
          signal_id TEXT PRIMARY KEY, candle_timestamp TEXT NOT NULL,
          symbol TEXT NOT NULL, direction TEXT NOT NULL, probability REAL,
          market_price REAL NOT NULL, status TEXT NOT NULL, reason TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS orders (
          order_id TEXT PRIMARY KEY, signal_id TEXT NOT NULL, symbol TEXT NOT NULL,
          side TEXT NOT NULL, status TEXT NOT NULL, expected_price REAL NOT NULL,
          fill_price REAL NOT NULL, notional REAL NOT NULL, fee REAL NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
          position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL UNIQUE,
          direction TEXT NOT NULL, entry_time TEXT NOT NULL, entry_price REAL NOT NULL,
          quantity REAL NOT NULL, notional REAL NOT NULL, collateral REAL NOT NULL,
          leverage REAL NOT NULL, stop_price REAL NOT NULL, target_price REAL NOT NULL,
          entry_fee REAL NOT NULL, signal_id TEXT NOT NULL, current_price REAL NOT NULL,
          unrealized_pnl REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS closed_trades (
          position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, direction TEXT NOT NULL,
          entry_time TEXT NOT NULL, exit_time TEXT NOT NULL, entry_price REAL NOT NULL,
          exit_price REAL NOT NULL, quantity REAL NOT NULL, gross_pnl REAL NOT NULL,
          fees REAL NOT NULL, net_pnl REAL NOT NULL, exit_reason TEXT NOT NULL,
          signal_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS equity_history (
          timestamp TEXT PRIMARY KEY, equity REAL NOT NULL, cash REAL NOT NULL,
          realized_pnl REAL NOT NULL, unrealized_pnl REAL NOT NULL
        );
        """)
        self.connection.execute(
            "INSERT OR IGNORE INTO account(id,starting_balance,cash,updated_at) VALUES(1,?,?,?)",
            (Config.PAPER_STARTING_BALANCE, Config.PAPER_STARTING_BALANCE, _now()),
        )
        self.connection.commit()

    @staticmethod
    def signal_id(row: Any) -> str:
        raw = f"{row['symbol']}|{row['timeframe']}|{row['timestamp']}|{row['model_direction']}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _account(self) -> sqlite3.Row:
        return self.connection.execute("SELECT * FROM account WHERE id=1").fetchone()

    def _positions(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM positions ORDER BY symbol").fetchall()

    def _equity(self) -> tuple[float, float]:
        unrealized = sum(float(row["unrealized_pnl"]) for row in self._positions())
        account = self._account()
        return float(account["cash"]) + sum(float(p["collateral"]) for p in self._positions()) + unrealized, unrealized

    def _reject(self, row: Any, signal_id: str, reason: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,?,?,?)",
            (signal_id, str(row["timestamp"]), str(row["symbol"]), str(row["model_direction"]),
             float(row["model_probability"]), float(row["price"]), "REJECTED", reason, _now()),
        )

    def _close_position(self, position: sqlite3.Row, price: float, reason: str, signal_id: str) -> None:
        slip = Config.PAPER_SLIPPAGE_BPS / 10000.0
        exit_price = price * (1 - slip if position["direction"] == "LONG" else 1 + slip)
        signed_move = exit_price - position["entry_price"]
        if position["direction"] == "SHORT":
            signed_move *= -1
        gross = signed_move * position["quantity"]
        exit_fee = position["notional"] * Config.PAPER_FEE_BPS / 10000.0
        net = gross - position["entry_fee"] - exit_fee
        self.connection.execute(
            "INSERT INTO closed_trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (position["position_id"], position["symbol"], position["direction"], position["entry_time"],
             _now(), position["entry_price"], exit_price, position["quantity"], gross,
             position["entry_fee"] + exit_fee, net, reason, signal_id),
        )
        self.connection.execute("DELETE FROM positions WHERE position_id=?", (position["position_id"],))
        self.connection.execute(
            "UPDATE account SET cash=cash+?+?, realized_pnl=realized_pnl+?, fees=fees+?, updated_at=? WHERE id=1",
            (position["collateral"], gross - exit_fee, net, exit_fee, _now()),
        )

    def _open_position(self, row: Any, signal_id: str) -> bool:
        equity, _ = self._equity()
        account = self._account()
        positions = self._positions()
        exposure = sum(float(p["notional"]) for p in positions)
        risk_sized_notional = equity * Config.PAPER_MAX_RISK_PER_TRADE / max(Config.PAPER_STOP_LOSS_PCT, 0.0001)
        notional = min(risk_sized_notional, equity * Config.PAPER_MAX_POSITION_PCT / 100.0,
                       equity * Config.PAPER_MAX_PORTFOLIO_EXPOSURE_PCT / 100.0 - exposure)
        leverage = max(1.0, Config.PAPER_LEVERAGE)
        collateral = notional / leverage
        cash_buffer = equity * Config.PAPER_CASH_BUFFER_PCT / 100.0
        if len(positions) >= Config.PAPER_MAX_OPEN_POSITIONS:
            self._reject(row, signal_id, "maximum open positions reached")
            return False
        if notional < Config.PAPER_MIN_ORDER_USD or float(account["cash"]) - collateral < cash_buffer:
            self._reject(row, signal_id, "insufficient exposure capacity or cash buffer")
            return False
        direction, price = str(row["model_direction"]), float(row["price"])
        slip = Config.PAPER_SLIPPAGE_BPS / 10000.0
        fill = price * (1 + slip if direction == "LONG" else 1 - slip)
        fee = notional * Config.PAPER_FEE_BPS / 10000.0
        quantity = notional / fill
        stop_factor = Config.PAPER_STOP_LOSS_PCT / 100.0
        target_factor = Config.PAPER_TAKE_PROFIT_PCT / 100.0
        stop = fill * (1 - stop_factor if direction == "LONG" else 1 + stop_factor)
        target = fill * (1 + target_factor if direction == "LONG" else 1 - target_factor)
        order_id, position_id = f"po_{signal_id}", f"pp_{signal_id}"
        self.connection.execute("INSERT INTO signals VALUES(?,?,?,?,?,?,?,?,?)",
            (signal_id, str(row["timestamp"]), str(row["symbol"]), direction, float(row["model_probability"]), price, "ACCEPTED", "", _now()))
        self.connection.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (order_id, signal_id, str(row["symbol"]), direction, "FILLED", price, fill, notional, fee, _now()))
        self.connection.execute("INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (position_id, str(row["symbol"]), direction, _now(), fill, quantity, notional, collateral,
             leverage, stop, target, fee, signal_id, fill, 0.0))
        self.connection.execute("UPDATE account SET cash=cash-?-?, fees=fees+?, updated_at=? WHERE id=1",
                                (collateral, fee, fee, _now()))
        return True

    def process_signals(self, signals: pd.DataFrame) -> PaperCycleSummary:
        """Process only each symbol's newest closed-candle prediction."""
        latest = signals.sort_values("timestamp").groupby("symbol", as_index=False).tail(1) if not signals.empty else signals
        opened = closed = rejected = generated = 0
        for _, row in latest.iterrows():
            symbol, price = str(row["symbol"]), float(row["price"])
            position = self.connection.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
            if position:
                pnl = (price - position["entry_price"]) * position["quantity"]
                if position["direction"] == "SHORT": pnl *= -1
                self.connection.execute("UPDATE positions SET current_price=?, unrealized_pnl=? WHERE symbol=?", (price, pnl, symbol))
                hit_stop = price <= position["stop_price"] if position["direction"] == "LONG" else price >= position["stop_price"]
                hit_target = price >= position["target_price"] if position["direction"] == "LONG" else price <= position["target_price"]
                reverse = row["model_direction"] in {"LONG", "SHORT"} and row["model_direction"] != position["direction"]
                if hit_stop or hit_target or reverse:
                    reason = "stop_loss" if hit_stop else "take_profit" if hit_target else "signal_reversal"
                    self._close_position(position, price, reason, self.signal_id(row)); closed += 1
                    position = None
            if row["model_direction"] not in {"LONG", "SHORT"}:
                continue
            generated += 1
            sid = self.signal_id(row)
            if self.connection.execute("SELECT 1 FROM signals WHERE signal_id=?", (sid,)).fetchone():
                continue
            if position:
                self._reject(row, sid, "position already open"); rejected += 1
            elif self._open_position(row, sid):
                opened += 1
            else:
                rejected += 1
        equity, unrealized = self._equity()
        account = self._account()
        self.connection.execute("INSERT OR REPLACE INTO equity_history VALUES(?,?,?,?,?)",
                                (_now(), equity, account["cash"], account["realized_pnl"], unrealized))
        self.connection.commit()
        return PaperCycleSummary(len(latest), generated, rejected, opened, closed, len(self._positions()),
                                 equity, float(account["realized_pnl"]), unrealized, float(account["fees"]))

    def sync(self) -> PaperCycleSummary:
        equity, unrealized = self._equity(); account = self._account()
        return PaperCycleSummary(0, 0, 0, 0, 0, len(self._positions()), equity,
                                 float(account["realized_pnl"]), unrealized, float(account["fees"]))
