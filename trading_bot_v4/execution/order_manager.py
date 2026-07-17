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
from collections import Counter

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
    deduplicated_signals: int = 0
    rejection_reasons: str = "none"
    risk_status: str = "normal"


class OrderManager:
    """Simulate fills and manage one persistent position per symbol."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or Config.PAPER_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, timeout=30.0)
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
        CREATE TABLE IF NOT EXISTS runtime_state (
          key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_risk (
          trading_date TEXT PRIMARY KEY, start_equity REAL NOT NULL,
          realized_pnl REAL NOT NULL DEFAULT 0, trades_opened INTEGER NOT NULL DEFAULT 0,
          failed_orders INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS monitor_heartbeats (
          id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
          status TEXT NOT NULL, open_positions INTEGER NOT NULL,
          symbols_checked INTEGER NOT NULL, error TEXT
        );
        CREATE TABLE IF NOT EXISTS position_exit_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
          position_id TEXT NOT NULL, symbol TEXT NOT NULL, reason TEXT NOT NULL,
          observed_price REAL NOT NULL, accounting_price REAL NOT NULL,
          observed_at TEXT NOT NULL, source TEXT NOT NULL
        );
        """)
        order_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(orders)")}
        if "order_kind" not in order_columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN order_kind TEXT NOT NULL DEFAULT 'ENTRY'")
        if "reduce_only" not in order_columns:
            self.connection.execute("ALTER TABLE orders ADD COLUMN reduce_only INTEGER NOT NULL DEFAULT 0")
        self.connection.execute(
            "INSERT OR IGNORE INTO account(id,starting_balance,cash,updated_at) VALUES(1,?,?,?)",
            (Config.PAPER_STARTING_BALANCE, Config.PAPER_STARTING_BALANCE, _now()),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO runtime_state VALUES('allow_new_entries','true',?)", (_now(),)
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

    def set_new_entries(self, allowed: bool) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO runtime_state VALUES('allow_new_entries',?,?)",
            ("true" if allowed else "false", _now()),
        )
        self.connection.commit()

    def new_entries_allowed(self) -> bool:
        row = self.connection.execute(
            "SELECT value FROM runtime_state WHERE key='allow_new_entries'"
        ).fetchone()
        return bool(row and row["value"] == "true")

    def _set_state(self, key: str, value: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO runtime_state VALUES(?,?,?)", (key, value, _now()))

    def _get_state(self, key: str, default: str = "") -> str:
        row = self.connection.execute("SELECT value FROM runtime_state WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def record_shutdown(self, mode: str, completed: bool, error: str | None = None) -> None:
        self._set_state("shutdown_mode", mode)
        self._set_state("shutdown_timestamp", _now())
        self._set_state("previous_shutdown_clean", "true" if completed else "false")
        self._set_state("shutdown_error", error or "")
        self.connection.commit()

    def restore_and_reconcile(self, execution_mode: str, whitelist: list[str] | None = None) -> dict[str, Any]:
        self.set_new_entries(False)
        previous_clean = self._get_state("previous_shutdown_clean", "true") == "true"
        # A running process is deliberately marked unclean. Only the shutdown
        # coordinator may mark it clean again, so power loss is detectable.
        self._set_state("previous_shutdown_clean", "false")
        self._set_state("shutdown_mode", "RUNNING")
        self._set_state("execution_mode", execution_mode)
        if whitelist is not None:
            self._set_state("current_whitelist", ",".join(whitelist))
        self.connection.commit()
        summary = self.sync()
        return {"previous_shutdown_clean": previous_clean, "balance": self._account()["cash"],
                "open_positions": summary.open_positions, "pending_orders": self.pending_order_count(),
                "equity": summary.equity}

    def pending_order_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM orders WHERE status='PENDING'").fetchone()[0])

    def open_position_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])

    def position_snapshots(self) -> list[dict[str, Any]]:
        fields = ("symbol", "direction", "entry_price", "current_price", "unrealized_pnl",
                  "stop_price", "target_price", "quantity", "notional", "leverage")
        return [{field: row[field] for field in fields} for row in self._positions()]

    def record_monitor_heartbeat(self, status: str, symbols_checked: int = 0, error: str = "") -> None:
        self.connection.execute(
            "INSERT INTO monitor_heartbeats(timestamp,status,open_positions,symbols_checked,error) VALUES(?,?,?,?,?)",
            (_now(), status, self.open_position_count(), int(symbols_checked), error[:1000]),
        )
        self._set_state("position_monitor_status", status)
        self._set_state("position_monitor_last_heartbeat", _now())
        self._set_state("position_monitor_last_error", error[:1000])
        self.connection.commit()

    def monitor_market_price(self, symbol: str, observed_price: float, observed_at: str,
                             source: str = "GMX_1M") -> dict[str, Any] | None:
        """Update one paper position and execute an immediate protective exit."""
        if observed_price <= 0:
            raise ValueError(f"invalid observed price for {symbol}: {observed_price}")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            position = self.connection.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
            if position is None:
                self.connection.commit()
                return None
            pnl = (observed_price - float(position["entry_price"])) * float(position["quantity"])
            if position["direction"] == "SHORT":
                pnl *= -1
            self.connection.execute(
                "UPDATE positions SET current_price=?,unrealized_pnl=? WHERE position_id=?",
                (observed_price, pnl, position["position_id"]),
            )
            hit_stop = observed_price <= position["stop_price"] if position["direction"] == "LONG" else observed_price >= position["stop_price"]
            hit_target = observed_price >= position["target_price"] if position["direction"] == "LONG" else observed_price <= position["target_price"]
            if not hit_stop and not hit_target:
                self.connection.commit()
                return None
            reason = "stop_loss" if hit_stop else "take_profit"
            accounting_price = float(position["stop_price"] if hit_stop else position["target_price"])
            monitor_id = f"monitor_{hashlib.sha256(f'{position['position_id']}|{observed_at}|{reason}'.encode()).hexdigest()[:16]}"
            self._close_position(position, accounting_price, reason, monitor_id)
            self.connection.execute(
                """INSERT INTO position_exit_events(
                    timestamp,position_id,symbol,reason,observed_price,accounting_price,observed_at,source
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (_now(), position["position_id"], symbol, reason, observed_price, accounting_price, observed_at, source),
            )
            self.save_state()
            self.connection.commit()
            return {
                "position_id": position["position_id"], "symbol": symbol,
                "direction": position["direction"], "reason": reason,
                "observed_price": observed_price, "accounting_price": accounting_price,
                "observed_at": observed_at, "source": source,
            }
        except Exception:
            self.connection.rollback()
            raise

    def update_whitelist(self, whitelist: list[str]) -> None:
        self._set_state("current_whitelist", ",".join(whitelist))
        self.connection.commit()

    def _daily_risk(self) -> sqlite3.Row:
        trading_date = datetime.now(timezone.utc).date().isoformat()
        equity, _ = self._equity()
        self.connection.execute(
            "INSERT OR IGNORE INTO daily_risk(trading_date,start_equity,updated_at) VALUES(?,?,?)",
            (trading_date, equity, _now()),
        )
        return self.connection.execute("SELECT * FROM daily_risk WHERE trading_date=?", (trading_date,)).fetchone()

    def _consecutive_losses(self) -> int:
        rows = self.connection.execute("SELECT net_pnl FROM closed_trades ORDER BY exit_time DESC").fetchall()
        losses = 0
        for row in rows:
            if float(row["net_pnl"]) >= 0:
                break
            losses += 1
        return losses

    def _drawdown_pct(self, equity: float) -> float:
        row = self.connection.execute("SELECT MAX(equity) AS peak FROM equity_history").fetchone()
        peak = max(float(row["peak"] or equity), equity)
        return ((peak - equity) / peak) * 100.0 if peak else 0.0

    def _risk_block_reason(self) -> str | None:
        daily = self._daily_risk()
        equity, _ = self._equity()
        if int(daily["trades_opened"]) >= Config.PAPER_MAX_TRADES_PER_DAY:
            return "maximum trades per day reached"
        if float(daily["realized_pnl"]) <= -(float(daily["start_equity"]) * Config.PAPER_MAX_DAILY_LOSS_PCT / 100.0):
            return "daily realized-loss limit reached"
        equity_loss = ((float(daily["start_equity"]) - equity) / float(daily["start_equity"])) * 100.0
        if equity_loss >= Config.PAPER_MAX_EQUITY_LOSS_PCT:
            return "daily equity-loss limit reached"
        if self._drawdown_pct(equity) >= Config.PAPER_MAX_DRAWDOWN_PCT:
            return "maximum account drawdown reached"
        if self._consecutive_losses() >= Config.PAPER_MAX_CONSECUTIVE_LOSSES:
            return "maximum consecutive losses reached"
        if int(daily["failed_orders"]) >= Config.PAPER_MAX_FAILED_ORDERS:
            return "maximum failed orders reached"
        return None

    def risk_status(self) -> str:
        return self._get_state("risk_shutdown", "normal") or "normal"

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        units = {"m": 60, "h": 3600, "d": 86400}
        text = str(timeframe).lower()
        if len(text) < 2 or text[-1] not in units:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        return int(text[:-1]) * units[text[-1]]

    def _validate_signal_candle(self, row: Any) -> str | None:
        required = ("open", "high", "low", "close")
        if not all(column in row.index for column in required):
            return None  # Compatibility with older persisted/test signal frames.
        values = {column: float(row[column]) for column in required}
        if any(not pd.notna(value) or value <= 0 for value in values.values()):
            return "invalid OHLC value"
        if values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]):
            return "inconsistent OHLC range"
        if abs(float(row["price"]) - values["close"]) > max(values["close"] * 1e-9, 1e-12):
            return "signal price does not match candle close"
        timestamp = pd.Timestamp(row["timestamp"])
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        seconds = self._timeframe_seconds(str(row["timeframe"]))
        if "candle_gap_seconds" in row.index and float(row["candle_gap_seconds"]) > seconds * 1.5:
            return "market-data gap before signal candle"
        age = (pd.Timestamp.now(tz="UTC") - timestamp).total_seconds()
        if age < seconds:
            return "candle is not closed"
        if age > seconds * Config.PAPER_MAX_CANDLE_AGE_MULTIPLIER:
            return "stale candle"
        return None

    def _cooldown_reason(self, row: Any) -> str | None:
        previous = self.connection.execute(
            "SELECT candle_timestamp FROM signals WHERE symbol=? AND status='ACCEPTED' ORDER BY candle_timestamp DESC LIMIT 1",
            (str(row["symbol"]),),
        ).fetchone()
        if not previous:
            return None
        current = pd.Timestamp(row["timestamp"])
        prior = pd.Timestamp(previous["candle_timestamp"])
        elapsed = (current - prior).total_seconds()
        minimum = self._timeframe_seconds(str(row["timeframe"])) * Config.PAPER_MIN_BARS_BETWEEN_TRADES
        return "symbol entry cooldown active" if elapsed < minimum else None

    def cancel_pending_entries(self) -> int:
        cursor = self.connection.execute(
            "UPDATE orders SET status='CANCELLED' WHERE status='PENDING' AND order_kind='ENTRY'"
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def close_all_positions(self, reason: str) -> int:
        positions = self._positions()
        for position in positions:
            shutdown_id = f"shutdown_{hashlib.sha256(f'{position['position_id']}|{_now()}'.encode()).hexdigest()[:16]}"
            self._close_position(position, float(position["current_price"]), reason, shutdown_id)
            self.connection.execute(
                "UPDATE orders SET status='CANCELLED' WHERE symbol=? AND status='PENDING' AND order_kind='EXIT'",
                (position["symbol"],),
            )
        self.connection.commit()
        return len(positions)

    def is_flat(self) -> bool:
        return not bool(self._positions())

    def save_state(self) -> None:
        equity, unrealized = self._equity()
        account = self._account()
        self.connection.execute(
            "INSERT OR REPLACE INTO equity_history VALUES(?,?,?,?,?)",
            (_now(), equity, account["cash"], account["realized_pnl"], unrealized),
        )
        self.connection.commit()

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
        self._daily_risk()
        slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
        exit_price = price * (1 - slip if position["direction"] == "LONG" else 1 + slip)
        signed_move = exit_price - position["entry_price"]
        if position["direction"] == "SHORT":
            signed_move *= -1
        gross = signed_move * position["quantity"]
        held_days = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(position["entry_time"])).total_seconds() / 86400.0)
        carrying_cost = position["notional"] * held_days * (
            Config.PAPER_FUNDING_BPS_PER_DAY + Config.PAPER_BORROWING_BPS_PER_DAY
        ) / 10000.0
        exit_fee = position["notional"] * Config.PAPER_FEE_BPS / 10000.0 + carrying_cost
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
        trading_date = datetime.now(timezone.utc).date().isoformat()
        self.connection.execute(
            "UPDATE daily_risk SET realized_pnl=realized_pnl+?, updated_at=? WHERE trading_date=?",
            (net, _now(), trading_date),
        )

    def _open_position(self, row: Any, signal_id: str) -> bool:
        if not self.new_entries_allowed():
            self._reject(row, signal_id, "new entries are blocked")
            return False
        cooldown_reason = self._cooldown_reason(row)
        if cooldown_reason:
            self._reject(row, signal_id, cooldown_reason)
            return False
        risk_reason = self._risk_block_reason()
        if risk_reason:
            self._set_state("risk_shutdown", risk_reason)
            self.set_new_entries(False)
            self._reject(row, signal_id, risk_reason)
            return False
        self._set_state("risk_shutdown", "normal")
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
        slip = (Config.PAPER_SLIPPAGE_BPS + Config.PAPER_PRICE_IMPACT_BPS) / 10000.0
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
        self.connection.execute("""INSERT INTO orders(
            order_id,signal_id,symbol,side,status,expected_price,fill_price,notional,fee,created_at,order_kind,reduce_only
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (order_id, signal_id, str(row["symbol"]), direction, "FILLED", price, fill, notional, fee, _now(), "ENTRY", 0))
        self.connection.execute("INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (position_id, str(row["symbol"]), direction, _now(), fill, quantity, notional, collateral,
             leverage, stop, target, fee, signal_id, fill, 0.0))
        self.connection.execute("UPDATE account SET cash=cash-?-?, fees=fees+?, updated_at=? WHERE id=1",
                                (collateral, fee, fee, _now()))
        trading_date = datetime.now(timezone.utc).date().isoformat()
        self.connection.execute(
            "UPDATE daily_risk SET trades_opened=trades_opened+1, updated_at=? WHERE trading_date=?",
            (_now(), trading_date),
        )
        return True

    def process_signals(self, signals: pd.DataFrame) -> PaperCycleSummary:
        """Process only each symbol's newest closed-candle prediction."""
        latest = signals.sort_values("timestamp").groupby("symbol", as_index=False).tail(1) if not signals.empty else signals
        opened = closed = rejected = generated = 0
        deduplicated = 0
        rejection_reasons: Counter[str] = Counter()
        for _, row in latest.iterrows():
            self._set_state(f"last_processed_candle:{row['symbol']}", str(row["timestamp"]))
            symbol, price = str(row["symbol"]), float(row["price"])
            candle_error = self._validate_signal_candle(row)
            sid = self.signal_id(row)
            if candle_error:
                if row["model_direction"] in {"LONG", "SHORT"} and not self.connection.execute(
                    "SELECT 1 FROM signals WHERE signal_id=?", (sid,)
                ).fetchone():
                    self._reject(row, sid, candle_error); rejected += 1
                    rejection_reasons[candle_error] += 1
                continue
            position = self.connection.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
            if position:
                pnl = (price - position["entry_price"]) * position["quantity"]
                if position["direction"] == "SHORT": pnl *= -1
                self.connection.execute("UPDATE positions SET current_price=?, unrealized_pnl=? WHERE symbol=?", (price, pnl, symbol))
                candle_high = float(row["high"]) if "high" in row.index else price
                candle_low = float(row["low"]) if "low" in row.index else price
                candle_open = float(row["open"]) if "open" in row.index else price
                hit_stop = candle_low <= position["stop_price"] if position["direction"] == "LONG" else candle_high >= position["stop_price"]
                hit_target = candle_high >= position["target_price"] if position["direction"] == "LONG" else candle_low <= position["target_price"]
                reverse = row["model_direction"] in {"LONG", "SHORT"} and row["model_direction"] != position["direction"]
                if hit_stop or hit_target or reverse:
                    stop_first = Config.PAPER_STOP_TARGET_PRIORITY == "STOP_FIRST"
                    use_stop = hit_stop and (not hit_target or stop_first)
                    reason = "stop_loss" if use_stop else "take_profit" if hit_target else "signal_reversal"
                    if reason == "stop_loss":
                        exit_reference = min(candle_open, position["stop_price"]) if position["direction"] == "LONG" else max(candle_open, position["stop_price"])
                    elif reason == "take_profit":
                        exit_reference = max(candle_open, position["target_price"]) if position["direction"] == "LONG" else min(candle_open, position["target_price"])
                    else:
                        exit_reference = price
                    self._close_position(position, exit_reference, reason, self.signal_id(row)); closed += 1
                    position = None
            if row["model_direction"] not in {"LONG", "SHORT"}:
                continue
            generated += 1
            if self.connection.execute("SELECT 1 FROM signals WHERE signal_id=?", (sid,)).fetchone():
                deduplicated += 1
                continue
            if position:
                self._reject(row, sid, "position already open"); rejected += 1; rejection_reasons["position already open"] += 1
            elif self._open_position(row, sid):
                opened += 1
            else:
                rejected += 1
                reason_row = self.connection.execute("SELECT reason FROM signals WHERE signal_id=?", (sid,)).fetchone()
                rejection_reasons[str(reason_row["reason"] if reason_row else "unknown")] += 1
        equity, unrealized = self._equity()
        account = self._account()
        self.connection.execute("INSERT OR REPLACE INTO equity_history VALUES(?,?,?,?,?)",
                                (_now(), equity, account["cash"], account["realized_pnl"], unrealized))
        self.connection.commit()
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(rejection_reasons.items())) or "none"
        return PaperCycleSummary(len(latest), generated, rejected, opened, closed, len(self._positions()),
                                 equity, float(account["realized_pnl"]), unrealized, float(account["fees"]),
                                 deduplicated, reasons, self.risk_status())

    def sync(self) -> PaperCycleSummary:
        equity, unrealized = self._equity(); account = self._account()
        return PaperCycleSummary(0, 0, 0, 0, 0, len(self._positions()), equity,
                                 float(account["realized_pnl"]), unrealized, float(account["fees"]))
