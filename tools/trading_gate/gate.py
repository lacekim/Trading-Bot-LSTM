"""
trading_gate.gate -- deterministic, FAIL-CLOSED pre-trade gate.

ISOLATION: this module imports nothing from trading_bot_v4, reads no bot database and writes no bot file.
Its only side effect is its OWN audit database (path given by the caller; ":memory:" for tests).

INTERFACE
    gate = TradingGate(limits=GateLimits(...), audit_path="gate_audit.sqlite3")
    decision = gate.evaluate(proposed_order=order, signal=signal, market_snapshot=market,
                             portfolio_state=portfolio, risk_state=risk)

The result is a dict:  approved (bool), decision ("APPROVED"/"REJECTED"), reasons (list[str]), warnings (list[str]),
order_id, order_kind, checks (dict[str, bool]), audit_id, evaluated_at.

FAIL CLOSED: a missing field, a wrong type, NaN/inf, a naive timestamp, an unknown order kind, a check that raises,
an internal error, a missing expected check, or (for opening orders) an audit-log write failure all produce REJECTED.

OPENING vs REDUCING: ENTRY orders must pass every check. EXIT orders (order_kind "EXIT", reduce_only True) are only
required to be well-formed, to reduce an existing position without exceeding it, and not to be duplicates. They are
NOT blocked by a kill switch, loss limits, a stale signal, a missing stop or stale market data (warnings are recorded).

CONVENTIONS (documented because the checks depend on them)
    order["side"]      : ENTRY -> "LONG" or "SHORT" (direction opened).  EXIT -> direction of the position being reduced.
    order["order_kind"]: "ENTRY" or "EXIT".   order["reduce_only"] must be True for EXIT and False for ENTRY.
    timestamps         : timezone-aware datetimes or ISO strings with an offset / "Z".  Naive timestamps are rejected.
    signal["candle_timestamp"] : OPEN time of the signal candle. The candle is closed (usable) once now >= open + timeframe,
                          and expires once now > open + timeframe * max_signal_age_multiplier.
    portfolio["positions"] items: {symbol, direction, quantity, notional}.
    portfolio["seen_idempotency_keys"]: collection of keys already submitted (the gate also keeps its own approved keys).
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
import sqlite3
import threading
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

TF_UNITS = {"m": 60, "h": 3600, "d": 86400}

EXPECTED_ENTRY_CHECKS = (
    "schema_valid", "data_fresh", "signal_matches_order", "signal_fresh", "duplicate_order", "risk_limits",
    "portfolio_rules", "short_borrow_ok", "stop_present", "kill_switch_ok", "daily_loss_ok", "drawdown_ok",
    "loss_streak_ok", "cooldown_ok",
)
EXPECTED_EXIT_CHECKS = ("schema_valid", "market_price_valid", "duplicate_order", "reduces_position")


class _Bad(Exception):
    """A check found a concrete problem; .reason is the machine-readable reason string."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------- helpers

def _num(m: Any, key: str, *, pos: bool = False, nonneg: bool = False, optional: bool = False):
    if not isinstance(m, Mapping) or key not in m or m[key] is None:
        if optional:
            return None
        raise _Bad(f"missing_field:{key}")
    v = m[key]
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise _Bad(f"non_numeric:{key}")
    v = float(v)
    if not math.isfinite(v):
        raise _Bad(f"non_finite:{key}")
    if pos and v <= 0:
        raise _Bad(f"non_positive:{key}")
    if nonneg and v < 0:
        raise _Bad(f"negative:{key}")
    return v


def _int(m: Any, key: str, *, nonneg: bool = True) -> int:
    v = _num(m, key)
    if v != int(v):
        raise _Bad(f"non_integer:{key}")
    if nonneg and v < 0:
        raise _Bad(f"negative:{key}")
    return int(v)


def _txt(m: Any, key: str, allowed: tuple | None = None) -> str:
    if not isinstance(m, Mapping) or key not in m or m[key] is None:
        raise _Bad(f"missing_field:{key}")
    v = m[key]
    if not isinstance(v, str) or not v.strip():
        raise _Bad(f"invalid_text:{key}")
    v = v.strip()
    if allowed is not None and v not in allowed:
        raise _Bad(f"invalid_value:{key}")
    return v


def _bool(m: Any, key: str) -> bool:
    if not isinstance(m, Mapping) or key not in m or m[key] is None:
        raise _Bad(f"missing_field:{key}")
    if not isinstance(m[key], bool):
        raise _Bad(f"non_boolean:{key}")
    return m[key]


def _ts(m: Any, key: str) -> datetime:
    if not isinstance(m, Mapping) or key not in m or m[key] is None:
        raise _Bad(f"missing_field:{key}")
    v = m[key]
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            raise _Bad(f"unparseable_timestamp:{key}")
    if not isinstance(v, datetime):
        raise _Bad(f"unparseable_timestamp:{key}")
    if v.tzinfo is None or v.utcoffset() is None:
        raise _Bad(f"timestamp_ambiguous:{key}")
    return v.astimezone(timezone.utc)


def _mapping(m: Any, name: str) -> Mapping:
    if not isinstance(m, Mapping):
        raise _Bad(f"missing_or_invalid:{name}")
    return m


def tf_seconds(tf: Any) -> int:
    found = re.fullmatch(r"(\d+)([mhd])", str(tf).strip().lower())
    if not found or int(found.group(1)) <= 0:
        raise _Bad("unsupported_timeframe")
    return int(found.group(1)) * TF_UNITS[found.group(2)]


# ---------------------------------------------------------------- limits

@dataclass(frozen=True)
class GateLimits:
    """Every risk-defining number is REQUIRED (no silent defaults). Percent fields are percent (5 = 5%)."""

    max_position_pct: float          # max single-order notional as % of equity
    max_exposure_pct: float          # max total notional (existing + new) as % of equity
    max_open_positions: int
    max_leverage: float
    min_order_usd: float
    max_data_age_s: float            # market snapshot must be at most this old
    max_spread_bps: float
    max_signal_age_multiplier: float  # signal may be at most this many timeframes old
    max_price_deviation_bps: float   # order/signal price vs market price
    notional_tolerance: float        # relative tolerance for notional == quantity * price
    min_stop_pct: float              # stop distance from price, % bounds
    max_stop_pct: float
    max_daily_loss_pct: float
    max_equity_loss_pct: float
    max_drawdown_pct: float
    max_consecutive_losses: int
    max_failed_orders: int
    max_trades_per_day: int          # 0 disables
    min_bars_between_entries: int    # 0 disables
    cash_buffer_pct: float
    require_borrow_check: bool       # True for live shorting
    max_net_exposure_pct: float | None = None  # None disables

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "max_net_exposure_pct" and v is None:
                continue
            if f.name == "require_borrow_check":
                if not isinstance(v, bool):
                    raise ValueError("require_borrow_check must be bool")
                continue
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
                raise ValueError(f"limit {f.name} must be a finite number")
            if float(v) < 0:
                raise ValueError(f"limit {f.name} must be >= 0")
        for name in ("max_position_pct", "max_exposure_pct", "max_leverage", "min_order_usd", "max_data_age_s",
                     "max_signal_age_multiplier", "max_open_positions", "min_stop_pct", "max_stop_pct",
                     "max_daily_loss_pct", "max_equity_loss_pct", "max_drawdown_pct",
                     "max_consecutive_losses", "max_failed_orders"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"limit {name} must be > 0")
        if self.min_stop_pct >= self.max_stop_pct:
            raise ValueError("min_stop_pct must be below max_stop_pct")


# ---------------------------------------------------------------- the gate

class TradingGate:
    def __init__(self, limits: GateLimits, audit_path: str, clock: Callable[[], datetime] | None = None):
        if not isinstance(limits, GateLimits):
            raise TypeError("limits must be a GateLimits")
        if not audit_path:
            raise ValueError("audit_path is required (use ':memory:' for tests)")
        self.limits = limits
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(audit_path, check_same_thread=False, isolation_level=None)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
              id INTEGER PRIMARY KEY AUTOINCREMENT, evaluated_at TEXT NOT NULL, order_id TEXT,
              idempotency_key TEXT, order_kind TEXT, approved INTEGER NOT NULL, reasons TEXT NOT NULL,
              warnings TEXT NOT NULL, checks TEXT NOT NULL, inputs_json TEXT NOT NULL, inputs_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS approved_keys (
              idempotency_key TEXT PRIMARY KEY, order_id TEXT, approved_at TEXT NOT NULL);
            """
        )

    # ------------------------------------------------------------ public

    def evaluate(self, proposed_order, signal, market_snapshot, portfolio_state, risk_state) -> dict:
        now = None
        reasons: list[str] = []
        warnings: list[str] = []
        checks: dict[str, bool] = {}
        kind = None
        order_id = None
        key = None
        try:
            now = self._clock()
            if not isinstance(now, datetime) or now.tzinfo is None:
                raise _Bad("clock_not_timezone_aware")
            if isinstance(proposed_order, Mapping):
                oid = proposed_order.get("order_id")
                order_id = oid if isinstance(oid, str) else None
                k = proposed_order.get("idempotency_key")
                key = k if isinstance(k, str) else None
                kk = proposed_order.get("order_kind")
                kind = kk if isinstance(kk, str) else None
            ctx = dict(order=proposed_order, signal=signal, market=market_snapshot, portfolio=portfolio_state,
                       risk=risk_state, now=now.astimezone(timezone.utc), warnings=warnings)
            if kind == "ENTRY":
                expected = EXPECTED_ENTRY_CHECKS
                self._run(checks, reasons, "schema_valid", self._chk_schema_entry, ctx)
                self._run(checks, reasons, "data_fresh", self._chk_data_fresh, ctx)
                self._run(checks, reasons, "signal_matches_order", self._chk_signal_matches, ctx)
                self._run(checks, reasons, "signal_fresh", self._chk_signal_fresh, ctx)
                self._run(checks, reasons, "duplicate_order", self._chk_duplicate, ctx)
                self._run(checks, reasons, "risk_limits", self._chk_risk_limits, ctx)
                self._run(checks, reasons, "portfolio_rules", self._chk_portfolio_rules, ctx)
                self._run(checks, reasons, "short_borrow_ok", self._chk_borrow, ctx)
                self._run(checks, reasons, "stop_present", self._chk_stop, ctx)
                self._run(checks, reasons, "kill_switch_ok", self._chk_kill_switch, ctx)
                self._run(checks, reasons, "daily_loss_ok", self._chk_daily_loss, ctx)
                self._run(checks, reasons, "drawdown_ok", self._chk_drawdown, ctx)
                self._run(checks, reasons, "loss_streak_ok", self._chk_loss_streak, ctx)
                self._run(checks, reasons, "cooldown_ok", self._chk_cooldown, ctx)
            elif kind == "EXIT":
                expected = EXPECTED_EXIT_CHECKS
                self._run(checks, reasons, "schema_valid", self._chk_schema_exit, ctx)
                self._run(checks, reasons, "market_price_valid", self._chk_exit_market, ctx)
                self._run(checks, reasons, "duplicate_order", self._chk_duplicate, ctx)
                self._run(checks, reasons, "reduces_position", self._chk_reduces, ctx)
            else:
                expected = ()
                checks["schema_valid"] = False
                reasons.append("unknown_order_kind")
            missing = [name for name in expected if name not in checks]
            if missing:                                  # a check was dropped or skipped: never approve
                reasons.append("expected_check_missing:" + ",".join(missing))
        except Exception as exc:  # noqa: BLE001 - fail closed on anything
            reason = exc.reason if isinstance(exc, _Bad) else f"gate_internal_error:{type(exc).__name__}"
            reasons.append(reason)
            checks.setdefault("schema_valid", False)
            expected = ()
        approved = bool(checks) and all(checks.values()) and not reasons and kind in ("ENTRY", "EXIT")
        return self._finalize(now, kind, order_id, key, approved, reasons, warnings, checks,
                              (proposed_order, signal, market_snapshot, portfolio_state, risk_state))

    # ------------------------------------------------------------ plumbing

    @staticmethod
    def _run(checks: dict, reasons: list, name: str, fn: Callable, ctx: dict) -> None:
        try:
            fn(ctx)
            checks[name] = True
        except _Bad as bad:
            checks[name] = False
            reasons.append(bad.reason)
        except Exception as exc:  # noqa: BLE001
            checks[name] = False
            reasons.append(f"check_error:{name}:{type(exc).__name__}")

    def _finalize(self, now, kind, order_id, key, approved, reasons, warnings, checks, inputs) -> dict:
        evaluated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        audit_id = None
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                if approved and key is not None:
                    dup = self._db.execute("SELECT 1 FROM approved_keys WHERE idempotency_key=?", (key,)).fetchone()
                    if dup:
                        approved = False
                        reasons.append("duplicate_order")
                        checks["duplicate_order"] = False
                try:
                    payload = json.dumps(inputs, default=str, sort_keys=True)
                except Exception:  # noqa: BLE001
                    payload = "UNSERIALIZABLE_INPUTS"
                digest = hashlib.sha256(payload.encode()).hexdigest()
                cur = self._db.execute(
                    "INSERT INTO evaluations(evaluated_at,order_id,idempotency_key,order_kind,approved,reasons,warnings,"
                    "checks,inputs_json,inputs_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (evaluated_at, order_id, key, kind, int(approved), json.dumps(reasons), json.dumps(warnings),
                     json.dumps(checks), payload, digest))
                audit_id = cur.lastrowid
                if approved and key is not None:
                    self._db.execute("INSERT INTO approved_keys VALUES(?,?,?)", (key, order_id, evaluated_at))
                self._db.execute("COMMIT")
            except Exception as exc:  # noqa: BLE001
                try:
                    self._db.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                audit_id = None
                if approved and kind == "ENTRY":          # opening trades need a trace: fail closed
                    approved = False
                    reasons.append("audit_write_failed")
                elif approved:                             # reducing risk is never blocked by logging
                    warnings.append("audit_write_failed_exit_allowed")
        return {
            "approved": bool(approved), "decision": "APPROVED" if approved else "REJECTED",
            "reasons": list(dict.fromkeys(reasons)), "warnings": list(dict.fromkeys(warnings)),
            "order_id": order_id, "order_kind": kind, "checks": dict(checks), "audit_id": audit_id,
            "evaluated_at": evaluated_at,
        }

    def audit_rows(self) -> list[dict]:
        with self._lock:
            cur = self._db.execute("SELECT id,evaluated_at,order_id,idempotency_key,order_kind,approved,reasons,"
                                   "warnings,checks,inputs_sha256 FROM evaluations ORDER BY id")
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------ shared order parsing

    @staticmethod
    def _order_core(ctx: dict, kinds: tuple) -> dict:
        o = _mapping(ctx["order"], "proposed_order")
        core = dict(
            order_id=_txt(o, "order_id"), key=_txt(o, "idempotency_key"), symbol=_txt(o, "symbol"),
            kind=_txt(o, "order_kind", kinds), side=_txt(o, "side", ("LONG", "SHORT")),
            reduce_only=_bool(o, "reduce_only"), qty=_num(o, "quantity", pos=True), price=_num(o, "price", pos=True),
        )
        return core

    # ------------------------------------------------------------ ENTRY checks

    def _chk_schema_entry(self, ctx):
        c = self._order_core(ctx, ("ENTRY",))
        if c["reduce_only"]:
            raise _Bad("entry_marked_reduce_only")
        o = ctx["order"]
        _num(o, "notional", pos=True)
        _num(o, "leverage", pos=True)
        tf_seconds(_txt(o, "timeframe"))
        _txt(o, "signal_id")

    def _chk_data_fresh(self, ctx):
        o, m, lim = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["market"], "market_snapshot"), self.limits
        if _txt(m, "symbol") != _txt(o, "symbol"):
            raise _Bad("market_symbol_mismatch")
        mp = _num(m, "price", pos=True)
        age = (ctx["now"] - _ts(m, "observed_at")).total_seconds()
        if age < -5.0:
            raise _Bad("market_data_in_future")
        if age > lim.max_data_age_s:
            raise _Bad("stale_market_data")
        bid, ask = _num(m, "bid", pos=True, optional=True), _num(m, "ask", pos=True, optional=True)
        if (bid is None) != (ask is None):
            raise _Bad("incomplete_quote")
        if bid is not None:
            if ask < bid:
                raise _Bad("crossed_quote")
            if (ask - bid) / ((ask + bid) / 2.0) * 1e4 > lim.max_spread_bps:
                raise _Bad("spread_too_wide")
        else:
            sb = _num(m, "spread_bps", nonneg=True, optional=True)
            if sb is not None and sb > lim.max_spread_bps:
                raise _Bad("spread_too_wide")
        if all(k in m and m[k] is not None for k in ("open", "high", "low", "close")):
            op, hi, lo, cl = (_num(m, k, pos=True) for k in ("open", "high", "low", "close"))
            if hi < max(op, cl) or lo > min(op, cl) or hi < lo:
                raise _Bad("invalid_ohlc")
        price = _num(o, "price", pos=True)
        if abs(price / mp - 1.0) * 1e4 > lim.max_price_deviation_bps:
            raise _Bad("order_price_deviates_from_market")

    def _chk_signal_matches(self, ctx):
        o, s = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["signal"], "signal")
        if _txt(s, "signal_id") != _txt(o, "signal_id"):
            raise _Bad("signal_order_mismatch:signal_id")
        if _txt(s, "symbol") != _txt(o, "symbol"):
            raise _Bad("signal_order_mismatch:symbol")
        if _txt(s, "direction", ("LONG", "SHORT")) != _txt(o, "side", ("LONG", "SHORT")):
            raise _Bad("signal_order_mismatch:direction")
        if _txt(s, "timeframe") != _txt(o, "timeframe"):
            raise _Bad("signal_order_mismatch:timeframe")

    def _chk_signal_fresh(self, ctx):
        s, m, lim = _mapping(ctx["signal"], "signal"), _mapping(ctx["market"], "market_snapshot"), self.limits
        seconds = tf_seconds(_txt(s, "timeframe"))
        age = (ctx["now"] - _ts(s, "candle_timestamp")).total_seconds()
        if age < seconds:
            raise _Bad("signal_candle_not_closed")
        if age > seconds * lim.max_signal_age_multiplier:
            raise _Bad("signal_expired")
        sp, mp = _num(s, "price", pos=True), _num(m, "price", pos=True)
        if abs(sp / mp - 1.0) * 1e4 > lim.max_price_deviation_bps:
            raise _Bad("signal_price_deviates_from_market")

    def _chk_duplicate(self, ctx):
        o, p = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["portfolio"], "portfolio_state")
        seen = p.get("seen_idempotency_keys")
        if seen is None or isinstance(seen, (str, bytes)):
            raise _Bad("missing_field:seen_idempotency_keys")
        if _txt(o, "idempotency_key") in set(seen):
            raise _Bad("duplicate_order")

    @staticmethod
    def _positions(ctx) -> list:
        p = _mapping(ctx["portfolio"], "portfolio_state")
        pos = p.get("positions")
        if not isinstance(pos, (list, tuple)):
            raise _Bad("missing_field:positions")
        out = []
        for item in pos:
            out.append(dict(symbol=_txt(item, "symbol"), direction=_txt(item, "direction", ("LONG", "SHORT")),
                            quantity=_num(item, "quantity", pos=True), notional=_num(item, "notional", nonneg=True)))
        return out

    def _chk_risk_limits(self, ctx):
        o, p, lim = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["portfolio"], "portfolio_state"), self.limits
        equity, cash = _num(p, "equity", pos=True), _num(p, "cash")
        qty, price, notional, lev = (_num(o, "quantity", pos=True), _num(o, "price", pos=True),
                                     _num(o, "notional", pos=True), _num(o, "leverage", pos=True))
        positions = self._positions(ctx)
        bad = []
        if abs(notional - qty * price) > lim.notional_tolerance * qty * price:
            bad.append("notional_mismatch")
        if notional < lim.min_order_usd:
            bad.append("below_min_order")
        if notional > equity * lim.max_position_pct / 100.0:
            bad.append("position_size_exceeds_limit")
        if lev > lim.max_leverage:
            bad.append("leverage_exceeds_limit")
        if sum(x["notional"] for x in positions) + notional > equity * lim.max_exposure_pct / 100.0:
            bad.append("portfolio_exposure_exceeds_limit")
        if len(positions) >= lim.max_open_positions:
            bad.append("max_open_positions_reached")
        if cash - notional / lev < equity * lim.cash_buffer_pct / 100.0:
            bad.append("insufficient_cash")
        if bad:
            raise _Bad(bad[0]) if len(bad) == 1 else _Bad("|".join(bad))

    def _chk_portfolio_rules(self, ctx):
        o, r, p = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["risk"], "risk_state"), _mapping(ctx["portfolio"], "portfolio_state")
        symbol, side = _txt(o, "symbol"), _txt(o, "side", ("LONG", "SHORT"))
        positions = self._positions(ctx)
        for pos in positions:
            if pos["symbol"] == symbol:
                raise _Bad("position_already_open" if pos["direction"] == side else "opposite_position_open_close_first")
        if side == "SHORT" and not _bool(r, "shorts_allowed"):
            raise _Bad("shorts_not_allowed")
        cap = self.limits.max_net_exposure_pct
        if cap is not None:
            equity = _num(p, "equity", pos=True)
            net = sum(x["notional"] * (1 if x["direction"] == "LONG" else -1) for x in positions)
            net += _num(o, "notional", pos=True) * (1 if side == "LONG" else -1)
            if abs(net) > equity * cap / 100.0:
                raise _Bad("net_exposure_exceeds_limit")

    def _chk_borrow(self, ctx):
        o, r = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["risk"], "risk_state")
        if _txt(o, "side", ("LONG", "SHORT")) == "SHORT" and self.limits.require_borrow_check:
            avail = r.get("borrow_available")
            if not isinstance(avail, Mapping) or avail.get(_txt(o, "symbol")) is not True:
                raise _Bad("short_borrow_unavailable")

    def _chk_stop(self, ctx):
        o, lim = _mapping(ctx["order"], "proposed_order"), self.limits
        if o.get("stop_price") is None:
            raise _Bad("missing_stop")
        stop, price, side = _num(o, "stop_price", pos=True), _num(o, "price", pos=True), _txt(o, "side", ("LONG", "SHORT"))
        if (side == "LONG" and stop >= price) or (side == "SHORT" and stop <= price):
            raise _Bad("stop_wrong_side")
        dist = abs(stop / price - 1.0) * 100.0
        if dist < lim.min_stop_pct:
            raise _Bad("stop_too_tight")
        if dist > lim.max_stop_pct:
            raise _Bad("stop_too_wide")
        tgt = _num(o, "target_price", pos=True, optional=True)
        if tgt is not None and ((side == "LONG" and tgt <= price) or (side == "SHORT" and tgt >= price)):
            raise _Bad("target_wrong_side")

    def _chk_kill_switch(self, ctx):
        if _bool(_mapping(ctx["risk"], "risk_state"), "kill_switch"):
            raise _Bad("kill_switch_active")

    def _chk_daily_loss(self, ctx):
        r, p, lim = _mapping(ctx["risk"], "risk_state"), _mapping(ctx["portfolio"], "portfolio_state"), self.limits
        start, realized, equity = _num(r, "day_start_equity", pos=True), _num(r, "daily_realized_pnl"), _num(p, "equity", pos=True)
        if realized <= -(start * lim.max_daily_loss_pct / 100.0):
            raise _Bad("daily_loss_limit")
        if (start - equity) / start * 100.0 >= lim.max_equity_loss_pct:
            raise _Bad("daily_equity_loss_limit")

    def _chk_drawdown(self, ctx):
        p = _mapping(ctx["portfolio"], "portfolio_state")
        equity, peak = _num(p, "equity", pos=True), _num(p, "peak_equity", pos=True)
        peak = max(peak, equity)
        if (peak - equity) / peak * 100.0 >= self.limits.max_drawdown_pct:
            raise _Bad("max_drawdown_limit")

    def _chk_loss_streak(self, ctx):
        r, lim = _mapping(ctx["risk"], "risk_state"), self.limits
        if _int(r, "consecutive_losses") >= lim.max_consecutive_losses:
            raise _Bad("max_consecutive_losses")
        if _int(r, "failed_orders_today") >= lim.max_failed_orders:
            raise _Bad("max_failed_orders")
        opened = _int(r, "trades_opened_today")
        if lim.max_trades_per_day > 0 and opened >= lim.max_trades_per_day:
            raise _Bad("max_trades_per_day")

    def _chk_cooldown(self, ctx):
        r, lim = _mapping(ctx["risk"], "risk_state"), self.limits
        if lim.min_bars_between_entries <= 0:
            return
        if "bars_since_last_entry" not in r:
            raise _Bad("missing_field:bars_since_last_entry")
        bars = r["bars_since_last_entry"]
        if bars is None:                       # explicit None == this symbol has never been entered
            return
        if _num(r, "bars_since_last_entry", nonneg=True) < lim.min_bars_between_entries:
            raise _Bad("entry_cooldown_active")

    # ------------------------------------------------------------ EXIT checks

    def _chk_schema_exit(self, ctx):
        c = self._order_core(ctx, ("EXIT",))
        if not c["reduce_only"]:
            raise _Bad("exit_not_reduce_only")

    def _chk_exit_market(self, ctx):
        o, m = _mapping(ctx["order"], "proposed_order"), _mapping(ctx["market"], "market_snapshot")
        _num(o, "price", pos=True)
        _num(m, "price", pos=True)
        if _txt(m, "symbol") != _txt(o, "symbol"):
            raise _Bad("market_symbol_mismatch")
        try:
            age = (ctx["now"] - _ts(m, "observed_at")).total_seconds()
            if age > self.limits.max_data_age_s:
                ctx["warnings"].append("stale_market_data_exit_allowed")
        except _Bad:
            ctx["warnings"].append("market_timestamp_unusable_exit_allowed")
        try:
            if _bool(_mapping(ctx["risk"], "risk_state"), "kill_switch"):
                ctx["warnings"].append("kill_switch_active_exit_allowed")
        except _Bad:
            pass

    def _chk_reduces(self, ctx):
        o = _mapping(ctx["order"], "proposed_order")
        symbol, direction, qty = _txt(o, "symbol"), _txt(o, "side", ("LONG", "SHORT")), _num(o, "quantity", pos=True)
        match = [x for x in self._positions(ctx) if x["symbol"] == symbol and x["direction"] == direction]
        if not match:
            raise _Bad("no_position_to_reduce")
        if qty > match[0]["quantity"] * (1.0 + 1e-9):
            raise _Bad("reduce_exceeds_position")
