"""
trading_gate.reconcile -- deterministic fill / position reconciliation. Read-only: it only inspects the data it is given.

    result = reconcile(orders, internal_positions, broker_positions, tolerances)
    result == {"ok": bool, "alerts": [ {"type":..., "symbol":..., "order_id":..., "detail":...}, ... ]}

orders:  [{order_id, symbol, side, order_kind, quantity, price, status, filled_quantity, avg_fill_price}]
         status in FILLED / PARTIAL / REJECTED / CANCELLED / EXPIRED.  side = direction LONG/SHORT.
positions (internal and broker): [{symbol, direction, quantity}]
tolerances (ALL required): {"qty_rel": 0.001, "slippage_bps": 25}

Alert types: partial_fill, rejected_fill, fill_price_slippage, position_missing_at_broker, unexpected_position_at_broker,
position_qty_mismatch, direction_mismatch, reconcile_error.   FAIL CLOSED: any malformed input -> ok False + reconcile_error.
"""
from __future__ import annotations

import math
import numbers
from typing import Any

_DONE_BAD = ("REJECTED", "CANCELLED", "EXPIRED")
_STATUSES = ("FILLED", "PARTIAL") + _DONE_BAD


class _Err(Exception):
    pass


def _f(d: Any, k: str, *, pos: bool = False, nonneg: bool = False) -> float:
    if not isinstance(d, dict) or k not in d or d[k] is None:
        raise _Err(f"missing:{k}")
    v = d[k]
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
        raise _Err(f"bad_number:{k}")
    v = float(v)
    if (pos and v <= 0) or (nonneg and v < 0):
        raise _Err(f"out_of_range:{k}")
    return v


def _s(d: Any, k: str, allowed: tuple | None = None) -> str:
    if not isinstance(d, dict) or not isinstance(d.get(k), str) or not d[k].strip():
        raise _Err(f"missing:{k}")
    v = d[k].strip()
    if allowed is not None and v not in allowed:
        raise _Err(f"bad_value:{k}")
    return v


def _alert(type_: str, symbol: str | None, order_id: str | None, detail: str) -> dict:
    return {"type": type_, "symbol": symbol, "order_id": order_id, "detail": detail}


def _book(items: Any, name: str) -> dict:
    if not isinstance(items, (list, tuple)):
        raise _Err(f"missing:{name}")
    out: dict = {}
    for it in items:
        sym, direction, q = _s(it, "symbol"), _s(it, "direction", ("LONG", "SHORT")), _f(it, "quantity", pos=True)
        if sym in out:
            raise _Err(f"duplicate_symbol:{name}:{sym}")
        out[sym] = (direction, q)
    return out


def reconcile(orders, internal_positions, broker_positions, tolerances) -> dict:
    alerts: list = []
    try:
        qty_rel = _f(tolerances, "qty_rel", nonneg=True)
        slip_bps = _f(tolerances, "slippage_bps", nonneg=True)
        if not isinstance(orders, (list, tuple)):
            raise _Err("missing:orders")
        for o in orders:
            oid, sym = _s(o, "order_id"), _s(o, "symbol")
            _s(o, "side", ("LONG", "SHORT"))
            status = _s(o, "status", _STATUSES)
            qty = _f(o, "quantity", pos=True)
            price = _f(o, "price", pos=True)
            filled = _f(o, "filled_quantity", nonneg=True)
            if filled > qty * (1 + qty_rel):
                alerts.append(_alert("reconcile_error", sym, oid, f"overfill filled={filled} > quantity={qty}"))
                continue
            if status in _DONE_BAD:
                if filled == 0:
                    alerts.append(_alert("rejected_fill", sym, oid, f"status={status} filled=0 of {qty}"))
                else:
                    alerts.append(_alert("partial_fill", sym, oid, f"status={status} filled={filled} of {qty}"))
            elif filled < qty * (1 - qty_rel):
                alerts.append(_alert("partial_fill", sym, oid, f"status={status} filled={filled} of {qty}"))
            if filled > 0:
                avg = _f(o, "avg_fill_price", pos=True)
                dev = abs(avg / price - 1.0) * 1e4
                if dev > slip_bps:
                    alerts.append(_alert("fill_price_slippage", sym, oid, f"slippage_bps={dev:.2f} > {slip_bps}"))
        internal, broker = _book(internal_positions, "internal_positions"), _book(broker_positions, "broker_positions")
        for sym, (d_i, q_i) in internal.items():
            if sym not in broker:
                alerts.append(_alert("position_missing_at_broker", sym, None, f"internal {d_i} {q_i}"))
                continue
            d_b, q_b = broker[sym]
            if d_i != d_b:
                alerts.append(_alert("direction_mismatch", sym, None, f"internal {d_i} vs broker {d_b}"))
            elif abs(q_i - q_b) > qty_rel * max(q_i, q_b):
                alerts.append(_alert("position_qty_mismatch", sym, None, f"internal {q_i} vs broker {q_b}"))
        for sym, (d_b, q_b) in broker.items():
            if sym not in internal:
                alerts.append(_alert("unexpected_position_at_broker", sym, None, f"broker {d_b} {q_b}"))
    except Exception as exc:  # noqa: BLE001 - fail closed
        alerts.append(_alert("reconcile_error", None, None, f"{type(exc).__name__}:{exc}"))
    return {"ok": not alerts, "alerts": alerts}
