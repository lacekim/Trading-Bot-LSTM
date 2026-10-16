"""
trading_gate.shadow -- READ-ONLY, NON-BLOCKING shadow adapter.

WHAT IT DOES
    Observes the bot's paper-order database (opened with SQLite mode=ro + PRAGMA query_only) and, for every real order
    the bot placed (ENTRY rows in `orders`, EXIT rows in `closed_trades`), builds the point-in-time inputs the gate needs,
    runs the gate, and logs what the gate WOULD have decided. It never blocks, never submits, never writes to the bot's
    database or any bot file. Its only outputs are its own files in `out_dir`:
        shadow_decisions.sqlite3   (decisions, captured stops, reconciliations, runs)
        shadow_decisions.jsonl     (one JSON line per new decision)
        gate_audit.sqlite3         (the gate's own audit log)

POINT-IN-TIME RULES (nothing after the event may influence its decision)
    * the bot's state before an event is rebuilt by replaying `orders` + `closed_trades` in time order, applying an event
      only AFTER it has been evaluated;
    * equity / peak equity come from `equity_history` rows STRICTLY BEFORE the event; the replay's realised-only equity
      is logged next to it (equity_replay) so the difference is visible;
    * signals, previously submitted keys, loss streak, trades-today and daily P&L are computed from earlier events only;
    * market spread comes from the latest `qualified_market_snapshots` row recorded at or before the event;
    * stop / target / leverage of an entry are attributes of the ORDER as placed; they are only stored in the bot's
      `positions` table, so the shadow runner captures them (captured_positions) while the position is open. An entry whose
      stop was never captured is logged as adapter data gap `stop_price_unavailable` and is NOT evaluated against a guess.
      (The stop is deliberately NOT back-inferred from the later exit price: that would be post-trade information.)

ASSUMPTIONS THE ADAPTER MUST MAKE (each is written into the decision's `flags`)
    * kill_switch=False for historical/observed entries (the bot accepted them, so entries were allowed at that moment);
    * leverage = captured value, else `assumed_leverage`; timeframe = `timeframe` argument (the bot's Config.TIMEFRAME);
    * signals.candle_timestamp without a timezone is UTC (the bot's own convention in _validate_signal_candle);
    * shorts_allowed=True, borrow check simulated (borrow_check="simulated" is logged for shorts).

CAMPAIGN LOOP (--loop N)
    Repeats a pass every N seconds until the FIRST of: every required coverage case was observed (stop early), --until
    (UTC, default 2026-10-19T23:59:59Z), the optional emergency --max-events limit, or Ctrl-C. It never forces or fabricates a
    case: it only observes the bot's own paper database. The campaign starts at the first pass on a shadow database (meta
    campaign_start, UTC); events already in the bot's database before that never count as coverage. Daily limits reset at the
    UTC date change (the bot's daily_risk.trading_date is also UTC). Logging is change-only: a `runs` row, a reconciliation row
    or a console line is written only when something changed; unchanged passes only update the `meta` counters. When the loop
    stops it records the reason in `meta` and writes shadow_report.json / shadow_report.md (with the coverage section).
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sqlite3
import sys
import time
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

try:
    from .gate import GateLimits, TradingGate
    from .reconcile import reconcile
    from .shadow_report import LIVE_STOP_SOURCE, compute_coverage, valid_protective_stop, write_report
except ImportError:  # executed as a script
    from gate import GateLimits, TradingGate
    from reconcile import reconcile
    from shadow_report import LIVE_STOP_SOURCE, compute_coverage, valid_protective_stop, write_report

# Default end of the multi-day shadow campaign (UTC). The loop also stops early once every required coverage case was observed.
DEFAULT_UNTIL = "2026-10-19T23:59:59Z"

# --------------------------------------------------------------------------------------- limits

LIMITS_STATED = dict(  # given by the user for paper integration testing (NOT live recommendations)
    max_position_pct=10, max_exposure_pct=50, max_net_exposure_pct=30, max_leverage=1.0, max_open_positions=5,
    min_stop_pct=0.5, max_stop_pct=5, max_daily_loss_pct=1, max_equity_loss_pct=1, max_drawdown_pct=5,
    require_borrow_check=False,
)
LIMITS_ASSUMED = dict(  # NOT specified by the user: mirrored from the bot's own Config where one exists
    min_order_usd=30,                # Config.PAPER_MIN_ORDER_USD
    max_signal_age_multiplier=2.5,   # Config.PAPER_MAX_CANDLE_AGE_MULTIPLIER
    max_consecutive_losses=5,        # Config.PAPER_MAX_CONSECUTIVE_LOSSES
    max_failed_orders=3,             # Config.PAPER_MAX_FAILED_ORDERS
    max_trades_per_day=0,            # Config.PAPER_MAX_TRADES_PER_DAY (0 = off)
    min_bars_between_entries=0,      # Config.PAPER_MIN_BARS_BETWEEN_TRADES
    cash_buffer_pct=0,               # Config.PAPER_CASH_BUFFER_PCT
    max_spread_bps=50,               # Config.QUALIFIED_MARKET_MAX_SPREAD_BPS
    max_data_age_s=60,               # twice Config.QUALIFIED_MARKET_SNAPSHOT_MAX_AGE_SECONDS (30)
    max_price_deviation_bps=100,     # chosen by the adapter author
    notional_tolerance=1e-6,         # notional is defined as quantity*fill in the bot
)


def paper_limits(**override) -> GateLimits:
    params = {**LIMITS_STATED, **LIMITS_ASSUMED, **override}
    return GateLimits(**params)


# --------------------------------------------------------------------------------------- schema

REQUIRED_COLUMNS = {
    "account": ("starting_balance", "cash", "realized_pnl", "fees"),
    "signals": ("signal_id", "candle_timestamp", "symbol", "direction", "probability", "market_price", "status", "reason", "created_at"),
    "orders": ("order_id", "signal_id", "symbol", "side", "status", "expected_price", "fill_price", "notional", "fee",
               "created_at", "order_kind", "reduce_only", "model_price", "observed_price", "observed_at", "price_source"),
    "positions": ("position_id", "symbol", "direction", "entry_time", "entry_price", "quantity", "notional", "collateral",
                  "leverage", "stop_price", "target_price", "entry_fee", "signal_id"),
    "closed_trades": ("position_id", "symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price",
                      "quantity", "gross_pnl", "fees", "net_pnl", "exit_reason", "signal_id"),
    "equity_history": ("timestamp", "equity", "cash", "realized_pnl", "unrealized_pnl"),
    "daily_risk": ("trading_date", "start_equity", "realized_pnl", "trades_opened", "failed_orders"),
}
OPTIONAL_COLUMNS = {
    "qualified_market_snapshots": ("recorded_at", "symbol", "spread_bps", "status"),
    "signal_model_lineage": ("signal_id", "model_side", "model_version"),
}

SHADOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT NOT NULL UNIQUE, shadow_id TEXT NOT NULL, run_id TEXT NOT NULL,
  event_time TEXT, kind TEXT NOT NULL, action TEXT, order_id TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL,
  notional REAL, signal_id TEXT, strategy TEXT, signal_direction TEXT, signal_probability REAL, exit_reason TEXT,
  equity REAL, gross_exposure_pct REAL, net_exposure_pct REAL, n_positions INTEGER, peak_equity REAL, drawdown_pct REAL,
  day_realized_pnl REAL,
  approved INTEGER, decision TEXT NOT NULL, reasons TEXT NOT NULL, checks TEXT NOT NULL, warnings TEXT NOT NULL,
  adapter_errors TEXT NOT NULL, data_gaps TEXT NOT NULL, rejection_class TEXT, borrow_check TEXT, flags TEXT NOT NULL,
  gate_audit_id INTEGER, intent_mismatch INTEGER NOT NULL DEFAULT 0, inputs_json TEXT, logged_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS captured_positions (
  position_id TEXT PRIMARY KEY, symbol TEXT, direction TEXT, stop_price REAL, target_price REAL, leverage REAL,
  collateral REAL, quantity REAL, notional REAL, first_seen_at TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'live_positions_table');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS reconciliations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL, ok INTEGER NOT NULL,
  details TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, paper_db TEXT, events_total INTEGER,
  events_new INTEGER, status TEXT, error TEXT);
"""


class _Map(Exception):
    """A mapping problem for one field (becomes an adapter error + data gap; never raised to the caller)."""


def _num(v: Any, name: str, *, pos: bool = False, nonneg: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise _Map(f"bad_number:{name}")
    v = float(v)
    if pos and v <= 0:
        raise _Map(f"non_positive:{name}")
    if nonneg and v < 0:
        raise _Map(f"negative:{name}")
    return v


def _dt(v: Any, name: str, *, naive_is_utc: bool = False) -> datetime:
    if not isinstance(v, str) or not v.strip():
        raise _Map(f"bad_timestamp:{name}")
    try:
        d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
    except ValueError:
        raise _Map(f"bad_timestamp:{name}")
    if d.tzinfo is None:
        if not naive_is_utc:
            raise _Map(f"naive_timestamp:{name}")
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _txt(v: Any, name: str, allowed: tuple | None = None) -> str:
    if not isinstance(v, str) or not v.strip():
        raise _Map(f"bad_text:{name}")
    v = v.strip()
    if allowed and v not in allowed:
        raise _Map(f"bad_value:{name}")
    return v


def open_readonly(path: str) -> sqlite3.Connection:
    """Open the bot's database so that writing is impossible (URI mode=ro + query_only)."""
    uri = "file:" + urllib.parse.quote(os.path.abspath(path)) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def schema_problems(conn: sqlite3.Connection) -> list[str]:
    problems = []
    for table, cols in REQUIRED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not have:
            problems.append(f"missing_table:{table}")
            continue
        for c in cols:
            if c not in have:
                problems.append(f"missing_column:{table}.{c}")
    return problems


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args)]


def _has_table(conn: sqlite3.Connection, table: str, cols: tuple) -> bool:
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    return bool(have) and all(c in have for c in cols)


# --------------------------------------------------------------------------------------- replay

class Replay:
    """Point-in-time state of the paper account rebuilt from events (apply an event only after evaluating it)."""

    def __init__(self, start_balance: float):
        self.start = float(start_balance)
        self.open: dict[str, dict] = {}            # position_id -> position
        self.net_closed = 0.0
        self.fees_closed = 0.0
        self.closed: list[tuple] = []              # (time, net_pnl, symbol)
        self.entry_keys: list[str] = []
        self.exit_keys: list[str] = []
        self.entries_by_date: Counter = Counter()
        self.notes: list[str] = []

    def positions(self) -> list[dict]:
        return [dict(symbol=p["symbol"], direction=p["direction"], quantity=p["quantity"], notional=p["notional"])
                for p in self.open.values()]

    def gross(self) -> float:
        return sum(p["notional"] for p in self.open.values())

    def net(self) -> float:
        return sum(p["notional"] * (1 if p["direction"] == "LONG" else -1) for p in self.open.values())

    def cash(self) -> float:
        return (self.start - sum(p["collateral"] for p in self.open.values()) + self.net_closed
                - sum(p["entry_fee"] for p in self.open.values()))

    def equity_realized(self) -> float:
        return self.start + self.net_closed - sum(p["entry_fee"] for p in self.open.values())

    def consecutive_losses(self) -> int:
        n = 0
        for _, net, _ in reversed(self.closed):
            if net >= 0:
                break
            n += 1
        return n

    def day_realized(self, date: str) -> float:
        return sum(net for t, net, _ in self.closed if t is not None and t.date().isoformat() == date)

    def apply_entry(self, pos: dict, key: str, t: datetime | None) -> None:
        self.open[pos["position_id"]] = pos
        self.entry_keys.append(key)
        if t is not None:
            self.entries_by_date[t.date().isoformat()] += 1

    def apply_exit(self, position_id: str, symbol: str, qty: float, net: float, fees: float, key: str,
                   t: datetime | None) -> str:
        outcome = "closed"
        pos = self.open.get(position_id)
        if pos is None:
            outcome = "unknown_position"
            self.notes.append(f"exit_for_unknown_position:{position_id}")
        elif qty >= pos["quantity"] * (1 - 1e-9):
            del self.open[position_id]
        else:
            ratio = (pos["quantity"] - qty) / pos["quantity"]
            for k in ("quantity", "notional", "collateral", "entry_fee"):
                pos[k] *= ratio
            outcome = "reduced"
            self.notes.append(f"partial_exit:{position_id}")
        self.net_closed += net
        self.fees_closed += fees
        self.closed.append((t, net, symbol))
        self.exit_keys.append(key)
        return outcome


class _Clock:
    t: datetime | None = None

    def __call__(self) -> datetime:
        if self.t is None:
            raise RuntimeError("clock not set")
        return self.t


# --------------------------------------------------------------------------------------- the observer

class ShadowObserver:
    def __init__(self, paper_db: str, out_dir: str, limits: GateLimits | None = None, timeframe: str = "1h",
                 assumed_leverage: float = 1.0, reset: bool = False, now: Callable[[], datetime] | None = None):
        self.paper_db = paper_db
        self.out_dir = out_dir
        self._now = now or (lambda: datetime.now(timezone.utc))   # UTC clock for every timestamp the shadow writes
        self._last_pass_sig: tuple | None = None
        self.limits = limits or paper_limits()
        self.timeframe = timeframe
        self.assumed_leverage = assumed_leverage
        os.makedirs(out_dir, exist_ok=True)
        if os.path.abspath(out_dir) == os.path.dirname(os.path.abspath(paper_db)) and not out_dir:
            raise ValueError("out_dir must be separate from the bot database folder")
        if reset:
            for name in ("shadow_decisions.sqlite3", "shadow_decisions.jsonl", "gate_audit.sqlite3"):
                p = os.path.join(out_dir, name)
                if os.path.exists(p):
                    os.remove(p)
        self.sdb = sqlite3.connect(os.path.join(out_dir, "shadow_decisions.sqlite3"), isolation_level=None)
        self.sdb.row_factory = sqlite3.Row
        self.sdb.executescript(SHADOW_SCHEMA)
        if "source" not in {r[1] for r in self.sdb.execute("PRAGMA table_info(captured_positions)")}:   # database from the older version
            self.sdb.execute("ALTER TABLE captured_positions ADD COLUMN source TEXT NOT NULL DEFAULT 'live_positions_table'")
        self.clock = _Clock()
        self.gate = TradingGate(self.limits, os.path.join(out_dir, "gate_audit.sqlite3"), clock=self.clock)
        self.jsonl_path = os.path.join(out_dir, "shadow_decisions.jsonl")

    def close(self) -> None:
        for c in (self.sdb, self.gate):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ public

    def run_once(self) -> dict:
        """One pass. NEVER raises: any failure is returned (and logged when it is a state change).

        Logging is change-only: a `runs` row is written only for the first pass of a process, when new decisions or a changed
        reconciliation were recorded, or when status/error changed; an unchanged pass only updates two `meta` counters."""
        run_id = f"run_{int(time.time() * 1000)}"
        started = self._now().astimezone(timezone.utc).isoformat(timespec="seconds")
        summary = dict(run_id=run_id, status="OK", events_total=0, events_new=0, error=None, schema_problems=[], recon_changed=[])
        conn = None
        try:
            self._ensure_campaign()
            conn = open_readonly(self.paper_db)
            problems = schema_problems(conn)
            if self._record_recon(run_id, "schema", not problems, dict(problems=problems)):
                summary["recon_changed"].append("schema")
            if problems:
                summary.update(status="SCHEMA_DRIFT", schema_problems=problems, error=";".join(problems))
                return summary
            self._capture_positions(conn)
            self._process(conn, run_id, summary)
        except Exception as exc:  # noqa: BLE001 - the shadow must never take the caller down
            summary.update(status="ERROR", error=f"{type(exc).__name__}:{exc}")
        finally:
            if conn is not None:
                conn.close()
            try:
                self._finish_pass(run_id, started, summary)
            except Exception as exc:  # noqa: BLE001
                summary.update(status="ERROR", error=f"{type(exc).__name__}:{exc}")
        return summary

    # ------------------------------------------------------------------ campaign metadata and change-only run log

    def _meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.sdb.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def _meta_set(self, key: str, value: str) -> None:
        self.sdb.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def _meta_delete(self, *keys: str) -> None:
        for k in keys:
            self.sdb.execute("DELETE FROM meta WHERE key=?", (k,))

    def _ensure_campaign(self) -> None:
        """The campaign starts at the first pass ever made on this shadow database (UTC). Events already in the bot's database
        before that instant are history and never count toward required coverage."""
        self.sdb.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('campaign_start',?)",
                         (self._now().astimezone(timezone.utc).isoformat(timespec="seconds"),))
        self._meta_set("timeframe", self.timeframe)

    def _finish_pass(self, run_id: str, started: str, summary: dict) -> None:
        finished = self._now().astimezone(timezone.utc).isoformat(timespec="seconds")
        sig = (summary["status"], summary["error"])
        if summary["events_new"] > 0 or summary["recon_changed"] or sig != self._last_pass_sig:
            self.sdb.execute(
                "INSERT OR REPLACE INTO runs(run_id,started_at,finished_at,paper_db,events_total,events_new,status,error) VALUES(?,?,?,?,?,?,?,?)",
                (run_id, started, finished, self.paper_db, summary["events_total"], summary["events_new"], summary["status"],
                 summary["error"]))
        self._last_pass_sig = sig
        self._meta_set("last_pass_at", finished)
        self._meta_set("last_status", summary["status"])
        self.sdb.execute("INSERT INTO meta(key,value) VALUES('passes','1') ON CONFLICT(key) DO UPDATE SET "
                         "value=CAST(meta.value AS INTEGER)+1")

    # ------------------------------------------------------------------ capture of order attributes held only in `positions`

    def _capture_positions(self, conn: sqlite3.Connection) -> None:
        now = self._now().astimezone(timezone.utc).isoformat(timespec="seconds")
        for p in _rows(conn, "SELECT * FROM positions"):
            self.sdb.execute(
                "INSERT OR IGNORE INTO captured_positions(position_id,symbol,direction,stop_price,target_price,leverage,collateral,"
                "quantity,notional,first_seen_at,source) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (p["position_id"], p["symbol"], p["direction"], p["stop_price"], p["target_price"], p["leverage"],
                 p["collateral"], p["quantity"], p["notional"], now, LIVE_STOP_SOURCE))

    # ------------------------------------------------------------------ main pass

    def _process(self, conn: sqlite3.Connection, run_id: str, summary: dict) -> None:
        account = _rows(conn, "SELECT * FROM account WHERE id=1")
        if not account:
            summary.update(status="ERROR", error="account_row_missing")
            return
        h = dict(
            account=account[0],
            signals={r["signal_id"]: r for r in _rows(conn, "SELECT rowid AS _rid, * FROM signals")},
            orders=_rows(conn, "SELECT rowid AS _rid, * FROM orders ORDER BY rowid"),
            closed=_rows(conn, "SELECT rowid AS _rid, * FROM closed_trades ORDER BY rowid"),
            positions=_rows(conn, "SELECT * FROM positions"),
            daily={r["trading_date"]: r for r in _rows(conn, "SELECT * FROM daily_risk")},
            lineage={},
            conn=conn,
            has_snapshots=_has_table(conn, "qualified_market_snapshots", OPTIONAL_COLUMNS["qualified_market_snapshots"]),
        )
        if _has_table(conn, "signal_model_lineage", OPTIONAL_COLUMNS["signal_model_lineage"]):
            h["lineage"] = {r["signal_id"]: r for r in _rows(conn, "SELECT * FROM signal_model_lineage")}
        eq_rows = []
        for r in _rows(conn, "SELECT * FROM equity_history"):
            try:
                eq_rows.append((_dt(r["timestamp"], "equity_history.timestamp"), _num(r["equity"], "equity_history.equity", pos=True)))
            except _Map:
                continue
        eq_rows.sort(key=lambda x: x[0])
        h["eq_times"] = [t for t, _ in eq_rows]
        h["eq_vals"] = [v for _, v in eq_rows]
        h["eq_peak"] = []
        peak = -math.inf
        for v in h["eq_vals"]:
            peak = max(peak, v)
            h["eq_peak"].append(peak)
        h["captured"] = {r["position_id"]: dict(r) for r in self.sdb.execute("SELECT * FROM captured_positions")}
        h["captured_now"] = {p["position_id"]: p for p in h["positions"]}

        events = self._build_events(h)
        summary["events_total"] = len(events)
        done = {r[0] for r in self.sdb.execute("SELECT event_key FROM decisions")}
        replay = Replay(_num(h["account"]["starting_balance"], "account.starting_balance", pos=True))
        for ev in events:
            if ev["key"] in done:
                self._apply(ev, replay, h, outcome_only=True)
                continue
            rec = self._evaluate_event(ev, replay, h, run_id)
            self._apply(ev, replay, h, outcome_only=False)
            self._store(rec)
            summary["events_new"] += 1
        summary["recon_changed"].extend(self._reconcile_all(run_id, h, replay))

    # ------------------------------------------------------------------ events

    def _build_events(self, h: dict) -> list[dict]:
        events = []
        for o in h["orders"]:
            if o.get("order_kind") == "ENTRY":
                events.append(dict(kind="ENTRY", key=f"entry:{o['order_id']}", row=o, rid=o["_rid"], tstr=o.get("created_at")))
        for c in h["closed"]:
            events.append(dict(kind="EXIT", key=f"exit:{c['position_id']}", row=c, rid=c["_rid"], tstr=c.get("exit_time")))
        far = datetime.max.replace(tzinfo=timezone.utc)
        for ev in events:
            try:
                ev["t"] = _dt(ev["tstr"], "event_time")
            except _Map as exc:
                ev["t"], ev["time_error"] = None, str(exc)
        events.sort(key=lambda e: (e["t"] or far, 0 if e["kind"] == "EXIT" else 1, e["rid"]))
        per_sec = Counter(e["t"].replace(microsecond=0) for e in events if e["t"] is not None)
        for e in events:
            e["same_second_events"] = per_sec[e["t"].replace(microsecond=0)] if e["t"] is not None else 0
        return events

    # ------------------------------------------------------------------ evaluation of one event

    def _evaluate_event(self, ev: dict, rp: Replay, h: dict, run_id: str) -> dict:
        rec = dict(event_key=ev["key"], run_id=run_id, kind=ev["kind"], event_time=ev["tstr"] if isinstance(ev["tstr"], str) else None,
                   adapter_errors=[], data_gaps=[], flags={}, reasons=[], checks={}, warnings=[], approved=None,
                   decision="NOT_EVALUATED", gate_audit_id=None, rejection_class=None, borrow_check="not_applicable",
                   intent_mismatch=0, inputs_json=None)
        try:
            if ev.get("time_error"):
                rec["adapter_errors"].append("event_time_unusable:" + ev["time_error"])
                self._fill_identity(rec, ev, h, rp)
                rec["data_gaps"].append("event_time")
                return rec
            if ev["same_second_events"] > 1:
                rec["flags"]["same_second_events"] = ev["same_second_events"]
                rec["flags"]["ordering_ambiguous"] = True
            self.clock.t = ev["t"]
            mapped = self._map_entry(ev, rp, h, rec) if ev["kind"] == "ENTRY" else self._map_exit(ev, rp, h, rec)
            decision = self.gate.evaluate(**mapped)
            rec.update(approved=int(bool(decision["approved"])), decision=decision["decision"], reasons=decision["reasons"],
                       checks=decision["checks"], warnings=decision["warnings"], gate_audit_id=decision["audit_id"])
            rec["rejection_class"] = self._classify(decision["reasons"], rec["data_gaps"]) if not decision["approved"] else None
            rec["inputs_json"] = json.dumps(self._trim(mapped), default=str, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            rec["adapter_errors"].append(f"adapter_exception:{type(exc).__name__}:{exc}")
            rec["approved"], rec["decision"] = None, "ADAPTER_ERROR"
        return rec

    @staticmethod
    def _trim(mapped: dict) -> dict:
        out = {k: v for k, v in mapped.items()}
        pf = out.get("portfolio_state")
        if isinstance(pf, dict):
            pf = dict(pf)
            pf["seen_idempotency_keys"] = f"<{len(pf.get('seen_idempotency_keys') or [])} keys>"
            out["portfolio_state"] = pf
        return out

    @staticmethod
    def _classify(reasons: list, gaps: list) -> str:
        def gap_reason(r: str) -> bool:
            return (r.startswith(("missing_field", "missing_or_invalid", "non_numeric", "non_finite", "unparseable_timestamp",
                                  "timestamp_ambiguous", "invalid_text", "check_error", "gate_internal_error"))
                    or (r == "missing_stop" and "stop_price" in gaps))
        flags = [gap_reason(r) for r in reasons] or [False]
        return "adapter_data_gap" if all(flags) else ("rule" if not any(flags) else "mixed")

    def _try(self, rec: dict, name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except _Map as exc:
            rec["adapter_errors"].append(str(exc))
            rec["data_gaps"].append(name)
        except (KeyError, TypeError) as exc:
            rec["adapter_errors"].append(f"missing_value:{name}:{type(exc).__name__}")
            rec["data_gaps"].append(name)
        return None

    def _fill_identity(self, rec: dict, ev: dict, h: dict, rp: Replay) -> None:
        r = ev["row"]
        rec["symbol"], rec["signal_id"] = r.get("symbol"), r.get("signal_id")
        rec["shadow_id"] = f"shadow:{ev['key']}"
        rec["order_id"] = r.get("order_id") or f"shadow_exit_{r.get('position_id')}"

    # ---- point-in-time market / account helpers

    def _pre_equity(self, t: datetime, h: dict, rp: Replay, rec: dict) -> tuple[float, float]:
        replay_eq = rp.equity_realized()
        i = bisect.bisect_left(h["eq_times"], t) - 1          # strictly before t
        rec["flags"]["equity_replay"] = round(replay_eq, 6)
        if i >= 0:
            rec["flags"]["equity_source"] = "equity_history"
            rec["flags"]["equity_asof"] = h["eq_times"][i].isoformat()
            rec["flags"]["equity_staleness_s"] = (t - h["eq_times"][i]).total_seconds()
            eq, peak = h["eq_vals"][i], h["eq_peak"][i]
        else:
            rec["flags"]["equity_source"] = "replay_realized_only"
            rec["adapter_errors"].append("equity_history_unavailable_before_event")
            eq, peak = replay_eq, replay_eq
        return eq, max(peak, eq)

    def _snapshot_spread(self, h: dict, symbol: str, t: datetime, rec: dict) -> float | None:
        if not h["has_snapshots"]:
            rec["flags"]["spread_source"] = "unavailable_table_missing"
            return None
        row = h["conn"].execute(
            "SELECT spread_bps, status FROM qualified_market_snapshots WHERE symbol=? AND recorded_at<=? "
            "ORDER BY recorded_at DESC, id DESC LIMIT 1", (symbol, t.isoformat(timespec="seconds"))).fetchone()
        if row is None:
            rec["flags"]["spread_source"] = "no_snapshot_at_or_before_event"
            return None
        rec["flags"]["spread_source"] = f"qualified_market_snapshots:{row['status']}"
        try:
            return _num(row["spread_bps"], "spread_bps", nonneg=True)
        except _Map as exc:
            rec["adapter_errors"].append(str(exc))
            return None

    def _risk_state(self, t: datetime, h: dict, rp: Replay, rec: dict, symbol: str | None, candle_ts: datetime | None) -> dict:
        date = t.date().isoformat()
        d = h["daily"].get(date)
        start_eq = self._try(rec, "day_start_equity", lambda: _num(d["start_equity"], "daily_risk.start_equity", pos=True)) \
            if d is not None else None
        if d is None:
            rec["adapter_errors"].append("daily_risk_row_missing")
            rec["data_gaps"].append("day_start_equity")
        failed = 0
        if d is not None:
            failed = self._try(rec, "failed_orders_today", lambda: int(_num(d["failed_orders"], "daily_risk.failed_orders", nonneg=True))) or 0
        bars = None
        if symbol is not None and candle_ts is not None:
            bars = self._bars_since_last_entry(h, symbol, candle_ts, rec)
        rec["flags"]["kill_switch_source"] = "assumed_false_entry_was_accepted_by_bot"
        rec["flags"]["utc_date"] = date                                    # daily limits are keyed by the UTC date of the event
        rec["flags"]["trades_opened_today"] = rp.entries_by_date.get(date, 0)
        return dict(kill_switch=False, day_start_equity=start_eq, daily_realized_pnl=rp.day_realized(date),
                    consecutive_losses=rp.consecutive_losses(), failed_orders_today=failed,
                    trades_opened_today=rp.entries_by_date.get(date, 0), bars_since_last_entry=bars,
                    shorts_allowed=True, borrow_available={})

    def _bars_since_last_entry(self, h: dict, symbol: str, candle_ts: datetime, rec: dict) -> float | None:
        prior = None
        for s in h["signals"].values():
            if s["symbol"] == symbol and s["status"] == "ACCEPTED":
                try:
                    c = _dt(s["candle_timestamp"], "signals.candle_timestamp", naive_is_utc=True)
                except _Map:
                    continue
                if c < candle_ts and (prior is None or c > prior):
                    prior = c
        if prior is None:
            return None
        from_tf = {"m": 60, "h": 3600, "d": 86400}[self.timeframe[-1]] * int(self.timeframe[:-1])
        return (candle_ts - prior).total_seconds() / from_tf

    # ---- ENTRY mapping

    def _map_entry(self, ev: dict, rp: Replay, h: dict, rec: dict) -> dict:
        o, t = ev["row"], ev["t"]
        rec["shadow_id"] = f"shadow:{ev['key']}"
        rec["order_id"] = o.get("order_id")
        sid = o.get("signal_id")
        rec["signal_id"] = sid
        rec["symbol"] = o.get("symbol")
        sig = h["signals"].get(sid)
        cap = h["captured"].get(f"pp_{sid}") or h["captured_now"].get(f"pp_{sid}")
        if sig is None:
            rec["adapter_errors"].append("signal_row_missing")
            rec["data_gaps"].append("signal")
        symbol = self._try(rec, "symbol", lambda: _txt(o["symbol"], "orders.symbol"))
        side = self._try(rec, "side", lambda: _txt(o["side"], "orders.side", ("LONG", "SHORT")))
        notional = self._try(rec, "notional", lambda: _num(o["notional"], "orders.notional", pos=True))
        fill = self._try(rec, "price", lambda: _num(o["fill_price"], "orders.fill_price", pos=True))
        qty = notional / fill if (notional and fill) else None
        if cap is not None and qty is not None:
            try:
                if abs(_num(cap["quantity"], "positions.quantity", pos=True) / qty - 1) > 1e-6:
                    rec["adapter_errors"].append("quantity_mismatch_vs_position_row")
            except _Map:
                pass
        stop = target = None
        if cap is not None:
            stop = self._try(rec, "stop_price", lambda: _num(cap["stop_price"], "positions.stop_price", pos=True))
            target = self._try(rec, "target_price", lambda: _num(cap["target_price"], "positions.target_price", pos=True))
            lev = self._try(rec, "leverage", lambda: _num(cap["leverage"], "positions.leverage", pos=True))
            rec["flags"]["stop_source"] = cap.get("source") or LIVE_STOP_SOURCE
            rec["flags"]["stop_valid_protective"] = valid_protective_stop(side, fill, stop)
        else:
            rec["flags"]["stop_source"] = "none_captured"
            rec["adapter_errors"].append("stop_price_unavailable")
            rec["data_gaps"].append("stop_price")
            lev = self.assumed_leverage
            rec["flags"]["leverage_assumed"] = True
        rec["flags"]["timeframe_assumed"] = self.timeframe
        order = dict(order_id=o.get("order_id"), idempotency_key=sid, symbol=symbol, order_kind="ENTRY", side=side,
                     reduce_only=False, quantity=qty, price=fill, notional=notional, leverage=lev,
                     timeframe=self.timeframe, signal_id=sid, stop_price=stop, target_price=target)
        # signal
        signal = None
        cts = None
        if sig is not None:
            cts = self._try(rec, "candle_timestamp", lambda: _dt(sig["candle_timestamp"], "signals.candle_timestamp", naive_is_utc=True))
            if isinstance(sig.get("candle_timestamp"), str):
                try:
                    if datetime.fromisoformat(sig["candle_timestamp"].replace("Z", "+00:00")).tzinfo is None:
                        rec["flags"]["candle_timestamp_naive_assumed_utc"] = True
                except ValueError:
                    pass
            sprice = self._try(rec, "signal_price", lambda: _num(sig["market_price"], "signals.market_price", pos=True))
            signal = dict(signal_id=sig["signal_id"], symbol=sig["symbol"], direction=sig["direction"], timeframe=self.timeframe,
                          price=sprice, candle_timestamp=cts)
            rec["signal_direction"] = sig["direction"]
            rec["signal_probability"] = sig["probability"] if isinstance(sig["probability"], (int, float)) else None
            if sig["direction"] != o.get("side"):
                rec["intent_mismatch"] = 1
        lin = h["lineage"].get(sid)
        rec["strategy"] = f"{lin['model_side']}:{lin['model_version']}" if lin else "UNKNOWN"
        # market
        mprice = o.get("observed_price")
        if mprice is None:
            mprice = o.get("expected_price")
            rec["flags"]["market_price_source"] = "expected_price_fallback"
        else:
            rec["flags"]["market_price_source"] = o.get("price_source") or "observed_price"
        market = dict(symbol=symbol, price=self._try(rec, "market_price", lambda: _num(mprice, "market.price", pos=True)),
                      observed_at=self._try(rec, "observed_at", lambda: _dt(o.get("observed_at"), "orders.observed_at")))
        spread = self._snapshot_spread(h, symbol, t, rec) if symbol else None
        if spread is not None:
            market["spread_bps"] = spread
        # portfolio / risk
        equity, peak = self._pre_equity(t, h, rp, rec)
        portfolio = dict(equity=equity, cash=rp.cash(), peak_equity=peak, positions=rp.positions(),
                         seen_idempotency_keys=list(rp.entry_keys))
        risk = self._risk_state(t, h, rp, rec, symbol, cts)
        rec.update(side=side, quantity=qty, price=fill, notional=notional, action="opened")
        self._snapshot_numbers(rec, equity, peak, rp, risk, extra_notional=0.0)
        rec["borrow_check"] = "simulated" if side == "SHORT" else "not_applicable"
        return dict(proposed_order=order, signal=signal, market_snapshot=market, portfolio_state=portfolio, risk_state=risk)

    # ---- EXIT mapping

    def _map_exit(self, ev: dict, rp: Replay, h: dict, rec: dict) -> dict:
        c, t = ev["row"], ev["t"]
        pid = c.get("position_id")
        rec["shadow_id"] = f"shadow:{ev['key']}"
        rec["order_id"] = f"shadow_exit_{pid}"
        rec["symbol"] = c.get("symbol")
        rec["signal_id"] = c.get("signal_id")
        rec["exit_reason"] = c.get("exit_reason")
        symbol = self._try(rec, "symbol", lambda: _txt(c["symbol"], "closed_trades.symbol"))
        direction = self._try(rec, "side", lambda: _txt(c["direction"], "closed_trades.direction", ("LONG", "SHORT")))
        qty = self._try(rec, "quantity", lambda: _num(c["quantity"], "closed_trades.quantity", pos=True))
        price = self._try(rec, "price", lambda: _num(c["exit_price"], "closed_trades.exit_price", pos=True))
        order = dict(order_id=rec["order_id"], idempotency_key=f"exit:{pid}", symbol=symbol, order_kind="EXIT", side=direction,
                     reduce_only=True, quantity=qty, price=price)
        market = dict(symbol=symbol, price=price, observed_at=self._try(rec, "observed_at", lambda: _dt(c["exit_time"], "closed_trades.exit_time")))
        rec["flags"]["exit_market_price_is_order_price"] = True
        net_pnl = c.get("net_pnl")
        if isinstance(net_pnl, (int, float)) and not isinstance(net_pnl, bool) and math.isfinite(float(net_pnl)):
            rec["flags"]["event_net_pnl"] = float(net_pnl)
        equity, peak = self._pre_equity(t, h, rp, rec)
        portfolio = dict(equity=equity, cash=rp.cash(), peak_equity=peak, positions=rp.positions(),
                         seen_idempotency_keys=list(rp.exit_keys))
        risk = self._risk_state(t, h, rp, rec, None, None)
        pos = rp.open.get(pid)
        if pos is None:
            rec["adapter_errors"].append("exit_for_unknown_position")
            rec["action"] = "unknown"
        else:
            rec["strategy"] = pos.get("strategy", "UNKNOWN")
            rec["action"] = "closed" if (qty or 0) >= pos["quantity"] * (1 - 1e-9) else "reduced"
        rec["strategy"] = rec.get("strategy") or "UNKNOWN"
        rec.update(side=direction, quantity=qty, price=price, notional=(qty * price) if (qty and price) else None)
        self._snapshot_numbers(rec, equity, peak, rp, risk, extra_notional=0.0)
        return dict(proposed_order=order, signal=None, market_snapshot=market, portfolio_state=portfolio, risk_state=risk)

    def _snapshot_numbers(self, rec: dict, equity: float, peak: float, rp: Replay, risk: dict, extra_notional: float) -> None:
        rec["equity"], rec["peak_equity"] = equity, peak
        rec["n_positions"] = len(rp.open)
        rec["gross_exposure_pct"] = rp.gross() / equity * 100.0 if equity else None
        rec["net_exposure_pct"] = rp.net() / equity * 100.0 if equity else None
        rec["drawdown_pct"] = (peak - equity) / peak * 100.0 if peak else None
        rec["day_realized_pnl"] = risk.get("daily_realized_pnl")

    # ---- applying the event to the replay AFTER evaluation

    def _apply(self, ev: dict, rp: Replay, h: dict, outcome_only: bool) -> None:
        """Apply an already-evaluated event to the replay. A malformed row is NOT replayed (it would poison every later
        decision); it is noted instead and the positions/account reconciliations will show the divergence."""
        try:
            if ev["kind"] == "ENTRY":
                o = ev["row"]
                notional = _num(o["notional"], "orders.notional", pos=True)
                fill = _num(o["fill_price"], "orders.fill_price", pos=True)
                fee = _num(o["fee"], "orders.fee", nonneg=True)
                cap = h["captured"].get(f"pp_{o['signal_id']}") or h["captured_now"].get(f"pp_{o['signal_id']}")
                lev = self.assumed_leverage
                if cap and isinstance(cap.get("leverage"), (int, float)) and not isinstance(cap["leverage"], bool) \
                        and math.isfinite(cap["leverage"]) and cap["leverage"] > 0:
                    lev = float(cap["leverage"])
                symbol = _txt(o["symbol"], "orders.symbol")
                side = _txt(o["side"], "orders.side", ("LONG", "SHORT"))
                lin = h["lineage"].get(o["signal_id"])
                rp.apply_entry(dict(position_id=f"pp_{o['signal_id']}", symbol=symbol, direction=side,
                                    quantity=notional / fill, notional=notional, collateral=notional / lev,
                                    entry_fee=fee, signal_id=o["signal_id"],
                                    strategy=f"{lin['model_side']}:{lin['model_version']}" if lin else "UNKNOWN"),
                               key=o["signal_id"], t=ev["t"])
            else:
                c = ev["row"]
                qty = _num(c["quantity"], "closed_trades.quantity", pos=True)
                net = _num(c["net_pnl"], "closed_trades.net_pnl")
                fees = _num(c["fees"], "closed_trades.fees", nonneg=True)
                rp.apply_exit(_txt(c["position_id"], "closed_trades.position_id"), _txt(c["symbol"], "closed_trades.symbol"),
                              qty, net, fees, key=f"exit:{c['position_id']}", t=ev["t"])
        except (_Map, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            rp.notes.append(f"event_not_replayed:{ev['key']}:{type(exc).__name__}:{exc}")

    # ------------------------------------------------------------------ store

    def _store(self, rec: dict) -> None:
        j = lambda v: json.dumps(v, default=str, sort_keys=True)  # noqa: E731
        cols = dict(
            event_key=rec["event_key"], shadow_id=rec.get("shadow_id") or f"shadow:{rec['event_key']}", run_id=rec["run_id"],
            event_time=rec.get("event_time"), kind=rec["kind"], action=rec.get("action"), order_id=rec.get("order_id"),
            symbol=rec.get("symbol"), side=rec.get("side"), quantity=rec.get("quantity"), price=rec.get("price"),
            notional=rec.get("notional"), signal_id=rec.get("signal_id"), strategy=rec.get("strategy"),
            signal_direction=rec.get("signal_direction"), signal_probability=rec.get("signal_probability"),
            exit_reason=rec.get("exit_reason"), equity=rec.get("equity"), gross_exposure_pct=rec.get("gross_exposure_pct"),
            net_exposure_pct=rec.get("net_exposure_pct"), n_positions=rec.get("n_positions"), peak_equity=rec.get("peak_equity"),
            drawdown_pct=rec.get("drawdown_pct"), day_realized_pnl=rec.get("day_realized_pnl"), approved=rec["approved"],
            decision=rec["decision"], reasons=j(rec["reasons"]), checks=j(rec["checks"]), warnings=j(rec["warnings"]),
            adapter_errors=j(rec["adapter_errors"]), data_gaps=j(sorted(set(rec["data_gaps"]))),
            rejection_class=rec["rejection_class"], borrow_check=rec["borrow_check"], flags=j(rec["flags"]),
            gate_audit_id=rec["gate_audit_id"], intent_mismatch=rec["intent_mismatch"], inputs_json=rec["inputs_json"],
            logged_at=self._now().astimezone(timezone.utc).isoformat(timespec="seconds"))
        names = ",".join(cols)
        self.sdb.execute(f"INSERT OR IGNORE INTO decisions({names}) VALUES({','.join('?' * len(cols))})", tuple(cols.values()))
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(cols, default=str, sort_keys=True) + "\n")

    # ------------------------------------------------------------------ reconciliations

    def _record_recon(self, run_id: str, kind: str, ok: bool, details: dict) -> bool:
        """Change-only: a row is written only if (ok, details) differs from the latest row of this kind. Returns True if written."""
        text = json.dumps(details, default=str, sort_keys=True)
        last = self.sdb.execute("SELECT ok, details FROM reconciliations WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        if last is not None and last[0] == int(ok) and last[1] == text:
            return False
        self.sdb.execute("INSERT INTO reconciliations(run_id,at,kind,ok,details) VALUES(?,?,?,?,?)",
                         (run_id, self._now().astimezone(timezone.utc).isoformat(timespec="seconds"), kind, int(ok), text))
        return True

    def _reconcile_all(self, run_id: str, h: dict, rp: Replay) -> list[str]:
        changed: list[str] = []
        _record = self._record_recon

        def _rec(kind: str, ok: bool, details: dict) -> None:
            if _record(run_id, kind, ok, details):
                changed.append(kind)
        # 1. replayed open positions vs the bot's actual positions table (exposure calculation check)
        actual = [dict(symbol=p["symbol"], direction=p["direction"], quantity=p["quantity"]) for p in h["positions"]]
        internal = [dict(symbol=p["symbol"], direction=p["direction"], quantity=p["quantity"]) for p in rp.open.values()]
        r = reconcile([], internal, actual, dict(qty_rel=1e-6, slippage_bps=25))
        gross_actual = sum(float(p["notional"]) for p in h["positions"])
        _rec("positions_vs_replay", r["ok"] and abs(gross_actual - rp.gross()) <= 1e-6 * max(1.0, gross_actual),
                           dict(alerts=r["alerts"], gross_actual=gross_actual, gross_replay=rp.gross(),
                                n_actual=len(actual), n_replay=len(internal)))
        # 2. replayed account (cash / realised P&L / fees) vs the bot's account row
        acct = h["account"]
        fees_replay = rp.fees_closed + sum(p["entry_fee"] for p in rp.open.values())
        diffs = dict(cash=rp.cash() - float(acct["cash"]), realized=rp.net_closed - float(acct["realized_pnl"]),
                     fees=fees_replay - float(acct["fees"]))
        _rec("account_vs_replay", all(abs(v) <= 1e-6 for v in diffs.values()),
                           dict(diffs=diffs, replay_cash=rp.cash(), account_cash=float(acct["cash"]), notes=rp.notes,
                                leverage_assumed=self.assumed_leverage))
        # 3. strategy intent (signals table) vs proposed orders (orders table)
        orders_by_sig = {}
        for o in h["orders"]:
            if o.get("order_kind") == "ENTRY":
                orders_by_sig.setdefault(o["signal_id"], []).append(o)
        sigs = h["signals"]
        accepted_no_order = sorted(s for s, r_ in sigs.items() if r_["status"] == "ACCEPTED" and s not in orders_by_sig)
        order_no_accepted = sorted(s for s in orders_by_sig if s not in sigs or sigs[s]["status"] != "ACCEPTED")
        dir_mismatch = sorted(s for s, os_ in orders_by_sig.items() if s in sigs and any(o["side"] != sigs[s]["direction"] for o in os_))
        dup_orders = sorted(s for s, os_ in orders_by_sig.items() if len(os_) > 1)
        bot_rejected = Counter(r_["reason"] for r_ in sigs.values() if r_["status"] == "REJECTED" and r_["direction"] in ("LONG", "SHORT"))
        _rec("intent_vs_orders",
                           not (accepted_no_order or order_no_accepted or dir_mismatch or dup_orders),
                           dict(accepted_without_order=accepted_no_order, orders_without_accepted_signal=order_no_accepted,
                                direction_mismatch=dir_mismatch, duplicate_signal_orders=dup_orders,
                                bot_rejected_by_reason=dict(bot_rejected), n_signals=len(sigs), n_entry_orders=len(h["orders"])))
        # 4. order fills (status / partial fills / slippage) via the gate package's reconcile()
        rec_orders = []
        for o in h["orders"]:
            if o.get("order_kind") != "ENTRY":
                continue
            try:
                q = float(o["notional"]) / float(o["fill_price"])
                filled = q if o["status"] == "FILLED" else 0.0
                rec_orders.append(dict(order_id=o["order_id"], symbol=o["symbol"], side=o["side"], quantity=q,
                                       price=float(o["expected_price"]), status=o["status"] if o["status"] in
                                       ("FILLED", "PARTIAL", "REJECTED", "CANCELLED", "EXPIRED") else "UNKNOWN",
                                       filled_quantity=filled, avg_fill_price=float(o["fill_price"])))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                rec_orders.append(dict(order_id=str(o.get("order_id")), malformed=True))
        fr = reconcile(rec_orders, internal, actual, dict(qty_rel=1e-6, slippage_bps=25))
        _rec("fills", fr["ok"], dict(alerts=fr["alerts"], n_orders=len(rec_orders),
                                                           status_counts=dict(Counter(o.get("status") for o in h["orders"]))))
        return changed


# --------------------------------------------------------------------------------------- campaign loop + CLI

def parse_until(text: str) -> datetime:
    """--until must carry an explicit timezone (e.g. 2026-10-19T23:59:59Z); it is converted to UTC."""
    d = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("--until needs an explicit timezone, e.g. 2026-10-19T23:59:59Z")
    return d.astimezone(timezone.utc)


def run_loop(obs: ShadowObserver, interval: float, until: datetime, max_events: int | None = None,
             now: Callable[[], datetime] | None = None, sleep: Callable[[float], None] | None = None,
             emit: Callable[[str], None] | None = None, heartbeat_s: float = 0.0) -> dict:
    """Repeat run_once() every `interval` seconds until the FIRST of:
         coverage_complete  every required case was observed in the live window (checked after each pass),
         deadline           `until` (UTC) has passed,
         max_events         optional emergency limit on the number of NEW decisions logged by this invocation,
         interrupted        Ctrl-C.
    Writes only state changes to the console (new decisions, a changed reconciliation, a status/error change, a newly observed
    coverage case); `heartbeat_s` > 0 additionally prints a liveness line at that interval (off by default).
    On stop it records stop_reason in `meta` and writes shadow_report.json/.md."""
    now = now or (lambda: datetime.now(timezone.utc))
    sleep = sleep or time.sleep
    emit = emit or (lambda line: print(line, flush=True))
    until = until.astimezone(timezone.utc)
    obs._meta_set("campaign_until", until.isoformat(timespec="seconds"))
    obs._meta_delete("stop_reason", "stopped_at")
    stamp = lambda: now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    total_new, code, reason = 0, 0, None
    prev_status: tuple | None = None
    prev_observed: set[str] = set()
    prev_cov_error: str | None = None
    last_emit = now()
    cov = None
    try:
        while True:
            s = obs.run_once()
            total_new += s["events_new"]
            if s["status"] == "SCHEMA_DRIFT":
                code = 2
            cov_error = None
            try:
                cov = compute_coverage(obs.sdb, now=now(), until=until)
            except Exception as exc:  # noqa: BLE001 - coverage failure keeps the loop running until the deadline
                cov, cov_error = None, f"{type(exc).__name__}:{exc}"
            changes = []
            status = (s["status"], s["error"])
            if status != prev_status:
                changes.append(f"status={s['status']}" + (f" error={s['error']}" if s["error"] else ""))
            if s["events_new"]:
                changes.append(f"new_decisions={s['events_new']}")
            if s["recon_changed"]:
                changes.append("reconciliation_changed=" + ",".join(s["recon_changed"]))
            if cov is not None:
                newly = [c for c in cov["observed"] if c not in prev_observed]
                if newly:
                    changes.append("coverage_observed=" + ",".join(newly))
                prev_observed = set(cov["observed"])
            if cov_error != prev_cov_error and cov_error:
                changes.append(f"coverage_error={cov_error}")
            prev_status, prev_cov_error = status, cov_error
            if changes:
                tail = f" coverage={cov['n_observed']}/{cov['n_required']}" if cov is not None else ""
                emit(f"[{stamp()}] " + "; ".join(changes) + tail)
                last_emit = now()
            if cov is not None and cov["complete"]:
                reason = "coverage_complete"
            elif now() >= until:
                reason = "deadline"
            elif max_events is not None and total_new >= max_events:
                reason = "max_events"
            if reason:
                break
            if heartbeat_s > 0 and (now() - last_emit).total_seconds() >= heartbeat_s:
                emit(f"[{stamp()}] heartbeat: running, no state change"
                     + (f", coverage={cov['n_observed']}/{cov['n_required']}" if cov is not None else ""))
                last_emit = now()
            sleep(max(0.0, min(float(interval), (until - now()).total_seconds())))
    except KeyboardInterrupt:
        reason = "interrupted"
    obs._meta_set("stop_reason", reason)
    obs._meta_set("stopped_at", stamp())
    report_error = None
    try:
        write_report(obs.out_dir, now=now())
        if cov is None:
            cov = compute_coverage(obs.sdb, now=now(), until=until)
    except Exception as exc:  # noqa: BLE001
        report_error = f"{type(exc).__name__}:{exc}"
    emit(f"[{stamp()}] STOP reason={reason}"
         + (f"; coverage {cov['n_observed']}/{cov['n_required']}; observed: {', '.join(cov['observed']) or 'none'}; "
            f"untested: {', '.join(cov['untested']) or 'none'}" if cov is not None else "")
         + (f"; report_error={report_error}" if report_error else f"; report written to {os.path.join(obs.out_dir, 'shadow_report.md')}"))
    return dict(stop_reason=reason, code=code, coverage=cov, report_error=report_error, new_decisions=total_new)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only, non-blocking shadow evaluation of the paper-order flow.")
    ap.add_argument("--paper-db", required=True, help="path to the bot's paper-order SQLite file (opened read-only)")
    ap.add_argument("--out", required=True, help="output folder for the shadow's own files (must not be the bot's data folder)")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--assumed-leverage", type=float, default=1.0)
    ap.add_argument("--loop", type=float, default=0.0,
                    help="repeat every N seconds until coverage is complete or --until passes (0 = run once)")
    ap.add_argument("--until", default=DEFAULT_UNTIL, help=f"loop deadline, ISO-8601 with timezone, UTC (default {DEFAULT_UNTIL})")
    ap.add_argument("--max-events", type=int, default=None,
                    help="optional emergency limit: stop the loop after this many NEW decisions logged by this invocation")
    ap.add_argument("--heartbeat", type=float, default=0.0, help="optional: print a liveness line every N seconds (default off)")
    ap.add_argument("--reset", action="store_true", help="delete the SHADOW's own previous output first (never the bot's files)")
    args = ap.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(os.path.dirname(os.path.abspath(args.paper_db))):
        print("refusing: --out must not be the folder that holds the bot's database", file=sys.stderr)
        return 2
    try:
        until = parse_until(args.until)
    except ValueError as exc:
        print(f"refusing: bad --until: {exc}", file=sys.stderr)
        return 2
    if args.max_events is not None and args.max_events <= 0:
        print("refusing: --max-events must be positive", file=sys.stderr)
        return 2
    obs = ShadowObserver(args.paper_db, args.out, timeframe=args.timeframe, assumed_leverage=args.assumed_leverage, reset=args.reset)
    code = 0
    try:
        if args.loop > 0:
            code = run_loop(obs, args.loop, until, max_events=args.max_events, heartbeat_s=args.heartbeat)["code"]
        else:
            s = obs.run_once()
            print(json.dumps(s, default=str))
            code = 2 if s["status"] == "SCHEMA_DRIFT" else 0
    finally:
        obs.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
