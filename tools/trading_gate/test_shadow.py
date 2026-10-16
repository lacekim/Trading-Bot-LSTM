"""
Tests for trading_gate.shadow / shadow_report. Hermetic: they build a synthetic copy of the bot's paper-order schema (DDL copied
from execution/order_manager.py) and mirror the bot's own accounting; nothing here imports or touches the bot.
Run from the folder that CONTAINS trading_gate/:   python3 -m unittest trading_gate.test_shadow
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

try:
    from . import shadow as S
    from . import shadow_report as R
    from . import verify_shadow_lookahead as V
except ImportError:
    import shadow as S
    import shadow_report as R
    import verify_shadow_lookahead as V

DDL = """
CREATE TABLE account (id INTEGER PRIMARY KEY CHECK(id=1), starting_balance REAL NOT NULL, cash REAL NOT NULL,
  realized_pnl REAL NOT NULL DEFAULT 0, fees REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
CREATE TABLE signals (signal_id TEXT PRIMARY KEY, candle_timestamp TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT NOT NULL,
  probability REAL, market_price REAL NOT NULL, status TEXT NOT NULL, reason TEXT, created_at TEXT NOT NULL);
CREATE TABLE orders (order_id TEXT PRIMARY KEY, signal_id TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
  status TEXT NOT NULL, expected_price REAL NOT NULL, fill_price REAL NOT NULL, notional REAL NOT NULL, fee REAL NOT NULL,
  created_at TEXT NOT NULL, order_kind TEXT NOT NULL DEFAULT 'ENTRY', reduce_only INTEGER NOT NULL DEFAULT 0,
  model_price REAL, observed_price REAL, observed_at TEXT, price_source TEXT, model_to_observed_bps REAL);
CREATE TABLE positions (position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL UNIQUE, direction TEXT NOT NULL, entry_time TEXT NOT NULL,
  entry_price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL, collateral REAL NOT NULL, leverage REAL NOT NULL,
  stop_price REAL NOT NULL, target_price REAL NOT NULL, entry_fee REAL NOT NULL, signal_id TEXT NOT NULL, current_price REAL NOT NULL,
  unrealized_pnl REAL NOT NULL DEFAULT 0);
CREATE TABLE closed_trades (position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, direction TEXT NOT NULL, entry_time TEXT NOT NULL,
  exit_time TEXT NOT NULL, entry_price REAL NOT NULL, exit_price REAL NOT NULL, quantity REAL NOT NULL, gross_pnl REAL NOT NULL,
  fees REAL NOT NULL, net_pnl REAL NOT NULL, exit_reason TEXT NOT NULL, signal_id TEXT NOT NULL);
CREATE TABLE equity_history (timestamp TEXT PRIMARY KEY, equity REAL NOT NULL, cash REAL NOT NULL, realized_pnl REAL NOT NULL,
  unrealized_pnl REAL NOT NULL);
CREATE TABLE daily_risk (trading_date TEXT PRIMARY KEY, start_equity REAL NOT NULL, realized_pnl REAL NOT NULL DEFAULT 0,
  trades_opened INTEGER NOT NULL DEFAULT 0, failed_orders INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
CREATE TABLE qualified_market_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
  symbol TEXT NOT NULL, min_price REAL NOT NULL, max_price REAL NOT NULL, spread_bps REAL NOT NULL, latency_ms REAL NOT NULL,
  age_seconds REAL NOT NULL, status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '');
CREATE TABLE signal_model_lineage (signal_id TEXT PRIMARY KEY, model_side TEXT NOT NULL, model_version TEXT NOT NULL, recorded_at TEXT NOT NULL);
"""

DAY = "2026-10-05"
FIXED_NOW = datetime(2026, 10, 5, 0, 0, 0, tzinfo=timezone.utc)      # campaign start used by the fixtures


def ts(h: int, m: int = 0, s: int = 0, day: str = DAY) -> str:
    return f"{day}T{h:02d}:{m:02d}:{s:02d}+00:00"


class Bot:
    """A tiny stand-in for the bot's paper DB that keeps the bot's own accounting identities."""

    def __init__(self, path: str, wal: bool = False, start: float = 10000.0):
        self.path, self.start = path, start
        self.c = sqlite3.connect(path, isolation_level=None)
        if wal:
            self.c.execute("PRAGMA journal_mode=WAL")
        self.c.executescript(DDL)
        self.c.execute("INSERT INTO account VALUES(1,?,?,0,0,?)", (start, start, ts(0)))
        self.pos: dict[str, dict] = {}
        self.n_sig = 0
        self.captures: list[tuple] = []          # what a running poller would have captured while the position was open
        self.simulate_poller = True

    def equity(self, when: str, eq: float) -> None:
        self.c.execute("INSERT OR REPLACE INTO equity_history VALUES(?,?,?,0,0)", (when, eq, eq))

    def daily(self, date: str = DAY, start_equity: float = 10000.0) -> None:
        self.c.execute("INSERT OR REPLACE INTO daily_risk VALUES(?,?,0,0,0,?)", (date, start_equity, ts(0)))

    def baseline(self) -> "Bot":
        self.equity(ts(0, 30), self.start)
        self.equity(ts(6), self.start)
        self.daily()
        return self

    def sig(self, sid, symbol, direction, candle, price, status="ACCEPTED", reason="", created=None) -> None:
        self.c.execute("INSERT INTO signals VALUES(?,?,?,?,?,?,?,?,?)",
                       (sid, candle, symbol, direction, 0.6, price, status, reason, created or ts(12, 0, 7)))

    def entry(self, sid, symbol, side, notional, created, price=100.0, stop_pct=2.0, leverage=1.0, open_position=True,
              observed_ago=8, with_signal=True, candle=None, signal_direction=None, stop=None, order_id=None,
              observed_at="auto", observed_price="auto", position_row=True, fill_price=None) -> dict:
        slip = 0.0007
        fill = fill_price if fill_price is not None else price * (1 + slip if side == "LONG" else 1 - slip)
        qty = notional / fill if isinstance(notional, (int, float)) and isinstance(fill, (int, float)) and fill else 0.0
        fee = notional * 5 / 1e4 if isinstance(notional, (int, float)) else 0.0
        collateral = notional / leverage if isinstance(notional, (int, float)) else 0.0
        try:
            t0 = datetime.fromisoformat(created)
        except ValueError:                                  # deliberately malformed timestamps in some tests
            t0 = datetime(2026, 10, 5, 10, 3, 7, tzinfo=timezone.utc)
        if observed_at == "auto":
            observed_at = (t0 - timedelta(seconds=observed_ago)).isoformat(timespec="seconds")
        if observed_price == "auto":
            observed_price = price
        if candle is None:
            candle = (t0.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        if with_signal:
            self.sig(sid, symbol, signal_direction or side, candle, price, created=created)
        order_id = order_id or f"po_{sid}"
        self.c.execute(
            "INSERT INTO orders(order_id,signal_id,symbol,side,status,expected_price,fill_price,notional,fee,created_at,order_kind,"
            "reduce_only,model_price,observed_price,observed_at,price_source,model_to_observed_bps) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (order_id, sid, symbol, side, "FILLED", price, fill, notional, fee, created, "ENTRY", 0, price, observed_price,
             observed_at, "GMX_TICKER_SNAPSHOT", 0.0))
        sgn = -1 if side == "LONG" else 1
        nf = fill if isinstance(fill, (int, float)) else price
        stop_price = stop if stop is not None else nf * (1 + sgn * stop_pct / 100)
        target = nf * (1 - sgn * 0.04)
        info = dict(symbol=symbol, direction=side, fill=fill, qty=qty, notional=notional, collateral=collateral, fee=fee,
                    created=created, sid=sid)
        self.pos[sid] = info
        if position_row:
            row = (f"pp_{sid}", symbol, side, created, fill, qty, notional, collateral, leverage, stop_price, target, fee, sid, fill, 0.0)
            self.captures.append(row)
            if open_position:
                self.c.execute("INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        self.c.execute("UPDATE account SET cash=cash-?-?, fees=fees+?", (collateral, fee, fee))
        return info

    def exit(self, sid, when, pnl=-5.0, qty_ratio=1.0, reason="stop_loss", position_id=None, remove_position=True,
             closing_signal=None) -> None:
        p = self.pos[sid]
        qty = p["qty"] * qty_ratio
        exit_fee = p["notional"] * qty_ratio * 5 / 1e4
        gross = pnl
        net = gross - p["fee"] * qty_ratio - exit_fee
        exit_price = p["fill"] * (1 + gross / (p["notional"] * qty_ratio) * (1 if p["direction"] == "LONG" else -1))
        pid = position_id or f"pp_{sid}"
        self.c.execute("INSERT INTO closed_trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (pid, p["symbol"], p["direction"], p["created"], when, p["fill"], exit_price, qty, gross,
                        p["fee"] * qty_ratio + exit_fee, net, reason, closing_signal or f"monitor_{sid}"))
        if remove_position and qty_ratio >= 1.0:
            self.c.execute("DELETE FROM positions WHERE position_id=?", (f"pp_{sid}",))
        self.c.execute("UPDATE account SET cash=cash+?+?, realized_pnl=realized_pnl+?, fees=fees+?",
                       (p["collateral"] * qty_ratio, gross - exit_fee, net, exit_fee))

    def close(self) -> None:
        self.c.close()


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.dbdir = os.path.join(self.dir, "bot")
        os.makedirs(self.dbdir)
        self.db = os.path.join(self.dbdir, "paper.sqlite3")
        self.out = os.path.join(self.dir, "shadow_out")
        self.bot = Bot(self.db).baseline()
        self._observers: list = []

    def tearDown(self):
        for o in self._observers:
            o.close()
        try:
            self.bot.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def run_shadow(self, out=None, limits=None, **kw):
        kw.setdefault("now", lambda: FIXED_NOW)
        obs = S.ShadowObserver(self.db, out or self.out, limits=limits, **kw)
        self._observers.append(obs)
        if self.bot.simulate_poller:
            for r in self.bot.captures:
                obs.sdb.execute("INSERT OR IGNORE INTO captured_positions(position_id,symbol,direction,stop_price,target_price,leverage,"
                                "collateral,quantity,notional,first_seen_at,source) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                (r[0], r[1], r[2], r[9], r[10], r[8], r[7], r[5], r[6], r[3], S.LIVE_STOP_SOURCE))
        summary = obs.run_once()
        return summary

    @staticmethod
    def read_text(path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def run_custom(self, edit=None, out=None, now=None, **kw):
        """Like run_shadow, but `edit(obs)` may change the captured rows before the pass (to simulate odd captures)."""
        obs = S.ShadowObserver(self.db, out or self.out, now=now or (lambda: FIXED_NOW), **kw)
        self._observers.append(obs)
        for r in self.bot.captures:
            obs.sdb.execute("INSERT OR IGNORE INTO captured_positions(position_id,symbol,direction,stop_price,target_price,leverage,"
                            "collateral,quantity,notional,first_seen_at,source) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (r[0], r[1], r[2], r[9], r[10], r[8], r[7], r[5], r[6], r[3], S.LIVE_STOP_SOURCE))
        if edit:
            edit(obs)
        return obs.run_once()

    def coverage(self, out=None, now=None, until=None) -> dict:
        db = sqlite3.connect(os.path.join(out or self.out, "shadow_decisions.sqlite3"))
        try:
            return R.compute_coverage(db, now=now or (FIXED_NOW + timedelta(days=3)), until=until)
        finally:
            db.close()

    def decisions(self, out=None) -> dict:
        db = sqlite3.connect(os.path.join(out or self.out, "shadow_decisions.sqlite3"))
        db.row_factory = sqlite3.Row
        res = {}
        for r in db.execute("SELECT * FROM decisions ORDER BY id"):
            d = dict(r)
            for k in ("reasons", "checks", "warnings", "adapter_errors", "data_gaps", "flags"):
                d[k] = json.loads(d[k])
            res[d["order_id"]] = d
        db.close()
        return res

    def recon(self, kind: str, out=None) -> dict:
        db = sqlite3.connect(os.path.join(out or self.out, "shadow_decisions.sqlite3"))
        row = db.execute("SELECT ok, details FROM reconciliations WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        db.close()
        return dict(ok=bool(row[0]), details=json.loads(row[1]))

    def one(self, out=None) -> dict:
        d = self.decisions(out)
        self.assertEqual(len(d), 1, list(d))
        return next(iter(d.values()))

    def assertRej(self, d, reason, cls=None):
        self.assertEqual(d["approved"], 0, d["reasons"])
        self.assertTrue(any(reason in r for r in d["reasons"]), f"{reason} not in {d['reasons']}")
        if cls:
            self.assertEqual(d["rejection_class"], cls)


class EntryMappingTests(Base):
    def test_valid_long_approved_all_checks_pass(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7))
        self.assertEqual(self.run_shadow()["status"], "OK")
        d = self.one()
        self.assertEqual(d["approved"], 1, (d["reasons"], d["adapter_errors"]))
        self.assertTrue(all(d["checks"].values()))
        self.assertEqual((d["kind"], d["action"], d["side"], d["borrow_check"]), ("ENTRY", "opened", "LONG", "not_applicable"))
        self.assertEqual(d["shadow_id"], "shadow:entry:po_s1")
        self.assertEqual(d["adapter_errors"], [])

    def test_valid_short_approved_and_borrow_simulated(self):
        self.bot.entry("s1", "AAA", "SHORT", 900, ts(12, 0, 7))
        self.run_shadow()
        d = self.one()
        self.assertEqual(d["approved"], 1, d["reasons"])
        self.assertEqual(d["borrow_check"], "simulated")

    def test_oversized_entry_is_a_rule_rejection(self):
        self.bot.entry("s1", "AAA", "LONG", 9400, ts(12, 0, 7))
        self.run_shadow()
        d = self.one()
        self.assertRej(d, "position_size_exceeds_limit", "rule")
        self.assertGreater(d["notional"], 9000)

    def test_missing_stop_is_logged_as_adapter_data_gap(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), position_row=False)
        self.run_shadow()
        d = self.one()
        self.assertRej(d, "missing_stop", "adapter_data_gap")
        self.assertIn("stop_price_unavailable", d["adapter_errors"])
        self.assertIn("stop_price", d["data_gaps"])

    def test_stale_signal_rejected(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), candle="2026-10-05 07:00:00")
        self.run_shadow()
        self.assertRej(self.one(), "signal_expired", "rule")

    def test_unclosed_candle_rejected(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), candle="2026-10-05 12:00:00")
        self.run_shadow()
        self.assertRej(self.one(), "signal_candle_not_closed")

    def test_naive_candle_timestamp_is_utc_and_flagged(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), candle="2026-10-05 11:00:00")
        self.run_shadow()
        d = self.one()
        self.assertTrue(d["flags"].get("candle_timestamp_naive_assumed_utc"))
        self.assertEqual(d["approved"], 1)

    def test_long_signal_short_order_rejected_and_intent_mismatch(self):
        self.bot.entry("s1", "AAA", "SHORT", 900, ts(12, 0, 7), signal_direction="LONG")
        self.run_shadow()
        d = self.one()
        self.assertRej(d, "signal_order_mismatch:direction")
        self.assertEqual(d["intent_mismatch"], 1)
        self.assertFalse(self.recon("intent_vs_orders")["ok"])
        self.assertEqual(self.recon("intent_vs_orders")["details"]["direction_mismatch"], ["s1"])

    def test_leverage_above_limit_from_captured_position(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), leverage=2.0)
        self.run_shadow()
        self.assertRej(self.one(), "leverage_exceeds_limit")

    def test_stop_distance_bounds(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), stop_pct=0.1)
        self.bot.entry("s2", "BBB", "LONG", 900, ts(12, 0, 8), stop_pct=8.0)
        self.run_shadow()
        d = self.decisions()
        self.assertRej(d["po_s1"], "stop_too_tight")
        self.assertRej(d["po_s2"], "stop_too_wide")

    def test_market_data_and_signal_price_checks(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7), observed_ago=300)
        self.bot.entry("s2", "BBB", "LONG", 900, ts(12, 0, 8), observed_price=103.0)
        self.run_shadow()
        d = self.decisions()
        self.assertRej(d["po_s1"], "stale_market_data")
        self.assertTrue(any("price_deviates" in r for r in d["po_s2"]["reasons"]), d["po_s2"]["reasons"])

    def test_spread_from_latest_snapshot_at_or_before_event(self):
        c = self.bot.c
        for rec, spread in ((ts(12, 0, 0), 5.0), (ts(12, 30, 0), 400.0)):          # the second is AFTER the order
            c.execute("INSERT INTO qualified_market_snapshots(observed_at,recorded_at,symbol,min_price,max_price,spread_bps,"
                      "latency_ms,age_seconds,status) VALUES(?,?,?,?,?,?,?,?,?)", (rec, rec, "AAA", 100, 100, spread, 1, 1, "healthy"))
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7))
        self.run_shadow()
        d = self.one()
        self.assertEqual(d["approved"], 1, d["reasons"])
        self.assertEqual(json.loads(d["inputs_json"])["market_snapshot"]["spread_bps"], 5.0)

    def test_wide_spread_rejected(self):
        self.bot.c.execute("INSERT INTO qualified_market_snapshots(observed_at,recorded_at,symbol,min_price,max_price,spread_bps,"
                           "latency_ms,age_seconds,status) VALUES(?,?,?,?,?,?,?,?,?)",
                           (ts(12, 0, 0), ts(12, 0, 0), "AAA", 100, 100, 400.0, 1, 1, "healthy"))
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7))
        self.run_shadow()
        self.assertRej(self.one(), "spread_too_wide")


class PortfolioStateTests(Base):
    def test_position_already_open_and_opposite_conflict(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.entry("b", "AAA", "SHORT", 900, ts(11, 0, 7), open_position=False)
        self.bot.entry("c", "AAA", "LONG", 900, ts(12, 0, 7), open_position=False)
        self.run_shadow()
        d = self.decisions()
        self.assertEqual(d["po_a"]["approved"], 1, d["po_a"]["reasons"])
        self.assertRej(d["po_b"], "opposite_position_open_close_first")
        self.assertRej(d["po_c"], "position_already_open")

    def test_net_exposure_cap_and_short_reduces_net(self):
        for i, sym in enumerate(("A", "B", "C")):
            self.bot.entry(f"p{i}", sym, "LONG", 900, ts(9, i, 7))
        self.bot.entry("n_long", "D", "LONG", 900, ts(12, 0, 7), open_position=False)
        self.bot.entry("n_short", "E", "SHORT", 900, ts(12, 1, 7), open_position=False)
        self.run_shadow()
        d = self.decisions()
        self.assertRej(d["po_n_long"], "net_exposure_exceeds_limit")
        self.assertEqual(d["po_n_short"]["approved"], 1, d["po_n_short"]["reasons"])

    def test_max_open_positions(self):
        for i in range(5):
            self.bot.entry(f"p{i}", f"S{i}", "LONG", 400, ts(9, i, 7))
        self.bot.entry("x", "ZZZ", "LONG", 400, ts(12, 0, 7), open_position=False)
        self.run_shadow()
        self.assertRej(self.decisions()["po_x"], "max_open_positions_reached")

    def test_daily_loss_blocks_entries_but_not_exits(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(8, 0, 7))
        self.bot.entry("c", "CCC", "LONG", 900, ts(8, 30, 7))
        self.bot.exit("a", ts(10, 0, 0), pnl=-150.0)                                 # 1.5% of 10,000 start equity
        self.bot.entry("b", "BBB", "LONG", 900, ts(12, 0, 7), open_position=False)
        self.bot.exit("c", ts(12, 30, 0), pnl=-10.0)
        self.run_shadow()
        d = self.decisions()
        self.assertEqual(d["shadow_exit_pp_a"]["approved"], 1)
        self.assertRej(d["po_b"], "daily_loss_limit", "rule")
        self.assertEqual(d["shadow_exit_pp_c"]["approved"], 1, d["shadow_exit_pp_c"]["reasons"])      # reduction still allowed
        self.assertEqual(d["shadow_exit_pp_c"]["action"], "closed")

    def test_drawdown_blocks_entries_but_not_exits(self):
        self.bot.equity(ts(7), 10000.0)
        self.bot.equity(ts(8, 30), 10000.0)
        self.bot.entry("a", "AAA", "LONG", 900, ts(9, 0, 7))
        self.bot.equity(ts(11), 9400.0)                                               # 6% below the 10,000 peak
        self.bot.entry("b", "BBB", "LONG", 900, ts(12, 0, 7), open_position=False)
        self.bot.exit("a", ts(12, 30, 0), pnl=-1.0)
        self.run_shadow()
        d = self.decisions()
        self.assertRej(d["po_b"], "max_drawdown_limit", "rule")
        self.assertEqual(d["shadow_exit_pp_a"]["approved"], 1)

    def test_equity_loss_limit_uses_start_of_day_equity(self):
        self.bot.daily(DAY, 10000.0)
        self.bot.equity(ts(11), 9850.0)                                               # -1.5% vs day start, no realised loss
        self.bot.entry("b", "BBB", "LONG", 900, ts(12, 0, 7))
        self.run_shadow()
        self.assertRej(self.one(), "daily_equity_loss_limit")

    def test_equity_is_point_in_time_future_rows_ignored(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(12, 0, 7))
        self.bot.equity(ts(13), 50000.0)                                              # AFTER the event: must not be used
        self.bot.equity(ts(12, 0, 7), 77777.0)                                        # same second: strictly-before rule excludes it
        self.run_shadow()
        d = self.one()
        self.assertEqual(d["equity"], 10000.0)
        self.assertEqual(d["flags"]["equity_source"], "equity_history")

    def test_later_events_do_not_affect_earlier_decisions(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(14, 0, 0), pnl=-500.0)                                  # big later loss
        self.run_shadow()
        d = self.decisions()
        self.assertEqual(d["po_a"]["day_realized_pnl"], 0.0)
        self.assertEqual(d["po_a"]["approved"], 1, d["po_a"]["reasons"])

    def test_consecutive_loss_and_trades_today_replayed(self):
        for i in range(5):
            self.bot.entry(f"l{i}", f"L{i}", "LONG", 400, ts(8, i, 7))
            self.bot.exit(f"l{i}", ts(8, i, 40), pnl=-1.0)
        self.bot.entry("z", "ZZZ", "LONG", 400, ts(12, 0, 7), open_position=False)
        self.run_shadow()
        d = self.decisions()["po_z"]
        self.assertRej(d, "max_consecutive_losses")
        self.assertEqual(json.loads(d["inputs_json"])["risk_state"]["trades_opened_today"], 5)

    def test_duplicate_signal_order_rejected(self):
        self.bot.entry("s1", "AAA", "LONG", 900, ts(12, 0, 7))
        self.bot.entry("s1", "BBB", "LONG", 900, ts(12, 0, 9), with_signal=False, order_id="po_dup", position_row=False)
        self.run_shadow()
        d = self.decisions()
        self.assertEqual(d["po_s1"]["approved"], 1, d["po_s1"]["reasons"])
        self.assertRej(d["po_dup"], "duplicate_order")
        self.assertFalse(self.recon("intent_vs_orders")["ok"])
        self.assertEqual(self.recon("intent_vs_orders")["details"]["duplicate_signal_orders"], ["s1"])

    def test_partial_exit_reduces_and_remainder_stays_open(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0, qty_ratio=0.5)
        self.run_shadow()
        d = self.decisions()["shadow_exit_pp_a"]
        self.assertEqual((d["action"], d["approved"]), ("reduced", 1), d["reasons"])
        notes = self.recon("account_vs_replay")["details"]["notes"]
        self.assertIn("partial_exit:pp_a", notes)
        pr = self.recon("positions_vs_replay")                                        # the bot's table has no remainder
        self.assertFalse(pr["ok"])

    def test_exit_mapping_fields(self):
        self.bot.entry("a", "AAA", "SHORT", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=3.0, reason="take_profit")
        self.run_shadow()
        d = self.decisions()["shadow_exit_pp_a"]
        self.assertEqual((d["approved"], d["warnings"], d["adapter_errors"]), (1, [], []))
        inp = json.loads(d["inputs_json"])
        o, m = inp["proposed_order"], inp["market_snapshot"]
        self.assertEqual((o["order_kind"], o["reduce_only"], o["side"], o["symbol"]), ("EXIT", True, "SHORT", "AAA"))
        self.assertEqual(o["idempotency_key"], "exit:pp_a")
        self.assertAlmostEqual(o["quantity"], 900 / (100.0 * 0.9993), places=6)
        self.assertEqual(m["observed_at"], "2026-10-05 11:00:00+00:00")
        self.assertEqual((d["exit_reason"], d["action"], d["event_time"]), ("take_profit", "closed", ts(11, 0, 0)))
        self.assertTrue(d["flags"]["exit_market_price_is_order_price"])

    def test_exit_larger_than_position_rejected(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0)
        self.bot.c.execute("UPDATE closed_trades SET quantity=quantity*2")
        self.run_shadow()
        self.assertRej(self.decisions()["shadow_exit_pp_a"], "reduce_exceeds_position")

    def test_exit_without_known_entry(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0)
        self.bot.c.execute("DELETE FROM orders")
        self.run_shadow()
        d = self.decisions()["shadow_exit_pp_a"]
        self.assertEqual(d["action"], "unknown")
        self.assertIn("exit_for_unknown_position", d["adapter_errors"])
        self.assertRej(d, "no_position_to_reduce")


class MalformedAndMissingTests(Base):
    def test_malformed_rows_are_logged_not_raised_and_later_rows_still_processed(self):
        self.bot.entry("bad1", "AAA", "LONG", 900, ts(10, 0, 7), fill_price="abc", position_row=False)
        self.bot.entry("bad2", "BBB", "LONG", -5, ts(10, 1, 7), position_row=False)
        self.bot.entry("bad3", "CCC", "LONG", 900, ts(10, 2, 7), observed_at="garbage", position_row=False)
        self.bot.entry("bad4", "DDD", "LONG", 900, "not-a-time", position_row=False)
        self.bot.entry("bad5", "EEE", "LONG", 900, ts(10, 4, 7), observed_price=float("nan"), position_row=False)
        self.bot.entry("good", "GGG", "LONG", 900, ts(12, 0, 7))
        s = self.run_shadow()
        self.assertEqual(s["status"], "OK", s)
        d = self.decisions()
        for k in ("po_bad1", "po_bad2", "po_bad3", "po_bad5"):
            self.assertEqual(d[k]["approved"], 0, k)
            self.assertTrue(d[k]["adapter_errors"], k)
        self.assertIn("bad_number:orders.fill_price", d["po_bad1"]["adapter_errors"])
        self.assertIn("non_positive:orders.notional", d["po_bad2"]["adapter_errors"])
        self.assertIn("bad_timestamp:orders.observed_at", d["po_bad3"]["adapter_errors"])
        self.assertIsNone(d["po_bad4"]["approved"])
        self.assertEqual(d["po_bad4"]["decision"], "NOT_EVALUATED")
        self.assertTrue(any(e.startswith("event_time_unusable") for e in d["po_bad4"]["adapter_errors"]))
        self.assertEqual(d["po_good"]["approved"], 1, d["po_good"]["reasons"])

    def test_missing_signal_row_daily_row_and_equity_history(self):
        self.bot.c.execute("DELETE FROM equity_history")
        self.bot.c.execute("DELETE FROM daily_risk")
        self.bot.entry("a", "AAA", "LONG", 900, ts(12, 0, 7), with_signal=False)
        self.run_shadow()
        d = self.one()
        self.assertEqual(d["approved"], 0)
        for e in ("signal_row_missing", "daily_risk_row_missing", "equity_history_unavailable_before_event"):
            self.assertIn(e, d["adapter_errors"])
        self.assertEqual(d["flags"]["equity_source"], "replay_realized_only")
        self.assertEqual(d["rejection_class"], "adapter_data_gap")

    def test_malformed_exit_rows(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0)
        self.bot.c.execute("UPDATE closed_trades SET exit_price='x', quantity=-1")
        s = self.run_shadow()
        self.assertEqual(s["status"], "OK")
        d = self.decisions()["shadow_exit_pp_a"]
        self.assertEqual(d["approved"], 0)
        self.assertTrue(d["adapter_errors"])

    def test_schema_drift_aborts_without_evaluating(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(12, 0, 7))
        self.bot.c.execute("ALTER TABLE orders RENAME COLUMN fee TO fee_renamed")
        s = self.run_shadow()
        self.assertEqual(s["status"], "SCHEMA_DRIFT")
        self.assertIn("missing_column:orders.fee", s["schema_problems"])
        self.assertEqual(self.decisions(), {})
        self.assertFalse(self.recon("schema")["ok"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(S.main(["--paper-db", self.db, "--out", os.path.join(self.dir, "o2")]), 2)

    def test_empty_database_runs_clean(self):
        s = self.run_shadow()
        self.assertEqual((s["status"], s["events_total"]), ("OK", 0))


class NonBlockingTests(Base):
    def test_bot_database_is_never_modified(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0)
        self.bot.entry("b", "BBB", "LONG", 900, ts(12, 0, 7))
        self.bot.c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        before = (md5(self.db), os.stat(self.db).st_mtime_ns, os.path.getsize(self.db))
        self.run_shadow()
        self.run_shadow()
        self.assertEqual((md5(self.db), os.stat(self.db).st_mtime_ns, os.path.getsize(self.db)), before)
        self.bot.c.execute("INSERT INTO equity_history VALUES('2030-01-01T00:00:00+00:00',1,1,0,0)")   # bot can still write

    def test_readonly_connection_cannot_write(self):
        conn = S.open_readonly(self.db)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO equity_history VALUES('x',1,1,0,0)")
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE evil(x)")
        conn.close()

    def test_wal_database_is_not_modified(self):
        path = os.path.join(self.dbdir, "wal.sqlite3")
        bot = Bot(path, wal=True).baseline()
        bot.entry("a", "AAA", "LONG", 900, ts(12, 0, 7))
        bot.c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        before = md5(path)
        obs = S.ShadowObserver(path, os.path.join(self.dir, "o_wal"))
        self._observers.append(obs)
        self.assertEqual(obs.run_once()["status"], "OK")
        self.assertEqual(md5(path), before)
        bot.close()

    def test_outputs_only_in_out_dir(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(12, 0, 7))
        listing_before = sorted(os.listdir(self.dbdir))
        self.run_shadow()
        self.assertEqual(sorted(os.listdir(self.dbdir)), listing_before)
        self.assertEqual(set(os.listdir(self.out)), {"shadow_decisions.sqlite3", "shadow_decisions.jsonl", "gate_audit.sqlite3"})

    def test_cli_refuses_out_dir_equal_to_bot_folder(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(S.main(["--paper-db", self.db, "--out", self.dbdir]), 2)

    def test_run_once_never_raises(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(12, 0, 7))
        with mock.patch.object(S, "open_readonly", side_effect=RuntimeError("disk gone")):
            s = self.run_shadow()
        self.assertEqual(s["status"], "ERROR")
        self.assertIn("RuntimeError", s["error"])
        with mock.patch.object(S.TradingGate, "evaluate", side_effect=RuntimeError("boom")):
            s = self.run_shadow(out=os.path.join(self.dir, "o3"))
        self.assertEqual(s["status"], "OK")
        d = self.one(os.path.join(self.dir, "o3"))
        self.assertEqual(d["decision"], "ADAPTER_ERROR")
        self.assertIsNone(d["approved"])
        self.assertTrue(any("adapter_exception:RuntimeError" in e for e in d["adapter_errors"]))

    def test_nonexistent_paper_db_does_not_raise_or_create_it(self):
        missing = os.path.join(self.dbdir, "nope.sqlite3")
        obs = S.ShadowObserver(missing, os.path.join(self.dir, "o4"))
        self._observers.append(obs)
        s = obs.run_once()
        self.assertEqual(s["status"], "ERROR")
        self.assertFalse(os.path.exists(missing))

    def test_gate_has_no_order_authority_in_shadow(self):
        # the observer exposes no way to submit or block: there is no callable that returns a decision to an order path
        public = [n for n in dir(S.ShadowObserver) if not n.startswith("_")]
        self.assertEqual(sorted(public), ["close", "run_once"])


class IdempotencyAndDeterminismTests(Base):
    def test_rerun_logs_nothing_new_and_incremental_events_use_prior_state(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.assertEqual(self.run_shadow()["events_new"], 1)
        self.assertEqual(self.run_shadow()["events_new"], 0)
        self.bot.entry("b", "AAA", "SHORT", 900, ts(12, 0, 7), open_position=False)       # conflicts with the still-open "a"
        s = self.run_shadow()
        self.assertEqual(s["events_new"], 1)
        self.assertRej(self.decisions()["po_b"], "opposite_position_open_close_first")
        with open(os.path.join(self.out, "shadow_decisions.jsonl")) as fh:
            self.assertEqual(len(fh.readlines()), 2)

    def test_deterministic_ids_and_decisions(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.entry("b", "BBB", "LONG", 9000, ts(11, 0, 7))
        self.bot.exit("a", ts(11, 30, 0), pnl=-1.0)
        self.run_shadow(out=os.path.join(self.dir, "x1"))
        self.run_shadow(out=os.path.join(self.dir, "x2"))
        a, b = self.decisions(os.path.join(self.dir, "x1")), self.decisions(os.path.join(self.dir, "x2"))
        key = lambda d: {k: (v["shadow_id"], v["approved"], v["reasons"], v["equity"]) for k, v in d.items()}  # noqa: E731
        self.assertEqual(key(a), key(b))

    def test_stop_capture_persists_after_position_closes(self):
        self.bot.simulate_poller = False               # rely on the observer's own capture
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.run_shadow()
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0)
        self.run_shadow()
        db = sqlite3.connect(os.path.join(self.out, "shadow_decisions.sqlite3"))
        row = db.execute("SELECT stop_price, leverage FROM captured_positions WHERE position_id='pp_a'").fetchone()
        db.close()
        self.assertIsNotNone(row)
        self.assertLess(row[0], 100.0)

    def test_same_second_events_flagged_and_exit_applied_before_entry(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(12, 0, 7), pnl=-1.0)
        self.bot.entry("b", "AAA", "SHORT", 900, ts(12, 0, 7))                       # reversal in the same second
        self.run_shadow()
        d = self.decisions()
        self.assertTrue(d["po_b"]["flags"].get("ordering_ambiguous"))
        self.assertEqual(d["po_b"]["approved"], 1, d["po_b"]["reasons"])             # exit first, so no conflict


class ExtraStateTests(Base):
    def test_prior_day_loss_does_not_count_toward_todays_limit(self):
        yesterday = "2026-10-04"
        self.bot.daily(yesterday, 10000.0)
        self.bot.equity(ts(20, day=yesterday), 10000.0)
        self.bot.entry("old", "OLD", "LONG", 900, ts(10, 0, 7, day=yesterday))
        self.bot.exit("old", ts(11, 0, 0, day=yesterday), pnl=-300.0)
        self.bot.equity(ts(23, day=yesterday), 10000.0)                            # equity recovered; only realised P&L differs
        self.bot.entry("new", "NEW", "LONG", 900, ts(12, 0, 7))
        self.run_shadow()
        d = self.decisions()["po_new"]
        self.assertEqual(d["day_realized_pnl"], 0.0)
        self.assertEqual(d["approved"], 1, d["reasons"])

    def test_loss_streak_counts_most_recent_losses(self):
        # win first, then five losses -> streak 5 (block); five losses, a win, one loss -> streak 1 (allow)
        self.bot.entry("w0", "W0", "LONG", 400, ts(7, 0, 7))
        self.bot.exit("w0", ts(7, 0, 40), pnl=+5.0)
        for i in range(5):
            self.bot.entry(f"l{i}", f"L{i}", "LONG", 400, ts(8, i, 7))
            self.bot.exit(f"l{i}", ts(8, i, 40), pnl=-1.0)
        self.bot.entry("z", "ZZZ", "LONG", 400, ts(9, 0, 7))
        self.run_shadow()
        self.assertRej(self.decisions()["po_z"], "max_consecutive_losses")
        # second scenario in a fresh DB
        self.bot.close()
        os.remove(self.db)
        self.bot = Bot(self.db).baseline()
        for i in range(5):
            self.bot.entry(f"l{i}", f"L{i}", "LONG", 400, ts(8, i, 7))
            self.bot.exit(f"l{i}", ts(8, i, 40), pnl=-1.0)
        self.bot.entry("w1", "W1", "LONG", 400, ts(9, 0, 7))
        self.bot.exit("w1", ts(9, 0, 40), pnl=+5.0)
        self.bot.entry("l9", "L9", "LONG", 400, ts(9, 1, 7))
        self.bot.exit("l9", ts(9, 1, 40), pnl=-1.0)
        self.bot.entry("z", "ZZZ", "LONG", 400, ts(9, 2, 7))
        self.run_shadow(out=os.path.join(self.dir, "o_b"))
        d = self.decisions(os.path.join(self.dir, "o_b"))["po_z"]
        self.assertEqual(json.loads(d["inputs_json"])["risk_state"]["consecutive_losses"], 1)
        self.assertEqual(d["approved"], 1, d["reasons"])

    def test_accepted_signal_without_order_is_reported(self):
        self.bot.sig("orphan", "AAA", "LONG", "2026-10-05 11:00:00", 100.0, status="ACCEPTED")
        self.run_shadow()
        r = self.recon("intent_vs_orders")
        self.assertFalse(r["ok"])
        self.assertEqual(r["details"]["accepted_without_order"], ["orphan"])
        rep = R.build_report(self.out) if self.decisions() else None
        self.assertIsNone(rep)                                                      # no decisions -> nothing to report

    def test_partial_exit_leaves_remaining_notional_in_gate_inputs(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-1.0, qty_ratio=0.25)
        self.bot.entry("b", "BBB", "LONG", 400, ts(12, 0, 7))
        self.run_shadow()
        pos = json.loads(self.decisions()["po_b"]["inputs_json"])["portfolio_state"]["positions"]
        self.assertEqual(len(pos), 1)
        self.assertAlmostEqual(pos[0]["notional"], 675.0, places=6)
        self.assertAlmostEqual(pos[0]["quantity"], 675.0 / (100.0 * 1.0007), places=6)

    def test_replay_equity_formula_logged(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-12.0)
        self.bot.entry("b", "BBB", "LONG", 400, ts(11, 30, 7))
        self.bot.entry("c", "CCC", "LONG", 400, ts(12, 0, 7))
        self.run_shadow()
        net_a = -12.0 - 900 * 5 / 1e4 - 900 * 5 / 1e4
        expected = 10000 + net_a - 400 * 5 / 1e4
        self.assertAlmostEqual(self.decisions()["po_c"]["flags"]["equity_replay"], expected, places=5)


class LookaheadTests(Base):
    def test_truncation_invariance_on_a_rich_history(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(8, 0, 7))
        self.bot.entry("b", "BBB", "SHORT", 800, ts(8, 30, 7))
        self.bot.exit("a", ts(9, 0, 0), pnl=-40.0)
        self.bot.equity(ts(9, 30), 9950.0)
        self.bot.entry("c", "AAA", "LONG", 9000, ts(10, 0, 7), open_position=False)       # oversized, accepted by the old bot
        self.bot.exit("b", ts(10, 30, 0), pnl=+10.0)
        self.bot.exit("c", ts(11, 0, 0), pnl=-100.0)
        self.bot.equity(ts(11, 30), 9800.0)
        self.bot.entry("d", "DDD", "LONG", 900, ts(12, 0, 7))
        self.bot.entry("e", "EEE", "LONG", 900, ts(12, 0, 7), signal_direction="SHORT", open_position=False)
        res = V.lookahead_audit(self.db, os.path.join(self.dir, "audit"))
        self.assertEqual(res["events_checked"], 8)
        self.assertEqual(res["mismatches"], [], res["mismatches"][:3])

    def test_audit_detects_a_deliberate_lookahead(self):
        # sanity check of the audit itself: if equity were read from the LAST row (future), the audit must flag a mismatch
        self.bot.entry("a", "AAA", "LONG", 900, ts(9, 0, 7))
        self.bot.equity(ts(10), 9000.0)
        self.bot.entry("b", "BBB", "LONG", 900, ts(12, 0, 7))
        with mock.patch.object(S.bisect, "bisect_left", side_effect=lambda a, x: len(a)):
            res = V.lookahead_audit(self.db, os.path.join(self.dir, "audit2"))
        self.assertFalse(res["ok"])


class ReconciliationTests(Base):
    def test_consistent_database_reconciles_exactly(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.entry("b", "BBB", "SHORT", 800, ts(10, 30, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-12.34)
        self.run_shadow()
        pv, av = self.recon("positions_vs_replay"), self.recon("account_vs_replay")
        self.assertTrue(pv["ok"], pv)
        self.assertTrue(av["ok"], av)
        for v in av["details"]["diffs"].values():
            self.assertLess(abs(v), 1e-9)
        self.assertTrue(self.recon("fills")["ok"])

    def test_tampered_positions_and_account_are_detected(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.c.execute("UPDATE positions SET quantity=quantity*1.5")
        self.run_shadow()
        pv = self.recon("positions_vs_replay")
        self.assertFalse(pv["ok"])
        self.assertEqual([a["type"] for a in pv["details"]["alerts"]], ["position_qty_mismatch"])
        self.bot.c.execute("UPDATE account SET cash=cash+50")
        self.run_shadow()
        av = self.recon("account_vs_replay")
        self.assertFalse(av["ok"])
        self.assertAlmostEqual(av["details"]["diffs"]["cash"], -50.0, places=6)

    def test_extra_or_missing_position_at_bot_detected(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.c.execute("DELETE FROM positions")
        self.bot.c.execute("INSERT INTO positions VALUES('pp_x','ZZZ','LONG',?,1,1,1,1,1,0.9,1.1,0,'x',1,0)", (ts(9),))
        self.run_shadow()
        types = sorted(a["type"] for a in self.recon("positions_vs_replay")["details"]["alerts"])
        self.assertEqual(types, ["position_missing_at_broker", "unexpected_position_at_broker"])

    def test_non_filled_order_status_raises_fill_alert(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.c.execute("UPDATE orders SET status='PARTIAL'")
        self.run_shadow()
        fr = self.recon("fills")
        self.assertFalse(fr["ok"])
        self.assertIn("partial_fill", [a["type"] for a in fr["details"]["alerts"]])
        self.bot.c.execute("UPDATE orders SET status='REJECTED'")
        self.run_shadow()
        self.assertIn("rejected_fill", [a["type"] for a in self.recon("fills")["details"]["alerts"]])


class ReportTests(Base):
    def test_report_counts_and_checklist(self):
        self.bot.entry("ok1", "AAA", "LONG", 900, ts(9, 0, 7))
        self.bot.entry("ok2", "BBB", "SHORT", 900, ts(9, 5, 7))
        self.bot.entry("big", "CCC", "LONG", 9000, ts(9, 10, 7), open_position=False)
        self.bot.entry("big2", "EEE", "LONG", 9000, ts(9, 12, 7), open_position=False)
        self.bot.entry("gap", "DDD", "LONG", 900, ts(9, 7, 7), position_row=False)
        self.bot.exit("ok1", ts(10, 0, 0), pnl=-1.0)
        self.bot.exit("ok2", ts(10, 1, 0), pnl=-1.0)
        self.bot.exit("big", ts(10, 2, 0), pnl=-1.0)
        self.bot.exit("big2", ts(10, 2, 30), pnl=-1.0)
        self.bot.exit("gap", ts(10, 3, 0), pnl=-1.0)
        self.run_shadow()
        rep = R.build_report(self.out)
        e = rep["decisions"]["entries"]
        self.assertEqual((e["evaluated"], e["approved"], e["rejected"]), (5, 2, 3))
        self.assertEqual(e["approval_rate_pct"], 40.0)
        self.assertEqual((e["rejected_rule_only"], e["rejected_data_gap_only"]), (2, 1))
        self.assertEqual(rep["decisions"]["exits"]["approved"], 5)
        self.assertEqual(rep["decisions"]["borrow_check_simulated"], 1)
        reasons = {r["reason"]: r["count"] for r in rep["top_rejection_reasons"]}
        self.assertEqual(reasons["position_size_exceeds_limit"], 2)
        self.assertEqual(reasons["missing_stop"], 1)
        self.assertEqual(rep["adapter_errors"]["by_error"]["stop_price_unavailable"], 1)
        status = {c["item"].split(".")[0]: c["status"] for c in rep["review_checklist"]}
        self.assertEqual(status["3"], "REVIEW")
        self.assertEqual(status["4"], "OK")
        self.assertEqual(status["5"], "OK")
        md = R.render_markdown(rep)
        self.assertIn("Shadow mode only: nothing was blocked", md)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(R.main(["--out", self.out]), 0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "shadow_report.json")))

    def test_report_flags_intent_and_duplicates_for_review(self):
        self.bot.entry("s1", "AAA", "SHORT", 900, ts(12, 0, 7), signal_direction="LONG")
        self.run_shadow()
        rep = R.build_report(self.out)
        status = {c["item"].split(".")[0]: c["status"] for c in rep["review_checklist"]}
        self.assertEqual(status["4"], "REVIEW")


def full_coverage_scenario(bot: "Bot") -> None:
    """Every required case occurs once in the bot's database (all events after FIXED_NOW, UTC)."""
    bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
    bot.entry("b", "BBB", "SHORT", 900, ts(10, 5, 7))
    bot.entry("c", "CCC", "LONG", 900, ts(10, 10, 7))
    bot.entry("d", "DDD", "LONG", 900, ts(10, 15, 7))
    bot.exit("a", ts(11, 0, 0), pnl=-5.0, reason="stop_loss")
    bot.exit("b", ts(11, 5, 0), pnl=5.0, reason="take_profit")
    bot.exit("c", ts(11, 10, 0), pnl=1.0, reason="window_exit")
    bot.exit("d", ts(12, 0, 0), pnl=-1.0, reason="signal_reversal", closing_signal="r1")
    bot.entry("r1", "DDD", "SHORT", 900, ts(12, 0, 0))                    # the closing signal opens the opposite side
    bot.daily("2026-10-06", 10000.0)
    bot.equity(ts(8, day="2026-10-06"), 10000.0)
    bot.entry("e", "EEE", "LONG", 900, ts(9, 0, 7, "2026-10-06"))         # next UTC date


class CoverageTests(Base):
    def test_each_exit_reason_and_short_entry_are_counted(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.entry("b", "BBB", "SHORT", 900, ts(10, 5, 7))
        self.bot.entry("c", "CCC", "LONG", 900, ts(10, 10, 7))
        self.bot.entry("d", "DDD", "LONG", 900, ts(10, 15, 7))
        self.bot.exit("a", ts(11, 0, 0), reason="stop_loss")
        self.bot.exit("b", ts(11, 5, 0), reason="take_profit")
        self.bot.exit("c", ts(11, 10, 0), reason="signal_reversal")
        self.bot.exit("d", ts(11, 15, 0), reason="window_exit")
        self.run_shadow()
        cov = self.coverage()
        self.assertEqual(sorted(cov["observed"]), sorted(["long_entry_stop_captured", "short_entry", "exit_stop_loss", "exit_take_profit",
                                                           "exit_signal_reversal", "exit_window_exit"]))
        self.assertEqual(sorted(cov["untested"]), ["daily_rollover_utc", "same_symbol_reversal"])
        self.assertFalse(cov["complete"])
        by = {c["id"]: c for c in cov["cases"]}
        self.assertEqual(by["exit_take_profit"]["evidence"][0]["order_id"], "shadow_exit_pp_b")
        self.assertEqual(by["short_entry"]["evidence"][0]["order_id"], "po_b")
        self.assertEqual(by["short_entry"]["first_at"], "2026-10-05T10:05:07+00:00")

    def test_untested_reason_no_signal_reached_gate(self):
        self.run_shadow()                                       # an empty bot database: nothing to judge
        cov = self.coverage()
        for c in cov["cases"]:
            self.assertEqual(c["untested_reason"], "NO_EVENT_REACHED_GATE", c["id"])
            self.assertEqual(c["gate_outcomes"]["related_events"], 0)
        md = R.render_markdown(R.build_report(self.out))
        self.assertIn("no signal reached the gate", md)
        self.assertIn("not a gate failure", md)

    def test_untested_reason_signal_rejected_by_gate(self):
        self.bot.entry("big", "AAA", "SHORT", 50000, ts(10, 0, 7))      # far above the position-size limit: gate rejects it
        self.run_shadow()
        by = {c["id"]: c for c in self.coverage()["cases"]}
        d = self.decisions()["po_big"]
        self.assertEqual(d["approved"], 0)
        self.assertEqual(by["short_entry"]["status"], "OBSERVED")        # the shadow observed it; the gate rejected it
        self.assertEqual((by["short_entry"]["gate_outcomes"]["gate_approved"], by["short_entry"]["gate_outcomes"]["gate_rejected"]), (0, 1))
        self.assertIsNone(by["short_entry"]["untested_reason"])
        self.assertEqual(by["exit_stop_loss"]["untested_reason"], "NO_EVENT_REACHED_GATE")
        self.assertEqual(by["long_entry_stop_captured"]["untested_reason"], "NO_EVENT_REACHED_GATE")

    def test_untested_reason_rejected_exit_and_condition_not_met(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7), position_row=False)    # approved long entry but no stop was ever captured
        self.bot.exit("a", ts(11, 0, 0), reason="signal_reversal", closing_signal="x9")
        self.run_shadow()
        by = {c["id"]: c for c in self.coverage()["cases"]}
        d = self.decisions()["po_a"]
        self.assertEqual(d["approved"], 0)                                           # the gate rejects an entry with no stop
        self.assertIn("missing_stop", "|".join(d["reasons"]))
        self.assertEqual(by["long_entry_stop_captured"]["untested_reason"], "SIGNAL_REJECTED_BY_GATE")
        self.assertEqual(by["long_entry_stop_captured"]["gate_outcomes"]["related_events"], 1)
        self.assertEqual(by["long_entry_stop_captured"]["gate_outcomes"]["gate_rejected"], 1)
        self.assertEqual(by["same_symbol_reversal"]["untested_reason"], "EVENT_SEEN_CONDITION_NOT_MET")
        self.assertIn("exit_signal_reversal", self.coverage()["observed"])
        md = R.render_markdown(R.build_report(self.out))
        self.assertIn("event seen, case condition not met", md)

    def test_untested_reason_is_specific_to_each_exit_reason(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), reason="stop_loss")
        self.run_shadow()
        by = {c["id"]: c for c in self.coverage()["cases"]}
        self.assertEqual(by["exit_stop_loss"]["gate_outcomes"]["related_events"], 1)
        for cid in ("exit_take_profit", "exit_signal_reversal", "exit_window_exit"):
            self.assertEqual(by[cid]["gate_outcomes"]["related_events"], 0, cid)
            self.assertEqual(by[cid]["untested_reason"], "NO_EVENT_REACHED_GATE", cid)

    def test_untested_reason_for_rollover_needs_two_utc_dates(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-5.0)
        self.run_shadow()                                       # every live event is on one UTC date
        case = {c["id"]: c for c in self.coverage()["cases"]}["daily_rollover_utc"]
        self.assertEqual((case["gate_outcomes"]["related_events"], case["untested_reason"]), (0, "NO_EVENT_REACHED_GATE"))

    def test_untested_reason_rollover_seen_but_condition_not_met(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-5.0)
        self.bot.entry("b", "BBB", "LONG", 900, ts(9, 0, 7, "2026-10-06"))      # a second UTC date, but no daily_risk row for it
        self.run_shadow()
        case = {c["id"]: c for c in self.coverage()["cases"]}["daily_rollover_utc"]
        self.assertEqual(case["status"], "UNTESTED")
        self.assertGreater(case["gate_outcomes"]["related_events"], 0)

    def test_untested_reason_is_empty_for_observed_cases(self):
        full_coverage_scenario(self.bot)
        self.run_shadow()
        for c in self.coverage()["cases"]:
            self.assertIsNone(c["untested_reason"], c["id"])

    def test_other_exit_reasons_do_not_count_as_required_reasons(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), reason="manual_close")
        self.run_shadow()
        cov = self.coverage()
        self.assertEqual(sorted(cov["observed"]), ["long_entry_stop_captured"])

    def test_full_scenario_is_complete(self):
        full_coverage_scenario(self.bot)
        self.run_shadow()
        cov = self.coverage()
        self.assertTrue(cov["complete"], cov["untested"])
        self.assertEqual((cov["n_observed"], cov["n_required"], cov["untested"], cov["status"]), (8, 8, [], "COMPLETE"))

    def test_long_entry_stop_captured_counts_only_with_live_valid_stop(self):
        self.bot.entry("ok", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.entry("nocap", "BBB", "LONG", 900, ts(10, 1, 7), position_row=False)
        self.bot.entry("wrongside", "CCC", "LONG", 900, ts(10, 2, 7), stop=105.0)               # stop above a long entry
        self.bot.entry("fixturesrc", "DDD", "LONG", 900, ts(10, 3, 7))
        self.bot.entry("early", "EEE", "LONG", 900, ts(10, 4, 7))
        self.bot.entry("shortvalid", "FFF", "SHORT", 900, ts(10, 5, 7))
        self.bot.exit("nocap", ts(11, 0, 0), reason="stop_loss")                                   # a later stop_loss exit is not enough

        def edit(obs):
            obs.sdb.execute("UPDATE captured_positions SET source='fixture' WHERE position_id='pp_fixturesrc'")
            obs.sdb.execute("UPDATE captured_positions SET first_seen_at='2026-10-04T23:59:59+00:00' WHERE position_id='pp_early'")
        self.run_custom(edit)
        cov = self.coverage()
        case = {c["id"]: c for c in cov["cases"]}["long_entry_stop_captured"]
        self.assertEqual([e["order_id"] for e in case["evidence"]], ["po_ok"])
        self.assertEqual(case["count"], 1)
        self.assertIn("exit_stop_loss", cov["observed"])
        d = self.decisions()
        self.assertTrue(d["po_ok"]["flags"]["stop_valid_protective"])
        self.assertEqual(d["po_ok"]["flags"]["stop_source"], S.LIVE_STOP_SOURCE)
        self.assertFalse(d["po_wrongside"]["flags"]["stop_valid_protective"])
        self.assertEqual(d["po_nocap"]["flags"]["stop_source"], "none_captured")
        self.assertEqual(d["po_fixturesrc"]["flags"]["stop_source"], "fixture")

    def test_real_capture_path_counts_as_live(self):
        self.bot.simulate_poller = False                      # the observer's own capture from the bot's positions table
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.run_shadow()
        db = sqlite3.connect(os.path.join(self.out, "shadow_decisions.sqlite3"))
        row = db.execute("SELECT source, first_seen_at FROM captured_positions WHERE position_id='pp_a'").fetchone()
        db.close()
        self.assertEqual(row, (S.LIVE_STOP_SOURCE, FIXED_NOW.isoformat(timespec="seconds")))
        self.assertIn("long_entry_stop_captured", self.coverage()["observed"])

    def test_valid_protective_stop_rules(self):
        v = R.valid_protective_stop
        self.assertTrue(v("LONG", 100, 98))
        self.assertFalse(v("LONG", 100, 100))
        self.assertFalse(v("LONG", 100, 102))
        self.assertTrue(v("SHORT", 100, 102))
        self.assertFalse(v("SHORT", 100, 98))
        for bad in (None, 0, -1, float("nan"), float("inf"), True, "98"):
            self.assertFalse(v("LONG", 100, bad), bad)
            self.assertFalse(v("LONG", bad, 98), bad)
        self.assertFalse(v("FLAT", 100, 98))

    def test_same_symbol_reversal_needs_the_same_signal_and_opposite_side(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(12, 0, 0), reason="signal_reversal", closing_signal="r1")
        self.bot.entry("r1", "AAA", "SHORT", 900, ts(12, 0, 0))                 # counts
        self.bot.entry("b", "BBB", "LONG", 900, ts(10, 1, 7))
        self.bot.exit("b", ts(12, 0, 0), reason="signal_reversal", closing_signal="x1")
        self.bot.entry("y1", "BBB", "SHORT", 900, ts(12, 0, 0))                 # different signal: not a reversal
        self.bot.entry("c", "CCC", "LONG", 900, ts(10, 2, 7))
        self.bot.exit("c", ts(12, 0, 0), reason="signal_reversal", closing_signal="z1")
        self.bot.entry("z1", "CCC", "LONG", 900, ts(12, 0, 0))                  # same side: not a reversal
        self.bot.entry("d", "DDD", "LONG", 900, ts(10, 3, 7))
        self.bot.exit("d", ts(12, 0, 0), reason="signal_reversal", closing_signal="w1")
        self.bot.entry("w1", "EEE", "SHORT", 900, ts(12, 0, 0))                 # other symbol: not a reversal
        self.bot.entry("f", "FFF", "LONG", 900, ts(10, 4, 7))
        self.bot.entry("v1", "FFF", "SHORT", 900, ts(11, 0, 0), open_position=False)
        self.bot.exit("f", ts(12, 30, 0), reason="signal_reversal", closing_signal="v1")   # entry BEFORE the exit: not a reversal
        self.run_shadow()
        case = {c["id"]: c for c in self.coverage()["cases"]}["same_symbol_reversal"]
        self.assertEqual([(e["symbol"], e["closed_side"], e["opened_side"]) for e in case["evidence"]], [("AAA", "LONG", "SHORT")])
        self.assertEqual(case["count"], 1)

    def _two_day_bot(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-5.0)
        self.bot.daily("2026-10-06", 10000.0)
        self.bot.equity(ts(8, day="2026-10-06"), 10000.0)
        self.bot.entry("b", "BBB", "LONG", 900, ts(9, 0, 7, "2026-10-06"))

    def test_daily_rollover_needs_nonzero_state_then_a_zero_state_next_utc_date(self):
        self._two_day_bot()
        self.run_shadow()
        case = {c["id"]: c for c in self.coverage()["cases"]}["daily_rollover_utc"]
        self.assertEqual(case["status"], "OBSERVED")
        self.assertEqual((case["evidence"][0]["from_utc_date"], case["evidence"][0]["to_utc_date"]), ("2026-10-05", "2026-10-06"))
        d = self.decisions()["po_b"]
        self.assertEqual((d["day_realized_pnl"], d["flags"]["trades_opened_today"], d["flags"]["utc_date"]), (0.0, 0, "2026-10-06"))

    def test_rollover_not_observed_for_one_date_or_missing_daily_row(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), pnl=-5.0)
        self.run_shadow()
        self.assertEqual(self.coverage()["cases"][7]["status"], "UNTESTED")                       # one UTC date only
        self.bot.entry("b", "BBB", "LONG", 900, ts(9, 0, 7, "2026-10-06"))                       # no daily_risk row for 10-06
        self.run_shadow()
        self.assertEqual(self.coverage()["cases"][7]["status"], "UNTESTED")
        self.assertIn("day_start_equity", self.decisions()["po_b"]["data_gaps"])

    def test_rollover_check_detects_a_state_that_does_not_reset(self):
        self._two_day_bot()
        with mock.patch.object(S.Replay, "day_realized", lambda self, date: sum(n for _, n, _ in self.closed)):
            self.run_shadow()
        self.assertEqual(self.coverage()["cases"][7]["status"], "UNTESTED")
        self.assertNotEqual(self.decisions()["po_b"]["day_realized_pnl"], 0.0)

    def test_rollover_carried_state_can_come_from_an_exit_only_day(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7, "2026-10-04"))                       # before the campaign: history
        self.bot.exit("a", ts(11, 0, 0), pnl=-5.0)                                                 # live, only decision on 10-05
        self.bot.daily("2026-10-06", 10000.0)
        self.bot.equity(ts(8, day="2026-10-06"), 10000.0)
        self.bot.entry("b", "BBB", "LONG", 900, ts(9, 0, 7, "2026-10-06"))
        self.run_shadow()
        self.assertEqual(self.coverage()["cases"][7]["status"], "OBSERVED")

    def test_rollover_not_counted_if_nothing_was_carried(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7, "2026-10-04"))
        self.bot.exit("a", ts(11, 0, 0), pnl=-5.0)
        self.bot.c.execute("UPDATE closed_trades SET net_pnl=0")                                  # an exit that realised nothing
        self.bot.daily("2026-10-06", 10000.0)
        self.bot.equity(ts(8, day="2026-10-06"), 10000.0)
        self.bot.entry("b", "BBB", "LONG", 900, ts(9, 0, 7, "2026-10-06"))
        self.run_shadow()
        self.assertEqual(self.coverage()["cases"][7]["status"], "UNTESTED")

    def test_exit_of_an_unknown_position_is_not_counted(self):
        self.bot.c.execute("INSERT INTO closed_trades VALUES('pp_ghost','GHOST','LONG',?,?,100,99,1,-1,0,-1,'stop_loss','monitor_ghost')",
                           (ts(9), ts(11)))
        self.run_shadow()
        self.assertEqual(self.decisions()["shadow_exit_pp_ghost"]["action"], "unknown")
        self.assertEqual(self.coverage()["cases"][2]["status"], "UNTESTED")

    def test_dates_are_utc_not_local(self):
        self.bot.daily("2026-10-06", 10000.0)
        self.bot.entry("a", "AAA", "LONG", 900, "2026-10-05T23:30:07-02:00")                    # = 2026-10-06T01:30:07Z
        self.run_shadow()
        d = self.decisions()["po_a"]
        self.assertEqual(d["flags"]["utc_date"], "2026-10-06")

    def test_history_before_campaign_start_and_events_after_deadline_never_count(self):
        full_coverage_scenario(self.bot)
        self.run_shadow(now=lambda: FIXED_NOW + timedelta(days=2))                              # campaign starts after all events
        cov = self.coverage()
        self.assertEqual((cov["n_observed"], cov["live_decisions"]), (0, 0))
        out2 = os.path.join(self.dir, "o2")
        self.run_shadow(out=out2)                                                                # campaign starts before them
        self.assertTrue(self.coverage(out=out2)["complete"])
        early = self.coverage(out=out2, until=datetime(2026, 10, 5, 10, 20, tzinfo=timezone.utc))
        self.assertEqual(sorted(early["observed"]), ["long_entry_stop_captured", "short_entry"])

    def test_database_without_campaign_metadata_reports_not_started(self):
        db = sqlite3.connect(":memory:")
        db.executescript(S.SHADOW_SCHEMA.replace("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);", ""))
        cov = R.compute_coverage(db, now=FIXED_NOW)
        self.assertEqual((cov["status"], cov["n_observed"], cov["complete"]), ("NOT_STARTED", 0, False))
        self.assertEqual(len(cov["untested"]), 8)

    def test_fixture_test_names_exist(self):
        self.assertEqual(set(R.FIXTURE_TESTS), set(R.CASE_IDS))
        for path in R.FIXTURE_TESTS.values():
            cls, name = path.split(".")
            self.assertTrue(hasattr(globals()[cls], name), path)

    def test_report_lists_observed_and_untested_cases_in_utc(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.bot.exit("a", ts(11, 0, 0), reason="stop_loss")
        self.run_shadow()
        rep = R.build_report(self.out, now=FIXED_NOW + timedelta(days=1))
        md = R.render_markdown(rep)
        self.assertIn("## Coverage of required cases (all times UTC)", md)
        self.assertIn("2 of 8 required cases observed", md)
        self.assertIn("### Observed (2)", md)
        self.assertIn("### Still untested (6)", md)
        for cid in ("short_entry", "exit_take_profit", "exit_signal_reversal", "exit_window_exit", "same_symbol_reversal", "daily_rollover_utc"):
            self.assertRegex(md, rf"- `{cid}`: .*(No signal reached the gate|signal was produced|condition was not satisfied).*Exercise this path with the deterministic")
        self.assertIn("fixture results are not live evidence", md)
        self.assertEqual(rep["coverage"]["campaign_start"], "2026-10-05T00:00:00+00:00")
        self.assertEqual(rep["campaign"]["passes"], 1)


class FakeTime:
    def __init__(self, start=FIXED_NOW):
        self.t = start
        self.sleeps: list[float] = []
        self.hook = None

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        if len(self.sleeps) > 5000:
            raise RuntimeError("the loop did not stop")                    # guard: a loop that never stops fails the test fast
        self.t += timedelta(seconds=s)
        if self.hook:
            self.hook(len(self.sleeps))


class LoopTests(Base):
    def loop(self, ft, until, interval=30.0, **kw):
        obs = S.ShadowObserver(self.db, self.out, now=ft.now)
        self._observers.append(obs)
        lines: list[str] = []
        res = S.run_loop(obs, interval, until, now=ft.now, sleep=ft.sleep, emit=lines.append, **kw)
        return obs, res, lines

    def meta(self):
        db = sqlite3.connect(os.path.join(self.out, "shadow_decisions.sqlite3"))
        m = dict(db.execute("SELECT key, value FROM meta"))
        db.close()
        return m

    def count(self, sql):
        db = sqlite3.connect(os.path.join(self.out, "shadow_decisions.sqlite3"))
        n = db.execute(sql).fetchone()[0]
        db.close()
        return n

    def test_stops_at_the_deadline_reports_untested_cases_and_logs_only_changes(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        ft = FakeTime()
        until = FIXED_NOW + timedelta(seconds=100)
        before = md5(self.db)
        _, res, lines = self.loop(ft, until)
        self.assertEqual(res["stop_reason"], "deadline")
        self.assertEqual(ft.t, until)                                      # the last sleep was capped at the deadline
        self.assertEqual(ft.sleeps, [30, 30, 30, 10])
        self.assertEqual(len(lines), 2, lines)                             # first pass (new state) + STOP; four quiet passes log nothing
        self.assertIn("new_decisions=1", lines[0])
        self.assertIn("STOP reason=deadline", lines[1])
        self.assertIn("untested:", lines[1])
        m = self.meta()
        self.assertEqual((m["stop_reason"], m["passes"], m["campaign_until"]), ("deadline", "5", "2026-10-05T00:01:40+00:00"))
        md = self.read_text(os.path.join(self.out, "shadow_report.md"))
        self.assertIn("### Still untested", md)
        self.assertIn("DEADLINE_REACHED", md)
        self.assertEqual(md5(self.db), before)                             # the bot's database is untouched by the whole loop

    def test_stops_early_only_when_every_case_was_observed(self):
        full_coverage_scenario(self.bot)
        ft = FakeTime()
        _, res, lines = self.loop(ft, FIXED_NOW + timedelta(days=14))
        self.assertEqual(res["stop_reason"], "coverage_complete")
        self.assertEqual(ft.sleeps, [])
        self.assertTrue(res["coverage"]["complete"])
        self.assertIn("coverage=8/8", lines[0])
        self.assertEqual(self.meta()["stop_reason"], "coverage_complete")
        self.assertIn("COMPLETE", self.read_text(os.path.join(self.out, "shadow_report.md")))

    def test_does_not_stop_early_with_seven_of_eight(self):
        full_coverage_scenario(self.bot)
        self.bot.c.execute("DELETE FROM closed_trades WHERE exit_reason='window_exit'")
        self.bot.c.execute("DELETE FROM positions WHERE position_id='pp_c'")
        ft = FakeTime()
        _, res, _ = self.loop(ft, FIXED_NOW + timedelta(days=2), interval=3600)
        self.assertEqual(res["stop_reason"], "deadline")
        self.assertEqual(res["coverage"]["untested"], ["exit_window_exit"])
        self.assertEqual(len(ft.sleeps), 48)

    def test_cases_observed_later_are_logged_once_and_end_the_loop(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(0, 10, 0))
        ft = FakeTime()

        def hook(n):
            if n == 2:                                                      # between passes, the bot produces the remaining cases
                self.bot.exit("a", ts(0, 20, 0), reason="stop_loss")
        ft.hook = hook
        _, res, lines = self.loop(ft, FIXED_NOW + timedelta(days=1), interval=1800)
        self.assertEqual(res["stop_reason"], "deadline")
        self.assertEqual([("coverage_observed=exit_stop_loss" in l) for l in lines].count(True), 1)

    def test_max_events_is_an_optional_emergency_limit(self):
        for i, sym in enumerate(("AAA", "BBB", "CCC")):
            self.bot.entry(f"s{i}", sym, "LONG", 300, ts(10, i, 7))
        ft = FakeTime()
        _, res, lines = self.loop(ft, FIXED_NOW + timedelta(days=14), max_events=2)
        self.assertEqual((res["stop_reason"], res["new_decisions"]), ("max_events", 3))
        self.assertEqual(ft.sleeps, [])

    def test_without_max_events_the_primary_rule_is_deadline_or_coverage(self):
        for i, sym in enumerate(("AAA", "BBB", "CCC")):
            self.bot.entry(f"s{i}", sym, "LONG", 300, ts(10, i, 7))
        ft = FakeTime()
        _, res, _ = self.loop(ft, FIXED_NOW + timedelta(seconds=60))
        self.assertEqual(res["stop_reason"], "deadline")

    def test_ctrl_c_stops_cleanly_and_writes_the_report(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        ft = FakeTime()

        def boom(n):
            raise KeyboardInterrupt
        ft.hook = boom
        _, res, lines = self.loop(ft, FIXED_NOW + timedelta(days=14))
        self.assertEqual(res["stop_reason"], "interrupted")
        self.assertEqual(self.meta()["stop_reason"], "interrupted")
        self.assertTrue(os.path.exists(os.path.join(self.out, "shadow_report.md")))

    def test_change_only_logging_of_runs_and_reconciliations(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        ft = FakeTime()

        def hook(n):
            if n == 3:
                self.bot.entry("b", "BBB", "LONG", 900, ts(10, 30, 7))
        ft.hook = hook
        _, res, lines = self.loop(ft, FIXED_NOW + timedelta(seconds=300))
        self.assertEqual(self.meta()["passes"], "11")
        self.assertEqual(self.count("SELECT COUNT(*) FROM runs"), 2)                       # first pass and the pass with the new order
        self.assertEqual(self.count("SELECT COUNT(*) FROM reconciliations WHERE kind='schema'"), 1)
        self.assertEqual(self.count("SELECT COUNT(*) FROM reconciliations"), 5 + 4)        # first pass: 5 kinds; new order changed 4; not 5 per pass
        self.assertEqual(len(lines), 3, lines)                                              # first pass, new order, STOP
        with open(os.path.join(self.out, "shadow_decisions.jsonl")) as fh:
            self.assertEqual(len(fh.readlines()), 2)

    def test_heartbeat_is_off_by_default_and_optional(self):
        ft = FakeTime()
        _, _, lines = self.loop(ft, FIXED_NOW + timedelta(seconds=300))
        self.assertFalse(any("heartbeat" in l for l in lines))
        ft2 = FakeTime()
        self.tearDown(), self.setUp()
        _, _, lines2 = self.loop(ft2, FIXED_NOW + timedelta(seconds=300), heartbeat_s=100)
        self.assertEqual(sum("heartbeat" in l for l in lines2), 2)

    def test_restart_after_a_stop_runs_again_and_clears_the_stop_reason(self):
        full_coverage_scenario(self.bot)
        ft = FakeTime()
        self.loop(ft, FIXED_NOW + timedelta(days=14))
        obs = S.ShadowObserver(self.db, self.out, now=ft.now)
        self._observers.append(obs)
        seen = {}
        res = S.run_loop(obs, 30, FIXED_NOW + timedelta(days=14), now=ft.now, sleep=ft.sleep, emit=lambda l: seen.setdefault("n", 0))
        self.assertEqual(res["stop_reason"], "coverage_complete")
        self.assertEqual(self.meta()["passes"], "2")
        self.assertEqual(self.meta()["campaign_start"], FIXED_NOW.isoformat(timespec="seconds"))      # campaign start is not reset

    def test_stale_stop_reason_is_cleared_while_a_new_loop_runs(self):
        ft = FakeTime()
        self.loop(ft, FIXED_NOW + timedelta(seconds=60))
        self.assertEqual(self.meta()["stop_reason"], "deadline")
        obs = S.ShadowObserver(self.db, self.out, now=ft.now)
        self._observers.append(obs)
        seen = []
        ft.hook = lambda n: seen.append(self.meta().get("stop_reason"))
        S.run_loop(obs, 30, ft.t + timedelta(seconds=60), now=ft.now, sleep=ft.sleep, emit=lambda l: None)
        self.assertEqual(seen[0], None)
        self.assertEqual(self.meta()["stop_reason"], "deadline")

    def test_schema_drift_in_loop_mode_returns_exit_code_2(self):
        self.bot.c.execute("ALTER TABLE orders RENAME COLUMN fee TO fee_renamed")
        ft = FakeTime()
        _, res, lines = self.loop(ft, FIXED_NOW + timedelta(seconds=60))
        self.assertEqual(res["code"], 2)
        self.assertEqual(sum("status=SCHEMA_DRIFT" in l for l in lines), 1)

    def test_loop_never_raises_when_the_paper_db_disappears(self):
        ft = FakeTime()
        os.remove(self.db)
        self.bot.c.close()
        obs = S.ShadowObserver(self.db, self.out, now=ft.now)
        self._observers.append(obs)
        lines: list[str] = []
        res = S.run_loop(obs, 30, FIXED_NOW + timedelta(seconds=60), now=ft.now, sleep=ft.sleep, emit=lines.append)
        self.assertEqual(res["stop_reason"], "deadline")
        self.assertTrue(any("status=ERROR" in l for l in lines))
        self.assertEqual(sum("status=" in l for l in lines), 1)                             # the error state is logged once, not every pass


class CliTests(Base):
    def test_parse_until(self):
        self.assertEqual(S.parse_until("2026-10-19T23:59:59Z"), datetime(2026, 10, 19, 23, 59, 59, tzinfo=timezone.utc))
        self.assertEqual(S.parse_until("2026-10-20T01:59:59+02:00"), datetime(2026, 10, 19, 23, 59, 59, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            S.parse_until("2026-10-19T23:59:59")                                           # no timezone: refused
        with self.assertRaises(ValueError):
            S.parse_until("tomorrow")
        self.assertEqual(S.DEFAULT_UNTIL, "2026-10-19T23:59:59Z")

    def test_cli_refuses_bad_until_and_max_events(self):
        base = ["--paper-db", self.db, "--out", os.path.join(self.dir, "o9")]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(S.main(base + ["--until", "2026-10-19T23:59:59"]), 2)
            self.assertEqual(S.main(base + ["--max-events", "0"]), 2)

    def test_cli_loop_stops_at_a_past_deadline_after_one_pass_and_writes_the_report(self):
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        out = os.path.join(self.dir, "o8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = S.main(["--paper-db", self.db, "--out", out, "--loop", "0.01", "--until", "2000-01-01T00:00:00Z"])
        self.assertEqual(code, 0)
        self.assertIn("STOP reason=deadline", buf.getvalue())
        self.assertTrue(os.path.exists(os.path.join(out, "shadow_report.md")))
        self.assertTrue(os.path.exists(os.path.join(out, "shadow_report.json")))

    def test_cli_single_pass_still_prints_one_json_summary(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(S.main(["--paper-db", self.db, "--out", os.path.join(self.dir, "o7")]), 0)
        self.assertEqual(json.loads(buf.getvalue())["status"], "OK")


class MigrationTests(Base):
    def _old_schema_db(self, out):
        os.makedirs(out, exist_ok=True)
        old = S.SHADOW_SCHEMA.replace(",\n  source TEXT NOT NULL DEFAULT 'live_positions_table');", ");")
        old = old.replace("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);", "")
        self.assertNotEqual(old, S.SHADOW_SCHEMA)
        self.assertNotIn("source", old)
        self.assertNotIn("meta", old)
        db = sqlite3.connect(os.path.join(out, "shadow_decisions.sqlite3"))
        db.executescript(old)
        db.execute("INSERT INTO captured_positions VALUES('pp_old','OLD','LONG',90,110,1,100,1,100,'2026-08-01T00:00:00+00:00')")
        db.commit()
        db.close()

    def test_old_shadow_database_is_migrated_in_place(self):
        out = os.path.join(self.dir, "old")
        self._old_schema_db(out)
        self.bot.entry("a", "AAA", "LONG", 900, ts(10, 0, 7))
        self.run_shadow(out=out)
        db = sqlite3.connect(os.path.join(out, "shadow_decisions.sqlite3"))
        cols = {r[1] for r in db.execute("PRAGMA table_info(captured_positions)")}
        old_row = db.execute("SELECT source FROM captured_positions WHERE position_id='pp_old'").fetchone()
        has_meta = db.execute("SELECT value FROM meta WHERE key='campaign_start'").fetchone()
        db.close()
        self.assertIn("source", cols)
        self.assertEqual(old_row[0], S.LIVE_STOP_SOURCE)
        self.assertEqual(has_meta[0], FIXED_NOW.isoformat(timespec="seconds"))
        self.assertEqual(self.decisions(out)["po_a"]["approved"], 1)

    def test_report_on_an_old_database_does_not_crash_and_says_not_started(self):
        out = os.path.join(self.dir, "old2")
        self._old_schema_db(out)
        rep = R.build_report(out, now=FIXED_NOW)
        self.assertEqual(rep["coverage"]["status"], "NOT_STARTED")
        self.assertIn("NOT_STARTED", R.render_markdown(rep))


class LimitsTests(unittest.TestCase):
    def test_paper_limits_match_the_values_the_user_set(self):
        L = S.paper_limits()
        self.assertEqual((L.max_position_pct, L.max_exposure_pct, L.max_net_exposure_pct, L.max_leverage, L.max_open_positions),
                         (10, 50, 30, 1.0, 5))
        self.assertEqual((L.min_stop_pct, L.max_stop_pct, L.max_daily_loss_pct, L.max_drawdown_pct), (0.5, 5, 1, 5))
        self.assertFalse(L.require_borrow_check)

    def test_assumed_limits_are_listed_separately(self):
        self.assertFalse(set(S.LIMITS_STATED) & set(S.LIMITS_ASSUMED))


if __name__ == "__main__":
    unittest.main(verbosity=2)
