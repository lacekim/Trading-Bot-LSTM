"""Tests for trading_gate. Written BEFORE any connection to an order path.  Run:  python3 -m unittest -v test_gate  (from this folder's parent: python3 -m unittest trading_gate.test_gate)"""
from __future__ import annotations

import copy
import itertools
import math
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

try:
    from . import gate as G
    from .gate import GateLimits, TradingGate
    from .reconcile import reconcile
except ImportError:  # run as a script from inside the folder
    import gate as G
    from gate import GateLimits, TradingGate
    from reconcile import reconcile

NOW = datetime(2026, 10, 5, 12, 5, 0, tzinfo=timezone.utc)


def limits(**kw) -> GateLimits:
    base = dict(max_position_pct=20, max_exposure_pct=80, max_open_positions=5, max_leverage=2.0, min_order_usd=10,
                max_data_age_s=120, max_spread_bps=50, max_signal_age_multiplier=2, max_price_deviation_bps=100,
                notional_tolerance=0.001, min_stop_pct=0.5, max_stop_pct=15, max_daily_loss_pct=3,
                max_equity_loss_pct=4, max_drawdown_pct=15, max_consecutive_losses=4, max_failed_orders=3,
                max_trades_per_day=10, min_bars_between_entries=2, cash_buffer_pct=5, require_borrow_check=True)
    base.update(kw)
    return GateLimits(**base)


def scenario(direction="LONG") -> dict:
    """A fully valid ENTRY scenario. direction LONG or SHORT."""
    long_ = direction == "LONG"
    price = 50000.0
    return dict(
        proposed_order=dict(order_id="o-1", idempotency_key="k-1", symbol="BTC/USD", order_kind="ENTRY", side=direction,
                            reduce_only=False, quantity=0.1, price=price, notional=5000.0, leverage=1.0,
                            timeframe="1h", signal_id="s-1",
                            stop_price=49000.0 if long_ else 51000.0, target_price=53000.0 if long_ else 47000.0),
        signal=dict(signal_id="s-1", symbol="BTC/USD", direction=direction, timeframe="1h", price=price,
                    candle_timestamp=NOW - timedelta(hours=1, minutes=5)),
        market_snapshot=dict(symbol="BTC/USD", price=50010.0, bid=50005.0, ask=50015.0,
                             observed_at=NOW - timedelta(seconds=5)),
        portfolio_state=dict(equity=100000.0, cash=100000.0, peak_equity=100000.0, positions=[],
                             seen_idempotency_keys=[]),
        risk_state=dict(kill_switch=False, day_start_equity=100000.0, daily_realized_pnl=0.0, consecutive_losses=0,
                        failed_orders_today=0, trades_opened_today=0, bars_since_last_entry=None,
                        shorts_allowed=True, borrow_available={"BTC/USD": True}),
    )


_n = itertools.count(1)


def exit_scenario(direction="LONG", qty=0.1, key=None) -> dict:
    """A valid EXIT scenario. Each call gets a fresh idempotency key unless one is given."""
    sc = scenario(direction)
    key = key or f"kx-{next(_n)}"
    sc["proposed_order"] = dict(order_id="x-1", idempotency_key=key, symbol="BTC/USD", order_kind="EXIT",
                                side=direction, reduce_only=True, quantity=qty, price=50000.0)
    sc["signal"] = None
    sc["portfolio_state"]["positions"] = [dict(symbol="BTC/USD", direction=direction, quantity=0.1, notional=5000.0)]
    return sc


class Base(unittest.TestCase):
    def setUp(self):
        self.gate = TradingGate(limits(), ":memory:", clock=lambda: NOW)

    def tearDown(self):
        self.gate.close()

    def ev(self, sc=None, gate=None):
        return (gate or self.gate).evaluate(**(sc if sc is not None else scenario()))

    def assertRejected(self, d, reason_part=None, check=None):
        self.assertFalse(d["approved"], d)
        self.assertEqual(d["decision"], "REJECTED")
        self.assertTrue(d["reasons"], "a rejection must carry a reason")
        if reason_part:
            self.assertTrue(any(reason_part in r for r in d["reasons"]), f"{reason_part!r} not in {d['reasons']}")
        if check:
            self.assertFalse(d["checks"].get(check, True), d["checks"])


class RequiredTests(Base):
    """The twelve tests the specification demands."""

    def test_01_valid_long_approved(self):
        d = self.ev(scenario("LONG"))
        self.assertTrue(d["approved"], d)
        self.assertEqual(d["reasons"], [])
        self.assertEqual(set(d["checks"]), set(G.EXPECTED_ENTRY_CHECKS))
        self.assertTrue(all(d["checks"].values()))
        self.assertEqual(d["order_id"], "o-1")

    def test_02_valid_short_approved(self):
        d = self.ev(scenario("SHORT"))
        self.assertTrue(d["approved"], d)

    def test_03_stale_signal_rejected(self):
        sc = scenario()
        sc["signal"]["candle_timestamp"] = NOW - timedelta(hours=5)
        self.assertRejected(self.ev(sc), "signal_expired", "signal_fresh")

    def test_04_duplicate_order_rejected(self):
        first = self.ev(scenario())
        self.assertTrue(first["approved"])
        second = self.ev(scenario())                      # same idempotency key: gate's own atomic store
        self.assertRejected(second, "duplicate_order", "duplicate_order")
        sc = scenario()
        sc["proposed_order"]["idempotency_key"] = "k-other"
        sc["portfolio_state"]["seen_idempotency_keys"] = ["k-other"]   # caller's own seen set
        self.assertRejected(self.ev(sc), "duplicate_order", "duplicate_order")

    def test_05_long_signal_short_order_rejected(self):
        sc = scenario("LONG")
        sc["proposed_order"]["side"] = "SHORT"
        sc["proposed_order"]["stop_price"], sc["proposed_order"]["target_price"] = 51000.0, 47000.0
        self.assertRejected(self.ev(sc), "signal_order_mismatch:direction", "signal_matches_order")

    def test_06_excessive_position_size_rejected(self):
        sc = scenario()
        sc["proposed_order"].update(quantity=0.5, notional=25000.0)          # 25% of equity > 20%
        self.assertRejected(self.ev(sc), "position_size_exceeds_limit", "risk_limits")

    def test_07_missing_stop_on_opening_order_rejected(self):
        for mutate in (lambda o: o.pop("stop_price"), lambda o: o.update(stop_price=None)):
            sc = scenario()
            mutate(sc["proposed_order"])
            self.assertRejected(self.ev(sc), "missing_stop", "stop_present")

    def test_08_daily_loss_limit_rejected(self):
        sc = scenario()
        sc["risk_state"]["daily_realized_pnl"] = -3000.0                      # exactly 3% of 100k: limit reached
        self.assertRejected(self.ev(sc), "daily_loss_limit", "daily_loss_ok")

    def test_09_malformed_price_or_quantity_rejected(self):
        bad = [("price", float("nan")), ("price", float("inf")), ("price", -1.0), ("price", 0.0), ("price", "50000"),
               ("price", None), ("price", True), ("quantity", float("nan")), ("quantity", 0.0), ("quantity", -0.1),
               ("quantity", "0.1"), ("quantity", None), ("quantity", [1])]
        for key, val in bad:
            sc = scenario()
            sc["proposed_order"][key] = val
            d = self.ev(sc)
            self.assertFalse(d["approved"], (key, val, d))
            self.assertTrue(d["reasons"])
        sc = scenario()
        del sc["proposed_order"]["quantity"]
        self.assertFalse(self.ev(sc)["approved"])

    def test_10_validator_exception_or_missing_data_rejected(self):
        # (a) a check that raises -> REJECTED with check_error, other checks still run
        with mock.patch.object(TradingGate, "_chk_risk_limits", side_effect=RuntimeError("boom")):
            d = self.ev(scenario())
        self.assertRejected(d, "check_error:risk_limits:RuntimeError", "risk_limits")
        self.assertTrue(d["checks"]["data_fresh"])
        # (b) the gate's own internals blow up before any check
        with mock.patch.object(TradingGate, "_run", side_effect=RuntimeError("boom")):
            d = self.ev(scenario())
        self.assertRejected(d, "gate_internal_error:RuntimeError")
        # (c) missing data
        for key in ("market_snapshot", "portfolio_state", "risk_state", "signal", "proposed_order"):
            sc = scenario()
            sc[key] = None
            self.assertFalse(self.ev(sc)["approved"], key)
        sc = scenario()
        sc["market_snapshot"].pop("price")
        self.assertRejected(self.ev(sc), "missing_field:price", "data_fresh")
        # (d) clock failure
        g = TradingGate(limits(), ":memory:", clock=lambda: (_ for _ in ()).throw(RuntimeError("clock")))
        self.assertFalse(self.ev(scenario(), g)["approved"])
        g.close()
        # (e) a dropped/skipped check is never an approval
        with mock.patch.object(G, "EXPECTED_ENTRY_CHECKS", G.EXPECTED_ENTRY_CHECKS + ("phantom_check",)):
            d = self.ev(scenario())
        self.assertRejected(d, "expected_check_missing:phantom_check")

    def test_11_reduction_allowed_during_kill_switch(self):
        sc = exit_scenario("LONG")
        sc["risk_state"]["kill_switch"] = True
        sc["risk_state"]["daily_realized_pnl"] = -50000.0                     # also far past every loss limit
        sc["market_snapshot"]["observed_at"] = NOW - timedelta(hours=3)        # and stale data
        d = self.ev(sc)
        self.assertTrue(d["approved"], d)
        self.assertIn("kill_switch_active_exit_allowed", d["warnings"])
        self.assertIn("stale_market_data_exit_allowed", d["warnings"])
        entry = scenario()
        entry["risk_state"]["kill_switch"] = True
        self.assertRejected(self.ev(entry), "kill_switch_active", "kill_switch_ok")   # but entries ARE blocked
        sc = exit_scenario("SHORT")
        sc["risk_state"]["kill_switch"] = True
        self.assertTrue(self.ev(sc)["approved"])

    def test_12_partial_or_rejected_fill_alerts(self):
        tol = dict(qty_rel=0.001, slippage_bps=25)
        pos = [dict(symbol="BTC/USD", direction="LONG", quantity=0.1)]
        clean = dict(order_id="o1", symbol="BTC/USD", side="LONG", quantity=0.1, price=50000.0, status="FILLED",
                     filled_quantity=0.1, avg_fill_price=50010.0)
        self.assertEqual(reconcile([clean], pos, pos, tol), {"ok": True, "alerts": []})
        partial = dict(clean, status="PARTIAL", filled_quantity=0.04)
        r = reconcile([partial], [dict(symbol="BTC/USD", direction="LONG", quantity=0.04)],
                      [dict(symbol="BTC/USD", direction="LONG", quantity=0.04)], tol)
        self.assertFalse(r["ok"])
        self.assertEqual([a["type"] for a in r["alerts"]], ["partial_fill"])
        rej = dict(clean, status="REJECTED", filled_quantity=0.0)
        r = reconcile([rej], [], [], tol)
        self.assertFalse(r["ok"])
        self.assertEqual([a["type"] for a in r["alerts"]], ["rejected_fill"])
        short_fill = dict(clean, status="FILLED", filled_quantity=0.05)       # "FILLED" but only half
        self.assertIn("partial_fill", [a["type"] for a in reconcile([short_fill], [], [], tol)["alerts"]])


class ExtraEntryTests(Base):
    def test_naive_timestamp_rejected(self):
        sc = scenario()
        sc["market_snapshot"]["observed_at"] = datetime(2026, 10, 5, 12, 4, 55)
        self.assertRejected(self.ev(sc), "timestamp_ambiguous", "data_fresh")
        sc = scenario()
        sc["signal"]["candle_timestamp"] = "2026-10-05T11:00:00"
        self.assertRejected(self.ev(sc), "timestamp_ambiguous")

    def test_iso_string_with_z_accepted(self):
        sc = scenario()
        sc["market_snapshot"]["observed_at"] = (NOW - timedelta(seconds=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertTrue(self.ev(sc)["approved"])

    def test_stale_market_data_rejected(self):
        sc = scenario()
        sc["market_snapshot"]["observed_at"] = NOW - timedelta(seconds=121)
        self.assertRejected(self.ev(sc), "stale_market_data", "data_fresh")

    def test_future_market_data_rejected(self):
        sc = scenario()
        sc["market_snapshot"]["observed_at"] = NOW + timedelta(seconds=60)
        self.assertRejected(self.ev(sc), "market_data_in_future", "data_fresh")

    def test_market_data_inconsistent_rejected(self):
        cases = [("crossed_quote", dict(bid=50020.0, ask=50010.0)), ("spread_too_wide", dict(bid=49000.0, ask=51000.0)),
                 ("incomplete_quote", dict(bid=None)), ("market_symbol_mismatch", dict(symbol="ETH/USD")),
                 ("invalid_ohlc", dict(open=50000.0, high=49000.0, low=48000.0, close=50000.0)),
                 ("non_finite", dict(price=float("nan"))), ("non_positive", dict(price=0.0)),
                 ("order_price_deviates_from_market", dict(price=60000.0, bid=59990.0, ask=60010.0))]
        for reason, patch in cases:
            sc = scenario()
            sc["market_snapshot"].update(patch)
            self.assertRejected(self.ev(sc), reason, "data_fresh")

    def test_signal_mismatches(self):
        for field, val, reason in (("signal_id", "s-999", "signal_id"), ("symbol", "ETH/USD", "symbol"),
                                   ("timeframe", "4h", "timeframe")):
            sc = scenario()
            sc["signal"][field] = val
            self.assertRejected(self.ev(sc), f"signal_order_mismatch:{reason}", "signal_matches_order")
        sc = scenario()
        sc["signal"]["direction"] = "HOLD"
        self.assertRejected(self.ev(sc), "invalid_value:direction", "signal_matches_order")

    def test_signal_candle_not_closed_and_price_drift(self):
        sc = scenario()
        sc["signal"]["candle_timestamp"] = NOW - timedelta(minutes=10)
        self.assertRejected(self.ev(sc), "signal_candle_not_closed", "signal_fresh")
        sc = scenario()
        sc["signal"]["price"] = 52000.0
        self.assertRejected(self.ev(sc), "signal_price_deviates_from_market", "signal_fresh")

    def test_notional_mismatch_and_min_order(self):
        sc = scenario()
        sc["proposed_order"]["notional"] = 4000.0
        self.assertRejected(self.ev(sc), "notional_mismatch", "risk_limits")
        sc = scenario()
        sc["proposed_order"].update(quantity=0.0001, notional=5.0)
        self.assertRejected(self.ev(sc), "below_min_order", "risk_limits")

    def test_leverage_exposure_positions_cash(self):
        sc = scenario()
        sc["proposed_order"]["leverage"] = 5.0
        self.assertRejected(self.ev(sc), "leverage_exceeds_limit", "risk_limits")
        sc = scenario()
        sc["portfolio_state"]["positions"] = [dict(symbol="ETH/USD", direction="LONG", quantity=1, notional=78000.0)]
        self.assertRejected(self.ev(sc), "portfolio_exposure_exceeds_limit", "risk_limits")
        sc = scenario()
        sc["portfolio_state"]["positions"] = [dict(symbol=f"S{i}/USD", direction="LONG", quantity=1, notional=100.0)
                                              for i in range(5)]
        self.assertRejected(self.ev(sc), "max_open_positions_reached", "risk_limits")
        sc = scenario()
        sc["portfolio_state"]["cash"] = 3000.0
        self.assertRejected(self.ev(sc), "insufficient_cash", "risk_limits")

    def test_position_already_open_and_opposite(self):
        sc = scenario()
        sc["portfolio_state"]["positions"] = [dict(symbol="BTC/USD", direction="LONG", quantity=0.1, notional=5000.0)]
        self.assertRejected(self.ev(sc), "position_already_open", "portfolio_rules")
        sc = scenario()
        sc["portfolio_state"]["positions"] = [dict(symbol="BTC/USD", direction="SHORT", quantity=0.1, notional=5000.0)]
        self.assertRejected(self.ev(sc), "opposite_position_open_close_first", "portfolio_rules")

    def test_net_exposure_cap(self):
        g = TradingGate(limits(max_net_exposure_pct=10), ":memory:", clock=lambda: NOW)
        sc = scenario()
        sc["portfolio_state"]["positions"] = [dict(symbol="ETH/USD", direction="LONG", quantity=1, notional=8000.0)]
        self.assertRejected(self.ev(sc, g), "net_exposure_exceeds_limit", "portfolio_rules")
        g.close()

    def test_short_rules(self):
        sc = scenario("SHORT")
        sc["risk_state"]["shorts_allowed"] = False
        self.assertRejected(self.ev(sc), "shorts_not_allowed", "portfolio_rules")
        sc = scenario("SHORT")
        sc["risk_state"]["borrow_available"] = {"BTC/USD": False}
        self.assertRejected(self.ev(sc), "short_borrow_unavailable", "short_borrow_ok")
        sc = scenario("SHORT")
        sc["risk_state"].pop("borrow_available")
        self.assertRejected(self.ev(sc), "short_borrow_unavailable", "short_borrow_ok")
        g = TradingGate(limits(require_borrow_check=False), ":memory:", clock=lambda: NOW)     # paper mode
        sc = scenario("SHORT")
        sc["risk_state"].pop("borrow_available")
        self.assertTrue(self.ev(sc, g)["approved"])
        g.close()
        sc = scenario("LONG")                                                                  # longs never need borrow
        sc["risk_state"].pop("borrow_available")
        self.assertTrue(self.ev(sc)["approved"])

    def test_stop_geometry(self):
        for side, stop, reason in (("LONG", 51000.0, "stop_wrong_side"), ("SHORT", 49000.0, "stop_wrong_side"),
                                   ("LONG", 49990.0, "stop_too_tight"), ("LONG", 30000.0, "stop_too_wide")):
            sc = scenario(side)
            sc["proposed_order"]["stop_price"] = stop
            self.assertRejected(self.ev(sc), reason, "stop_present")
        sc = scenario("LONG")
        sc["proposed_order"]["target_price"] = 48000.0
        self.assertRejected(self.ev(sc), "target_wrong_side", "stop_present")
        sc = scenario("LONG")
        sc["proposed_order"]["stop_price"] = float("nan")
        self.assertRejected(self.ev(sc), "non_finite", "stop_present")

    def test_equity_loss_and_drawdown(self):
        sc = scenario()
        sc["portfolio_state"]["equity"] = 95000.0
        sc["portfolio_state"]["cash"] = 95000.0
        self.assertRejected(self.ev(sc), "daily_equity_loss_limit", "daily_loss_ok")
        sc = scenario()
        sc["portfolio_state"]["peak_equity"] = 120000.0                       # 16.7% below peak
        self.assertRejected(self.ev(sc), "max_drawdown_limit", "drawdown_ok")

    def test_loss_streak_failed_orders_trades_per_day(self):
        for key, val, reason in (("consecutive_losses", 4, "max_consecutive_losses"),
                                 ("failed_orders_today", 3, "max_failed_orders"),
                                 ("trades_opened_today", 10, "max_trades_per_day")):
            sc = scenario()
            sc["risk_state"][key] = val
            self.assertRejected(self.ev(sc), reason, "loss_streak_ok")
        sc = scenario()
        sc["risk_state"]["consecutive_losses"] = 1.5
        self.assertRejected(self.ev(sc), "non_integer", "loss_streak_ok")

    def test_cooldown(self):
        sc = scenario()
        sc["risk_state"]["bars_since_last_entry"] = 1
        self.assertRejected(self.ev(sc), "entry_cooldown_active", "cooldown_ok")
        sc = scenario()
        sc["risk_state"]["bars_since_last_entry"] = 2
        self.assertTrue(self.ev(sc)["approved"])
        sc = scenario()
        sc["risk_state"].pop("bars_since_last_entry")                          # absent != "never entered"
        self.assertRejected(self.ev(sc), "missing_field:bars_since_last_entry", "cooldown_ok")

    def test_all_reasons_are_collected(self):
        sc = scenario()
        sc["proposed_order"].pop("stop_price")
        sc["risk_state"]["kill_switch"] = True
        sc["signal"]["candle_timestamp"] = NOW - timedelta(days=2)
        d = self.ev(sc)
        for part in ("missing_stop", "kill_switch_active", "signal_expired"):
            self.assertTrue(any(part in r for r in d["reasons"]), (part, d["reasons"]))

    def test_unknown_or_missing_order_kind_rejected(self):
        for kind in ("BUY", None, "", 3, "entry"):
            sc = scenario()
            sc["proposed_order"]["order_kind"] = kind
            self.assertRejected(self.ev(sc), "unknown_order_kind")
        sc = scenario()
        sc["proposed_order"]["reduce_only"] = True                            # an "ENTRY" disguised as reduce-only
        self.assertRejected(self.ev(sc), "entry_marked_reduce_only")

    def test_empty_and_garbage_inputs_fail_closed(self):
        self.assertFalse(self.gate.evaluate({}, {}, {}, {}, {})["approved"])
        self.assertFalse(self.gate.evaluate(None, None, None, None, None)["approved"])
        self.assertFalse(self.gate.evaluate("x", 1, [], 2.0, object())["approved"])
        d = self.gate.evaluate(**dict(scenario(), proposed_order=dict(order_kind="ENTRY")))
        self.assertFalse(d["approved"])

    def test_result_structure(self):
        d = self.ev(scenario())
        for key in ("approved", "decision", "reasons", "warnings", "order_id", "order_kind", "checks", "audit_id",
                    "evaluated_at"):
            self.assertIn(key, d)
        self.assertIsInstance(d["approved"], bool)
        self.assertIsInstance(d["reasons"], list)
        self.assertIsInstance(d["checks"], dict)

    def test_inputs_not_mutated(self):
        sc = scenario()
        before = copy.deepcopy(sc)
        self.ev(sc)
        self.assertEqual(sc, before)

    def test_rejected_order_key_can_be_resubmitted_after_fix(self):
        sc = scenario()
        sc["proposed_order"].pop("stop_price")
        self.assertFalse(self.ev(sc)["approved"])
        self.assertTrue(self.ev(scenario())["approved"])           # only APPROVED keys are burned


class HelperTests(unittest.TestCase):
    """Direct tests of the parsing primitives, so a weakened primitive cannot hide behind other checks."""

    def bad(self, fn, *a, **k):
        with self.assertRaises(G._Bad):
            fn(*a, **k)

    def test_num_rejects_bool_string_nan_inf_missing(self):
        for v in (True, False, "1", [1], float("nan"), float("inf"), float("-inf"), None):
            self.bad(G._num, {"a": v}, "a")
        self.bad(G._num, {}, "a")
        self.bad(G._num, None, "a")
        self.assertEqual(G._num({"a": 3}, "a"), 3.0)
        self.assertIsNone(G._num({}, "a", optional=True))
        self.bad(G._num, {"a": 0}, "a", pos=True)
        self.bad(G._num, {"a": -1}, "a", nonneg=True)
        self.assertEqual(G._num({"a": 0}, "a", nonneg=True), 0.0)

    def test_ts_rejects_naive_garbage_and_missing(self):
        for v in (datetime(2026, 1, 1), "2026-01-01T00:00:00", "garbage", "", 12345, None, ["x"]):
            self.bad(G._ts, {"t": v}, "t")
        self.assertEqual(G._ts({"t": "2026-01-01T00:00:00+02:00"}, "t"), datetime(2025, 12, 31, 22, tzinfo=timezone.utc))

    def test_txt_bool_int_helpers(self):
        for v in ("", "  ", 5, None):
            self.bad(G._txt, {"a": v}, "a")
        self.bad(G._txt, {"a": "x"}, "a", ("y",))
        self.bad(G._bool, {"a": 1}, "a")
        self.bad(G._bool, {}, "a")
        self.bad(G._int, {"a": 1.5}, "a")
        self.bad(G._int, {"a": -1}, "a")
        self.assertEqual(G._int({"a": 2.0}, "a"), 2)

    def test_tf_seconds(self):
        self.assertEqual((G.tf_seconds("1h"), G.tf_seconds("15m"), G.tf_seconds("1d")), (3600, 900, 86400))
        for v in ("", "h", "0h", "1w", "-1h", None, 5):
            self.bad(G.tf_seconds, v)


class ExtraBoundaryTests(Base):
    def test_unparseable_timestamps_rejected_in_entry(self):
        sc = scenario()
        sc["market_snapshot"]["observed_at"] = "not a date"
        self.assertRejected(self.ev(sc), "unparseable_timestamp", "data_fresh")
        sc = scenario()
        sc["signal"]["candle_timestamp"] = "garbage"
        self.assertRejected(self.ev(sc), "unparseable_timestamp", "signal_fresh")

    def test_negative_values_where_nonnegative_required(self):
        sc = scenario()
        sc["portfolio_state"]["positions"] = [dict(symbol="ETH/USD", direction="LONG", quantity=1, notional=-5000.0)]
        d = self.ev(sc)
        self.assertFalse(d["approved"], d)
        sc = scenario()
        sc["market_snapshot"].update(bid=None, ask=None, spread_bps=-5.0)
        self.assertRejected(self.ev(sc), "negative:spread_bps", "data_fresh")
        sc = scenario()
        sc["risk_state"]["bars_since_last_entry"] = -3
        self.assertRejected(self.ev(sc), "negative:bars_since_last_entry", "cooldown_ok")

    def test_spread_bps_without_quote(self):
        sc = scenario()
        sc["market_snapshot"].update(bid=None, ask=None, spread_bps=500.0)
        self.assertRejected(self.ev(sc), "spread_too_wide", "data_fresh")
        sc = scenario()
        sc["market_snapshot"].update(bid=None, ask=None, spread_bps=5.0)
        self.assertTrue(self.ev(sc)["approved"])

    def test_boolean_numbers_rejected_in_order_and_portfolio(self):
        for where, key in (("proposed_order", "quantity"), ("proposed_order", "notional"),
                           ("proposed_order", "leverage"), ("portfolio_state", "equity"),
                           ("portfolio_state", "cash"), ("risk_state", "day_start_equity")):
            sc = scenario()
            sc[where][key] = True
            self.assertFalse(self.ev(sc)["approved"], (where, key))

    def test_garbage_timestamp_in_exit_is_warning_not_block(self):
        sc = exit_scenario()
        sc["market_snapshot"]["observed_at"] = "garbage"
        d = self.ev(sc)
        self.assertTrue(d["approved"], d)
        self.assertIn("market_timestamp_unusable_exit_allowed", d["warnings"])

    def test_gate_does_not_mutate_inputs_even_on_rejection(self):
        for sc in (scenario(), exit_scenario(), dict(scenario(), proposed_order={"order_kind": "ENTRY"})):
            before = copy.deepcopy(sc)
            self.ev(sc)
            self.assertEqual(sc, before)


class ExitTests(Base):
    def test_valid_exit_long_and_short(self):
        for direction in ("LONG", "SHORT"):
            d = self.ev(exit_scenario(direction))
            self.assertTrue(d["approved"], d)
            self.assertEqual(set(d["checks"]), set(G.EXPECTED_EXIT_CHECKS))

    def test_partial_reduction_allowed(self):
        self.assertTrue(self.ev(exit_scenario("LONG", qty=0.04))["approved"])

    def test_exit_needs_no_stop_signal_or_fresh_data(self):
        sc = exit_scenario()
        sc["market_snapshot"]["observed_at"] = NOW - timedelta(days=1)
        d = self.ev(sc)
        self.assertTrue(d["approved"], d)
        sc = exit_scenario()
        sc["market_snapshot"].pop("observed_at")
        d = self.ev(sc)
        self.assertTrue(d["approved"], d)
        self.assertIn("market_timestamp_unusable_exit_allowed", d["warnings"])

    def test_exit_without_position_rejected(self):
        sc = exit_scenario()
        sc["portfolio_state"]["positions"] = []
        self.assertRejected(self.ev(sc), "no_position_to_reduce", "reduces_position")
        sc = exit_scenario("LONG")
        sc["proposed_order"]["side"] = "SHORT"                                # would flip, not reduce
        self.assertRejected(self.ev(sc), "no_position_to_reduce", "reduces_position")

    def test_exit_exceeding_position_rejected(self):
        self.assertRejected(self.ev(exit_scenario("LONG", qty=0.2)), "reduce_exceeds_position", "reduces_position")

    def test_exit_must_be_reduce_only(self):
        sc = exit_scenario()
        sc["proposed_order"]["reduce_only"] = False
        self.assertRejected(self.ev(sc), "exit_not_reduce_only", "schema_valid")
        sc = exit_scenario()
        sc["proposed_order"].pop("reduce_only")
        self.assertRejected(self.ev(sc), "missing_field:reduce_only", "schema_valid")

    def test_exit_malformed_and_missing_market_rejected(self):
        sc = exit_scenario()
        sc["market_snapshot"]["price"] = float("nan")
        self.assertRejected(self.ev(sc), "non_finite", "market_price_valid")
        sc = exit_scenario()
        sc["market_snapshot"] = None
        self.assertFalse(self.ev(sc)["approved"])
        sc = exit_scenario()
        sc["proposed_order"]["quantity"] = -1
        self.assertFalse(self.ev(sc)["approved"])

    def test_exit_duplicate_rejected(self):
        self.assertTrue(self.ev(exit_scenario(key="kx-same"))["approved"])
        self.assertRejected(self.ev(exit_scenario(key="kx-same")), "duplicate_order", "duplicate_order")


class AuditTests(Base):
    def test_every_decision_is_traced(self):
        a = self.ev(scenario())
        sc = scenario()
        sc["proposed_order"].update(order_id="o-2", idempotency_key="k-2")
        sc["proposed_order"].pop("stop_price")
        b = self.ev(sc)
        rows = self.gate.audit_rows()
        self.assertEqual([r["order_id"] for r in rows], ["o-1", "o-2"])
        self.assertEqual([r["approved"] for r in rows], [1, 0])
        self.assertEqual(a["audit_id"], rows[0]["id"])
        self.assertEqual(b["audit_id"], rows[1]["id"])
        self.assertEqual(len(rows[0]["inputs_sha256"]), 64)

    def test_malformed_inputs_are_still_logged(self):
        self.gate.evaluate(None, None, None, None, None)
        self.assertEqual(len(self.gate.audit_rows()), 1)

    def test_audit_failure_blocks_entry_but_not_exit(self):
        g = TradingGate(limits(), ":memory:", clock=lambda: NOW)
        g._db.execute("DROP TABLE evaluations")
        d = self.ev(scenario(), g)
        self.assertRejected(d, "audit_write_failed")
        d = self.ev(exit_scenario(), g)
        self.assertTrue(d["approved"], d)
        self.assertIn("audit_write_failed_exit_allowed", d["warnings"])
        g.close()

    def test_audit_persists_to_file_and_dedupe_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audit.sqlite3")
            g1 = TradingGate(limits(), path, clock=lambda: NOW)
            self.assertTrue(self.ev(scenario(), g1)["approved"])
            g1.close()
            g2 = TradingGate(limits(), path, clock=lambda: NOW)
            self.assertRejected(self.ev(scenario(), g2), "duplicate_order")
            self.assertEqual(len(g2.audit_rows()), 2)
            g2.close()

    def test_gate_writes_nothing_but_its_own_audit_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(os.listdir(tmp))
            g = TradingGate(limits(), os.path.join(tmp, "a.sqlite3"), clock=lambda: NOW)
            self.ev(scenario(), g)
            g.close()
            self.assertEqual(set(os.listdir(tmp)) - before, {"a.sqlite3"})


class LimitsTests(unittest.TestCase):
    def test_limits_are_required_and_validated(self):
        with self.assertRaises(TypeError):
            GateLimits()                                                          # no silent defaults
        for bad in (dict(max_position_pct=0), dict(max_leverage=float("nan")), dict(max_data_age_s=-1),
                    dict(min_stop_pct=20, max_stop_pct=10), dict(require_borrow_check="yes"),
                    dict(max_position_pct=True)):
            with self.assertRaises(ValueError, msg=str(bad)):
                limits(**bad)

    def test_gate_constructor_validation(self):
        with self.assertRaises(TypeError):
            TradingGate({"max_position_pct": 5}, ":memory:")
        with self.assertRaises(ValueError):
            TradingGate(limits(), "")

    def test_naive_clock_fails_closed(self):
        g = TradingGate(limits(), ":memory:", clock=lambda: datetime(2026, 10, 5, 12, 5))
        self.assertRejected_(g.evaluate(**scenario()))
        g.close()

    def assertRejected_(self, d):
        self.assertFalse(d["approved"])
        self.assertIn("clock_not_timezone_aware", d["reasons"])


class ReconcileTests(unittest.TestCase):
    tol = dict(qty_rel=0.001, slippage_bps=25)
    P = staticmethod(lambda s, d, q: dict(symbol=s, direction=d, quantity=q))

    def types(self, orders, internal, broker, tol=None):
        return sorted(a["type"] for a in reconcile(orders, internal, broker, tol or self.tol)["alerts"])

    def test_clean(self):
        p = [self.P("BTC/USD", "LONG", 0.1)]
        self.assertTrue(reconcile([], p, p, self.tol)["ok"])

    def test_position_mismatches(self):
        i = [self.P("BTC/USD", "LONG", 0.1)]
        self.assertEqual(self.types([], i, []), ["position_missing_at_broker"])
        self.assertEqual(self.types([], [], i), ["unexpected_position_at_broker"])
        self.assertEqual(self.types([], i, [self.P("BTC/USD", "SHORT", 0.1)]), ["direction_mismatch"])
        self.assertEqual(self.types([], i, [self.P("BTC/USD", "LONG", 0.08)]), ["position_qty_mismatch"])

    def test_slippage_alert(self):
        o = dict(order_id="o", symbol="X", side="LONG", quantity=1, price=100.0, status="FILLED", filled_quantity=1,
                 avg_fill_price=101.0)
        p = [self.P("X", "LONG", 1)]
        self.assertEqual(self.types([o], p, p), ["fill_price_slippage"])

    def test_malformed_input_fails_closed(self):
        for args in ((None, [], [], self.tol), ([], None, [], self.tol), ([], [], None, self.tol),
                     ([], [], [], {}), ([], [], [], dict(qty_rel=float("nan"), slippage_bps=1)),
                     ([dict(order_id="o")], [], [], self.tol),
                     ([], [self.P("X", "LONG", float("nan"))], [], self.tol),
                     ([], [self.P("X", "LONG", 1), self.P("X", "LONG", 1)], [], self.tol)):
            r = reconcile(*args)
            self.assertFalse(r["ok"], args)
            self.assertIn("reconcile_error", [a["type"] for a in r["alerts"]])

    def test_overfill_is_an_error(self):
        o = dict(order_id="o", symbol="X", side="LONG", quantity=1, price=100.0, status="FILLED", filled_quantity=2,
                 avg_fill_price=100.0)
        self.assertEqual(self.types([o], [], []), ["reconcile_error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
