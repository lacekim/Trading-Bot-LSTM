import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_bot_v4.execution.order_manager import OrderManager
from trading_bot_v4.risk.market_cap_tiers import AssetRiskDecision


def signal(timestamp="2026-01-01T01:00:00Z", direction="LONG", price=100.0, symbol="BTC", **ohlc):
    row = {"timestamp": timestamp, "symbol": symbol, "timeframe": "1h",
           "model_direction": direction, "model_probability": 0.8, "price": price}
    row.update(ohlc)
    return pd.DataFrame([row])


class PaperOrderManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "paper.sqlite3"
        self.market_cap_risk = patch(
            "trading_bot_v4.execution.order_manager.Config.MARKET_CAP_RISK_ENABLED", False
        )
        self.market_cap_risk.start()

    def tearDown(self):
        self.market_cap_risk.stop()
        self.temp.cleanup()

    def test_persists_and_deduplicates_position(self):
        manager = OrderManager(self.db)
        first = manager.process_signals(signal())
        again = manager.process_signals(signal())
        manager.close()
        restored = OrderManager(self.db).sync()
        self.assertEqual(first.orders_opened, 1)
        self.assertEqual(again.orders_opened, 0)
        self.assertEqual(restored.open_positions, 1)

    def test_reversal_closes_and_respects_symbol_cooldown(self):
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MIN_BARS_BETWEEN_TRADES", 4):
            manager = OrderManager(self.db)
            manager.process_signals(signal())
            result = manager.process_signals(signal("2026-01-01T02:00:00Z", "SHORT", 101.0))
            self.assertEqual(result.orders_closed, 1)
            self.assertEqual(result.orders_opened, 0)
            self.assertIn("cooldown", result.rejection_reasons)
            manager.close()

    def test_exit_only_reversal_closes_without_opening_unqualified_short(self):
        manager = OrderManager(self.db)
        manager.process_signals(signal())
        reversal = signal("2026-01-01T02:00:00Z", "SHORT", 101.0)
        reversal["_entry_eligible"] = False
        result = manager.process_signals(reversal)
        self.assertEqual(result.orders_closed, 1)
        self.assertEqual(result.orders_opened, 0)
        self.assertEqual(result.open_positions, 0)
        manager.close()

    def test_configured_window_exit_is_enforced(self):
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_HOLD_CANDLES", 1):
            manager = OrderManager(self.db)
            manager.process_signals(signal())
            result = manager.process_signals(signal("2026-01-01T02:00:00Z", "HOLD", 101.0))
            trade = manager.connection.execute("SELECT exit_reason FROM closed_trades").fetchone()
            self.assertEqual(result.orders_closed, 1)
            self.assertEqual(result.open_positions, 0)
            self.assertEqual(trade["exit_reason"], "window_exit")
            manager.close()

    def test_risk_limit_rejects_new_order(self):
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_OPEN_POSITIONS", 0):
            manager = OrderManager(self.db)
            result = manager.process_signals(signal())
            self.assertEqual(result.signals_rejected, 1)
            self.assertEqual(result.open_positions, 0)
            manager.close()

    def test_intrabar_low_triggers_long_stop_even_when_close_is_above_stop(self):
        manager = OrderManager(self.db)
        manager.process_signals(signal())
        timestamp = (pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)).isoformat()
        result = manager.process_signals(signal(timestamp, "HOLD", 100.0, open=100.0, high=101.0, low=97.0, close=100.0))
        trade = manager.connection.execute("SELECT exit_reason FROM closed_trades").fetchone()
        self.assertEqual(result.orders_closed, 1)
        self.assertEqual(trade["exit_reason"], "stop_loss")
        manager.close()

    def test_stale_closed_candle_blocks_entry(self):
        manager = OrderManager(self.db)
        stale = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5)).isoformat()
        result = manager.process_signals(signal(stale, "LONG", 100.0, open=100.0, high=101.0, low=99.0, close=100.0))
        reason = manager.connection.execute("SELECT reason FROM signals").fetchone()["reason"]
        self.assertEqual(result.orders_opened, 0)
        self.assertEqual(reason, "stale candle")
        manager.close()

    def test_market_data_gap_blocks_entry(self):
        manager = OrderManager(self.db)
        timestamp = (pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)).isoformat()
        result = manager.process_signals(signal(timestamp, "LONG", 100.0, open=100.0, high=101.0,
                                                low=99.0, close=100.0, candle_gap_seconds=7200.0))
        self.assertEqual(result.orders_opened, 0)
        self.assertIn("market-data gap", result.rejection_reasons)
        manager.close()

    def test_maximum_trades_per_day_blocks_additional_entry(self):
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_TRADES_PER_DAY", 1):
            manager = OrderManager(self.db)
            manager.process_signals(signal(symbol="BTC"))
            result = manager.process_signals(signal("2026-01-01T02:00:00Z", symbol="ETH"))
            self.assertEqual(result.orders_opened, 0)
            self.assertIn("maximum trades per day", result.rejection_reasons)
            manager.close()

    def test_small_cap_tier_caps_position_and_persists_entry_risk(self):
        manager = OrderManager(self.db)
        observed_at = pd.Timestamp.now(tz="UTC").isoformat()
        manager.record_qualified_market_snapshot("BTC", 99.9, 100.0, observed_at, 10.0, 5.0, 0.0)
        decision = AssetRiskDecision(True, "SMALL", 50_000_000.0, .25, 5.0)
        with patch("trading_bot_v4.execution.order_manager.Config.MARKET_CAP_RISK_ENABLED", True), \
             patch("trading_bot_v4.execution.order_manager.evaluate_asset_risk", return_value=decision):
            result = manager.process_signals(signal())
        position = manager.connection.execute("SELECT position_id,notional FROM positions").fetchone()
        metadata = manager.connection.execute(
            "SELECT tier,market_cap_usd,risk_pct,max_position_pct FROM position_risk_metadata"
        ).fetchone()
        self.assertEqual(result.orders_opened, 1)
        self.assertLessEqual(position["notional"], 500.0)
        self.assertEqual(metadata["tier"], "SMALL")
        self.assertEqual(metadata["market_cap_usd"], 50_000_000.0)
        self.assertEqual(metadata["risk_pct"], .25)
        manager.close()

    def test_whitelist_update_is_persistent(self):
        manager = OrderManager(self.db)
        manager.update_whitelist(["LDO", "SYRUP"])
        value = manager.connection.execute("SELECT value FROM runtime_state WHERE key='current_whitelist'").fetchone()["value"]
        self.assertEqual(value, "LDO,SYRUP")
        manager.close()

    def test_shadow_challenger_requires_confirmation_and_persists_trade(self):
        manager = OrderManager(self.db)
        end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
        rows = []
        for offset in range(25):
            price = 100.0 + offset
            rows.append({
                "timestamp": end - pd.Timedelta(hours=24 - offset), "symbol": "BTC", "timeframe": "1h",
                "model_direction": "LONG", "model_probability": 0.8, "price": price,
                "open": price - 1.0, "high": price + 0.2, "low": price - 1.2, "close": price,
            })
        history = pd.DataFrame(rows)
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_CANDLE_AGE_MULTIPLIER", 10.0):
            manager.process_challenger_signals(history)
        self.assertEqual(manager.connection.execute("SELECT COUNT(*) FROM challenger_positions").fetchone()[0], 1)

        newest = history.copy()
        newest.loc[len(newest)] = {
            "timestamp": end + pd.Timedelta(hours=1), "symbol": "BTC", "timeframe": "1h",
            "model_direction": "HOLD", "model_probability": 0.2, "price": 120.0,
            "open": 121.0, "high": 121.0, "low": 118.0, "close": 120.0,
        }
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_CANDLE_AGE_MULTIPLIER", 10.0):
            manager.process_challenger_signals(newest)
        trade = manager.connection.execute("SELECT exit_reason,return_pct FROM challenger_trades").fetchone()
        self.assertEqual(trade["exit_reason"], "stop_loss")
        self.assertLess(trade["return_pct"], 0.0)
        manager.close()

    def test_shadow_challenger_can_open_and_profit_from_short(self):
        manager = OrderManager(self.db)
        end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
        rows = []
        for offset in range(25):
            price = 125.0 - offset
            rows.append({
                "timestamp": end - pd.Timedelta(hours=24 - offset), "symbol": "ETH", "timeframe": "1h",
                "model_direction": "SHORT", "model_probability": 0.8, "price": price,
                "open": price + 1.0, "high": price + 1.2, "low": price - 0.2, "close": price,
            })
        history = pd.DataFrame(rows)
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_CANDLE_AGE_MULTIPLIER", 10.0):
            manager.process_challenger_signals(history)
        position = manager.connection.execute("SELECT direction FROM challenger_positions").fetchone()
        self.assertEqual(position["direction"], "SHORT")

        newest = history.copy()
        newest.loc[len(newest)] = {
            "timestamp": end + pd.Timedelta(hours=1), "symbol": "ETH", "timeframe": "1h",
            "model_direction": "HOLD", "model_probability": 0.2, "price": 96.0,
            "open": 97.0, "high": 97.0, "low": 95.0, "close": 96.0,
        }
        with patch("trading_bot_v4.execution.order_manager.Config.PAPER_MAX_CANDLE_AGE_MULTIPLIER", 10.0):
            manager.process_challenger_signals(newest)
        trade = manager.connection.execute("SELECT direction,exit_reason,return_pct FROM challenger_trades").fetchone()
        self.assertEqual(trade["direction"], "SHORT")
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertGreater(trade["return_pct"], 0.0)
        manager.close()


if __name__ == "__main__":
    unittest.main()
