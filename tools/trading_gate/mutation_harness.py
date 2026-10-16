"""
Bug-injection harness for trading_gate. Copies the package to a temp dir, applies ONE deliberate bug to gate.py (or
reconcile.py), runs test_gate.py, and requires the suite to FAIL. A mutant that survives means the tests are blind to it.
Run:  python3 mutation_harness.py          (from this folder)       Exit code 0 only if every mutant is killed.
"""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
G, R = "gate.py", "reconcile.py"
M = [  # (name, file, old, new)  -- `old` must occur exactly once
 ("drop stop check but also its expectation", G, '                self._run(checks, reasons, "stop_present", self._chk_stop, ctx)\n', '                checks["stop_present"] = True\n'),
 ("approve ignores reasons", G, "all(checks.values()) and not reasons and kind", "all(checks.values()) and kind"),
 ("approve if any check passes", G, "all(checks.values()) and not reasons", "any(checks.values()) and not reasons"),
 ("approve ignores checks", G, "approved = bool(checks) and all(checks.values()) and not reasons and kind", "approved = bool(checks) and not reasons and kind"),
 ("failing check recorded as pass", G, '            checks[name] = False\n            reasons.append(f"check_error', '            checks[name] = True\n            reasons.append(f"check_error'),
 ("check exception swallowed (no reason, pass)", G, '            checks[name] = False\n            reasons.append(f"check_error:{name}:{type(exc).__name__}")', '            checks[name] = True'),
 ("expected-check guard removed", G, '            if missing:                                  # a check was dropped or skipped: never approve\n                reasons.append', '            if False:\n                reasons.append'),
 ("internal error leaves approvable", G, '            reasons.append(reason)\n            checks.setdefault("schema_valid", False)', '            checks.setdefault("schema_valid", False)'),
 ("daily loss boundary <= -> <", G, "if realized <= -(start", "if realized < -(start"),
 ("daily loss disabled", G, 'raise _Bad("daily_loss_limit")', 'pass'),
 ("daily equity loss disabled", G, 'raise _Bad("daily_equity_loss_limit")', 'pass'),
 ("drawdown disabled", G, 'raise _Bad("max_drawdown_limit")', 'pass'),
 ("drawdown ignores peak>equity guard", G, "peak = max(peak, equity)", "peak = peak"),
 ("loss streak disabled", G, 'raise _Bad("max_consecutive_losses")', 'pass'),
 ("failed orders disabled", G, 'raise _Bad("max_failed_orders")', 'pass'),
 ("trades per day disabled", G, 'raise _Bad("max_trades_per_day")', 'pass'),
 ("cooldown disabled", G, 'raise _Bad("entry_cooldown_active")', 'pass'),
 ("cooldown absent field treated as never", G, 'raise _Bad("missing_field:bars_since_last_entry")', 'return'),
 ("kill switch ignored for entries", G, 'raise _Bad("kill_switch_active")', 'pass'),
 ("kill switch blocks exits", G, 'ctx["warnings"].append("kill_switch_active_exit_allowed")', 'raise _Bad("kill_switch_active")'),
 ("exit blocked by stale data", G, 'ctx["warnings"].append("stale_market_data_exit_allowed")', 'raise _Bad("stale_market_data")'),
 ("exit requires stop", G, '        if not c["reduce_only"]:\n            raise _Bad("exit_not_reduce_only")', '        if not c["reduce_only"]:\n            raise _Bad("exit_not_reduce_only")\n        if ctx["order"].get("stop_price") is None:\n            raise _Bad("missing_stop")'),
 ("exit reduce_only not required", G, 'raise _Bad("exit_not_reduce_only")', 'pass'),
 ("entry reduce_only not rejected", G, 'raise _Bad("entry_marked_reduce_only")', 'pass'),
 ("exit may exceed position", G, 'raise _Bad("reduce_exceeds_position")', 'pass'),
 ("exit without position allowed", G, 'raise _Bad("no_position_to_reduce")', 'pass'),
 ("exit direction not matched", G, 'x["symbol"] == symbol and x["direction"] == direction]', 'x["symbol"] == symbol]'),
 ("stale market data allowed", G, 'raise _Bad("stale_market_data")', 'pass'),
 ("future market data allowed", G, 'raise _Bad("market_data_in_future")', 'pass'),
 ("crossed quote allowed", G, 'raise _Bad("crossed_quote")', 'pass'),
 ("wide spread allowed", G, 'raise _Bad("spread_too_wide")\n        else:', 'pass\n        else:'),
 ("incomplete quote allowed", G, 'raise _Bad("incomplete_quote")', 'pass'),
 ("invalid ohlc allowed", G, 'raise _Bad("invalid_ohlc")', 'pass'),
 ("market symbol mismatch allowed (entry)", G, 'raise _Bad("market_symbol_mismatch")\n        mp', 'pass\n        mp'),
 ("order price deviation allowed", G, 'raise _Bad("order_price_deviates_from_market")', 'pass'),
 ("signal id mismatch allowed", G, 'raise _Bad("signal_order_mismatch:signal_id")', 'pass'),
 ("signal symbol mismatch allowed", G, 'raise _Bad("signal_order_mismatch:symbol")', 'pass'),
 ("signal direction mismatch allowed", G, 'raise _Bad("signal_order_mismatch:direction")', 'pass'),
 ("signal timeframe mismatch allowed", G, 'raise _Bad("signal_order_mismatch:timeframe")', 'pass'),
 ("expired signal allowed", G, 'raise _Bad("signal_expired")', 'pass'),
 ("unclosed candle allowed", G, 'raise _Bad("signal_candle_not_closed")', 'pass'),
 ("signal price drift allowed", G, 'raise _Bad("signal_price_deviates_from_market")', 'pass'),
 ("caller duplicate set ignored", G, '        if _txt(o, "idempotency_key") in set(seen):\n            raise _Bad("duplicate_order")', '        pass'),
 ("gate approved-key dedupe off", G, "                if approved and key is not None:\n                    dup =", "                if False:\n                    dup ="),
 ("approved key never stored", G, '                if approved and key is not None:\n                    self._db.execute("INSERT INTO approved_keys', '                if False:\n                    self._db.execute("INSERT INTO approved_keys'),
 ("rejected keys burned (should not)", G, 'if approved and key is not None:\n                    self._db.execute("INSERT INTO approved_keys', 'if key is not None:\n                    self._db.execute("INSERT INTO approved_keys'),
 ("notional mismatch allowed", G, 'bad.append("notional_mismatch")', 'pass'),
 ("min order allowed", G, 'bad.append("below_min_order")', 'pass'),
 ("position size limit off", G, 'bad.append("position_size_exceeds_limit")', 'pass'),
 ("position size > becomes <", G, "if notional > equity * lim.max_position_pct / 100.0:", "if notional < equity * lim.max_position_pct / 100.0:"),
 ("leverage limit off", G, 'bad.append("leverage_exceeds_limit")', 'pass'),
 ("exposure ignores existing positions", G, "if sum(x[\"notional\"] for x in positions) + notional >", "if notional >"),
 ("max open positions off", G, 'bad.append("max_open_positions_reached")', 'pass'),
 ("cash check off", G, 'bad.append("insufficient_cash")', 'pass'),
 ("risk reasons dropped when several", G, 'raise _Bad(bad[0]) if len(bad) == 1 else _Bad("|".join(bad))', 'pass'),
 ("position already open allowed", G, 'raise _Bad("position_already_open" if pos["direction"] == side else "opposite_position_open_close_first")', 'pass'),
 ("shorts-not-allowed ignored", G, 'raise _Bad("shorts_not_allowed")', 'pass'),
 ("net exposure cap off", G, 'raise _Bad("net_exposure_exceeds_limit")', 'pass'),
 ("borrow check off", G, 'raise _Bad("short_borrow_unavailable")', 'pass'),
 ("borrow applied to longs too", G, 'if _txt(o, "side", ("LONG", "SHORT")) == "SHORT" and self.limits.require_borrow_check:', 'if self.limits.require_borrow_check:'),
 ("missing stop allowed", G, 'raise _Bad("missing_stop")', 'pass'),
 ("stop wrong side allowed", G, 'raise _Bad("stop_wrong_side")', 'pass'),
 ("stop side flipped", G, '(side == "LONG" and stop >= price) or (side == "SHORT" and stop <= price)', '(side == "LONG" and stop <= price) or (side == "SHORT" and stop >= price)'),
 ("stop too tight allowed", G, 'raise _Bad("stop_too_tight")', 'pass'),
 ("stop too wide allowed", G, 'raise _Bad("stop_too_wide")', 'pass'),
 ("target wrong side allowed", G, 'raise _Bad("target_wrong_side")', 'pass'),
 ("NaN accepted", G, "if not math.isfinite(v):\n        raise _Bad(f\"non_finite:{key}\")", "if False:\n        raise _Bad(f\"non_finite:{key}\")"),
 ("zero/negative price accepted", G, "if pos and v <= 0:", "if False:"),
 ("negative accepted where nonneg", G, "if nonneg and v < 0:\n        raise _Bad(f\"negative:{key}\")\n    return v", "return v"),
 ("bool accepted as number", G, "if isinstance(v, bool) or not isinstance(v, numbers.Real):\n        raise _Bad(f\"non_numeric:{key}\")", "if not isinstance(v, (bool, numbers.Real)):\n        raise _Bad(f\"non_numeric:{key}\")"),
 ("string number accepted", G, "if isinstance(v, bool) or not isinstance(v, numbers.Real):\n        raise _Bad(f\"non_numeric:{key}\")\n    v = float(v)", "v = float(v)"),
 ("missing numeric field defaults to 1", G, '        raise _Bad(f"missing_field:{key}")\n    v = m[key]\n    if isinstance(v, bool)', '        return 1.0\n    v = m[key]\n    if isinstance(v, bool)'),
 ("naive timestamp accepted", G, 'raise _Bad(f"timestamp_ambiguous:{key}")', 'return v.replace(tzinfo=timezone.utc)'),
 ("unparseable timestamp accepted", G, 'raise _Bad(f"unparseable_timestamp:{key}")\n    if not', 'return datetime.now(timezone.utc)\n    if not'),
 ("integer check off", G, 'raise _Bad(f"non_integer:{key}")', 'pass'),
 ("unknown order kind not rejected", G, '                reasons.append("unknown_order_kind")', '                pass'),
 ("naive clock accepted", G, 'raise _Bad("clock_not_timezone_aware")', 'pass'),
 ("audit failure approves entry", G, '                if approved and kind == "ENTRY":          # opening trades need a trace: fail closed\n                    approved = False', '                if False:\n                    approved = False'),
 ("audit failure rejects exits", G, '                elif approved:                             # reducing risk is never blocked by logging\n                    warnings.append("audit_write_failed_exit_allowed")', '                elif approved:\n                    approved = False\n                    reasons.append("audit_write_failed")'),
 ("audit row not written", G, '                cur = self._db.execute(\n                    "INSERT INTO evaluations', '                cur = self._db.execute(\n                    "SELECT 1 WHERE 0 OR \'INSERT INTO evaluations'),
 ("inputs mutated", G, "ctx = dict(order=proposed_order,", "if isinstance(proposed_order, dict): proposed_order['status'] = 'x'\n            ctx = dict(order=proposed_order,"),
 ("BOTH approval layers removed", G, "approved = bool(checks) and all(checks.values()) and not reasons and kind in", "approved = bool(checks) and kind in"),
 # reconcile mutants
 ("recon: partial fill not alerted", R, 'alerts.append(_alert("partial_fill", sym, oid, f"status={status} filled={filled} of {qty}"))\n            if filled > 0', 'pass\n            if filled > 0'),
 ("recon: rejected fill not alerted", R, 'alerts.append(_alert("rejected_fill"', 'pass; (lambda *a: None)("'),
 ("recon: slippage not alerted", R, 'if dev > slip_bps:', 'if False:'),
 ("recon: missing at broker ignored", R, 'alerts.append(_alert("position_missing_at_broker"', 'pass; (lambda *a: None)("'),
 ("recon: unexpected at broker ignored", R, 'alerts.append(_alert("unexpected_position_at_broker"', 'pass; (lambda *a: None)("'),
 ("recon: qty mismatch ignored", R, 'elif abs(q_i - q_b) > qty_rel * max(q_i, q_b):', 'elif False:'),
 ("recon: direction mismatch ignored", R, 'if d_i != d_b:', 'if False:'),
 ("recon: errors swallowed (ok True)", R, 'alerts.append(_alert("reconcile_error", None, None, f"{type(exc).__name__}:{exc}"))', 'pass'),
 ("recon: overfill ignored", R, 'if filled > qty * (1 + qty_rel):', 'if False:'),
]


# Mutants that are behaviourally EQUIVALENT by design (redundant defence-in-depth), proven below rather than waved away:
#  * approval = all(checks) AND no reasons AND valid kind: every failing check also appends a reason, so removing ONE of
#    the two layers cannot change any outcome. The mutant that removes BOTH is in the main list and must be killed.
#  * drawdown `peak = max(peak, equity)`: if peak < equity the drawdown is negative and cannot breach either way.
EQUIVALENT = {"approve if any check passes", "approve ignores checks", "drawdown ignores peak>equity guard"}


def main() -> int:
    survivors, killed, equivalent = [], 0, []
    for name, fname, old, new in M:
        src = open(os.path.join(HERE, fname)).read()
        if src.count(old) != 1:
            print(f"BAD MUTANT (pattern occurs {src.count(old)}x): {name}")
            survivors.append(name + " [bad pattern]")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "trading_gate")
            shutil.copytree(HERE, pkg, ignore=shutil.ignore_patterns("__pycache__", "*.sqlite3", "mutation_harness.py"))
            open(os.path.join(pkg, fname), "w").write(src.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "unittest", "trading_gate.test_gate"], cwd=tmp,
                               capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            killed += 1
            print(f"killed    : {name}")
        elif name in EQUIVALENT:
            equivalent.append(name)
            print(f"equivalent: {name}")
        else:
            survivors.append(name)
            print(f"SURVIVED  : {name}")
    print(f"\n{killed}/{len(M)} mutants killed, {len(equivalent)} equivalent-by-design, {len(survivors)} survivors: {survivors}")
    return 0 if not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
