"""
Bug-injection harness for trading_gate.shadow / shadow_report. Copies the package to a temp dir, applies ONE deliberate bug,
runs test_shadow.py, and requires it to FAIL. Run:  python3 mutation_harness_shadow.py   (exit 0 only if all killed)
"""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SH, RP = "shadow.py", "shadow_report.py"
M = [  # (name, file, old, new)  -- `old` must occur exactly once
 ("bot DB opened read-write-create", SH, '"?mode=ro"', '"?mode=rwc"'),
 ("query_only dropped (alone)", SH, 'conn.execute("PRAGMA query_only=ON")', 'pass'),
 ("BOTH read-only protections dropped", SH, 'uri = "file:" + urllib.parse.quote(os.path.abspath(path)) + "?mode=ro"\n    conn = sqlite3.connect(uri, uri=True, timeout=10)\n    conn.execute("PRAGMA query_only=ON")', 'uri = "file:" + urllib.parse.quote(os.path.abspath(path)) + "?mode=rwc"\n    conn = sqlite3.connect(uri, uri=True, timeout=10)'),
 ("equity row at the same second used", SH, 'bisect.bisect_left(h["eq_times"], t) - 1', 'bisect.bisect_right(h["eq_times"], t) - 1'),
 ("latest (future) equity row used", SH, 'i = bisect.bisect_left(h["eq_times"], t) - 1          # strictly before t', 'i = len(h["eq_times"]) - 1'),
 ("peak equity = current equity", SH, 'h["eq_peak"].append(peak)', 'h["eq_peak"].append(v)'),
 ("snapshot after the event used", SH, 'recorded_at<=? "', 'recorded_at>=? "'),
 ("kill switch assumed True", SH, 'return dict(kill_switch=False,', 'return dict(kill_switch=True,'),
 ("seen keys not passed (entry)", SH, 'seen_idempotency_keys=list(rp.entry_keys))\n        risk = self._risk_state(t, h, rp, rec, symbol, cts)', 'seen_idempotency_keys=[])\n        risk = self._risk_state(t, h, rp, rec, symbol, cts)'),
 ("event applied BEFORE evaluation", SH, '            rec = self._evaluate_event(ev, replay, h, run_id)\n            self._apply(ev, replay, h, outcome_only=False)', '            self._apply(ev, replay, h, outcome_only=False)\n            rec = self._evaluate_event(ev, replay, h, run_id)'),
 ("entry sorted before exit in same second", SH, '0 if e["kind"] == "EXIT" else 1', '1 if e["kind"] == "EXIT" else 0'),
 ("naive candle timestamp rejected", SH, '_dt(sig["candle_timestamp"], "signals.candle_timestamp", naive_is_utc=True))\n            if isinstance', '_dt(sig["candle_timestamp"], "signals.candle_timestamp"))\n            if isinstance'),
 ("daily P&L ignores the date", SH, 'if t is not None and t.date().isoformat() == date)', 'if t is not None)'),
 ("trades_opened_today constant", SH, 'trades_opened_today=rp.entries_by_date.get(date, 0)', 'trades_opened_today=0'),
 ("loss streak counted from oldest", SH, 'for _, net, _ in reversed(self.closed):', 'for _, net, _ in self.closed:'),
 ("cash ignores entry fees", SH, '                - sum(p["entry_fee"] for p in self.open.values()))\n\n    def equity_realized', '                )\n\n    def equity_realized'),
 ("replay equity ignores open entry fees", SH, 'return self.start + self.net_closed - sum(p["entry_fee"] for p in self.open.values())', 'return self.start + self.net_closed'),
 ("partial exit remainder wrong", SH, 'ratio = (pos["quantity"] - qty) / pos["quantity"]', 'ratio = qty / pos["quantity"]'),
 ("intent mismatch not flagged", SH, 'rec["intent_mismatch"] = 1', 'pass'),
 ("stop gap never classified", SH, '(r == "missing_stop" and "stop_price" in gaps)', '(r == "missing_stop" and False)'),
 ("stop gap not recorded", SH, 'rec["data_gaps"].append("stop_price")\n            lev = self.assumed_leverage', 'lev = self.assumed_leverage'),
 ("adapter error counted as rejection", SH, 'rec["approved"], rec["decision"] = None, "ADAPTER_ERROR"', 'rec["approved"], rec["decision"] = 0, "ADAPTER_ERROR"'),
 ("run_once exceptions escape", SH, 'except Exception as exc:  # noqa: BLE001 - the shadow must never take the caller down', 'except ValueError as exc:  # noqa'),
 ("schema drift ignored", SH, '            if problems:\n                summary.update(status="SCHEMA_DRIFT"', '            if False:\n                summary.update(status="SCHEMA_DRIFT"'),
 ("already-logged events re-evaluated", SH, '            if ev["key"] in done:', '            if False:'),
 ("account reconciliation always ok", SH, 'all(abs(v) <= 1e-6 for v in diffs.values())', 'True'),
 ("positions reconciliation ignores alerts", SH, 'r["ok"] and abs(gross_actual - rp.gross())', 'True or abs(gross_actual - rp.gross())'),
 ("fills treated as always filled", SH, 'filled = q if o["status"] == "FILLED" else 0.0', 'filled = q'),
 ("accepted-without-order not checked", SH, 'accepted_no_order = sorted(s for s, r_ in sigs.items() if r_["status"] == "ACCEPTED" and s not in orders_by_sig)', 'accepted_no_order = []'),
 ("duplicate signal orders not checked", SH, 'dup_orders = sorted(s for s, os_ in orders_by_sig.items() if len(os_) > 1)', 'dup_orders = []'),
 ("borrow check not marked simulated", SH, '"simulated" if side == "SHORT" else "not_applicable"', '"not_applicable"'),
 ("out dir == bot folder allowed", SH, '        print("refusing: --out must not be the folder that holds the bot\'s database", file=sys.stderr)\n        return 2', '        pass'),
 ("position capture disabled", SH, '            self._capture_positions(conn)\n            self._process', '            self._process'),
 ("stop ignored from captured row", SH, 'stop = self._try(rec, "stop_price", lambda: _num(cap["stop_price"], "positions.stop_price", pos=True))', 'stop = None'),
 ("leverage ignored from captured row", SH, 'lev = self._try(rec, "leverage", lambda: _num(cap["leverage"], "positions.leverage", pos=True))', 'lev = self.assumed_leverage'),
 ("same-second ambiguity not flagged", SH, 'if ev["same_second_events"] > 1:', 'if ev["same_second_events"] > 99:'),
 ("reduced exits reported as closed", SH, '"closed" if (qty or 0) >= pos["quantity"] * (1 - 1e-9) else "reduced"', '"closed"'),
 ("daily realised P&L not passed", SH, 'daily_realized_pnl=rp.day_realized(date),', 'daily_realized_pnl=0.0,'),
 ("day start equity not mapped", SH, 'start_eq = self._try(rec, "day_start_equity", lambda: _num(d["start_equity"], "daily_risk.start_equity", pos=True)) \\\n            if d is not None else None', 'start_eq = None'),
 ("malformed rows poison the replay", SH, 'notional = _num(o["notional"], "orders.notional", pos=True)\n                fill = _num(o["fill_price"], "orders.fill_price", pos=True)\n                fee = _num(o["fee"], "orders.fee", nonneg=True)', 'notional = float(o["notional"]); fill = float(o["fill_price"]); fee = float(o["fee"])'),
 ("jsonl log not written", SH, '        with open(self.jsonl_path, "a", encoding="utf-8") as fh:\n            fh.write(json.dumps(cols, default=str, sort_keys=True) + "\\n")', '        pass'),
 ("max position limit loosened", SH, 'max_position_pct=10,', 'max_position_pct=20,'),
 ("net exposure limit loosened", SH, 'max_net_exposure_pct=30,', 'max_net_exposure_pct=60,'),
 ("daily loss limit loosened", SH, 'max_daily_loss_pct=1,', 'max_daily_loss_pct=5,'),
 ("drawdown limit loosened", SH, 'max_drawdown_pct=5,', 'max_drawdown_pct=50,'),
 ("exit market price from future", SH, 'market = dict(symbol=symbol, price=price, observed_at=self._try(rec, "observed_at", lambda: _dt(c["exit_time"], "closed_trades.exit_time")))', 'market = dict(symbol=symbol, price=price, observed_at=None)'),
 ("report intent check removed entirely", RP, 'bad = bool(i["entry_orders_whose_side_differs_from_signal"] or rc.get("accepted_without_order")\n               or rc.get("orders_without_accepted_signal") or rc.get("direction_mismatch"))', 'bad = False'),
 # ---- coverage / campaign loop (this change)
 ("campaign start moves every pass", SH, "INSERT OR IGNORE INTO meta(key,value) VALUES('campaign_start',?)", "INSERT OR REPLACE INTO meta(key,value) VALUES('campaign_start',?)"),
 ("loop ignores coverage completion", SH, 'if cov is not None and cov["complete"]:', 'if False:'),
 ("loop stops at 7 of 8", SH, 'if cov is not None and cov["complete"]:', 'if cov is not None and cov["n_observed"] >= cov["n_required"] - 1:'),
 ("loop ignores the deadline", SH, 'elif now() >= until:', 'elif False:'),
 ("last sleep not capped at the deadline", SH, 'min(float(interval), (until - now()).total_seconds())', 'float(interval)'),
 ("max-events limit ignored", SH, 'elif max_events is not None and total_new >= max_events:', 'elif False:'),
 ("new decisions not counted for max-events", SH, 'total_new += s["events_new"]', 'total_new += 0'),
 ("report not written at stop", SH, 'write_report(obs.out_dir, now=now())', 'pass'),
 ("stop reason not recorded", SH, 'obs._meta_set("stop_reason", reason)', 'pass'),
 ("stale stop reason kept on restart", SH, 'obs._meta_delete("stop_reason", "stopped_at")', 'pass'),
 ("console line on every pass", SH, '            if changes:\n                tail', '            if True:\n                tail'),
 ("run row on every pass", SH, 'if summary["events_new"] > 0 or summary["recon_changed"] or sig != self._last_pass_sig:', 'if True:'),
 ("reconciliation row on every pass", SH, 'if last is not None and last[0] == int(ok) and last[1] == text:\n            return False', 'if False:\n            return False'),
 ("schema drift exit code lost in loop", SH, '            if s["status"] == "SCHEMA_DRIFT":\n                code = 2\n            cov_error', '            cov_error'),
 ("captured stops not marked live", SH, 'p["collateral"], p["quantity"], p["notional"], now, LIVE_STOP_SOURCE))', 'p["collateral"], p["quantity"], p["notional"], now, "fixture"))'),
 ("old database not migrated", SH, 'if "source" not in {r[1] for r in self.sdb.execute("PRAGMA table_info(captured_positions)")}:', 'if False:'),
 ("utc_date flag wrong", SH, 'rec["flags"]["utc_date"] = date ', 'rec["flags"]["utc_date"] = "x" '),
 ("trades_opened_today flag never resets", SH, 'rec["flags"]["trades_opened_today"] = rp.entries_by_date.get(date, 0)', 'rec["flags"]["trades_opened_today"] = sum(rp.entries_by_date.values())'),
 ("event_net_pnl flag dropped", SH, 'rec["flags"]["event_net_pnl"] = float(net_pnl)', 'pass'),
 ("naive --until accepted", SH, '    if d.tzinfo is None:\n        raise ValueError("--until needs', '    if False:\n        raise ValueError("--until needs'),
 ("default deadline changed", SH, 'DEFAULT_UNTIL = "2026-10-19T23:59:59Z"', 'DEFAULT_UNTIL = "2026-10-19T23:59:58Z"'),
 ("max-events zero accepted", SH, 'if args.max_events is not None and args.max_events <= 0:', 'if args.max_events is not None and args.max_events < 0:'),
 ("live window start ignored", RP, 'if t is None or t < start or (end is not None and t > end):', 'if t is None or (end is not None and t > end):'),
 ("live window end ignored", RP, 'if t is None or t < start or (end is not None and t > end):', 'if t is None or t < start:'),
 ("stop source not required", RP, 'c["source"] == LIVE_STOP_SOURCE and c["direction"]', 'True and c["direction"]'),
 ("stop first-seen time not required", RP, 'seen is not None and seen >= start', 'seen is not None'),
 ("stop validity not required", RP, '\n                            and valid_protective_stop("LONG", r["price"], c["stop_price"])):', '):'),
 ("later stop_loss exit counted as stop captured", RP, 'c = cap.get(f"pp_{r[\'signal_id\']}")', 'c = cap.get(f"pp_{r[\'signal_id\']}") or dict(source=LIVE_STOP_SOURCE, direction="LONG", stop_price=r["price"] * 0.98, first_seen_at=r["event_time"])'),
 ("unknown-position exits counted", RP, 'elif r["action"] in ("closed", "reduced"):', 'else:'),
 ("reversal ignores the signal link", RP, 'and y["signal_id"] == x["signal_id"]', ''),
 ("reversal ignores the symbol", RP, 'y["symbol"] == x["symbol"] and', 'True and'),
 ("reversal allows the same side", RP, 'and y["side"] in ("LONG", "SHORT") and x["side"] in ("LONG", "SHORT") and y["side"] != x["side"]):', 'and y["side"] in ("LONG", "SHORT") and x["side"] in ("LONG", "SHORT")):'),
 ("coverage sorts entries before exits", RP, '0 if r["kind"] == "EXIT" else 1, r["id"]))', '1 if r["kind"] == "EXIT" else 0, r["id"]))'),
 ("rollover ignores carried state", RP, 'if (carried and f["flags"]', 'if (True and f["flags"]'),
 ("rollover ignores non-zero P&L at day start", RP, 'abs(pnl) <= 1e-9 and trades == 0.0', 'True and trades == 0.0'),
 ("rollover ignores missing day-start equity", RP, '\n                    and "day_start_equity" not in f["data_gaps"]):', '):'),
 ("deadline status never shown", RP, 'elif end is not None and now >= end:', 'elif False:'),
 ("coverage section missing from the report", RP, '    lines += render_coverage(rep)\n', ''),
 # report mutants
 ("report counts rejected as approved", RP, 'ok = [r for r in ev if r["approved"] == 1]', 'ok = [r for r in ev if r["approved"] == 0]'),
 ("report hides adapter errors", RP, 'add("3. adapter errors", "REVIEW" if n_err or d["not_evaluated"] else "OK",', 'add("3. adapter errors", "OK",'),
 ("report swaps rule/gap counts", RP, 'rejected_rule_only=cls.get("rule", 0), rejected_data_gap_only=cls.get("adapter_data_gap", 0)', 'rejected_rule_only=cls.get("adapter_data_gap", 0), rejected_data_gap_only=cls.get("rule", 0)'),
 ("report ignores intent mismatch", RP, 'bad = bool(i["entry_orders_whose_side_differs_from_signal"] or', 'bad = bool('),
 ("report reasons not split", RP, 'for atom in str(reason).split("|"):', 'for atom in [str(reason)]:'),
 # why-untested distinction (no signal reached the gate vs signal rejected by gate vs condition not met)
 ("untested: short case related to long entries", RP, 'return [r for r in live if r["kind"] == "ENTRY" and r["side"] == "SHORT"]', 'return [r for r in live if r["kind"] == "ENTRY" and r["side"] == "LONG"]'),
 ("untested: rejected-only never reported", RP, 'elif not approved:', 'elif False:'),
 ("untested: no-event never reported", RP, 'elif not rel:', 'elif False:'),
 ("untested: gate_rejected counts approved", RP, 'rejected = [r for r in rel if r["approved"] == 0]', 'rejected = [r for r in rel if r["approved"] == 1]'),
 ("untested: rollover related with one date", RP, 'if len({r["t"].date() for r in live}) >= 2', 'if len({r["t"].date() for r in live}) >= 1'),
 ("untested: reason set on observed cases", RP, 'untested_reason = None', 'untested_reason = "NO_EVENT_REACHED_GATE"'),
 ("untested: exit related ignores the exit reason", RP, 'return [r for r in live if r["kind"] == "EXIT" and r["exit_reason"] == reason]', 'return [r for r in live if r["kind"] == "EXIT"]'),
 ("untested: why text missing from report", RP, 'why = _REASON_TEXT[c["untested_reason"]]', 'why = ""'),
 ("untested: condition-not-met mislabelled", RP, 'untested_reason = "EVENT_SEEN_CONDITION_NOT_MET"', 'untested_reason = "SIGNAL_REJECTED_BY_GATE"'),
]


# Behaviourally equivalent BY DESIGN (redundant defence in depth); the mutant removing every layer is in the main list.
#  * query_only: mode=ro already makes SQLite refuse writes; "BOTH read-only protections dropped" must be killed.
#  * report intent check: the side-differs list and reconciliation.direction_mismatch carry the same information.
EQUIVALENT = {"query_only dropped (alone)", "report ignores intent mismatch"}


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
            shutil.copytree(HERE, pkg, ignore=shutil.ignore_patterns("__pycache__", "*.sqlite3", "mutation_harness*.py"))
            open(os.path.join(pkg, fname), "w").write(src.replace(old, new))
            try:
                rc = subprocess.run([sys.executable, "-m", "unittest", "trading_gate.test_shadow"], cwd=tmp,
                                    capture_output=True, text=True, timeout=120).returncode
            except subprocess.TimeoutExpired:
                rc = -1                                             # a mutant that makes the suite hang is detected too
        if rc != 0:
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
