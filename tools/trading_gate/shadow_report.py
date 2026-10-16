"""
trading_gate.shadow_report -- approve/reject summary and rejection-rate report for the shadow run.

Reads ONLY the shadow's own database (never the bot's). Writes shadow_report.json and shadow_report.md next to it.
    python3 -m trading_gate.shadow_report --out runtime/trading_gate_shadow

The six review questions asked before any blocking mode:
  1 percentage of real orders approved          2 most common rejection reasons       3 adapter errors
  4 strategy intent vs proposed orders          5 exposure / account reconciliation   6 duplicate and partial-fill handling
The report states numbers and which checks need human review. It does not decide whether blocking mode may start.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 2) if d else None


# ------------------------------------------------------------------------------------------ required coverage
# Every timestamp here is UTC. A case counts only if it was seen in the LIVE window of the campaign:
#     campaign_start (first pass of the campaign, stored in the shadow's `meta` table)  <=  event_time  <=  campaign_until.
# Events that were already in the bot's database before the campaign started (historical replay) never count.
# Fixtures and replays are NOT live evidence; they are listed separately as the way to exercise an untested path.

LIVE_STOP_SOURCE = "live_positions_table"

REQUIRED_CASES = (
    ("long_entry_stop_captured", "Long entry whose valid protective stop was observed by the shadow while the position was open"),
    ("short_entry", "Short entry"),
    ("exit_stop_loss", "Exit with exit_reason stop_loss"),
    ("exit_take_profit", "Exit with exit_reason take_profit"),
    ("exit_signal_reversal", "Exit with exit_reason signal_reversal"),
    ("exit_window_exit", "Exit with exit_reason window_exit"),
    ("same_symbol_reversal", "Same-symbol reversal: a closing exit and, from the same signal, an opposite-direction entry on the same symbol"),
    ("daily_rollover_utc", "Daily-limit rollover at midnight UTC: daily state that was non-zero on one UTC date is zero at the first decision of the next"),
)
CASE_IDS = tuple(c for c, _ in REQUIRED_CASES)

# deterministic fixture tests that exercise each path (test_shadow.CoverageTests); fixtures are not live evidence
FIXTURE_TESTS = {
    "long_entry_stop_captured": "CoverageTests.test_long_entry_stop_captured_counts_only_with_live_valid_stop",
    "short_entry": "CoverageTests.test_each_exit_reason_and_short_entry_are_counted",
    "exit_stop_loss": "CoverageTests.test_each_exit_reason_and_short_entry_are_counted",
    "exit_take_profit": "CoverageTests.test_each_exit_reason_and_short_entry_are_counted",
    "exit_signal_reversal": "CoverageTests.test_each_exit_reason_and_short_entry_are_counted",
    "exit_window_exit": "CoverageTests.test_each_exit_reason_and_short_entry_are_counted",
    "same_symbol_reversal": "CoverageTests.test_same_symbol_reversal_needs_the_same_signal_and_opposite_side",
    "daily_rollover_utc": "CoverageTests.test_daily_rollover_needs_nonzero_state_then_a_zero_state_next_utc_date",
}


_REASON_LABEL = {
    None: "-",
    "NO_EVENT_REACHED_GATE": "no signal reached the gate",
    "SIGNAL_REJECTED_BY_GATE": "signal rejected by gate",
    "EVENT_SEEN_CONDITION_NOT_MET": "event seen, case condition not met",
}
_REASON_TEXT = {
    "NO_EVENT_REACHED_GATE": ("No signal reached the gate for this case in the live window: the shadow saw no related paper order or exit, "
                              "so there was nothing to judge. The model produced no tradable signal or the bot did not act on one. "
                              "This is missing eligible trades, not a gate failure."),
    "SIGNAL_REJECTED_BY_GATE": "A signal was produced and reached the gate, and the gate rejected every related event.",
    "EVENT_SEEN_CONDITION_NOT_MET": ("Related events reached the gate and at least one was approved, but the case's own condition "
                                     "was not satisfied (see the definitions below)."),
}


def valid_protective_stop(direction, entry_price, stop) -> bool:
    """A stop is a valid protective stop if it is a finite positive number strictly on the losing side of the entry price."""
    def ok(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0
    if not (ok(entry_price) and ok(stop)):
        return False
    if direction == "LONG":
        return float(stop) < float(entry_price)
    if direction == "SHORT":
        return float(stop) > float(entry_price)
    return False


def _to_utc(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(timezone.utc) if d.tzinfo is not None else None


def _iso(d: datetime | None) -> str | None:
    return d.astimezone(timezone.utc).isoformat(timespec="seconds") if d is not None else None


def _query(db: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    cur = db.execute(sql, args)
    names = [c[0] for c in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def _cols(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _num_flag(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) else None


def compute_coverage(db: sqlite3.Connection, now: datetime | None = None, until: datetime | None = None) -> dict:
    """Which of the required cases occurred in the live window, from the shadow's own database only (pure; writes nothing)."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    meta = {}
    if _cols(db, "meta"):
        meta = {k: v for k, v in db.execute("SELECT key, value FROM meta")}
    start = _to_utc(meta.get("campaign_start"))
    end = until.astimezone(timezone.utc) if until is not None else _to_utc(meta.get("campaign_until"))
    found: dict[str, list[dict]] = {c: [] for c in CASE_IDS}

    live: list[dict] = []
    if start is not None:
        for r in _query(db, "SELECT id, event_time, kind, action, order_id, symbol, side, price, signal_id, exit_reason, "
                            "day_realized_pnl, data_gaps, flags, approved, reasons FROM decisions WHERE approved IS NOT NULL ORDER BY id"):
            t = _to_utc(r["event_time"])
            if t is None or t < start or (end is not None and t > end):
                continue
            r["t"] = t
            try:
                r["reasons"] = json.loads(r["reasons"]) if r["reasons"] else []
            except ValueError:
                r["reasons"] = []
            try:
                r["flags"] = json.loads(r["flags"]) if r["flags"] else {}
            except ValueError:
                r["flags"] = {}
            try:
                r["data_gaps"] = json.loads(r["data_gaps"]) if r["data_gaps"] else []
            except ValueError:
                r["data_gaps"] = []
            live.append(r)
        live.sort(key=lambda r: (r["t"], 0 if r["kind"] == "EXIT" else 1, r["id"]))

        cap: dict[str, dict] = {}
        have = _cols(db, "captured_positions")
        if have:
            src = "source" if "source" in have else "NULL AS source"        # unknown source never counts as live
            cap = {r["position_id"]: r for r in _query(
                db, f"SELECT position_id, direction, stop_price, first_seen_at, {src} FROM captured_positions")}

        def ev(r: dict, **extra) -> dict:
            return dict(order_id=r["order_id"], symbol=r["symbol"], at=_iso(r["t"]), **extra)

        for r in live:
            if r["kind"] == "ENTRY":
                if r["side"] == "SHORT":
                    found["short_entry"].append(ev(r))
                if r["side"] == "LONG":
                    c = cap.get(f"pp_{r['signal_id']}")
                    seen = _to_utc(c["first_seen_at"]) if c else None
                    if (c and c["source"] == LIVE_STOP_SOURCE and c["direction"] == "LONG" and seen is not None and seen >= start
                            and valid_protective_stop("LONG", r["price"], c["stop_price"])):
                        found["long_entry_stop_captured"].append(ev(r, stop_price=c["stop_price"], stop_first_seen_at=_iso(seen)))
            elif r["action"] in ("closed", "reduced"):
                key = {"stop_loss": "exit_stop_loss", "take_profit": "exit_take_profit",
                       "signal_reversal": "exit_signal_reversal", "window_exit": "exit_window_exit"}.get(r["exit_reason"])
                if key:
                    found[key].append(ev(r))

        # same-symbol reversal: the closing signal of a closed position is the signal that opened the opposite entry
        for i, x in enumerate(live):
            if x["kind"] != "EXIT" or x["action"] != "closed" or not x["signal_id"]:
                continue
            for y in live[i + 1:]:
                if (y["kind"] == "ENTRY" and y["symbol"] == x["symbol"] and y["signal_id"] == x["signal_id"]
                        and y["side"] in ("LONG", "SHORT") and x["side"] in ("LONG", "SHORT") and y["side"] != x["side"]):
                    found["same_symbol_reversal"].append(dict(order_id=y["order_id"], symbol=y["symbol"], at=_iso(y["t"]),
                                                              closed_side=x["side"], opened_side=y["side"], exit_order_id=x["order_id"]))
                    break

        # daily rollover: state carried on one UTC date, zero at the first decision of the next live UTC date
        by_date: dict[str, list[dict]] = {}
        for r in live:
            by_date.setdefault(r["t"].date().isoformat(), []).append(r)
        dates = sorted(by_date)
        for d1, d2 in zip(dates, dates[1:]):
            carried = any(r["kind"] == "ENTRY" for r in by_date[d1]) or any(
                (_num_flag(r["flags"].get("event_net_pnl")) or 0.0) != 0.0 for r in by_date[d1] if r["kind"] == "EXIT")
            f = by_date[d2][0]
            pnl, trades = _num_flag(f["day_realized_pnl"]), _num_flag(f["flags"].get("trades_opened_today"))
            if (carried and f["flags"].get("utc_date") == d2 and pnl is not None and abs(pnl) <= 1e-9 and trades == 0.0
                    and "day_start_equity" not in f["data_gaps"]):
                found["daily_rollover_utc"].append(dict(order_id=f["order_id"], symbol=f["symbol"], at=_iso(f["t"]),
                                                        from_utc_date=d1, to_utc_date=d2))

    def related(cid: str) -> list[dict]:
        """Live decisions that could have exercised this case: whatever reached the gate for it, approved or rejected."""
        if cid == "short_entry":
            return [r for r in live if r["kind"] == "ENTRY" and r["side"] == "SHORT"]
        if cid == "long_entry_stop_captured":
            return [r for r in live if r["kind"] == "ENTRY" and r["side"] == "LONG"]
        if cid.startswith("exit_"):
            reason = cid[len("exit_"):]
            return [r for r in live if r["kind"] == "EXIT" and r["exit_reason"] == reason]
        if cid == "same_symbol_reversal":
            return [r for r in live if r["kind"] == "EXIT" and r["action"] == "closed"]
        if cid == "daily_rollover_utc":
            return list(live) if len({r["t"].date() for r in live}) >= 2 else []
        return []

    cases = []
    for cid, desc in REQUIRED_CASES:
        hits = found[cid]
        rel = related(cid)
        rejected = [r for r in rel if r["approved"] == 0]
        approved = [r for r in rel if r["approved"] == 1]
        top = Counter(atom for r in rejected for reason in r["reasons"] for atom in str(reason).split("|")).most_common(3)
        outcomes = dict(related_events=len(rel), gate_approved=len(approved), gate_rejected=len(rejected),
                        top_rejection_reasons=[dict(reason=k, count=v) for k, v in top])
        if hits:
            untested_reason = None
        elif not rel:
            untested_reason = "NO_EVENT_REACHED_GATE"          # nothing arrived for the gate to judge: missing eligible trades
        elif not approved:
            untested_reason = "SIGNAL_REJECTED_BY_GATE"        # events arrived and the gate rejected every one of them
        else:
            untested_reason = "EVENT_SEEN_CONDITION_NOT_MET"   # events arrived but the case's own condition was not satisfied
        cases.append(dict(id=cid, description=desc, status="OBSERVED" if hits else "UNTESTED", count=len(hits),
                          first_at=hits[0]["at"] if hits else None, evidence=hits[:3],
                          fixture_test=FIXTURE_TESTS[cid], gate_outcomes=outcomes, untested_reason=untested_reason))
    observed = [c["id"] for c in cases if c["status"] == "OBSERVED"]
    untested = [c["id"] for c in cases if c["status"] == "UNTESTED"]
    complete = start is not None and not untested
    if start is None:
        status = "NOT_STARTED"
    elif complete:
        status = "COMPLETE"
    elif end is not None and now >= end:
        status = "DEADLINE_REACHED"
    else:
        status = "IN_PROGRESS"
    return dict(campaign_start=_iso(start), campaign_until=_iso(end), as_of=_iso(now), status=status, complete=complete,
                n_required=len(cases), n_observed=len(observed), observed=observed, untested=untested, cases=cases,
                live_decisions=len(live))


def build_report(out_dir: str, now: datetime | None = None) -> dict:
    path = os.path.join(out_dir, "shadow_decisions.sqlite3")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM decisions ORDER BY id")]
    for r in rows:
        for k in ("reasons", "checks", "warnings", "adapter_errors", "data_gaps", "flags"):
            r[k] = json.loads(r[k]) if r[k] else ([] if k != "checks" and k != "flags" else {})
    entries = [r for r in rows if r["kind"] == "ENTRY"]
    exits = [r for r in rows if r["kind"] == "EXIT"]
    evaluated = [r for r in rows if r["approved"] is not None]
    not_evaluated = [r for r in rows if r["approved"] is None]

    def split(rs: list[dict]) -> dict:
        ev = [r for r in rs if r["approved"] is not None]
        ok = [r for r in ev if r["approved"] == 1]
        no = [r for r in ev if r["approved"] == 0]
        cls = Counter(r["rejection_class"] for r in no)
        return dict(total=len(rs), evaluated=len(ev), approved=len(ok), rejected=len(no),
                    approval_rate_pct=_pct(len(ok), len(ev)), rejection_rate_pct=_pct(len(no), len(ev)),
                    rejected_rule_only=cls.get("rule", 0), rejected_data_gap_only=cls.get("adapter_data_gap", 0),
                    rejected_mixed=cls.get("mixed", 0),
                    rule_involved_rejection_rate_pct=_pct(cls.get("rule", 0) + cls.get("mixed", 0), len(ev)))

    reasons = Counter()
    reason_class: dict[str, str] = {}
    for r in evaluated:
        if r["approved"] == 0:
            for reason in r["reasons"]:
                for atom in str(reason).split("|"):
                    reasons[atom] += 1
                    reason_class[atom] = r["rejection_class"] or ""
    errors = Counter()
    for r in rows:
        for e in r["adapter_errors"]:
            errors[e] += 1
    flags = Counter()
    for r in rows:
        for k, v in r["flags"].items():
            if v is True:
                flags[k] += 1
    # equity: bot-recorded history vs replay realised-only (point-in-time)
    diffs = []
    for r in rows:
        f = r["flags"]
        if f.get("equity_source") == "equity_history" and r["equity"] and f.get("equity_replay") is not None:
            diffs.append(abs(r["equity"] - f["equity_replay"]) / r["equity"] * 100.0)
    recon_latest: dict = {}
    for kind in ("positions_vs_replay", "account_vs_replay", "intent_vs_orders", "fills", "schema"):
        row = db.execute("SELECT ok, details, run_id, at FROM reconciliations WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        if row:
            recon_latest[kind] = dict(ok=bool(row["ok"]), run_id=row["run_id"], at=row["at"], details=json.loads(row["details"]))
    runs = [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY started_at")]
    dup_rejections = [r["order_id"] for r in rows if any("duplicate_order" in x for x in r["reasons"])]
    reduced = [r["order_id"] for r in exits if r["action"] == "reduced"]
    unknown_exits = [r["order_id"] for r in exits if r["action"] == "unknown"]
    intent_rows = [r["order_id"] for r in entries if r["intent_mismatch"]]
    intent = recon_latest.get("intent_vs_orders", {}).get("details", {})
    n_sig = intent.get("n_signals")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    meta = {k: v for k, v in db.execute("SELECT key, value FROM meta")} if _cols(db, "meta") else {}
    rep = dict(
        generated_at=now.isoformat(timespec="seconds"),
        coverage=compute_coverage(db, now=now),
        campaign=dict(start=meta.get("campaign_start"), until=meta.get("campaign_until"), last_pass_at=meta.get("last_pass_at"),
                      passes=int(meta["passes"]) if str(meta.get("passes", "")).isdigit() else None,
                      stop_reason=meta.get("stop_reason"), stopped_at=meta.get("stopped_at"), timeframe=meta.get("timeframe")),
        decisions=dict(total=len(rows), entries=split(entries), exits=split(exits), overall=split(rows),
                       not_evaluated=len(not_evaluated), borrow_check_simulated=sum(1 for r in rows if r["borrow_check"] == "simulated")),
        top_rejection_reasons=[dict(reason=k, count=v, class_of_rejections_it_appears_in=reason_class.get(k)) for k, v in reasons.most_common(15)],
        adapter_errors=dict(events_with_errors=sum(1 for r in rows if r["adapter_errors"]), by_error=dict(errors.most_common()),
                            events_not_evaluated=[dict(order_id=r["order_id"], errors=r["adapter_errors"]) for r in not_evaluated]),
        assumption_flags=dict(flags),
        intent_vs_orders=dict(entry_orders_whose_side_differs_from_signal=intent_rows, reconciliation=intent),
        exposure_and_account=dict(
            positions_vs_replay=recon_latest.get("positions_vs_replay"), account_vs_replay=recon_latest.get("account_vs_replay"),
            equity_history_vs_replay_realized_only_pct=dict(n=len(diffs), mean=round(sum(diffs) / len(diffs), 4) if diffs else None,
                                                             max=round(max(diffs), 4) if diffs else None),
            max_gross_exposure_pct_seen=max([r["gross_exposure_pct"] for r in rows if r["gross_exposure_pct"] is not None] or [None]) if rows else None),
        duplicates_and_partials=dict(gate_duplicate_rejections=dup_rejections, duplicate_signal_orders=intent.get("duplicate_signal_orders"),
                                     partial_exits_reduced=reduced, exits_without_known_position=unknown_exits,
                                     fills_reconciliation=recon_latest.get("fills")),
        runs=dict(count=len(runs), last=runs[-1] if runs else None),
    )
    rep["review_checklist"] = _checklist(rep, rows)
    return rep


def _checklist(rep: dict, rows: list[dict]) -> list[dict]:
    d, ex = rep["decisions"], rep["exposure_and_account"]
    out = []

    def add(item: str, status: str, detail: str) -> None:
        out.append(dict(item=item, status=status, detail=detail))

    e = d["entries"]
    add("1. share of real orders approved", "INFO",
        f"entries approved {e['approved']}/{e['evaluated']} ({e['approval_rate_pct']}%); exits approved "
        f"{d['exits']['approved']}/{d['exits']['evaluated']} ({d['exits']['approval_rate_pct']}%)")
    top = rep["top_rejection_reasons"][:3]
    add("2. most common rejection reasons", "INFO", ", ".join(f"{t['reason']} x{t['count']}" for t in top) or "none")
    n_err = rep["adapter_errors"]["events_with_errors"]
    add("3. adapter errors", "REVIEW" if n_err or d["not_evaluated"] else "OK",
        f"{n_err} events carried adapter errors; {d['not_evaluated']} events could not be evaluated")
    i = rep["intent_vs_orders"]
    rc = i["reconciliation"] or {}
    bad = bool(i["entry_orders_whose_side_differs_from_signal"] or rc.get("accepted_without_order")
               or rc.get("orders_without_accepted_signal") or rc.get("direction_mismatch"))
    add("4. strategy intent vs proposed orders", "REVIEW" if bad else "OK",
        f"side differs from signal: {len(i['entry_orders_whose_side_differs_from_signal'])}; accepted signals without an order: "
        f"{len(i['reconciliation'].get('accepted_without_order', []))}; orders without an accepted signal: "
        f"{len(i['reconciliation'].get('orders_without_accepted_signal', []))}")
    pv, av = ex["positions_vs_replay"], ex["account_vs_replay"]
    ok5 = bool(pv and pv["ok"] and av and av["ok"])
    add("5. exposure/account match the paper account", "OK" if ok5 else "REVIEW",
        f"positions_vs_replay ok={pv['ok'] if pv else None}; account_vs_replay ok={av['ok'] if av else None}; "
        f"equity_history vs replay mean diff {ex['equity_history_vs_replay_realized_only_pct']['mean']}%")
    dp = rep["duplicates_and_partials"]
    fills = dp["fills_reconciliation"]
    ok6 = bool(fills and fills["ok"]) and not dp["duplicate_signal_orders"] and not dp["exits_without_known_position"]
    add("6. duplicate and partial-fill handling", "OK" if ok6 else "REVIEW",
        f"gate duplicate rejections {len(dp['gate_duplicate_rejections'])}; duplicate signal orders {len(dp['duplicate_signal_orders'] or [])}; "
        f"partial exits {len(dp['partial_exits_reduced'])}; fill alerts {len(fills['details']['alerts']) if fills else 'n/a'}")
    return out


def render_coverage(rep: dict) -> list[str]:
    cov, camp = rep["coverage"], rep["campaign"]
    lines = ["## Coverage of required cases (all times UTC)", "",
             f"Status: **{cov['status']}** - {cov['n_observed']} of {cov['n_required']} required cases observed in the live window. "
             f"Window: {cov['campaign_start']} to {cov['campaign_until'] or 'deadline not set (loop not started)'}. Last pass: {camp['last_pass_at']} ({camp['passes']} passes). "
             f"Stop reason: {camp['stop_reason'] or 'none recorded (still running or killed)'}.", "",
             "| case | status | live events | first seen (UTC) | events that reached the gate (approved / rejected) | why untested |",
             "|---|---|---|---|---|---|"]
    for c in cov["cases"]:
        g = c["gate_outcomes"]
        lines.append(f"| `{c['id']}` | {c['status']} | {c['count']} | {c['first_at'] or '-'} | "
                     f"{g['related_events']} ({g['gate_approved']} / {g['gate_rejected']}) | {_REASON_LABEL.get(c['untested_reason'], '-')} |")
    lines += ["", f"### Observed ({len(cov['observed'])})", ""]
    for c in cov["cases"]:
        if c["status"] == "OBSERVED":
            ev = "; ".join(f"{e['order_id']} {e['symbol']} {e['at']}" for e in c["evidence"])
            g = c["gate_outcomes"]
            lines.append(f"- `{c['id']}`: {c['description']}. Evidence (first {len(c['evidence'])}): {ev}. "
                         f"Gate outcome on related events: {g['gate_approved']} approved, {g['gate_rejected']} rejected.")
    if not cov["observed"]:
        lines.append("- none")
    lines += ["", f"### Still untested ({len(cov['untested'])})", ""]
    for c in cov["cases"]:
        if c["status"] == "UNTESTED":
            g = c["gate_outcomes"]
            why = _REASON_TEXT[c["untested_reason"]]
            if c["untested_reason"] != "NO_EVENT_REACHED_GATE":
                tops = ", ".join(f"{t['reason']} x{t['count']}" for t in g["top_rejection_reasons"]) or "none"
                why += f" Related events: {g['related_events']} ({g['gate_approved']} approved, {g['gate_rejected']} rejected); top rejection reasons: {tops}."
            lines.append(f"- `{c['id']}`: {c['description']}. {why} Exercise this path with the deterministic "
                         f"fixture test `{c['fixture_test']}` (fixture results are not live evidence).")
    if not cov["untested"]:
        lines.append("- none")
    lines += ["", "Definitions: live window = campaign start (first pass of this campaign) through the deadline; events already in the "
              "bot's database before the campaign started never count. A long entry counts only if the shadow saw a finite, positive "
              "stop on the losing side of the entry price in the bot's `positions` table while the position was open (a later "
              "stop_loss exit is not enough). Reversal = closing exit and an opposite-direction entry on the same symbol from the "
              "same signal id. Rollover = a UTC date with entries or realised P&L followed by a first decision on a later UTC date "
              "with daily realised P&L 0, trades opened today 0 and a daily_risk start equity present.", ""]
    return lines


def render_markdown(rep: dict) -> str:
    d = rep["decisions"]
    lines = ["# Shadow-mode gate report", "", f"Generated {rep['generated_at']} (UTC). Shadow mode only: nothing was blocked.", ""]
    lines += render_coverage(rep)
    lines += ["## Approve / reject", "","| | evaluated | approved | rejected | approval % | rejection % | rule-only | data-gap-only | mixed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in ("entries", "exits", "overall"):
        s = d[name]
        lines.append(f"| {name} | {s['evaluated']} | {s['approved']} | {s['rejected']} | {s['approval_rate_pct']} | "
                     f"{s['rejection_rate_pct']} | {s['rejected_rule_only']} | {s['rejected_data_gap_only']} | {s['rejected_mixed']} |")
    lines += ["", f"Events not evaluated: {d['not_evaluated']}. Shorts logged with borrow_check=simulated: {d['borrow_check_simulated']}.", "",
              "## Most common rejection reasons", ""]
    lines += [f"- `{r['reason']}` x{r['count']}" for r in rep["top_rejection_reasons"]] or ["- none"]
    lines += ["", "## Adapter errors", ""]
    lines += [f"- `{k}` x{v}" for k, v in rep["adapter_errors"]["by_error"].items()] or ["- none"]
    lines += ["", "## Review checklist", ""]
    lines += [f"- **{c['item']}** [{c['status']}]: {c['detail']}" for c in rep["review_checklist"]]
    lines += ["", "## Assumption flags (count of decisions)", ""]
    lines += [f"- `{k}`: {v}" for k, v in rep["assumption_flags"].items()] or ["- none"]
    return "\n".join(lines) + "\n"


def write_report(out_dir: str, now: datetime | None = None) -> tuple[dict, str]:
    """Build the report and write shadow_report.json / shadow_report.md next to the shadow database."""
    rep = build_report(out_dir, now=now)
    with open(os.path.join(out_dir, "shadow_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    md = render_markdown(rep)
    with open(os.path.join(out_dir, "shadow_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    return rep, md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the shadow output folder")
    args = ap.parse_args(argv)
    _, md = write_report(args.out)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
