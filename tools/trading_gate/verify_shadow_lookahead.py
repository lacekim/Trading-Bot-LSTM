"""
Look-ahead audit for the shadow adapter (truncation-invariance test).

For every event E in a paper-order database: make a COPY of the database, delete every row that was written AFTER E's
timestamp, re-run the shadow on the truncated copy, and require E's decision AND the exact gate inputs to be identical to
the full-history run. If any later information leaked into E's inputs, the two runs would differ.
The bot's own database is only read (copied through SQLite's backup API from a read-only connection).

    python3 -m trading_gate.verify_shadow_lookahead --paper-db data/v5_paper_trading.sqlite3 [--work /tmp/dir]
Exit code 0 = every event identical.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

try:
    from . import shadow as S
except ImportError:
    import shadow as S

COMPARE = ("approved", "decision", "reasons", "checks", "warnings", "adapter_errors", "data_gaps", "equity", "peak_equity",
           "day_realized_pnl", "n_positions", "gross_exposure_pct", "net_exposure_pct", "inputs_json", "action", "strategy")


def _copy_ro(src: str, dst: str) -> None:
    ro = S.open_readonly(src)
    out = sqlite3.connect(dst)
    try:
        ro.backup(out)
    finally:
        out.close()
        ro.close()


def _decisions(out_dir: str) -> dict:
    db = sqlite3.connect(os.path.join(out_dir, "shadow_decisions.sqlite3"))
    db.row_factory = sqlite3.Row
    res = {r["event_key"]: dict(r) for r in db.execute("SELECT * FROM decisions")}
    db.close()
    return res


def _truncate(path: str, t: datetime) -> None:
    iso, date = t.isoformat(timespec="seconds"), t.date().isoformat()
    db = sqlite3.connect(path)
    for sql, arg in (("DELETE FROM orders WHERE created_at > ?", iso), ("DELETE FROM closed_trades WHERE exit_time > ?", iso),
                     ("DELETE FROM signals WHERE created_at > ?", iso), ("DELETE FROM equity_history WHERE timestamp > ?", iso),
                     ("DELETE FROM positions WHERE entry_time > ?", iso), ("DELETE FROM daily_risk WHERE trading_date > ?", date)):
        db.execute(sql, (arg,))
    try:
        db.execute("DELETE FROM qualified_market_snapshots WHERE recorded_at > ?", (iso,))
    except sqlite3.OperationalError:
        pass
    db.commit()
    db.close()


def lookahead_audit(paper_db: str, work_dir: str, limits=None, **kw) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    full_db = os.path.join(work_dir, "full.sqlite3")
    _copy_ro(paper_db, full_db)
    full_out = os.path.join(work_dir, "full_out")
    obs = S.ShadowObserver(full_db, full_out, limits=limits, **kw)
    obs.run_once()
    obs.close()
    full = _decisions(full_out)
    mismatches, checked = [], 0
    for i, (key, rec) in enumerate(sorted(full.items(), key=lambda kv: (kv[1]["event_time"] or "", kv[1]["id"]))):
        if not rec["event_time"] or rec["approved"] is None and rec["decision"] == "NOT_EVALUATED":
            continue
        t = datetime.fromisoformat(rec["event_time"]).astimezone(timezone.utc)
        cut_db = os.path.join(work_dir, f"cut_{i}.sqlite3")
        _copy_ro(paper_db, cut_db)
        _truncate(cut_db, t)
        cut_out = os.path.join(work_dir, f"cut_{i}_out")
        o2 = S.ShadowObserver(cut_db, cut_out, limits=limits, **kw)
        o2.run_once()
        o2.close()
        cut = _decisions(cut_out).get(key)
        checked += 1
        if cut is None:
            mismatches.append(dict(event=key, field="<missing in truncated run>"))
            continue
        for f in COMPARE:
            if cut[f] != rec[f]:
                mismatches.append(dict(event=key, field=f, full=rec[f], truncated=cut[f]))
    return dict(events_checked=checked, mismatches=mismatches, ok=not mismatches)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-db", required=True)
    ap.add_argument("--work", default=None, help="scratch folder (default: a temp dir)")
    args = ap.parse_args(argv)
    work = args.work or tempfile.mkdtemp(prefix="shadow_lookahead_")
    res = lookahead_audit(args.paper_db, work)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
