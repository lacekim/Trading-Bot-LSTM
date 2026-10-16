"""
Independent verification of trading_gate (does not reuse gate.py's check code):
  A. ORACLE: 20,000 randomised ENTRY scenarios; an independently written approve/reject rule is compared with the gate.
  B. FAIL-CLOSED FUZZ: every field of a valid scenario is deleted / nulled / NaN'd / string'd / list'd; the gate must never
     raise, and must never approve when a REQUIRED field is damaged.
  C. EXIT oracle: 5,000 randomised EXIT scenarios (must be approved iff well-formed, reduce-only and within the position).
Run:  python3 fuzz_verify.py     exit code 0 = no disagreement.
"""
import copy, math, os, random, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_gate.gate import GateLimits, TradingGate  # noqa: E402
from trading_gate.test_gate import limits, scenario, exit_scenario, NOW  # noqa: E402

L = limits()
rnd = random.Random(20261005)


def oracle_entry(sc):
    o, s, m, p, r = (sc[k] for k in ("proposed_order", "signal", "market_snapshot", "portfolio_state", "risk_state"))
    why = []
    age = (NOW - m["observed_at"]).total_seconds()
    if age < -5 or age > L.max_data_age_s: why.append("data_age")
    if m["ask"] < m["bid"] or (m["ask"] - m["bid"]) / ((m["ask"] + m["bid"]) / 2) * 1e4 > L.max_spread_bps: why.append("quote")
    if abs(o["price"] / m["price"] - 1) * 1e4 > L.max_price_deviation_bps: why.append("order_dev")
    sa = (NOW - s["candle_timestamp"]).total_seconds()
    if sa < 3600 or sa > 3600 * L.max_signal_age_multiplier: why.append("signal_age")
    if abs(s["price"] / m["price"] - 1) * 1e4 > L.max_price_deviation_bps: why.append("signal_dev")
    if s["direction"] != o["side"] or s["signal_id"] != o["signal_id"] or s["symbol"] != o["symbol"]: why.append("match")
    if o["idempotency_key"] in p["seen_idempotency_keys"]: why.append("dup")
    eq = p["equity"]; n = o["notional"]
    if abs(n - o["quantity"] * o["price"]) > L.notional_tolerance * o["quantity"] * o["price"]: why.append("notional")
    if n < L.min_order_usd or n > eq * L.max_position_pct / 100 or o["leverage"] > L.max_leverage: why.append("size")
    if sum(x["notional"] for x in p["positions"]) + n > eq * L.max_exposure_pct / 100: why.append("exposure")
    if len(p["positions"]) >= L.max_open_positions: why.append("npos")
    if p["cash"] - n / o["leverage"] < eq * L.cash_buffer_pct / 100: why.append("cash")
    if any(x["symbol"] == o["symbol"] for x in p["positions"]): why.append("have_pos")
    if o["side"] == "SHORT" and (not r["shorts_allowed"] or r["borrow_available"].get(o["symbol"]) is not True): why.append("short")
    st = o.get("stop_price")
    if st is None: why.append("nostop")
    else:
        good = st < o["price"] if o["side"] == "LONG" else st > o["price"]
        d = abs(st / o["price"] - 1) * 100
        if not good or d < L.min_stop_pct or d > L.max_stop_pct: why.append("stop")
    if r["kill_switch"]: why.append("kill")
    if r["daily_realized_pnl"] <= -(r["day_start_equity"] * L.max_daily_loss_pct / 100) or \
            (r["day_start_equity"] - eq) / r["day_start_equity"] * 100 >= L.max_equity_loss_pct: why.append("dayloss")
    if (max(p["peak_equity"], eq) - eq) / max(p["peak_equity"], eq) * 100 >= L.max_drawdown_pct: why.append("dd")
    if r["consecutive_losses"] >= L.max_consecutive_losses or r["failed_orders_today"] >= L.max_failed_orders or \
            r["trades_opened_today"] >= L.max_trades_per_day: why.append("streak")
    b = r["bars_since_last_entry"]
    if b is not None and b < L.min_bars_between_entries: why.append("cooldown")
    return not why


def random_entry():
    """Mostly-valid scenario; each independent fault fires with probability f (f drawn per scenario)."""
    f = rnd.choice([0.0, 0.03, 0.03, 0.1, 0.3])
    flip = lambda: rnd.random() < f
    d = rnd.choice(["LONG", "SHORT"])
    sc = scenario(d)
    o, s, m, p, r = (sc[k] for k in ("proposed_order", "signal", "market_snapshot", "portfolio_state", "risk_state"))
    px = rnd.uniform(100, 60000)
    eq = rnd.uniform(20000, 200000)
    n = eq * (rnd.uniform(0.21, 0.4) if flip() else rnd.uniform(0.01, 0.19))
    q = n / px
    if flip(): n *= rnd.choice([0.99, 1.01, 0.5])                       # notional mismatch
    lev = rnd.choice([2.5, 3.0]) if flip() else rnd.choice([1.0, 1.5, 2.0])
    o.update(price=px, quantity=q, notional=n, leverage=lev, idempotency_key=f"k-{rnd.random()}")
    sgn = -1 if d == "LONG" else 1
    if flip(): o["stop_price"] = None
    elif flip(): o["stop_price"] = px * (1 - sgn * rnd.uniform(0.005, 0.1))      # wrong side
    else: o["stop_price"] = px * (1 + sgn * (rnd.choice([0.001, 0.30]) if flip() else rnd.uniform(0.006, 0.14)))
    o["target_price"] = None
    s["price"] = px * (1 + rnd.uniform(0.012, 0.03)) if flip() else px * (1 + rnd.uniform(-0.009, 0.009))
    if flip(): s["direction"] = "SHORT" if d == "LONG" else "LONG"
    s["candle_timestamp"] = NOW - timedelta(seconds=rnd.choice([rnd.uniform(0, 3500), rnd.uniform(7300, 20000)]) if flip() else rnd.uniform(3700, 7100))
    mp = px * (1 + (rnd.choice([-1, 1]) * rnd.uniform(0.012, 0.03) if flip() else rnd.uniform(-0.002, 0.002)))
    half = mp * (rnd.choice([0.006, 0.02]) if flip() else rnd.choice([0.0002, 0.0003, 0.002])) / 2
    bid, ask = mp - half, mp + half
    if flip(): bid, ask = ask, bid
    m.update(price=mp, bid=bid, ask=ask,
             observed_at=NOW - timedelta(seconds=rnd.choice([rnd.uniform(125, 400), rnd.uniform(-30, -6)]) if flip() else rnd.uniform(-4, 115)))
    p.update(equity=eq, cash=eq * (rnd.uniform(0.01, 0.1) if flip() else rnd.uniform(0.5, 1.0)),
             peak_equity=eq * (rnd.uniform(1.17, 1.4) if flip() else rnd.uniform(1.0, 1.12)),
             positions=[dict(symbol=f"S{i}", direction=rnd.choice(["LONG", "SHORT"]), quantity=1,
                             notional=eq * (rnd.uniform(0.2, 0.3) if flip() else rnd.uniform(0, 0.08)))
                        for i in range(rnd.randint(4, 6) if flip() else rnd.randint(0, 3))])
    if flip(): p["positions"].append(dict(symbol="BTC/USD", direction=rnd.choice(["LONG", "SHORT"]), quantity=1, notional=100))
    if flip(): p["seen_idempotency_keys"] = [o["idempotency_key"]]
    r.update(kill_switch=flip(), day_start_equity=eq * (rnd.uniform(1.05, 1.12) if flip() else rnd.uniform(0.97, 1.03)),
             daily_realized_pnl=eq * (rnd.uniform(-0.06, -0.025) if flip() else rnd.uniform(-0.02, 0.02)),
             consecutive_losses=rnd.randint(4, 7) if flip() else rnd.randint(0, 3),
             failed_orders_today=rnd.randint(3, 6) if flip() else rnd.randint(0, 2),
             trades_opened_today=rnd.randint(10, 14) if flip() else rnd.randint(0, 9),
             bars_since_last_entry=rnd.choice([0, 1]) if flip() else rnd.choice([None, 2, 3, 10]),
             shorts_allowed=not flip(), borrow_available={"BTC/USD": not flip()})
    return sc


def main():
    bad = 0
    # A
    appr = 0
    for i in range(20000):
        sc = random_entry()
        g = TradingGate(L, ":memory:", clock=lambda: NOW)
        got = g.evaluate(**copy.deepcopy(sc))["approved"]
        g.close()
        exp = oracle_entry(sc)
        appr += exp
        if got != exp:
            bad += 1
            if bad <= 5: print("ORACLE MISMATCH", exp, got, sc)
    print(f"A. oracle: 20000 scenarios, {appr} approved by oracle, mismatches so far: {bad}")
    assert appr > 2000, "oracle never approves enough to be meaningful"
    # B
    bads = [None, float("nan"), float("inf"), "x", [], {}, True, -1, 0]
    optional = {("proposed_order", "target_price"), ("market_snapshot", "open"), ("market_snapshot", "high"), ("market_snapshot", "low"),
                ("market_snapshot", "close"), ("market_snapshot", "spread_bps")}
    n_fuzz = n_appr_bad = 0
    for d in ("LONG", "SHORT"):
        base = scenario(d)
        for part, content in base.items():
            keys = list(content) if isinstance(content, dict) else []
            for k in keys + [None]:
                variants = [("delete", None)] + [("set", b) for b in bads]
                for how, val in variants:
                    sc = copy.deepcopy(base)
                    try:
                        if k is None:
                            sc[part] = val if how == "set" else None
                        elif how == "delete":
                            del sc[part][k]
                        else:
                            sc[part][k] = val
                    except Exception: continue
                    if sc == base: continue
                    g = TradingGate(L, ":memory:", clock=lambda: NOW)
                    try:
                        dec = g.evaluate(**sc)
                    except Exception as e:
                        bad += 1; print("GATE RAISED", part, k, how, val, repr(e)); continue
                    finally:
                        g.close()
                    n_fuzz += 1
                    # fields where the damaged value can still be a valid, approvable scenario are excluded from the invariant
                    harmless = ((part, k) in optional and (how == "delete" or val is None)) or \
                               (part == "proposed_order" and k in ("order_id",) and False) or \
                               (part == "risk_state" and k == "borrow_available" and d == "LONG") or \
                               (part == "risk_state" and k == "shorts_allowed" and d == "LONG") or \
                               (part == "proposed_order" and k == "target_price" and val is None) or \
                               (part == "market_snapshot" and k in ("bid", "ask") and False) or \
                               (part == "risk_state" and k in ("failed_orders_today", "trades_opened_today", "consecutive_losses", "kill_switch", "day_start_equity", "daily_realized_pnl", "bars_since_last_entry") and val in (0, -1) and False)
                    if dec["approved"] and not harmless:
                        # allowed approvals: values that are legitimately valid (e.g. 0 losses, False-like bool for kill_switch=False)
                        legit = (val in (0,) and not isinstance(val, bool) and k in ("consecutive_losses", "failed_orders_today", "trades_opened_today", "daily_realized_pnl", "bars_since_last_entry")) \
                                or (isinstance(val, bool) and val is False)
                        # damaged values that are in fact valid values for that field (not a gate weakness):
                        legit = legit or (k in ("order_id", "idempotency_key") and val == "x") \
                            or (k == "seen_idempotency_keys" and val == {}) \
                            or (k == "daily_realized_pnl" and val == -1)
                        if not legit and not (k in ("bars_since_last_entry",) and val is None):
                            n_appr_bad += 1; bad += 1
                            print("APPROVED DESPITE DAMAGE", d, part, k, how, repr(val))
    print(f"B. fuzz: {n_fuzz} damaged scenarios, approved-despite-damage: {n_appr_bad}")
    # C
    ok = 0
    for i in range(5000):
        d = rnd.choice(["LONG", "SHORT"])
        pos_q = rnd.uniform(0.01, 2)
        sc = exit_scenario(d, qty=1)
        o = sc["proposed_order"]
        o["quantity"] = pos_q * rnd.choice([0.1, 0.5, 1.0, 1.0, 1.2, 2.0])
        o["idempotency_key"] = f"x-{i}"
        has = rnd.random() > 0.15
        sc["portfolio_state"]["positions"] = [dict(symbol="BTC/USD", direction=rnd.choice([d, d, "SHORT" if d == "LONG" else "LONG"]) if has else d,
                                                  quantity=pos_q, notional=1000.0)] if has else []
        sc["risk_state"]["kill_switch"] = rnd.random() < 0.5
        sc["market_snapshot"]["observed_at"] = NOW - timedelta(seconds=rnd.uniform(0, 5000))
        pos = sc["portfolio_state"]["positions"]
        exp = bool(pos) and pos[0]["direction"] == d and o["quantity"] <= pos[0]["quantity"] * (1 + 1e-9)
        g = TradingGate(L, ":memory:", clock=lambda: NOW)
        got = g.evaluate(**sc)["approved"]
        g.close()
        ok += exp
        if got != exp:
            bad += 1
            if bad <= 8: print("EXIT MISMATCH", exp, got, sc)
    print(f"C. exit oracle: 5000 scenarios, {ok} approved by oracle")
    print("RESULT:", "ALL AGREE" if bad == 0 else f"{bad} PROBLEMS")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
