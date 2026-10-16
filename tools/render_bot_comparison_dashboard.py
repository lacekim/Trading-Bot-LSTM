"""Static page comparing the two live dry-run bots side by side:
golden_cross_crypto (127.0.0.1:8080, unfiltered) vs.
golden_cross_crypto_regime_filtered (127.0.0.1:8081, regime-gated) --
the actual forward test of tools/run_intermarket_regime_filter_test.py's
backtested finding, on genuinely unseen future data.

Unlike the other dashboards in this project, this one needs no Python-side
data prep -- both bots' REST APIs are live, real, and already CORS-enabled
for this page's origin (see CORS_origins in both bots' config.json). All
data fetching happens client-side via polling, so the numbers are always
current, not a snapshot from whenever this was last rendered. This script
only needs to run once to produce the static HTML shell; re-running it is
only necessary if the page's own layout/styling changes.
"""
from __future__ import annotations

from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "bot_comparison_dashboard.html"

HTML = r"""<title>Golden Cross Forward Test</title>
<style>
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  :root {
    --bg: oklch(21.8% 0.008 223.9);
    --bg-elevated: oklch(27.5% 0.011 216.9);
    --border: oklch(27.5% 0.011 216.9);
    --text: oklch(92.5% 0.005 214.3);
    --text-muted: oklch(72.3% 0.014 214.4);
    --text-highlighted: #fff;
    --primary: #00c3e6;
    --positive: oklch(44.8% 0.119 151.328);
    --positive-text: #d1fae5;
    --negative: oklch(57% 0.245 27.33);
    --negative-text: #fee2e2;
    --sans: -apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", "Noto Sans", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; line-height: 1.5; }
  .topbar { background: var(--bg-elevated); border-bottom: 1px solid var(--border); padding: 10px 16px;
            display: flex; align-items: center; justify-content: space-between; }
  .topbar h1 { font-size: 15px; margin: 0; color: var(--text-highlighted); font-weight: 600; }
  .topbar .sub { color: var(--text-muted); font-size: 12px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .panel h3 { margin: 0; padding: 8px 12px; font-size: 12px; font-weight: 600; border-bottom: 1px solid var(--border);
              color: var(--text-highlighted); display: flex; align-items: center; justify-content: space-between; }
  .panel .body { padding: 10px 12px; }
  .full { grid-column: 1 / -1; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--text-muted); font-weight: 500; padding: 5px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 5px 8px; border-bottom: 1px solid color-mix(in oklch, var(--border) 50%, transparent); font-variant-numeric: tabular-nums; }
  .pill { display: inline-flex; align-items: center; gap: 3px; padding: 1px 7px; border-radius: 999px; font-weight: 600; font-size: 11px; }
  .pos { background: color-mix(in oklch, var(--positive) 22%, transparent); color: var(--positive-text); }
  .neg { background: color-mix(in oklch, var(--negative) 22%, transparent); color: var(--negative-text); }
  .na { background: color-mix(in oklch, var(--text-muted) 18%, transparent); color: var(--text-muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot.online { background: #34d399; }
  .dot.offline { background: #f87171; }
  .stat-row { display: flex; gap: 16px; margin-bottom: 8px; flex-wrap: wrap; }
  .stat { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; min-width: 110px; }
  .stat .label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }
  .stat .value { font-size: 18px; font-weight: 600; color: var(--text-highlighted); margin-top: 2px; }
  .empty-state { color: var(--text-muted); text-align: center; padding: 16px; }
  code { font-family: ui-monospace, monospace; background: var(--bg); padding: 1px 5px; border-radius: 4px; }
  .regime-line { display: flex; gap: 8px; align-items: center; padding: 3px 0; }
  .regime-line .k { color: var(--text-muted); min-width: 170px; }
</style>

<div class="topbar">
  <h1>Golden Cross &mdash; Forward Test</h1>
  <div class="sub" id="last-updated">loading&hellip;</div>
</div>

<div class="grid">
  <div class="panel">
    <h3><span><span class="dot" id="dot-a"></span> golden_cross_crypto <span style="color:var(--text-muted);font-weight:400;">(unfiltered, :8080)</span></span></h3>
    <div class="body">
      <div class="stat-row" id="stats-a"></div>
      <table><thead><tr><th>Pair</th><th>Amount</th><th>Open rate</th><th>Current rate</th><th>Profit %</th><th>Open date</th></tr></thead>
        <tbody id="trades-a"><tr><td colspan="6" class="empty-state">loading&hellip;</td></tr></tbody></table>
    </div>
  </div>

  <div class="panel">
    <h3><span><span class="dot" id="dot-b"></span> golden_cross_crypto_regime_filtered <span style="color:var(--text-muted);font-weight:400;">(gated, :8081)</span></span></h3>
    <div class="body">
      <div class="stat-row" id="stats-b"></div>
      <table><thead><tr><th>Pair</th><th>Amount</th><th>Open rate</th><th>Current rate</th><th>Profit %</th><th>Open date</th></tr></thead>
        <tbody id="trades-b"><tr><td colspan="6" class="empty-state">loading&hellip;</td></tr></tbody></table>
    </div>
  </div>

  <div class="panel full">
    <h3>Live regime state <span style="color:var(--text-muted);font-weight:400;">(what the gated bot is reading right now &mdash; runtime/regime_state.json)</span></h3>
    <div class="body" id="regime-body">loading&hellip;</div>
  </div>
</div>

<script>
  const BOTS = {
    a: { base: "http://127.0.0.1:8080", user: "freqtrade", pass: "tradeFreq" },
    b: { base: "http://127.0.0.1:8081", user: "freqtrade", pass: "tradeFreq" },
  };

  function authHeader(bot) { return "Basic " + btoa(`${bot.user}:${bot.pass}`); }

  function fmtAbs(v) {
    const a = Math.abs(v);
    const s = a >= 1e6 ? (a/1e6).toFixed(2)+"M" : a >= 1e3 ? (a/1e3).toFixed(1)+"k" : a.toFixed(2);
    return (v >= 0 ? "" : "-") + s;
  }
  function pill(pctValue, absValue) {
    if (pctValue === null || pctValue === undefined || isNaN(pctValue)) return `<span class="pill na">n/a</span>`;
    const cls = pctValue >= 0 ? "pos" : "neg";
    const arrow = pctValue >= 0 ? "&#9650;" : "&#9660;";
    const absPart = absValue === undefined ? "" : ` (${fmtAbs(absValue)})`;
    return `<span class="pill ${cls}">${arrow} ${(pctValue>=0?"+":"")+pctValue.toFixed(2)}%${absPart}</span>`;
  }

  async function fetchJson(bot, path) {
    const res = await fetch(bot.base + path, { headers: { Authorization: authHeader(bot) } });
    if (!res.ok) throw new Error(`${path} -> ${res.status}`);
    return res.json();
  }

  async function refreshBot(key) {
    const bot = BOTS[key];
    const dot = document.getElementById(`dot-${key}`);
    const statsEl = document.getElementById(`stats-${key}`);
    const tradesEl = document.getElementById(`trades-${key}`);
    try {
      const [status, profit] = await Promise.all([
        fetchJson(bot, "/api/v1/status"),
        fetchJson(bot, "/api/v1/profit"),
      ]);
      dot.className = "dot online";

      statsEl.innerHTML = `
        <div class="stat"><div class="label">Open trades</div><div class="value">${status.length}</div></div>
        <div class="stat"><div class="label">Closed trades</div><div class="value">${profit.closed_trade_count}</div></div>
        <div class="stat"><div class="label">Win rate</div><div class="value">${(profit.winrate*100).toFixed(0)}%</div></div>
        <div class="stat"><div class="label">Total profit</div><div class="value">${pill(profit.profit_all_percent)}</div></div>
      `;

      if (status.length === 0) {
        tradesEl.innerHTML = `<tr><td colspan="6" class="empty-state">Currently no open trades.</td></tr>`;
      } else {
        tradesEl.innerHTML = status.map(t => `
          <tr>
            <td>${t.pair}</td>
            <td>${t.amount < 1 ? t.amount.toFixed(4) : t.amount.toFixed(2)}</td>
            <td>${t.open_rate.toFixed(2)}</td>
            <td>${(t.current_rate ?? t.open_rate).toFixed(2)}</td>
            <td>${pill(t.profit_pct, t.profit_abs)}</td>
            <td>${t.open_date.slice(0,16).replace("T"," ")}</td>
          </tr>
        `).join("");
      }
    } catch (e) {
      dot.className = "dot offline";
      statsEl.innerHTML = `<div class="stat"><div class="label">Status</div><div class="value" style="color:var(--negative-text)">offline</div></div>`;
      tradesEl.innerHTML = `<tr><td colspan="6" class="empty-state">${e.message}</td></tr>`;
    }
  }

  async function refreshRegime() {
    const el = document.getElementById("regime-body");
    try {
      const res = await fetch("regime_state.json?_=" + Date.now());
      if (!res.ok) throw new Error("regime_state.json -> " + res.status);
      const s = await res.json();
      const ageMin = (Date.now() - new Date(s.computed_at).getTime()) / 60000;
      el.innerHTML = `
        <div class="regime-line"><span class="k">Risk-on (entries allowed)</span> ${s.risk_on ? '<span class="pill pos">&#9650; RISK_ON</span>' : '<span class="pill neg">&#9660; RISK_OFF / BLOCKED</span>'}</div>
        <div class="regime-line"><span class="k">Credit risk-on (HYG/IEF &gt; 200MA)</span> ${s.credit_risk_on}</div>
        <div class="regime-line"><span class="k">Dollar not a headwind (UUP &le; 200MA)</span> ${s.dollar_not_headwind}</div>
        <div class="regime-line"><span class="k">In FOMC event blackout</span> ${s.in_fomc_event_window}</div>
        <div class="regime-line"><span class="k">HYG/IEF ratio / 200MA</span> <code>${s.hyg_ief_ratio?.toFixed(4)}</code> / <code>${s.hyg_ief_ratio_ma200?.toFixed(4)}</code></div>
        <div class="regime-line"><span class="k">UUP close / 200MA</span> <code>${s.uup_close?.toFixed(2)}</code> / <code>${s.uup_ma200?.toFixed(2)}</code></div>
        <div class="regime-line"><span class="k">Cache age</span> ${ageMin.toFixed(0)} min ${ageMin > 360 ? '<span class="pill neg">STALE &mdash; bot will deny all entries</span>' : ''}</div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">${e.message}</div>`;
    }
  }

  async function refreshAll() {
    await Promise.all([refreshBot("a"), refreshBot("b"), refreshRegime()]);
    document.getElementById("last-updated").textContent = "last updated " + new Date().toLocaleTimeString();
  }

  refreshAll();
  setInterval(refreshAll, 30000);
</script>
"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(HTML, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
