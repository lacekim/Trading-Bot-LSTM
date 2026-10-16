"""Renders trade_tracer_data.json (see generate_trade_tracer.py) into a
local HTML dashboard -- overlaid intra-trade price paths per asset, plus a
sortable trade table. Served locally via serve_dashboard.py, same pattern
as the cross-asset dashboard."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "runtime" / "trade_tracer_data.json"
OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "trade_tracer_dashboard.html"


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    trades = data["trades"]
    assets = sorted(set(t["asset"] for t in trades))
    # keep the session's established asset ordering where possible
    order = ["BTC", "ETH", "XMR", "PAXG", "SOL", "TRX"]
    assets = [a for a in order if a in assets] + [a for a in assets if a not in order]

    total = len(trades)
    wins = sum(1 for t in trades if t["net_ret_pct"] > 0)

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(trades)) \
        .replace("__ASSETS__", json.dumps(assets)) \
        .replace("__GENERATED_AT__", data["generated_at"][:16].replace("T", " ") + " UTC") \
        .replace("__TOTAL__", str(total)) \
        .replace("__WINS__", str(wins)) \
        .replace("__WINRATE__", f"{wins/total*100:.0f}" if total else "0")

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


HTML_TEMPLATE = r"""<title>Trade Tracer</title>
<style>
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

  :root {
    --bg: #10131a;
    --surface: #171b24;
    --surface-alt: #1c212c;
    --border: #262c3a;
    --text: #e6e9f0;
    --muted: #8891a3;
    --muted-2: #5d6577;
    --accent: #d99a4e;
    --accent-dim: #d99a4e33;
    --bull: #4fb877;
    --bull-dim: #4fb87726;
    --bear: #e2636d;
    --bear-dim: #e2636d26;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.5; padding: 28px 20px 80px; }
  .wrap { max-width: 1360px; margin: 0 auto; }

  header.page { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 22px; padding-bottom: 18px; border-bottom: 1px solid var(--border); }
  header.page h1 { font-family: var(--mono); font-size: 20px; letter-spacing: 0.04em; margin: 0; font-weight: 600; }
  header.page h1 .dim { color: var(--muted-2); font-weight: 400; }
  header.page .meta { font-family: var(--mono); font-size: 12px; color: var(--muted); }

  .stat-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); margin-bottom: 26px; border-radius: 6px; overflow: hidden; }
  .stat { background: var(--surface); padding: 16px 20px; }
  .stat .label { font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 6px; }
  .stat .value { font-family: var(--mono); font-size: 28px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat.wins .value { color: var(--bull); }

  .jump { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 24px; }
  .jump a { font-family: var(--mono); font-size: 11px; color: var(--muted); text-decoration: none; padding: 5px 10px; border-radius: 4px; border: 1px solid var(--border); }
  .jump a:hover { color: var(--accent); border-color: var(--accent); }

  section.asset-block { margin-top: 34px; }
  section.asset-block h2 { font-family: var(--mono); font-size: 13px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); border-left: 3px solid var(--accent); padding-left: 10px; margin: 0 0 4px; }
  section.asset-block .sub { font-family: var(--mono); font-size: 11px; color: var(--muted); padding-left: 13px; margin: 0 0 12px; }

  .chart-wrap { border: 1px solid var(--border); border-radius: 6px; background: var(--surface); padding: 12px; margin-bottom: 12px; }
  canvas { display: block; width: 100%; height: 280px; }
  .chart-legend { display: flex; gap: 16px; font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 6px; padding: 0 4px; }
  .chart-legend span.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }

  .table-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12.5px; min-width: 760px; }
  thead th { text-align: right; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; font-size: 10.5px; color: var(--muted); background: var(--surface-alt); padding: 9px 12px; border-bottom: 1px solid var(--border); cursor: pointer; white-space: nowrap; }
  thead th:first-child, thead th:nth-child(2) { text-align: left; }
  thead th:hover { color: var(--text); }
  thead th.sorted { color: var(--accent); }
  thead th .arrow { font-size: 9px; margin-left: 3px; opacity: 0.7; }
  tbody td { text-align: right; padding: 7px 12px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; white-space: nowrap; }
  tbody td:first-child, tbody td:nth-child(2) { text-align: left; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--surface-alt); cursor: pointer; }
  tbody tr.win td:nth-child(4) { color: var(--bull); }
  tbody tr.loss td:nth-child(4) { color: var(--bear); }

  footer.note { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--muted-2); line-height: 1.7; }
</style>

<div class="wrap">
  <header class="page">
    <h1>TRADE TRACER <span class="dim">// golden cross, entry &rarr; exit</span></h1>
    <div class="meta">generated __GENERATED_AT__ &middot; real 4h Kraken history, fee+slip adjusted</div>
  </header>

  <div class="stat-strip">
    <div class="stat"><div class="label">Total trades</div><div class="value">__TOTAL__</div></div>
    <div class="stat wins"><div class="label">Wins</div><div class="value">__WINS__</div></div>
    <div class="stat"><div class="label">Win rate</div><div class="value">__WINRATE__%</div></div>
  </div>

  <div class="jump" id="jump"></div>
  <div id="sections"></div>

  <footer class="note">
    Each line is one real historical trade's price path from entry to exit, normalized to % return from entry
    and resampled across % of that trade's own duration (trades ranged 4 hours to ~6 months, so raw time
    wouldn't overlay comparably). Green = closed profitable after fees/slippage, red = closed at a loss.
    Click a table row to highlight its path on the chart above it. Click a column header to sort.
  </footer>
</div>

<script>
  const TRADES = __DATA__;
  const ASSETS = __ASSETS__;
  const BULL = "#4fb877", BEAR = "#e2636d", GRID = "#262c3a", TEXT_MUTED = "#5d6577";

  const COLS = [
    { key: "entry_time", label: "Entry", type: "text" },
    { key: "exit_time", label: "Exit", type: "text" },
    { key: "duration_days", label: "Days", type: "num" },
    { key: "net_ret_pct", label: "Net Return", type: "pct" },
    { key: "max_drawdown_pct", label: "Max DD", type: "pct" },
    { key: "max_runup_pct", label: "Max Run-up", type: "pct" },
  ];

  const sectionsEl = document.getElementById("sections");
  const jumpEl = document.getElementById("jump");
  const state = { sort: {}, highlighted: {} };

  function fmtDate(iso) { return iso.slice(0, 10); }
  function fmtPct(v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "%"; }

  ASSETS.forEach(asset => {
    const rows = TRADES.filter(t => t.asset === asset);
    if (!rows.length) return;
    const wins = rows.filter(t => t.net_ret_pct > 0).length;

    const a = document.createElement("a");
    a.href = "#asset-" + asset;
    a.textContent = `${asset} (${rows.length})`;
    jumpEl.appendChild(a);

    const section = document.createElement("section");
    section.className = "asset-block";
    section.id = "asset-" + asset;
    section.innerHTML = `
      <h2>${asset}</h2>
      <p class="sub">${rows.length} trades &middot; ${wins} wins (${(wins/rows.length*100).toFixed(0)}%)</p>
      <div class="chart-wrap">
        <canvas id="canvas-${asset}"></canvas>
        <div class="chart-legend">
          <span><span class="sw" style="background:${BULL}"></span>profitable close</span>
          <span><span class="sw" style="background:${BEAR}"></span>loss close</span>
          <span>x: % of trade duration elapsed &middot; y: % return from entry</span>
        </div>
      </div>
      <div class="table-scroll"><table><thead><tr></tr></thead><tbody></tbody></table></div>
    `;
    sectionsEl.appendChild(section);

    const trh = section.querySelector("thead tr");
    COLS.forEach(c => {
      const th = document.createElement("th");
      th.textContent = c.label;
      th.dataset.key = c.key;
      th.tabIndex = 0;
      th.addEventListener("click", () => sortBy(asset, c.key));
      th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sortBy(asset, c.key); } });
      trh.appendChild(th);
    });

    state.sort[asset] = { key: "entry_time", dir: 1 };
    renderAsset(asset, rows);
  });

  function drawChart(asset, rows) {
    const canvas = document.getElementById("canvas-" + asset);
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 900, cssH = 280;
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 46, padR = 12, padT = 10, padB = 20;
    const plotW = cssW - padL - padR, plotH = cssH - padT - padB;

    let yMin = 0, yMax = 0;
    rows.forEach(t => t.path.forEach(v => { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }));
    yMin -= 2; yMax += 2;
    const yRange = yMax - yMin || 1;

    const xToPx = pct => padL + (pct / 100) * plotW;
    const yToPx = val => padT + (1 - (val - yMin) / yRange) * plotH;

    // gridlines + labels
    ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.font = "10px " + getComputedStyle(document.body).getPropertyValue("--mono") || "monospace";
    ctx.fillStyle = TEXT_MUTED;
    const steps = 4;
    for (let i = 0; i <= steps; i++) {
      const val = yMin + (yRange * i / steps);
      const y = yToPx(val);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(cssW - padR, y); ctx.stroke();
      ctx.fillText(val.toFixed(0) + "%", 4, y + 3);
    }
    // zero line, emphasized
    if (yMin < 0 && yMax > 0) {
      ctx.strokeStyle = "#8891a355";
      ctx.beginPath(); ctx.moveTo(padL, yToPx(0)); ctx.lineTo(cssW - padR, yToPx(0)); ctx.stroke();
    }

    const highlightKey = state.highlighted[asset];
    const drawOne = (t, emphasize) => {
      const isWin = t.net_ret_pct > 0;
      ctx.strokeStyle = isWin ? BULL : BEAR;
      ctx.globalAlpha = emphasize ? 1 : (highlightKey ? 0.08 : 0.35);
      ctx.lineWidth = emphasize ? 2.4 : 1;
      ctx.beginPath();
      t.path.forEach((v, i) => {
        const x = xToPx((i / (t.path.length - 1)) * 100);
        const y = yToPx(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    rows.forEach(t => { if (rowKey(t) !== highlightKey) drawOne(t, false); });
    if (highlightKey) {
      const hit = rows.find(t => rowKey(t) === highlightKey);
      if (hit) drawOne(hit, true);
    }
    ctx.globalAlpha = 1;
  }

  function rowKey(t) { return t.entry_time + "|" + t.asset; }

  function renderAsset(asset, allRows) {
    const section = document.getElementById("asset-" + asset);
    const tbody = section.querySelector("tbody");
    const { key, dir } = state.sort[asset];
    const sorted = [...allRows].sort((a, b) => {
      let av = a[key], bv = b[key];
      if (typeof av === "string") return dir * av.localeCompare(bv);
      return dir * (av - bv);
    });

    section.querySelectorAll("th").forEach(th => {
      th.classList.toggle("sorted", th.dataset.key === key);
      const ex = th.querySelector(".arrow"); if (ex) ex.remove();
      if (th.dataset.key === key) {
        const arrow = document.createElement("span");
        arrow.className = "arrow"; arrow.textContent = dir === 1 ? "▲" : "▼";
        th.appendChild(arrow);
      }
    });

    tbody.innerHTML = "";
    sorted.forEach(t => {
      const tr = document.createElement("tr");
      tr.className = t.net_ret_pct > 0 ? "win" : "loss";
      tr.innerHTML = `
        <td>${fmtDate(t.entry_time)}</td>
        <td>${fmtDate(t.exit_time)}</td>
        <td>${t.duration_days.toFixed(1)}</td>
        <td>${fmtPct(t.net_ret_pct)}</td>
        <td>${fmtPct(t.max_drawdown_pct)}</td>
        <td>${fmtPct(t.max_runup_pct)}</td>
      `;
      tr.addEventListener("click", () => {
        const k = rowKey(t);
        state.highlighted[asset] = state.highlighted[asset] === k ? null : k;
        drawChart(asset, allRows);
      });
      tbody.appendChild(tr);
    });

    drawChart(asset, allRows);
  }

  function sortBy(asset, key) {
    const cur = state.sort[asset];
    state.sort[asset] = cur.key === key ? { key, dir: -cur.dir } : { key, dir: key.includes("time") ? 1 : -1 };
    renderAsset(asset, TRADES.filter(t => t.asset === asset));
  }

  window.addEventListener("resize", () => {
    ASSETS.forEach(asset => drawChart(asset, TRADES.filter(t => t.asset === asset)));
  });
</script>
"""

if __name__ == "__main__":
    main()
