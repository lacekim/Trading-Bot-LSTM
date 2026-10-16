"""Generates a static HTML dashboard from the cross-asset tracker's snapshot
CSV. Run cross_asset_tracker.py first to produce a fresh snapshot, then this
to render it -- kept as a separate step so the fetch (slow, network-bound)
and the render (instant) don't have to happen together every time."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_snapshot.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_dashboard.html"

ASSET_CLASS_ORDER = ["Equities", "Commodities", "Bonds/Rates", "Forex", "Crypto"]


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    records = [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    counts = df["trend"].value_counts().to_dict()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(records)) \
        .replace("__ASSET_CLASSES__", json.dumps(ASSET_CLASS_ORDER)) \
        .replace("__GENERATED_AT__", generated_at) \
        .replace("__BULL__", str(int(counts.get("BULL", 0)))) \
        .replace("__BEAR__", str(int(counts.get("BEAR", 0)))) \
        .replace("__TRANS__", str(int(counts.get("TRANSITION", 0)))) \
        .replace("__TOTAL__", str(len(df)))

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


HTML_TEMPLATE = r"""<title>Cross-Asset Regime Board</title>
<style>
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

  :root {
    --bg: #15171f; --panel: #1c1f2b; --panel-alt: #21242f; --border: #2c3040;
    --text: #d7dae0; --muted: #8890a0; --muted-2: #5c6373;
    --navbar: #0a9396; --navbar-dot: #7cffcb;
    --bull: #26a69a; --bull-bg: #26a69a1f; --bear: #ef5350; --bear-bg: #ef53501f;
    --transition-c: #7d8aa8; --transition-bg: #7d8aa81f;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; line-height: 1.5; }

  .navbar { background: var(--navbar); color: #fff; display: flex; align-items: center; gap: 20px; padding: 0 20px; height: 46px; }
  .navbar .brand { font-weight: 700; font-size: 15px; display: flex; align-items: center; gap: 8px; }
  .navbar .brand .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--navbar-dot); }
  .navbar .spacer { flex: 1; }
  .navbar .status { font-family: var(--mono); font-size: 12px; display: flex; align-items: center; gap: 8px; opacity: 0.95; }
  .navbar .status .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--navbar-dot); }

  .wrap { max-width: 1500px; margin: 0 auto; padding: 16px 16px 60px; }
  .subtitle { font-family: var(--mono); font-size: 11px; color: var(--muted-2); margin-bottom: 14px; }

  .stat-strip {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: var(--border); border: 1px solid var(--border);
    margin-bottom: 18px; border-radius: 6px; overflow: hidden;
  }
  .stat { background: var(--panel); padding: 12px 16px; }
  .stat .label { font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-bottom: 5px; }
  .stat .value { font-family: var(--mono); font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat.bull .value { color: var(--bull); }
  .stat.bear .value { color: var(--bear); }
  .stat.transition .value { color: var(--transition-c); }
  .stat.total .value { color: #4fd1d9; }

  .controls {
    position: sticky; top: 0; z-index: 5; display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
    background: var(--bg); padding: 10px 0 14px; margin-bottom: 8px; border-bottom: 1px solid var(--border);
  }
  .chip {
    font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.02em;
    padding: 5px 12px; border-radius: 4px; border: 1px solid var(--border);
    background: var(--panel); color: var(--muted); cursor: pointer; user-select: none;
  }
  .chip:hover { border-color: var(--muted-2); color: var(--text); }
  .chip.active { background: #0a939624; border-color: #4fd1d9; color: #4fd1d9; }

  #search {
    font-family: var(--mono); font-size: 11.5px; background: var(--panel); border: 1px solid var(--border);
    color: var(--text); padding: 6px 12px; border-radius: 4px; width: 180px; margin-left: auto;
  }
  #search::placeholder { color: var(--muted-2); }

  .jump { display: flex; gap: 6px; flex-wrap: wrap; }
  .jump a { font-family: var(--mono); font-size: 11px; color: var(--muted); text-decoration: none; padding: 5px 10px; border-radius: 4px; border: 1px solid transparent; }
  .jump a:hover { color: #4fd1d9; border-color: var(--border); }

  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; margin-top: 14px; }
  .panel h2 {
    margin: 0; padding: 10px 14px; font-size: 12px; font-weight: 700; font-family: var(--mono);
    color: var(--text); border-bottom: 1px solid var(--border); background: var(--panel-alt);
  }
  .panel h2 .count { color: var(--muted-2); font-weight: 400; margin-left: 6px; }

  .table-scroll { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; min-width: 760px; }
  thead th {
    text-align: right; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; font-size: 10px;
    color: var(--muted); padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer; white-space: nowrap;
  }
  thead th:first-child { text-align: left; }
  thead th:hover { color: var(--text); }
  thead th.sorted { color: #4fd1d9; }
  thead th .arrow { font-size: 9px; margin-left: 3px; }

  tbody td { text-align: right; padding: 7px 12px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; white-space: nowrap; }
  tbody td:first-child { text-align: left; font-weight: 700; color: var(--text); }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--panel-alt); }
  tbody tr.hidden { display: none; }

  .pill {
    display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px; border-radius: 3px;
    font-size: 11px; font-weight: 700; border: 1px solid;
  }
  .pill.bull { background: var(--bull-bg); color: var(--bull); border-color: var(--bull); }
  .pill.bear { background: var(--bear-bg); color: var(--bear); border-color: var(--bear); }
  .pill.transition { background: var(--transition-bg); color: var(--transition-c); border-color: var(--transition-c); }

  .ret-cell { border-radius: 3px; }

  footer.note { margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--muted-2); line-height: 1.7; }

  .empty-msg { padding: 24px; text-align: center; color: var(--muted-2); font-family: var(--mono); font-size: 12px; display: none; }
</style>

<div class="navbar">
  <div class="brand"><span class="dot"></span>Cross-Asset Regime Board</div>
  <div class="spacer"></div>
  <div class="status"><span class="dot"></span>116 instruments &middot; snapshot</div>
</div>

<div class="wrap">
  <div class="subtitle">generated __GENERATED_AT__ &middot; 50/200d MA trend state, Yahoo Finance daily bars &middot; standalone, not wired into any trading decision</div>

  <div class="stat-strip">
    <div class="stat total"><div class="label">Tracked</div><div class="value">__TOTAL__</div></div>
    <div class="stat bull"><div class="label">Bull</div><div class="value">__BULL__</div></div>
    <div class="stat bear"><div class="label">Bear</div><div class="value">__BEAR__</div></div>
    <div class="stat transition"><div class="label">Transition</div><div class="value">__TRANS__</div></div>
  </div>

  <div class="controls">
    <button class="chip active" data-filter="ALL">All</button>
    <button class="chip" data-filter="BULL">Bull</button>
    <button class="chip" data-filter="BEAR">Bear</button>
    <button class="chip" data-filter="TRANSITION">Transition</button>
    <div class="jump" id="jump"></div>
    <input id="search" type="text" placeholder="filter ticker&hellip;" autocomplete="off" />
  </div>

  <div id="sections"></div>
  <div class="empty-msg" id="empty">no instruments match the current filter</div>

  <footer class="note">
    Trend: BULL = 50MA &gt; 200MA and price &gt; 50MA &middot; BEAR = 50MA &lt; 200MA and price &lt; 50MA &middot; TRANSITION = mixed.<br />
    Return cells are heat-tinted by magnitude (green = positive, red = negative). Click a column header to sort.
  </footer>
</div>

<script>
  const DATA = __DATA__;
  const ASSET_CLASSES = __ASSET_CLASSES__;

  const COLS = [
    { key: "symbol", label: "Symbol", type: "text" },
    { key: "trend", label: "Trend", type: "trend" },
    { key: "price", label: "Price", type: "price" },
    { key: "dist_from_200ma_pct", label: "vs 200MA", type: "pct" },
    { key: "ret_1m_pct", label: "1M", type: "pct" },
    { key: "ret_3m_pct", label: "3M", type: "pct" },
    { key: "ret_6m_pct", label: "6M", type: "pct" },
    { key: "ret_1y_pct", label: "1Y", type: "pct" },
    { key: "pct_off_52w_high", label: "Off 52wHi", type: "pct" },
  ];

  const state = { filter: "ALL", search: "", sort: {} };

  function fmtPrice(v) {
    if (v === null || v === undefined) return "&mdash;";
    const abs = Math.abs(v);
    const dp = abs >= 100 ? 2 : abs >= 1 ? 3 : 5;
    return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }
  function fmtPct(v) {
    if (v === null || v === undefined) return "&mdash;";
    return (v >= 0 ? "+" : "") + v.toFixed(1) + "%";
  }
  function heat(v) {
    if (v === null || v === undefined) return "";
    const mag = Math.min(Math.abs(v) / 40, 1);
    const alpha = (mag * 0.32).toFixed(2);
    return v >= 0 ? `rgba(79,184,119,${alpha})` : `rgba(226,99,109,${alpha})`;
  }
  function pillClass(trend) {
    return trend === "BULL" ? "bull" : trend === "BEAR" ? "bear" : "transition";
  }
  function glyph(trend) {
    return trend === "BULL" ? "&#9650;" : trend === "BEAR" ? "&#9660;" : "&#8213;";
  }

  const sectionsEl = document.getElementById("sections");
  const jumpEl = document.getElementById("jump");

  ASSET_CLASSES.forEach(cls => {
    const rows = DATA.filter(d => d.asset_class === cls);
    if (!rows.length) return;

    const a = document.createElement("a");
    a.href = "#sec-" + cls.replace(/[^a-z0-9]/gi, "");
    a.textContent = cls;
    jumpEl.appendChild(a);

    const section = document.createElement("div");
    section.className = "panel";
    section.id = "sec-" + cls.replace(/[^a-z0-9]/gi, "");
    section.dataset.class = cls;

    const h2 = document.createElement("h2");
    h2.innerHTML = `${cls} <span class="count">(${rows.length})</span>`;
    section.appendChild(h2);

    const scroll = document.createElement("div");
    scroll.className = "table-scroll";
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    COLS.forEach(c => {
      const th = document.createElement("th");
      th.textContent = c.label;
      th.dataset.key = c.key;
      th.tabIndex = 0;
      th.addEventListener("click", () => sortBy(cls, c.key));
      th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sortBy(cls, c.key); } });
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    const tbody = document.createElement("tbody");
    table.appendChild(thead);
    table.appendChild(tbody);
    scroll.appendChild(table);
    section.appendChild(scroll);
    sectionsEl.appendChild(section);

    state.sort[cls] = { key: "ret_3m_pct", dir: -1 };
    renderTable(cls, rows);
  });

  function renderTable(cls, rows) {
    const section = document.querySelector(`.panel[data-class="${CSS.escape(cls)}"]`);
    const tbody = section.querySelector("tbody");
    const { key, dir } = state.sort[cls];

    const sorted = [...rows].sort((a, b) => {
      let av = a[key], bv = b[key];
      if (key === "symbol" || key === "trend") { av = av || ""; bv = bv || ""; return dir * av.localeCompare(bv); }
      av = av === null || av === undefined ? -Infinity : av;
      bv = bv === null || bv === undefined ? -Infinity : bv;
      return dir * (av - bv);
    });

    section.querySelectorAll("th").forEach(th => {
      th.classList.toggle("sorted", th.dataset.key === key);
      const existing = th.querySelector(".arrow");
      if (existing) existing.remove();
      if (th.dataset.key === key) {
        const arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = dir === 1 ? "▲" : "▼";
        th.appendChild(arrow);
      }
    });

    tbody.innerHTML = "";
    let visibleCount = 0;
    sorted.forEach(r => {
      const matchesFilter = state.filter === "ALL" || r.trend === state.filter;
      const matchesSearch = !state.search || r.symbol.toLowerCase().includes(state.search);
      const tr = document.createElement("tr");
      if (!matchesFilter || !matchesSearch) tr.classList.add("hidden");
      else visibleCount++;

      tr.innerHTML = `
        <td>${r.symbol}</td>
        <td><span class="pill ${pillClass(r.trend)}">${glyph(r.trend)} ${r.trend}</span></td>
        <td>${fmtPrice(r.price)}</td>
        <td>${fmtPct(r.dist_from_200ma_pct)}</td>
        <td class="ret-cell" style="background:${heat(r.ret_1m_pct)}">${fmtPct(r.ret_1m_pct)}</td>
        <td class="ret-cell" style="background:${heat(r.ret_3m_pct)}">${fmtPct(r.ret_3m_pct)}</td>
        <td class="ret-cell" style="background:${heat(r.ret_6m_pct)}">${fmtPct(r.ret_6m_pct)}</td>
        <td class="ret-cell" style="background:${heat(r.ret_1y_pct)}">${fmtPct(r.ret_1y_pct)}</td>
        <td>${fmtPct(r.pct_off_52w_high)}</td>
      `;
      tbody.appendChild(tr);
    });

    section.style.display = visibleCount === 0 ? "none" : "";
  }

  function sortBy(cls, key) {
    const cur = state.sort[cls];
    state.sort[cls] = cur.key === key ? { key, dir: -cur.dir } : { key, dir: key === "symbol" ? 1 : -1 };
    renderTable(cls, DATA.filter(d => d.asset_class === cls));
  }

  function renderAll() {
    ASSET_CLASSES.forEach(cls => {
      const rows = DATA.filter(d => d.asset_class === cls);
      if (rows.length) renderTable(cls, rows);
    });
    const anyVisible = Array.from(document.querySelectorAll("#sections .panel"))
      .some(s => s.style.display !== "none");
    document.getElementById("empty").style.display = anyVisible ? "none" : "block";
  }

  document.querySelectorAll(".chip[data-filter]").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip[data-filter]").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      state.filter = chip.dataset.filter;
      renderAll();
    });
  });

  document.getElementById("search").addEventListener("input", e => {
    state.search = e.target.value.trim().toLowerCase();
    renderAll();
  });
</script>
"""

if __name__ == "__main__":
    main()
