#!/usr/bin/env python3
"""PREMISE TEST for long-only U.S. stock momentum, on survivorship-free data.

Question asked (and ONLY this): after publication and in the recent period,
has simply holding the top-momentum decile of U.S. stocks, long-only, beaten
the market on a risk-adjusted basis by enough to survive a plausible
implementation-cost allowance? This does NOT test the proposed 70/30
momentum+quality score, the $2B/$5/$20M screens, the 30-stock equal-weight
portfolio, the sector caps, the rank-60 buffer, or any lookback other than
12-2. Passing here would only say the premise is not dead; it would not
validate that strategy. Failing here would be bad news for it.

DATA: Kenneth French's Data Library (free, built from CRSP, includes delisted
stocks, no survivorship bias):
  - 10_Portfolios_Prior_12_2     (value- and equal-weighted monthly returns of
    10 portfolios sorted on prior 12-2 month return: skip the latest month,
    i.e. the standard "12-1" momentum definition)
  - F-F_Research_Data_Factors    (Mkt-RF and RF monthly)
Raw zips are cached in runtime/french_data/ with md5s printed, so a re-run
uses identical inputs. Returns are TOTAL returns, GROSS of trading costs.

PREDECLARED (before any result was seen):
  Strategy under test: Hi PRIOR decile, VALUE-weighted (primary; closer to
    something implementable than the micro-cap-heavy equal-weighted version).
    Equal-weighted Hi PRIOR is reported alongside for information, no verdict.
  Benchmark (primary): the U.S. market, total return = Mkt-RF + RF.
    (60/40 cannot be built from these files; the market is the fair bar for a
    100%-equity strategy anyway.) Also shown: the average of the 10 deciles
    as a rough equal-weight-universe proxy (approximate, information only).
  Sub-periods: pre-2012 (start -> 2011-12) and post-2012 (2012-01 -> end),
    each on its own. For information I also show 1927-1992 vs 1993-2011 vs
    2012-end (1993 = publication of the original momentum paper); that split
    carries no verdict.
  Cost allowance (crude, predeclared): a flat 2.0%/yr haircut subtracted from
    the strategy only, because the files give no turnover. Haircuts of 0/1/3%
    are shown for sensitivity; the verdict uses 2%. The break-even haircut
    (where strategy CAGR equals market CAGR) is also reported.
  PREMISE PASSES only if, in BOTH sub-periods:
     (i)   strategy Calmar (CAGR / |max drawdown|, monthly data) > market Calmar
     (ii)  strategy CAGR after the 2% haircut > market CAGR
     (iii) strategy beats the market on more than 50% of rolling 60-month
           windows lying fully inside the sub-period (gross, no haircut)
  Otherwise FAIL. No parameter is tuned after seeing results.
  June 2026 requirement: the data are monthly, so the May 15-Jul 1 window
    cannot be isolated; May 2026 and June 2026 monthly returns are printed
    separately as the closest available view (if the data reach that far --
    French's files lag by a month or two).

KNOWN LIMITS (disclosed): French's portfolios include small/micro caps and
apply no price, liquidity or size screens; breakpoints use NYSE stocks;
drawdowns are measured on month-end data (understating intramonth lows);
costs are the crude haircut above; sector concentration, lookback
sensitivity (9-12 months) and the quality score are untestable here.

USAGE (javier env; network needed only on first run to fetch the zips):
    cd /Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera
    python tools/run_french_momentum_premise.py
If a download fails, fetch the two zips from the French library page by hand
and put them in runtime/french_data/ with the file names printed below.
Writes runtime/french_momentum_premise_results.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "tools") \
    else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "runtime" / "french_data"
OUT_PATH = PROJECT_ROOT / "runtime" / "french_momentum_premise_results.json"

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "mom": "10_Portfolios_Prior_12_2_CSV.zip",
    "ff": "F-F_Research_Data_Factors_CSV.zip",
}

SPLIT = pd.Period("2012-01", "M")
HAIRCUT_VERDICT = 0.02
HAIRCUT_GRID = [0.0, 0.01, 0.02, 0.03]
ROLL = 60


# ---------------- download + parse ----------------

def fetch(name: str) -> bytes:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / FILES[name]
    if not path.exists():
        url = BASE + FILES[name]
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                path.write_bytes(resp.read())
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"download failed ({exc}). Fetch {url} manually and save it as {path}")
    raw = path.read_bytes()
    print(f"  {FILES[name]}: {len(raw)} bytes, md5 {hashlib.md5(raw).hexdigest()}")
    return raw


def zip_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        member = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        return zf.read(member).decode("latin-1")


def parse_sections(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Return [(title, header_cols, rows)] for every block containing monthly
    (YYYYMM) data rows. Annual (YYYY) rows are ignored."""
    sections = []
    title, header, rows = "", None, []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if rows:
                sections.append((title, header, rows))
                rows = []
                header = None
            continue
        first = s.split(",")[0].strip()
        if re.fullmatch(r"\d{6}", first):
            rows.append([c.strip() for c in s.split(",")])
        elif re.fullmatch(r"\d{4}", first):
            continue
        elif s.startswith(","):
            header = [c.strip() for c in s.split(",")][1:]
        else:
            title = s
    if rows:
        sections.append((title, header, rows))
    return sections


def to_frame(header, rows) -> pd.DataFrame:
    idx = pd.PeriodIndex([pd.Period(f"{r[0][:4]}-{r[0][4:]}", "M") for r in rows])
    vals = np.array([[float(x) if x not in ("", "-99.99", "-999") else np.nan for x in r[1:]] for r in rows])
    vals[vals <= -99] = np.nan
    return pd.DataFrame(vals / 100.0, index=idx, columns=header[: vals.shape[1]])


def load_data():
    print("=== loading French data ===")
    mom_text = zip_text(fetch("mom"))
    ff_text = zip_text(fetch("ff"))
    mom_secs = parse_sections(mom_text)
    vw = ew = None
    for title, header, rows in mom_secs:
        t = title.lower()
        if "value weighted" in t and "monthly" in t and vw is None:
            vw = to_frame(header, rows)
        elif "equal weighted" in t and "monthly" in t and ew is None:
            ew = to_frame(header, rows)
    if vw is None or ew is None:
        raise SystemExit(f"could not find monthly VW/EW sections; titles seen: {[s[0] for s in mom_secs]}")
    assert vw.shape[1] == 10 and ew.shape[1] == 10, f"expected 10 deciles, got {vw.shape[1]}/{ew.shape[1]}"
    ff_secs = parse_sections(ff_text)
    ff = to_frame(ff_secs[0][1], ff_secs[0][2])
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("MKT")
    rf = ff["RF"].rename("RF")
    print(f"  decile columns: {list(vw.columns)}")
    print(f"  momentum data: {vw.index.min()} -> {vw.index.max()} ({len(vw)} months)")
    print(f"  factor data  : {ff.index.min()} -> {ff.index.max()}")
    return vw, ew, mkt, rf


# ---------------- metrics ----------------

def cagr_of(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (12.0 / len(r)) - 1) if len(r) else float("nan")


def metrics(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return dict(months=len(r), cagr=float("nan"), vol=float("nan"), sharpe=float("nan"),
                    max_dd=float("nan"), calmar=float("nan"))
    eq = (1 + r).cumprod()
    cagr = cagr_of(r)
    vol = float(r.std() * np.sqrt(12))
    ex = r - rf.reindex(r.index)
    sharpe = float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    calmar = float(cagr / abs(dd)) if dd < 0 else float("nan")
    return dict(months=int(len(r)), cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd, calmar=calmar)


def fmt(m: dict) -> str:
    return (f"months={m['months']:4d}  cagr={m['cagr']*100:6.2f}%  vol={m['vol']*100:5.2f}%  "
            f"sharpe={m['sharpe']:5.2f}  maxDD={m['max_dd']*100:6.1f}%  calmar={m['calmar']:5.2f}")


def rolling_cagr(r: pd.Series, window: int = ROLL) -> pd.Series:
    lg = np.log1p(r.dropna())
    return np.expm1(lg.rolling(window).sum() * (12.0 / window)).dropna()


def breakeven_haircut(strat: pd.Series, mkt: pd.Series) -> float:
    target = cagr_of(mkt)
    lo, hi = -0.10, 0.50
    for _ in range(60):
        mid = (lo + hi) / 2
        if cagr_of(strat - mid / 12.0) > target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def slice_period(s: pd.Series, a, b) -> pd.Series:
    return s[(s.index >= a) & (s.index <= b)]


def evaluate(vw: pd.DataFrame, ew: pd.DataFrame, mkt: pd.Series, rf: pd.Series, write: bool = True) -> dict:
    common = vw.index.intersection(mkt.index).intersection(ew.index)
    vw, ew, mkt, rf = vw.loc[common], ew.loc[common], mkt.loc[common], rf.loc[common]
    hi_vw, hi_ew = vw.iloc[:, -1], ew.iloc[:, -1]
    lo_vw = vw.iloc[:, 0]
    allavg = vw.mean(axis=1)
    start, end = common.min(), common.max()
    print(f"\nanalysis window: {start} -> {end} ({len(common)} months)")

    periods = {
        "FULL": (start, end),
        "PRE-2012": (start, SPLIT - 1),
        "POST-2012": (SPLIT, end),
    }
    info_periods = {
        "1927-1992": (start, pd.Period("1992-12", "M")),
        "1993-2011": (pd.Period("1993-01", "M"), pd.Period("2011-12", "M")),
        "2012-end": (SPLIT, end),
    }
    series = {"TOP-DECILE VW": hi_vw, "TOP-DECILE EW": hi_ew, "MARKET": mkt,
              "ALL-DECILE AVG (approx EW universe)": allavg, "BOTTOM-DECILE VW (reference)": lo_vw}
    out = dict(window=[str(start), str(end)], periods={}, info_periods={}, verdict={})

    for label, (a, b) in {**periods}.items():
        print(f"\n=== {label} ({a} -> {b}) ===")
        rfp = slice_period(rf, a, b)
        out["periods"][label] = {}
        for name, s in series.items():
            m = metrics(slice_period(s, a, b), rfp)
            out["periods"][label][name] = m
            print(f"  {name:36s} {fmt(m)}")
        for h in HAIRCUT_GRID:
            m = metrics(slice_period(hi_vw, a, b) - h / 12.0, rfp)
            out["periods"][label][f"TOP-DECILE VW minus {h*100:.0f}%/yr"] = m
            print(f"  {'TOP-DECILE VW minus %d%%/yr' % round(h*100):36s} {fmt(m)}")
        be = breakeven_haircut(slice_period(hi_vw, a, b), slice_period(mkt, a, b))
        out["periods"][label]["breakeven_haircut_vs_market"] = be
        print(f"  break-even annual cost haircut vs market (VW top decile): {be*100:.2f}%/yr")

    print("\n=== DECILE CAGR BY PERIOD (value-weighted, gross; shows whether the momentum ordering exists) ===")
    print(f"  {'period':12s}" + "".join(f"{'D'+str(i+1):>8s}" for i in range(10)))
    for label, (a, b) in {**periods}.items():
        row = [cagr_of(slice_period(vw.iloc[:, i], a, b)) * 100 for i in range(10)]
        print(f"  {label:12s}" + "".join(f"{x:8.1f}" for x in row))
        out["periods"][label]["decile_cagr_vw_pct"] = row

    print("\n=== INFORMATION ONLY: three-way split (no verdict attached) ===")
    for label, (a, b) in info_periods.items():
        rfp = slice_period(rf, a, b)
        mt = metrics(slice_period(hi_vw, a, b), rfp)
        mm = metrics(slice_period(mkt, a, b), rfp)
        out["info_periods"][label] = dict(top_vw=mt, market=mm)
        print(f"  {label:10s} TOP-VW {fmt(mt)}")
        print(f"  {'':10s} MARKET {fmt(mm)}")

    print(f"\n=== ROLLING {ROLL}-MONTH CAGR: top-decile VW (gross) vs market, windows fully inside each period ===")
    roll_win = {}
    for label, (a, b) in periods.items():
        rs = rolling_cagr(slice_period(hi_vw, a, b)); rm = rolling_cagr(slice_period(mkt, a, b))
        both = rs.index.intersection(rm.index)
        if len(both) == 0:
            roll_win[label] = (float("nan"), 0)
            continue
        win = float((rs.loc[both] > rm.loc[both]).mean())
        med = float((rs.loc[both] - rm.loc[both]).median())
        roll_win[label] = (win, len(both))
        out["periods"][label]["rolling60"] = dict(windows=int(len(both)), strat_beats_market_frac=win,
                                                   median_cagr_diff=med,
                                                   strat_negative_frac=float((rs.loc[both] < 0).mean()),
                                                   market_negative_frac=float((rm.loc[both] < 0).mean()))
        print(f"  {label:10s} windows={len(both):4d}  strat>market in {win*100:5.1f}%  median CAGR diff {med*100:+.2f} pts  "
              f"strat<0 in {(rs.loc[both] < 0).mean()*100:.1f}%  market<0 in {(rm.loc[both] < 0).mean()*100:.1f}%")

    print("\n=== CALENDAR YEARS: top-decile VW vs market (last 20 full years, then worst/best 5 for the strategy) ===")
    yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1  # noqa: E731
    ys, ym = yr(hi_vw), yr(mkt)
    counts = hi_vw.groupby(hi_vw.index.year).size()
    full_years = [y for y in ys.index if counts[y] == 12]
    for y in full_years[-20:]:
        print(f"  {y}: strategy {ys[y]*100:7.1f}%   market {ym[y]*100:7.1f}%   diff {(ys[y]-ym[y])*100:+6.1f}")
    srt = ys.loc[full_years].sort_values()
    print("  worst 5 years for the strategy: " + ", ".join(f"{y} ({v*100:.0f}% vs mkt {ym[y]*100:.0f}%)" for y, v in srt.head(5).items()))
    print("  best 5 years for the strategy : " + ", ".join(f"{y} ({v*100:.0f}% vs mkt {ym[y]*100:.0f}%)" for y, v in srt.tail(5).items()))
    out["calendar_years"] = {int(y): dict(strategy=float(ys[y]), market=float(ym[y])) for y in ys.index}

    print("\n=== LATEST MONTHS (closest available view of the June 2026 window; monthly resolution only) ===")
    for p in list(common[-6:]):
        print(f"  {p}: top-VW {hi_vw[p]*100:6.2f}%   bottom-VW {lo_vw[p]*100:6.2f}%   market {mkt[p]*100:6.2f}%")
    for p in [pd.Period("2026-05", "M"), pd.Period("2026-06", "M")]:
        if p in common:
            out.setdefault("june_2026_proxy", {})[str(p)] = dict(top_vw=float(hi_vw[p]), market=float(mkt[p]))
    if "june_2026_proxy" not in out:
        print(f"  May/June 2026 are not in the data yet (last month available: {end}).")

    print("\n=== VERDICT (predeclared: all three conditions, in BOTH sub-periods; 2%/yr haircut) ===")
    parts = {}
    for label in ["PRE-2012", "POST-2012"]:
        P = out["periods"][label]
        c1 = P["TOP-DECILE VW"]["calmar"] > P["MARKET"]["calmar"]
        c2 = P["TOP-DECILE VW minus 2%/yr"]["cagr"] > P["MARKET"]["cagr"]
        win, nwin = roll_win[label]
        c3 = bool(np.isfinite(win) and win > 0.5)
        parts[label] = dict(calmar_beats_market=bool(c1), cagr_beats_market_after_2pct=bool(c2),
                            rolling60_win_over_50pct=c3, passed=bool(c1 and c2 and c3))
        print(f"  {label}: (i) Calmar {P['TOP-DECILE VW']['calmar']:.2f} vs mkt {P['MARKET']['calmar']:.2f} -> {c1};  "
              f"(ii) CAGR after 2% haircut {P['TOP-DECILE VW minus 2%/yr']['cagr']*100:.2f}% vs mkt {P['MARKET']['cagr']*100:.2f}% -> {c2};  "
              f"(iii) rolling-60m win rate {win*100:.1f}% -> {c3}")
    passed = all(v["passed"] for v in parts.values())
    out["verdict"] = dict(per_period=parts, passed=passed)
    print(f"\n  OVERALL PREMISE VERDICT: {'PASS' if passed else 'FAIL'}  "
          f"(this tests the premise only -- not the 70/30 score, the screens, or the top-30 implementation)")

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {OUT_PATH}")
    return out


def main():
    vw, ew, mkt, rf = load_data()
    evaluate(vw, ew, mkt, rf)


if __name__ == "__main__":
    main()
