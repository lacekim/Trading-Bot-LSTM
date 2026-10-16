"""Universe and concentration diagnostics (item 5, the final tractable item
of the post-exit-logic validation/deployment-readiness sequence), per the
user's own exact spec:

  "Test reasonable alternate universe definitions using the same locked
  strategy. Attribute winner P&L by symbol and sector. Keep top-winner
  exclusions strictly descriptive; they should not become a rationale for
  changing the strategy."

PART A -- UNIVERSE ROBUSTNESS. Four universe definitions, predeclared
before this script was run against real data, all built from the SAME
existing data source (runtime/cross_asset_golden_cross_results.json) that
every prior script's universe already comes from -- no new data source, no
survivorship-bias claim implied or resolved (see the module-level note at
the bottom of this docstring):

  1. CURRENT               -- the exact existing universe every prior
                               script has used: robust==True AND
                               asset_class != "Crypto". A DISCLOSED FINDING
                               from building this diagnostic: despite being
                               referred to throughout this project as "the
                               equities universe," CURRENT is actually 57
                               Equities + 2 Commodities (GC=F gold, PA=F
                               palladium) + 1 Bonds/Rates (IEF, a 7-10yr
                               Treasury ETF) = 60 symbols. This was already
                               true of every prior script in this whole
                               sequence -- it is being surfaced here, not
                               changed here.
  2. PURE_EQUITIES          -- robust==True AND asset_class=="Equities"
                               only (57 symbols): CURRENT with those same 3
                               non-equity members removed, nothing else
                               changed.
  3. ALL_EQUITIES_UNFILTERED -- asset_class=="Equities", robust flag IGNORED
                               (72 symbols): same asset class as #2, but the
                               upstream "robust" pre-screen from the
                               cross-asset golden-cross study is removed.
  4. ALL_NONCRYPTO_UNFILTERED -- asset_class != "Crypto", robust flag
                               IGNORED (102 symbols): the broadest,
                               uncurated non-crypto universe available in
                               this data source -- adds 17 commodity
                               futures, 8 FX pairs, and 4 more bonds/rates
                               instruments on top of #3.

Each of the four is run through the exact same locked strategy (unchanged
run_backtest_exit_variant: BASELINE golden-cross entry, EQUAL-DOLLAR
sizing, fixed -12% stop, death-cross exit, SAME_CLOSE/exact-stop-price
fills, 12bps/12bps cost) -- nothing about the strategy itself is retuned
for any universe.

PART B -- CONCENTRATION DIAGNOSTICS, run on CURRENT's own trade log (the
actual production universe, since that is the reference specification
every other item in this sequence measures against):
  - total realized P&L attributed by SYMBOL (ranked, top winners shown);
  - total realized P&L attributed by SECTOR/CATEGORY, using a static,
    predeclared classification (see SECTOR_MAP below) -- GICS-like sectors
    for the 57-72 individual equities, plus explicit "Broad Index ETF"
    (DIA/IWM/QQQ/SPY/VTI -- these track a whole index, not one business,
    and are kept out of any single equity sector), "Commodities",
    "Bonds/Rates", and "Forex" buckets for the non-equity members. This
    classification is this script's own, disclosed, non-authoritative
    labeling -- built for concentration-attribution purposes only;
  - the % of total POSITIVE realized P&L contributed by the top 3/5/10
    winning symbols;
  - a DESCRIPTIVE-ONLY re-run of the exact same locked strategy on
    CURRENT-minus-its-own-top-3-winning-symbols, to quantify (not to
    justify changing) how much of the control's performance rides on a
    small number of names. PER THE USER'S OWN INSTRUCTION, this exclusion
    is reported as a descriptive fact about concentration, and is NOT, and
    must never be read as, a recommendation to drop those symbols or
    otherwise alter the strategy.

WHAT THIS DOES NOT DO, DISCLOSED: none of these four universes adds
delisted securities, point-in-time constituents, or any data source other
than the one every prior script already uses. This item tests sensitivity
to which subset of the EXISTING data source's symbols is included -- it
does not touch, resolve, or imply resolution of the survivorship-bias
question, which stays exactly where the user left it: an explicit,
separate, still-pending data-source fork (current-source/survivorship-
biased vs. point-in-time-vendor), never to be conflated with this
universe-subset sensitivity check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tools.run_sizing_allocation_backtest_equities import (
    ENTRY_COST, EXIT_COST, STARTING_CAPITAL, BASELINE_STOP_PCT,
    ENTRY_SETS, build_states, make_sizing_fn,
)
from tools.run_pullback_entry_backtest_equities import fetch_ohlcv_batch
from tools.run_time_based_exit_backtest_equities import (
    run_backtest_exit_variant as run_control,
)

RESULTS_PATH = Path(__file__).resolve().parents[1] / "runtime" / "cross_asset_golden_cross_results.json"
OUT_PATH = Path(__file__).resolve().parents[1] / "runtime" / "universe_concentration_diagnostics_results.json"

# ---------------------------------------------------------------------------
# Predeclared static sector/category map. Non-authoritative, this script's
# own labeling, built solely to attribute concentration -- see docstring.
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "ADBE": "Technology", "AMD": "Technology", "AVGO": "Technology",
    "CRM": "Technology", "MSFT": "Technology", "NVDA": "Technology", "ORCL": "Technology",
    # Health Care
    "ABBV": "Health Care", "ABT": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "MRK": "Health Care", "PFE": "Health Care", "TMO": "Health Care", "UNH": "Health Care",
    # Financials
    "AXP": "Financials", "BAC": "Financials", "BRK-B": "Financials", "GS": "Financials",
    "JPM": "Financials", "MA": "Financials", "MS": "Financials", "V": "Financials", "WFC": "Financials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    # Consumer Staples
    "COST": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "PG": "Consumer Staples", "PM": "Consumer Staples", "WMT": "Consumer Staples",
    # Communication Services
    "CMCSA": "Communication Services", "DIS": "Communication Services", "GOOGL": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "LMT": "Industrials", "RTX": "Industrials", "UPS": "Industrials",
    # Energy
    "COP": "Energy", "CVX": "Energy", "EOG": "Energy", "SLB": "Energy", "XOM": "Energy",
    # Materials
    "FCX": "Materials", "LIN": "Materials", "NEM": "Materials", "SHW": "Materials",
    # Utilities
    "DUK": "Utilities", "NEE": "Utilities", "SO": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "SPG": "Real Estate",
    # Broad Index ETFs -- track a whole index, not one business/sector
    "DIA": "Broad Index ETF", "IWM": "Broad Index ETF", "QQQ": "Broad Index ETF",
    "SPY": "Broad Index ETF", "VTI": "Broad Index ETF",
    # Bonds/Rates
    "IEF": "Bonds/Rates", "SHY": "Bonds/Rates", "TLT": "Bonds/Rates", "^TNX": "Bonds/Rates", "^TYX": "Bonds/Rates",
    # Commodities (kept as one bucket; sub-typing by energy/metal/ag is not
    # needed at this universe's concentration granularity)
    "BZ=F": "Commodities", "CC=F": "Commodities", "CL=F": "Commodities", "CT=F": "Commodities",
    "GC=F": "Commodities", "HE=F": "Commodities", "HG=F": "Commodities", "KC=F": "Commodities",
    "LE=F": "Commodities", "NG=F": "Commodities", "PA=F": "Commodities", "PL=F": "Commodities",
    "SB=F": "Commodities", "SI=F": "Commodities", "ZC=F": "Commodities", "ZS=F": "Commodities",
    "ZW=F": "Commodities",
    # Forex
    "AUDUSD=X": "Forex", "DX-Y.NYB": "Forex", "EURUSD=X": "Forex", "GBPUSD=X": "Forex",
    "NZDUSD=X": "Forex", "USDCAD=X": "Forex", "USDCHF=X": "Forex", "USDJPY=X": "Forex",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Unclassified")


def _load_universe(filter_fn) -> list[str]:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return sorted({r["symbol"] for r in data["results"] if filter_fn(r)})


UNIVERSE_DEFINITIONS = {
    "CURRENT": lambda r: bool(r.get("robust")) and r.get("asset_class") != "Crypto",
    "PURE_EQUITIES": lambda r: bool(r.get("robust")) and r.get("asset_class") == "Equities",
    "ALL_EQUITIES_UNFILTERED": lambda r: r.get("asset_class") == "Equities",
    "ALL_NONCRYPTO_UNFILTERED": lambda r: r.get("asset_class") != "Crypto",
}


def summarize(trades: pd.DataFrame, nav_series: pd.Series) -> dict:
    if nav_series.empty or len(nav_series) < 2:
        return {}
    nav_end = float(nav_series.iloc[-1])
    total_return = (nav_end / STARTING_CAPITAL - 1) * 100
    years = max((nav_series.index[-1] - nav_series.index[0]).days / 365.25, 1e-9)
    cagr = (nav_end / STARTING_CAPITAL) ** (1 / years) - 1
    daily_ret = nav_series.pct_change().dropna()
    ann_factor = np.sqrt(252)
    mean_d, std_d = (float(daily_ret.mean()), float(daily_ret.std(ddof=1))) if len(daily_ret) > 1 else (float("nan"), float("nan"))
    sharpe = (mean_d / std_d) * ann_factor if std_d and std_d > 0 else float("nan")
    downside = daily_ret[daily_ret < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean_d / downside_std) * ann_factor if downside_std and downside_std > 0 else float("nan")
    running_max = nav_series.cummax()
    max_dd = float(((nav_series / running_max - 1) * 100).min())
    n_trades = 0 if trades.empty else len(trades)
    return {"cagr_pct": cagr * 100, "total_return_pct": total_return, "max_dd_pct": max_dd,
            "sharpe": sharpe, "sortino": sortino, "n_trades": n_trades}


def run_universe(symbols: list[str], entry_filter_fn, sizing_fn) -> tuple[dict, pd.DataFrame, int, int]:
    price = fetch_ohlcv_batch(symbols)
    states = build_states(price)
    trades, nav = run_control(states, entry_filter_fn, sizing_fn, stop_mode="FIXED",
                               entry_cost=ENTRY_COST, exit_cost=EXIT_COST)
    summary = summarize(trades, nav)
    return summary, trades, len(symbols), len(price)


def concentration_diagnostics(trades: pd.DataFrame, current_symbols: list[str], entry_filter_fn, sizing_fn) -> dict:
    by_symbol = trades.groupby("asset")["pnl"].sum().sort_values(ascending=False)
    by_sector = trades.assign(sector=trades["asset"].map(sector_of)).groupby("sector")["pnl"].sum().sort_values(ascending=False)

    total_positive_pnl = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    top_n_pct = {}
    for n in (3, 5, 10):
        top_symbols = by_symbol.head(n)
        top_positive = top_symbols[top_symbols > 0].sum()
        top_n_pct[n] = float(top_positive / total_positive_pnl * 100) if total_positive_pnl > 0 else float("nan")

    top3_symbols = list(by_symbol.head(3).index)
    minus_top3_symbols = [s for s in current_symbols if s not in top3_symbols]
    summary_minus_top3, _, _, _ = run_universe(minus_top3_symbols, entry_filter_fn, sizing_fn)

    return {
        "pnl_by_symbol_top15": by_symbol.head(15).to_dict(),
        "pnl_by_sector": by_sector.to_dict(),
        "total_positive_pnl": float(total_positive_pnl),
        "pct_of_positive_pnl_from_top_n": top_n_pct,
        "top3_winning_symbols": top3_symbols,
        "descriptive_only_minus_top3_summary": summary_minus_top3,
    }


def main() -> None:
    entry_filter_fn = ENTRY_SETS["BASELINE"]
    sizing_fn = make_sizing_fn("EQUAL-DOLLAR", 0.0, 0.0)

    print("=" * 108)
    print("PART A -- universe robustness: same locked strategy, four predeclared universe definitions")
    print("=" * 108)
    results = {}
    trades_by_universe = {}
    universe_symbols = {}
    for name, filter_fn in UNIVERSE_DEFINITIONS.items():
        symbols = _load_universe(filter_fn)
        universe_symbols[name] = symbols
        summary, trades, n_defined, n_usable = run_universe(symbols, entry_filter_fn, sizing_fn)
        results[name] = summary
        trades_by_universe[name] = trades
        print(f"  {name:<26} n_symbols={n_defined:>4} usable={n_usable:>4}  n_trades={summary.get('n_trades', 0):>4}  "
              f"cagr={summary.get('cagr_pct', float('nan')):>6.2f}%  sharpe={summary.get('sharpe', float('nan')):>5.2f}  "
              f"sortino={summary.get('sortino', float('nan')):>5.2f}  max_dd={summary.get('max_dd_pct', float('nan')):>7.2f}%")

    baseline = results["CURRENT"]
    print("\nchange vs. CURRENT (each alternate universe independently):")
    for name in ["PURE_EQUITIES", "ALL_EQUITIES_UNFILTERED", "ALL_NONCRYPTO_UNFILTERED"]:
        s = results[name]
        cagr_delta = s.get("cagr_pct", float("nan")) - baseline.get("cagr_pct", float("nan"))
        dd_delta = s.get("max_dd_pct", float("nan")) - baseline.get("max_dd_pct", float("nan"))
        print(f"  {name:<26} CAGR {cagr_delta:+.3f}pp   max_dd {dd_delta:+.3f}pp (negative = worse/deeper)")

    print("\n" + "=" * 108)
    print("PART B -- concentration diagnostics on CURRENT's own trades (descriptive only)")
    print("=" * 108)
    conc = concentration_diagnostics(trades_by_universe["CURRENT"], universe_symbols["CURRENT"], entry_filter_fn, sizing_fn)

    print("\ntop winners by symbol (realized P&L, $):")
    for sym, pnl in list(conc["pnl_by_symbol_top15"].items())[:10]:
        print(f"    {sym:<10} ${pnl:>12,.2f}")

    print("\nrealized P&L by sector/category ($):")
    for sector, pnl in conc["pnl_by_sector"].items():
        print(f"    {sector:<22} ${pnl:>12,.2f}")

    print(f"\n% of total POSITIVE realized P&L from top 3 / 5 / 10 winning symbols: "
          f"{conc['pct_of_positive_pnl_from_top_n'][3]:.1f}% / "
          f"{conc['pct_of_positive_pnl_from_top_n'][5]:.1f}% / "
          f"{conc['pct_of_positive_pnl_from_top_n'][10]:.1f}%")

    print(f"\nDESCRIPTIVE ONLY -- not a proposal to change the strategy: CURRENT universe with its own top-3 "
          f"winning symbols ({', '.join(conc['top3_winning_symbols'])}) excluded entirely:")
    s = conc["descriptive_only_minus_top3_summary"]
    cagr_delta = s.get("cagr_pct", float("nan")) - baseline.get("cagr_pct", float("nan"))
    dd_delta = s.get("max_dd_pct", float("nan")) - baseline.get("max_dd_pct", float("nan"))
    print(f"    n_trades={s.get('n_trades', 0)}  cagr={s.get('cagr_pct', float('nan')):.2f}%  "
          f"sharpe={s.get('sharpe', float('nan')):.2f}  max_dd={s.get('max_dd_pct', float('nan')):.2f}%  "
          f"(CAGR {cagr_delta:+.3f}pp, max_dd {dd_delta:+.3f}pp vs. CURRENT)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "universe_definitions": {k: v for k, v in universe_symbols.items()},
        "part_a_results": results,
        "part_b_concentration": conc,
    }, default=str, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
