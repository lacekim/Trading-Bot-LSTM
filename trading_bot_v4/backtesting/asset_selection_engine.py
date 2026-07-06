"""Asset selection ranking for V4 analysis-only workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot import list_gmx_symbols, load_gmx_ohlc
from trading_bot_v4.config_v4 import V4Config as Config
from trading_bot_v4.core.smc_swings import SMC_FEATURE_COLUMNS
from trading_bot_v4.execution.paper_model_comparison import _predict_original_model_signals
from trading_bot_v4.execution.paper_model_performance import PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH
from trading_bot_v4.features.smc_feature_builder import _build_smc_feature_frame
from trading_bot_v4.utils.logger import build_logger
from trading_bot_v4.utils.model_cache import ModelScalerCache


logger = build_logger("v4_asset_selection")

ASSET_RANKINGS_PATH = Path("models/asset_rankings.csv")
WALK_FORWARD_SUMMARY_PATH = Path("v4_walk_forward_summary.csv")


@dataclass(frozen=True)
class AssetRankingResult:
    output_path: Path
    rankings: pd.DataFrame
    top_assets: list[str]
    worst_assets: list[str]
    suggested_live_whitelist: list[str]
    suggested_blacklist: list[str]


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _profit_factor_from_returns(returns: pd.Series) -> float:
    clean = _clean_numeric(returns).dropna()
    wins = clean[clean > 0.0]
    losses = clean[clean < 0.0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    if gross_loss <= 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def _max_drawdown_pct(close: pd.Series) -> float:
    prices = _clean_numeric(close).dropna()
    if prices.empty:
        return 0.0
    equity = prices / prices.iloc[0]
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0) * 100.0
    return float(drawdown.min())


def _sortino_ratio(returns: pd.Series) -> float:
    clean = _clean_numeric(returns).dropna()
    downside = clean[clean < 0.0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    if downside_std <= 0.0:
        return 0.0
    return float(clean.mean() / downside_std)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _atr(df: pd.DataFrame, period: int | None = None) -> pd.Series:
    atr_period = period or Config.ATR_PERIOD
    high = _clean_numeric(df["High"])
    low = _clean_numeric(df["Low"])
    close = _clean_numeric(df["Close"])
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=atr_period, min_periods=1).mean()


def _trend_strength(close: pd.Series, atr: pd.Series) -> float:
    prices = _clean_numeric(close).dropna().tail(120)
    if len(prices) < 20:
        return 0.0
    x = np.arange(len(prices), dtype=float)
    slope = float(np.polyfit(x, prices.to_numpy(dtype=float), 1)[0])
    latest_atr = float(_clean_numeric(atr).reindex(prices.index).dropna().tail(20).mean() or 0.0)
    return abs(_safe_ratio(slope, latest_atr))


def _ohlc_metrics(symbol: str, timeframe: str) -> dict[str, float | int | str]:
    raw = load_gmx_ohlc(symbol, timeframe).copy()
    if raw.empty:
        raise ValueError(f"No OHLC rows for {symbol} {timeframe}")

    close = _clean_numeric(raw["Close"])
    high = _clean_numeric(raw["High"])
    low = _clean_numeric(raw["Low"])
    volume = _clean_numeric(raw["Volume"]) if "Volume" in raw.columns else pd.Series(0.0, index=raw.index)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    atr = _atr(raw)
    historical_return_pct = ((float(close.dropna().iloc[-1]) / float(close.dropna().iloc[0])) - 1.0) * 100.0
    volatility_pct = float(returns.std(ddof=0) * 100.0) if len(returns) > 1 else 0.0
    sharpe_ratio = _safe_ratio(float(returns.mean()), float(returns.std(ddof=0))) if len(returns) > 1 else 0.0
    sortino_ratio = _sortino_ratio(returns)
    max_drawdown_pct = _max_drawdown_pct(close)
    calmar_ratio = _safe_ratio(historical_return_pct, abs(max_drawdown_pct))
    atr_latest = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0
    atr_pct = _safe_ratio(atr_latest, float(close.dropna().iloc[-1])) * 100.0
    liquidity = float((close * volume).replace([np.inf, -np.inf], np.nan).dropna().tail(240).median() or 0.0)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": int(len(raw)),
        "historical_return_pct": historical_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "profit_factor": _profit_factor_from_returns(returns),
        "max_drawdown_pct": max_drawdown_pct,
        "volatility_pct": volatility_pct,
        "atr": atr_latest,
        "atr_pct": atr_pct,
        "trend_strength": _trend_strength(close, atr),
        "liquidity": liquidity,
    }


def _smc_score(symbol: str, timeframe: str) -> float:
    smc = _build_smc_feature_frame(symbol, timeframe)
    if smc.empty:
        return 0.0
    recent = smc[SMC_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).tail(240)
    bullish_columns = [
        "bullish_bos",
        "bullish_choch",
        "bullish_liquidity_sweep",
        "bullish_order_block",
        "bullish_fvg",
    ]
    bearish_columns = [
        "bearish_bos",
        "bearish_choch",
        "bearish_liquidity_sweep",
        "bearish_order_block",
        "bearish_fvg",
    ]
    bullish = float(recent[[column for column in bullish_columns if column in recent.columns]].sum(axis=1).mean())
    bearish = float(recent[[column for column in bearish_columns if column in recent.columns]].sum(axis=1).mean())
    trending = float(recent.get("regime_trending", pd.Series(0.0, index=recent.index)).mean())
    regime = float(recent.get("regime_score", pd.Series(0.0, index=recent.index)).mean())
    raw_score = 50.0 + ((bullish - bearish) * 25.0) + (trending * 15.0) + (regime * 10.0)
    return float(np.clip(raw_score, 0.0, 100.0))


def _prediction_metrics(model: Any, scaler: Any, symbol: str, timeframe: str) -> dict[str, float | int]:
    signals = _predict_original_model_signals(model, scaler, symbol, timeframe)
    if signals.empty:
        return {
            "cnn_lstm_confidence": 0.0,
            "trade_frequency_pct": 0.0,
            "prediction_count": 0,
        }
    probabilities = _clean_numeric(signals["original_probability"]).dropna()
    confidence = (probabilities - 0.5).abs() * 2.0
    candidates = signals["original_candidate"].astype(bool)
    return {
        "cnn_lstm_confidence": float(confidence.tail(240).mean() if len(confidence) else 0.0),
        "trade_frequency_pct": float(candidates.mean() * 100.0),
        "prediction_count": int(len(signals)),
    }


def _load_constrained_performance() -> pd.DataFrame:
    if not PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH.exists():
        return pd.DataFrame()
    performance = pd.read_csv(PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH)
    performance["symbol"] = performance["symbol"].astype(str).str.upper()
    return performance


def _performance_overrides(symbol: str, performance: pd.DataFrame) -> dict[str, float]:
    if performance.empty:
        return {}
    rows = performance.loc[performance["symbol"].eq(symbol.upper())]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "profit_factor": float(row.get("smc_profit_factor", row.get("original_profit_factor", 0.0))),
        "trade_frequency_pct": _safe_ratio(float(row.get("smc_trade_count", 0.0)), float(row.get("shared_timestamps", 0.0))) * 100.0,
    }


def _load_walk_forward_stability(timeframe: str) -> dict[str, float]:
    if not WALK_FORWARD_SUMMARY_PATH.exists():
        return {}
    summary = pd.read_csv(WALK_FORWARD_SUMMARY_PATH)
    if summary.empty:
        return {}
    summary["symbol"] = summary["symbol"].astype(str).str.upper()
    summary = summary.loc[summary["timeframe"].astype(str).eq(str(timeframe))]
    if "strategy" in summary.columns:
        preferred = summary.loc[summary["strategy"].astype(str).eq("smc_filtered")]
        if not preferred.empty:
            summary = preferred

    stability: dict[str, float] = {}
    for symbol, group in summary.groupby("symbol"):
        returns = _clean_numeric(group["return_pct"]).dropna()
        profit_factors = _clean_numeric(group["profit_factor"]).dropna()
        if returns.empty:
            stability[symbol] = 0.0
            continue
        positive_rate = float((returns > 0.0).mean())
        mean_return = float(returns.mean())
        return_std = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
        consistency = 1.0 / (1.0 + max(return_std, 0.0) / 25.0)
        profit_factor_bonus = float((profit_factors > 1.0).mean()) if len(profit_factors) else 0.0
        score = (positive_rate * 50.0) + (consistency * 30.0) + (np.tanh(mean_return / 25.0) * 10.0) + (profit_factor_bonus * 10.0)
        stability[symbol] = float(np.clip(score, 0.0, 100.0))
    return stability


def _percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = _clean_numeric(series).replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = clean.rank(pct=True, method="average") * 100.0
    ranks = ranks.fillna(0.0)
    return ranks if higher_is_better else 100.0 - ranks


def _add_ranking_scores(rankings: pd.DataFrame) -> pd.DataFrame:
    result = rankings.copy()
    result["historical_return_score"] = _percentile_score(result["historical_return_pct"])
    result["sharpe_score"] = _percentile_score(result["sharpe_ratio"])
    result["sortino_score"] = _percentile_score(result["sortino_ratio"])
    result["calmar_score"] = _percentile_score(result["calmar_ratio"])
    result["profit_factor_score"] = _percentile_score(result["profit_factor"])
    result["drawdown_score"] = _percentile_score(result["max_drawdown_pct"])
    result["trade_frequency_score"] = _percentile_score(result["trade_frequency_pct"])
    result["volatility_score"] = _percentile_score(result["volatility_pct"], higher_is_better=False)
    result["atr_score"] = _percentile_score(result["atr_pct"], higher_is_better=False)
    result["trend_strength_score"] = _percentile_score(result["trend_strength"])
    result["liquidity_score"] = _percentile_score(np.log1p(result["liquidity"].clip(lower=0.0)))
    result["smc_component_score"] = _clean_numeric(result["smc_score"]).clip(0.0, 100.0).fillna(0.0)
    result["confidence_score"] = _percentile_score(result["cnn_lstm_confidence"])
    result["walk_forward_stability_score"] = _clean_numeric(result["walk_forward_stability"]).clip(0.0, 100.0).fillna(0.0)

    weights = {
        "historical_return_score": 0.10,
        "sharpe_score": 0.08,
        "sortino_score": 0.08,
        "calmar_score": 0.08,
        "profit_factor_score": 0.10,
        "drawdown_score": 0.10,
        "trade_frequency_score": 0.06,
        "volatility_score": 0.05,
        "atr_score": 0.04,
        "trend_strength_score": 0.07,
        "liquidity_score": 0.06,
        "smc_component_score": 0.08,
        "confidence_score": 0.05,
        "walk_forward_stability_score": 0.05,
    }
    result["ranking_score"] = sum(result[column] * weight for column, weight in weights.items())
    return result.sort_values("ranking_score", ascending=False).reset_index(drop=True)


def run_asset_ranking(args: Any) -> AssetRankingResult:
    """Rank all local GMX assets for analysis-only asset selection."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    symbols = [str(symbol).upper() for symbol in list_gmx_symbols(timeframe)]
    performance = _load_constrained_performance()
    walk_forward_stability = _load_walk_forward_stability(timeframe)

    model = scaler = None
    try:
        model, scaler = ModelScalerCache().load()
    except Exception as exc:
        logger.warning("CNN/LSTM confidence unavailable; continuing without model confidence: %s", exc)

    rows: list[dict[str, float | int | str]] = []
    for symbol in symbols:
        try:
            row = _ohlc_metrics(symbol, timeframe)
            row["smc_score"] = _smc_score(symbol, timeframe)
            if model is not None and scaler is not None:
                row.update(_prediction_metrics(model, scaler, symbol, timeframe))
            else:
                row.update({"cnn_lstm_confidence": 0.0, "trade_frequency_pct": 0.0, "prediction_count": 0})
            row.update(_performance_overrides(symbol, performance))
            row["walk_forward_stability"] = walk_forward_stability.get(symbol, 0.0)
            rows.append(row)
        except Exception as exc:
            logger.warning("Skipping %s during asset ranking: %s", symbol, exc)

    rankings = pd.DataFrame(rows)
    if rankings.empty:
        rankings = pd.DataFrame(columns=["symbol", "timeframe", "ranking_score"])
    else:
        rankings = _add_ranking_scores(rankings)
        rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))

    ASSET_RANKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(ASSET_RANKINGS_PATH, index=False)

    top_assets = rankings["symbol"].head(20).astype(str).tolist() if not rankings.empty else []
    worst_assets = rankings["symbol"].tail(20).astype(str).tolist() if not rankings.empty else []
    whitelist = rankings.loc[
        (rankings["ranking_score"] >= 60.0)
        & (rankings["max_drawdown_pct"] >= -50.0)
        & (rankings["profit_factor"] >= 0.7)
        & (rankings["walk_forward_stability"] >= 50.0),
        "symbol",
    ].head(20).astype(str).tolist() if not rankings.empty else []

    blacklist_frame = rankings.loc[
        (rankings["ranking_score"] <= 35.0)
        | (rankings["max_drawdown_pct"] <= -75.0)
        | (rankings["profit_factor"] <= 0.35),
    ].copy() if not rankings.empty else pd.DataFrame()
    if not blacklist_frame.empty:
        blacklist_frame = blacklist_frame.loc[~blacklist_frame["symbol"].isin(whitelist)]
    blacklist = blacklist_frame.sort_values("ranking_score").head(30)["symbol"].astype(str).tolist() if not blacklist_frame.empty else []
    if len(blacklist) < 10:
        fallback = [symbol for symbol in worst_assets[-10:] if symbol not in whitelist and symbol not in blacklist]
        blacklist.extend(fallback)

    return AssetRankingResult(
        output_path=ASSET_RANKINGS_PATH,
        rankings=rankings,
        top_assets=top_assets,
        worst_assets=worst_assets,
        suggested_live_whitelist=whitelist,
        suggested_blacklist=blacklist,
    )
