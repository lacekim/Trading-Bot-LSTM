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
ASSET_RANKINGS_VALIDATED_PATH = Path("models/asset_rankings_validated.csv")
WALK_FORWARD_SUMMARY_PATH = Path("v4_walk_forward_summary.csv")


@dataclass(frozen=True)
class AssetRankingResult:
    output_path: Path
    rankings: pd.DataFrame
    top_assets: list[str]
    worst_assets: list[str]
    suggested_live_whitelist: list[str]
    suggested_blacklist: list[str]


@dataclass(frozen=True)
class AssetRankingValidationResult:
    rankings_path: Path
    performance_path: Path
    merged: pd.DataFrame
    top_ranked_with_performance: pd.DataFrame
    top_performers_with_rank: pd.DataFrame
    ranking_return_correlation: float
    component_correlations: pd.DataFrame
    whitelist_diagnostics: pd.DataFrame
    trx_diagnostics: pd.DataFrame
    recommended_weights: dict[str, float]


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
        "constrained_smc_return_pct": float(row.get("smc_return_pct", np.nan)),
        "constrained_smc_profit_factor": float(row.get("smc_profit_factor", np.nan)),
        "constrained_smc_max_drawdown_pct": float(row.get("smc_max_drawdown_pct", np.nan)),
        "smc_vs_original_improvement_pct": float(row.get("return_difference_pct", np.nan)),
        "constrained_smc_trade_count": float(row.get("smc_trade_count", np.nan)),
        "shared_timestamps": float(row.get("shared_timestamps", np.nan)),
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


def _trade_count_sanity_score(trade_count: pd.Series, shared_timestamps: pd.Series) -> pd.Series:
    count = _clean_numeric(trade_count).fillna(0.0)
    timestamps = _clean_numeric(shared_timestamps).replace(0.0, np.nan)
    frequency = (count / timestamps).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    count_ok = count.between(80, 700)
    frequency_ok = frequency.between(0.02, 0.20)
    score = pd.Series(100.0, index=count.index)
    score.loc[~count_ok] -= 35.0
    score.loc[~frequency_ok] -= 35.0
    score.loc[count < 30] = 0.0
    return score.clip(lower=0.0, upper=100.0)


def _add_validated_ranking_scores(rankings: pd.DataFrame) -> pd.DataFrame:
    result = rankings.copy()
    result["constrained_smc_return_score"] = _percentile_score(result["constrained_smc_return_pct"])
    result["constrained_smc_profit_factor_score"] = _percentile_score(result["constrained_smc_profit_factor"])
    result["constrained_smc_drawdown_score"] = _percentile_score(result["constrained_smc_max_drawdown_pct"])
    result["walk_forward_stability_score"] = _clean_numeric(result["walk_forward_stability"]).clip(0.0, 100.0).fillna(0.0)
    result["smc_improvement_score"] = _percentile_score(result["smc_vs_original_improvement_pct"])
    result["trade_count_sanity_score"] = _trade_count_sanity_score(
        result["constrained_smc_trade_count"],
        result["shared_timestamps"],
    )
    result["liquidity_score"] = _percentile_score(np.log1p(result["liquidity"].clip(lower=0.0)))
    result["trend_strength_score"] = _percentile_score(result["trend_strength"])
    result["confidence_score"] = _percentile_score(result["cnn_lstm_confidence"])
    result["secondary_filter_score"] = (
        (result["liquidity_score"] * 0.34)
        + (result["trend_strength_score"] * 0.33)
        + (result["confidence_score"] * 0.33)
    )

    weights = {
        "constrained_smc_return_score": 0.34,
        "constrained_smc_profit_factor_score": 0.20,
        "constrained_smc_drawdown_score": 0.18,
        "walk_forward_stability_score": 0.12,
        "smc_improvement_score": 0.08,
        "trade_count_sanity_score": 0.05,
        "secondary_filter_score": 0.03,
    }
    result["validated_score"] = sum(result[column] * weight for column, weight in weights.items())
    result["ranking_score"] = result["validated_score"]
    return result.sort_values("validated_score", ascending=False).reset_index(drop=True)


def _whitelist_condition_frame(rankings: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=rankings.index)
    result["score_at_least_60"] = _clean_numeric(rankings["ranking_score"]) >= 60.0
    result["drawdown_at_least_minus_50"] = _clean_numeric(rankings["max_drawdown_pct"]) >= -50.0
    result["profit_factor_at_least_0_7"] = _clean_numeric(rankings["profit_factor"]) >= 0.7
    result["walk_forward_stability_at_least_50"] = _clean_numeric(rankings["walk_forward_stability"]) >= 50.0
    result["passes_whitelist"] = result.all(axis=1)
    return result


def _current_weight_table() -> dict[str, float]:
    return {
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


def _recommended_weight_table() -> dict[str, float]:
    return {
        "constrained_smc_return_score": 0.24,
        "constrained_smc_profit_factor_score": 0.14,
        "constrained_smc_drawdown_score": 0.12,
        "walk_forward_stability_score": 0.12,
        "historical_return_score": 0.08,
        "sharpe_score": 0.06,
        "sortino_score": 0.06,
        "calmar_score": 0.05,
        "smc_component_score": 0.05,
        "confidence_score": 0.03,
        "trend_strength_score": 0.02,
        "liquidity_score": 0.01,
        "trade_frequency_score": 0.01,
        "volatility_score": 0.005,
        "atr_score": 0.005,
    }


def _load_required_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{description} is empty: {path}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame


def _component_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    target = _clean_numeric(merged["smc_return_pct"])
    rows = []
    for component, weight in _current_weight_table().items():
        if component not in merged.columns:
            continue
        values = _clean_numeric(merged[component])
        valid = values.notna() & target.notna()
        correlation = float(values.loc[valid].corr(target.loc[valid])) if valid.sum() > 2 else float("nan")
        rows.append(
            {
                "component": component,
                "current_weight": weight,
                "correlation_to_constrained_smc_return": correlation,
            }
        )
    return pd.DataFrame(rows).sort_values("correlation_to_constrained_smc_return", ascending=False, na_position="last")


def _whitelist_diagnostics(merged: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    conditions = _whitelist_condition_frame(merged)
    rows = []
    for symbol in symbols:
        matches = merged.loc[merged["symbol"].eq(symbol.upper())]
        if matches.empty:
            rows.append({"symbol": symbol.upper(), "reason": "missing from rankings"})
            continue
        index = matches.index[0]
        failed = [column for column in conditions.columns if column != "passes_whitelist" and not bool(conditions.loc[index, column])]
        row = matches.loc[index]
        rows.append(
            {
                "symbol": symbol.upper(),
                "rank": int(row.get("rank", 0)),
                "ranking_score": float(row.get("ranking_score", np.nan)),
                "smc_return_pct": float(row.get("smc_return_pct", np.nan)),
                "smc_profit_factor": float(row.get("smc_profit_factor", np.nan)),
                "max_drawdown_pct": float(row.get("max_drawdown_pct", np.nan)),
                "walk_forward_stability": float(row.get("walk_forward_stability", np.nan)),
                "passes_whitelist": bool(conditions.loc[index, "passes_whitelist"]),
                "failed_conditions": ", ".join(failed) if failed else "none",
            }
        )
    return pd.DataFrame(rows)


def _trx_diagnostics(merged: pd.DataFrame) -> pd.DataFrame:
    matches = merged.loc[merged["symbol"].eq("TRX")]
    if matches.empty:
        return pd.DataFrame([{"symbol": "TRX", "reason": "missing from rankings"}])
    columns = [
        "symbol",
        "rank",
        "ranking_score",
        "smc_return_pct",
        "historical_return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "smc_profit_factor",
        "max_drawdown_pct",
        "trade_frequency_pct",
        "volatility_pct",
        "trend_strength",
        "liquidity",
        "smc_score",
        "cnn_lstm_confidence",
        "walk_forward_stability",
        "historical_return_score",
        "sharpe_score",
        "sortino_score",
        "calmar_score",
        "drawdown_score",
        "trade_frequency_score",
        "volatility_score",
        "atr_score",
        "trend_strength_score",
        "liquidity_score",
        "smc_component_score",
        "confidence_score",
        "walk_forward_stability_score",
    ]
    return matches[[column for column in columns if column in matches.columns]].copy()


def validate_asset_rankings(args: Any) -> AssetRankingValidationResult:
    """Compare saved asset rankings against constrained paper performance."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    rankings = _load_required_csv(ASSET_RANKINGS_PATH, "Asset rankings")
    performance = _load_required_csv(PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH, "Constrained paper performance")
    rankings = rankings.loc[rankings["timeframe"].astype(str).eq(timeframe)].copy()
    performance = performance.loc[performance["timeframe"].astype(str).eq(timeframe)].copy()
    if rankings.empty:
        raise ValueError(f"No asset rankings for timeframe {timeframe}")
    if performance.empty:
        raise ValueError(f"No constrained paper performance rows for timeframe {timeframe}")

    merged = rankings.merge(
        performance[
            [
                "symbol",
                "timeframe",
                "smc_return_pct",
                "original_return_pct",
                "return_difference_pct",
                "smc_profit_factor",
                "smc_max_drawdown_pct",
                "smc_win_rate_pct",
                "smc_trade_count",
            ]
        ],
        on=["symbol", "timeframe"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping symbols between asset rankings and constrained paper performance")

    correlation = float(_clean_numeric(merged["ranking_score"]).corr(_clean_numeric(merged["smc_return_pct"])))
    top_ranked = merged.sort_values("rank").head(20).copy()
    top_performers = merged.sort_values("smc_return_pct", ascending=False).head(20).copy()

    return AssetRankingValidationResult(
        rankings_path=ASSET_RANKINGS_PATH,
        performance_path=PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH,
        merged=merged,
        top_ranked_with_performance=top_ranked,
        top_performers_with_rank=top_performers,
        ranking_return_correlation=correlation,
        component_correlations=_component_correlations(merged),
        whitelist_diagnostics=_whitelist_diagnostics(merged, ["AIXBT", "DYDX", "PENGU"]),
        trx_diagnostics=_trx_diagnostics(merged),
        recommended_weights=_recommended_weight_table(),
    )


def run_asset_ranking(args: Any) -> AssetRankingResult:
    """Rank all local GMX assets for analysis-only asset selection."""
    timeframe = str(getattr(args, "timeframe", Config.TIMEFRAME))
    validated = bool(getattr(args, "validated", False))
    symbols = [str(symbol).upper() for symbol in list_gmx_symbols(timeframe)]
    performance = _load_constrained_performance()
    if validated and performance.empty:
        raise FileNotFoundError(f"Validated ranking requires constrained performance: {PAPER_MODEL_PERFORMANCE_CONSTRAINED_CSV_PATH}")
    walk_forward_stability = _load_walk_forward_stability(timeframe)

    model = getattr(args, "model", None)
    scaler = getattr(args, "scaler", None)
    if model is None or scaler is None:
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
    elif validated:
        required = [
            "constrained_smc_return_pct",
            "constrained_smc_profit_factor",
            "constrained_smc_max_drawdown_pct",
            "smc_vs_original_improvement_pct",
            "constrained_smc_trade_count",
            "shared_timestamps",
        ]
        rankings = rankings.dropna(subset=required)
        rankings = _add_validated_ranking_scores(rankings)
        rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    else:
        rankings = _add_ranking_scores(rankings)
        rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))

    output_path = ASSET_RANKINGS_VALIDATED_PATH if validated else ASSET_RANKINGS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_path, index=False)

    top_assets = rankings["symbol"].head(20).astype(str).tolist() if not rankings.empty else []
    worst_assets = rankings["symbol"].tail(20).astype(str).tolist() if not rankings.empty else []
    if validated and not rankings.empty:
        whitelist = rankings.loc[
            (_clean_numeric(rankings["constrained_smc_return_pct"]) > 0.0)
            & (_clean_numeric(rankings["constrained_smc_profit_factor"]) >= 1.0)
            & (_clean_numeric(rankings["constrained_smc_max_drawdown_pct"]) >= -15.0)
            & (_clean_numeric(rankings["trade_count_sanity_score"]) >= 70.0),
            "symbol",
        ].head(20).astype(str).tolist()
        if not whitelist:
            whitelist = top_assets[:10]
    else:
        conditions = _whitelist_condition_frame(rankings) if not rankings.empty else pd.DataFrame()
        whitelist = rankings.loc[
            conditions["passes_whitelist"],
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
        output_path=output_path,
        rankings=rankings,
        top_assets=top_assets,
        worst_assets=worst_assets,
        suggested_live_whitelist=whitelist,
        suggested_blacklist=blacklist,
    )
