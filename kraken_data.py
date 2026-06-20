from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests

from config import Config


OHLC_COLUMNS = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'trades']


def interval_to_minutes(interval):
    if interval.endswith('m'):
        return int(interval[:-1])
    if interval.endswith('h'):
        return int(interval[:-1]) * 60
    if interval.endswith('d'):
        return int(interval[:-1]) * 1440
    raise ValueError(f"Intervalo no soportado: {interval}")


def period_to_days(period):
    if period.endswith('mo'):
        return int(period[:-2]) * 30
    unit = period[-1].lower()
    value = int(period[:-1])
    if unit == 'd':
        return value
    if unit == 'y':
        return value * 365
    raise ValueError(f"Periodo no soportado: {period}")


def _normalize_ohlc(df):
    df = df.copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def _filter_period(df, period):
    if df.empty:
        return df
    days = period_to_days(period)
    cutoff = df.index.max() - timedelta(days=days)
    return df[df.index >= cutoff]


def latest_contiguous_segment(df, max_gap_hours=None):
    if df.empty:
        return df
    max_gap_hours = max_gap_hours or Config.KRAKEN_MAX_GAP_HOURS
    gaps = df.index.to_series().diff()
    break_points = gaps[gaps > pd.Timedelta(hours=max_gap_hours)]
    if break_points.empty:
        return df
    last_break = break_points.index[-1]
    return df[df.index >= last_break]


def local_ohlc_path(pair=None, interval=None):
    pair = pair or Config.KRAKEN_PAIR
    interval = interval or Config.TIMEFRAME
    minutes = interval_to_minutes(interval)
    return Path(Config.KRAKEN_OHLC_DIR) / f"{pair}_{minutes}.csv"


def load_local_ohlc(pair=None, period=None, interval=None, require_contiguous=False):
    pair = pair or Config.KRAKEN_PAIR
    period = period or Config.LOOKBACK_PERIOD
    interval = interval or Config.TIMEFRAME
    path = local_ohlc_path(pair, interval)

    if not path.exists():
        raise FileNotFoundError(f"Local Kraken OHLC not found: {path}")

    df = pd.read_csv(path, header=None, names=OHLC_COLUMNS)
    df = _normalize_ohlc(df)
    df = _filter_period(df, period)
    if require_contiguous:
        df = latest_contiguous_segment(df)
    return df


def fetch_live_ohlc(pair=None, period=None, interval=None):
    pair = pair or Config.KRAKEN_PAIR
    period = period or Config.LOOKBACK_PERIOD
    interval = interval or Config.TIMEFRAME
    minutes = interval_to_minutes(interval)
    days = period_to_days(period)
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    response = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={'pair': pair, 'interval': minutes, 'since': since},
        timeout=20
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('error'):
        raise RuntimeError(f"Kraken OHLC error: {payload['error']}")

    result = payload.get('result', {})
    ohlc_key = next((key for key in result.keys() if key != 'last'), None)
    if not ohlc_key:
        raise ValueError("Kraken OHLC returned no data")

    rows = result[ohlc_key]
    if not rows:
        raise ValueError("Kraken OHLC returned empty data")

    df = pd.DataFrame(
        rows,
        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'trades']
    )
    return _normalize_ohlc(df)


def load_kraken_ohlc(pair=None, period=None, interval=None, prefer_live=False, require_contiguous=False, logger=None):
    if prefer_live:
        try:
            df = fetch_live_ohlc(pair, period, interval)
            if not df.empty:
                if logger:
                    logger.info(f"✔️ Kraken live OHLC: {len(df)} candles")
                return df
        except Exception as e:
            if logger:
                logger.warning(f"Kraken live OHLC failed: {e}. Using local CSV.")

    df = load_local_ohlc(pair, period, interval, require_contiguous=require_contiguous)
    if logger:
        source = local_ohlc_path(pair, interval)
        age_days = (time.time() - source.stat().st_mtime) / 86400
        logger.info(f"✔️ Kraken local OHLC: {len(df)} candles from {source} ({age_days:.1f}d since modification)")
    return df


def update_local_ohlc(pair=None, interval=None, path=None, pause_seconds=1.0, logger=None):
    pair = pair or Config.KRAKEN_PAIR
    interval = interval or Config.TIMEFRAME
    minutes = interval_to_minutes(interval)
    path = Path(path) if path else local_ohlc_path(pair, interval)

    if not path.exists():
        raise FileNotFoundError(f"Local Kraken OHLC not found: {path}")

    existing = pd.read_csv(path, header=None, names=OHLC_COLUMNS)
    if existing.empty:
        raise ValueError(f"Local Kraken OHLC is empty: {path}")

    existing['timestamp'] = pd.to_numeric(existing['timestamp'], errors='coerce')
    existing = existing.dropna(subset=['timestamp'])
    existing['timestamp'] = existing['timestamp'].astype('int64')
    last_timestamp = int(existing['timestamp'].max())
    since = last_timestamp + minutes * 60
    new_rows = []

    if logger:
        last_dt = datetime.fromtimestamp(last_timestamp, tz=timezone.utc)
        logger.info(f"Actualizando {path} desde {last_dt.isoformat()}")

    while True:
        response = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={'pair': pair, 'interval': minutes, 'since': since},
            timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get('error'):
            raise RuntimeError(f"Kraken OHLC error: {payload['error']}")

        result = payload.get('result', {})
        ohlc_key = next((key for key in result.keys() if key != 'last'), None)
        rows = result.get(ohlc_key, []) if ohlc_key else []
        batch = []
        for row in rows:
            timestamp = int(float(row[0]))
            if timestamp > last_timestamp:
                batch.append([
                    timestamp,
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[6]),
                    int(row[7]),
                ])

        if not batch:
            break

        new_rows.extend(batch)
        last_timestamp = max(row[0] for row in batch)
        next_since = int(result.get('last', last_timestamp + minutes * 60))
        if next_since <= since:
            next_since = last_timestamp + minutes * 60
        since = next_since

        if len(batch) < 720:
            break
        time.sleep(pause_seconds)

    if not new_rows:
        if logger:
            logger.info("Local Kraken OHLC was already up to date")
        return 0, path

    updates = pd.DataFrame(new_rows, columns=OHLC_COLUMNS)
    combined = pd.concat([existing[OHLC_COLUMNS], updates], ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
    combined.to_csv(path, header=False, index=False)

    if logger:
        first = datetime.fromtimestamp(int(updates['timestamp'].min()), tz=timezone.utc)
        last = datetime.fromtimestamp(int(updates['timestamp'].max()), tz=timezone.utc)
        logger.info(f"Added {len(updates)} candles to {path}: {first.isoformat()} -> {last.isoformat()}")

    return len(updates), path
