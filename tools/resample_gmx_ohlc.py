#!/usr/bin/env python3
"""
Resample GMX OHLC CSV files from one timeframe to another (e.g. 15m -> 1h).

Usage:
  python tools/resample_gmx_ohlc.py --input-dir data/GMX_OHLCVT --from 15m --to 1h

By default the script writes files alongside the originals with the new timeframe
in the filename (e.g. gmx_arbitrum_ADA_1h.csv). Use --force to overwrite existing
target files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def resample_file(path: Path, to_tf: str, out_path: Path, force: bool = False):
    if out_path.exists() and not force:
        print(f'Skipping existing {out_path}')
        return

    df = pd.read_csv(path)

    # determine time column
    if 'open_time' in df.columns:
        time_col = 'open_time'
    elif 'Date' in df.columns:
        time_col = 'Date'
    else:
        raise ValueError(f'No timestamp column found in {path}')

    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
    df = df.set_index(time_col).sort_index()

    # normalize column names to lowercase for aggregation
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    for col in ('open', 'high', 'low', 'close', 'volume'):
        if col not in df.columns:
            df[col] = pd.NA

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }

    res = df.resample(to_tf).agg(agg)

    # drop periods with no price data
    res = res.dropna(subset=['open', 'close'])

    # write with original-style column names ('open_time', 'open', 'high', ...)
    res = res.reset_index()
    res = res.rename(columns={res.columns[0]: 'open_time'})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(res)} rows)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/GMX_OHLCVT')
    parser.add_argument('--from', dest='from_tf', default='15m', help='Source timeframe suffix in filenames')
    parser.add_argument('--to', dest='to_tf', default='1h', help='Target resample timeframe (pandas offset alias, e.g. 1h)')
    parser.add_argument('--pattern-prefix', default='gmx_arbitrum_', help='Filename prefix')
    parser.add_argument('--force', action='store_true', help='Overwrite existing target files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f'Input directory not found: {input_dir}')

    tf = args.from_tf
    pattern = f"{args.pattern_prefix}*_{tf}.csv"
    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f'No files matching {pattern} in {input_dir}')
        return

    for p in files:
        stem = p.stem
        # remove prefix and suffix to get symbol
        name = stem.removeprefix(args.pattern_prefix).removesuffix(f'_{tf}')
        out_name = f"{args.pattern_prefix}{name}_{args.to_tf}.csv"
        out_path = input_dir / out_name
        try:
            resample_file(p, args.to_tf, out_path, force=args.force)
        except Exception as e:
            print(f'Error processing {p}: {e}')


if __name__ == '__main__':
    main()
