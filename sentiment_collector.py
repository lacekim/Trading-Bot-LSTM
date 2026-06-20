#!/usr/bin/env python3
"""
Generate social sentiment CSV for the trading bot from Twitter search results.

This script can use either snscrape or the Twitter API.
For Twitter API mode, provide a bearer token via --bearer-token or TWITTER_BEARER_TOKEN.
It uses a simple lexicon scorer to estimate sentiment.

Example:
    python sentiment_collector.py \
      --query "ADA OR Cardano OR $ADA" \
      --start 2026-06-01 \
      --end 2026-06-16 \
      --interval 15m \
      --max-tweets 2000

Output:
    data/social_sentiment.csv

If snscrape is not installed:
    pip install snscrape

If Twitter scraping is blocked, you can also use a local tweet export:
    --input-file data/tweets.csv --text-column text --timestamp-column timestamp
"""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from config import Config
except Exception:
    Config = None

load_dotenv()

# Compatibility shim for snscrape on Python 3.12+
# The package still uses the deprecated FileFinder.find_module API.
if not hasattr(importlib.machinery.FileFinder, 'find_module'):
    def _find_module(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        return spec.loader if spec else None

    importlib.machinery.FileFinder.find_module = _find_module

DEFAULT_NITTER_INSTANCES = [
    'https://nitter.net',
    'https://nitter.it',
    'https://nitter.eu',
]

POSITIVE_WORDS = {
    'bull', 'bullish', 'moon', 'pump', 'long', 'buy', 'green', 'breakout', 'rocket',
    'rise', 'up', 'gain', 'gains', 'support', 'strong', 'buying', 'buying pressure',
    'bought', 'bullrun', 'bull market', 'moonshot'
}

NEGATIVE_WORDS = {
    'bear', 'bearish', 'dump', 'sell', 'red', 'drop', 'down', 'loss', 'losses',
    'weak', 'resistance', 'selloff', 'panic', 'fear', 'weakness', 'short',
    'bear market', 'crash', 'dumping'
}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
NON_WORD_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')


def clean_text(text: str) -> str:
    text = text or ''
    text = URL_PATTERN.sub(' ', text)
    text = MENTION_PATTERN.sub(' ', text)
    text = HASHTAG_PATTERN.sub(' ', text)
    text = NON_WORD_PATTERN.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def score_text(text: str) -> float:
    text = clean_text(text)
    if not text:
        return 0.0

    tokens = text.split()
    score = 0
    for token in tokens:
        if token in POSITIVE_WORDS:
            score += 1
        if token in NEGATIVE_WORDS:
            score -= 1

    if len(tokens) == 0:
        return 0.0

    return max(-1.0, min(1.0, score / (len(tokens) ** 0.5)))


def _parse_tweet_file(path: str, text_column: str = 'text', timestamp_column: str = 'timestamp'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Tweet file not found: {path}')

    suffix = path.suffix.lower()
    if suffix in ('.csv', '.tsv'):
        df = pd.read_csv(path)
    elif suffix == '.json':
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path)
    else:
        raise ValueError(
            'Unsupported tweet file format. Use CSV or JSON with text and timestamp columns.'
        )

    if text_column not in df.columns:
        candidates = [c for c in ['text', 'content', 'tweet', 'full_text'] if c in df.columns]
        if not candidates:
            raise ValueError(
                f'No text column found in tweet file. Expected one of: text, content, tweet, full_text.'
            )
        text_column = candidates[0]

    if timestamp_column not in df.columns:
        candidates = [c for c in ['timestamp', 'date', 'created_at'] if c in df.columns]
        if not candidates:
            raise ValueError(
                f'No timestamp column found in tweet file. Expected one of: timestamp, date, created_at.'
            )
        timestamp_column = candidates[0]

    records = []
    for _, row in df.iterrows():
        text = str(row[text_column]) if not pd.isna(row[text_column]) else ''
        ts = row[timestamp_column]
        records.append({
            'timestamp': pd.to_datetime(ts, utc=True).to_pydatetime(),
            'text': text,
            'tweet_id': str(row.get('tweet_id', '')),
            'username': str(row.get('username', '')),
        })

    return records


def get_snscrape_tweets(query: str, since: str, until: str, max_tweets: int):
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        raise ImportError(
            'snscrape is required to fetch tweets. Install with: pip install snscrape'
        )

    query_string = f"{query} since:{since} until:{until}"
    tweets = []

    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query_string).get_items()):
            if i >= max_tweets:
                break

            tweets.append({
                'timestamp': tweet.date.replace(tzinfo=None),
                'text': tweet.content,
                'tweet_id': tweet.id,
                'username': tweet.user.username,
            })
    except Exception as e:
        raise RuntimeError(
            'snscrape failed to collect tweets. Twitter guest search may be blocked or the API changed. '
            'Try the --input-file mode with a pre-downloaded tweet export, or use a different environment.'
        ) from e

    return tweets


def _twitter_bearer_token(token: str | None = None) -> str:
    token = token or os.getenv('TWITTER_BEARER_TOKEN', '')
    if not token:
        raise ValueError(
            'Twitter bearer token is required for source=twitter. '
            'Provide --bearer-token or set TWITTER_BEARER_TOKEN in your environment.'
        )
    return token


def _twitter_iso_datetime(date_string: str, end_of_day: bool = False) -> str:
    parsed = datetime.fromisoformat(date_string)
    if end_of_day:
        parsed = parsed + timedelta(days=1)
    return parsed.strftime('%Y-%m-%dT%H:%M:%SZ')


def get_twitter_api_tweets(
    query: str,
    since: str,
    until: str,
    max_tweets: int,
    bearer_token: str | None = None,
    full_archive: bool = False,
):
    bearer_token = _twitter_bearer_token(bearer_token)

    endpoint = (
        'https://api.twitter.com/2/tweets/search/all'
        if full_archive
        else 'https://api.twitter.com/2/tweets/search/recent'
    )

    params = {
        'query': query,
        'max_results': 100,
        'tweet.fields': 'created_at,author_id',
        'expansions': 'author_id',
        'user.fields': 'username',
        'start_time': _twitter_iso_datetime(since, end_of_day=False),
        'end_time': _twitter_iso_datetime(until, end_of_day=True),
    }

    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Accept': 'application/json',
    }

    tweets = []
    users = {}
    collected = 0
    next_token = None

    while collected < max_tweets:
        if next_token:
            params['next_token'] = next_token

        response = requests.get(endpoint, params=params, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f'Twitter API request failed ({response.status_code}): {response.text}'
            )

        data = response.json()
        if 'data' not in data:
            break

        for user in data.get('includes', {}).get('users', []):
            users[user['id']] = user.get('username', '')

        for tweet in data['data']:
            if collected >= max_tweets:
                break
            tweets.append({
                'timestamp': tweet.get('created_at'),
                'text': tweet.get('text', ''),
                'tweet_id': tweet.get('id'),
                'username': users.get(tweet.get('author_id', ''), ''),
            })
            collected += 1

        next_token = data.get('meta', {}).get('next_token')
        if not next_token:
            break

    return tweets


def get_tweets(
    query: str,
    since: str,
    until: str,
    max_tweets: int,
    source: str = 'snscrape',
    input_file: str | None = None,
    text_column: str = 'text',
    timestamp_column: str = 'timestamp',
    bearer_token: str | None = None,
    twitter_full_archive: bool = False,
):
    if input_file:
        return _parse_tweet_file(input_file, text_column=text_column, timestamp_column=timestamp_column)

    if source == 'snscrape':
        return get_snscrape_tweets(query, since, until, max_tweets)

    if source == 'twitter':
        return get_twitter_api_tweets(
            query,
            since,
            until,
            max_tweets,
            bearer_token=bearer_token,
            full_archive=twitter_full_archive,
        )

    raise ValueError('Unsupported source. Choose snscrape, twitter, or use --input-file for local data.')


def _read_asset_symbols(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Asset summary file not found: {path}')

    df = pd.read_csv(path)
    if 'symbol' not in df.columns:
        raise ValueError('Asset summary file must contain a symbol column')

    symbols = [str(symbol).strip() for symbol in df['symbol'].dropna() if str(symbol).strip()]
    return sorted(dict.fromkeys(symbols))


def _safe_symbol_filename(symbol: str) -> str:
    safe = ''.join(ch if ch.isalnum() else '_' for ch in str(symbol).strip())
    return '_'.join(part for part in safe.split('_') if part)


def _symbol_aliases(symbol: str):
    raw_symbol = str(symbol).strip()
    clean_symbol = raw_symbol.split()[0].replace('[', '').replace(']', '')
    clean_symbol = clean_symbol.replace('.V2', '').replace('.E', '')
    aliases = [
        raw_symbol,
        clean_symbol,
        raw_symbol.upper(),
        clean_symbol.upper(),
        f"${clean_symbol.upper()}",
        f"#{clean_symbol.upper()}",
    ]

    alias_map = getattr(Config, 'SENTIMENT_SYMBOL_ALIASES', {}) if Config else {}
    for key in (raw_symbol.upper(), clean_symbol.upper()):
        configured = alias_map.get(key, [])
        aliases.extend(configured)
        aliases.extend(f"${alias}" for alias in configured)
        aliases.extend(f"#{alias}" for alias in configured)

    seen = set()
    result = []
    for alias in aliases:
        key = str(alias).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(str(alias))
    return result


def _format_query_alias(alias: str) -> str:
    alias = str(alias).strip()
    if not alias:
        return ''
    if any(ch.isspace() for ch in alias) or '[' in alias or ']' in alias:
        return '"' + alias.replace('"', '') + '"'
    return alias


def build_symbol_query(symbol: str, query_template: str):
    aliases = [_format_query_alias(alias) for alias in _symbol_aliases(symbol)]
    aliases = [alias for alias in aliases if alias]
    alias_query = ' OR '.join(aliases)
    return query_template.format(symbol=symbol, aliases=alias_query)


def aggregate_sentiment(tweets, interval='15min'):
    df = pd.DataFrame(tweets)
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'sentiment_score', 'mention_count'])

    df['sentiment_score'] = df['text'].apply(score_text)
    df['mention_count'] = 1
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    grouped = df.resample(interval).agg(
        sentiment_score=('sentiment_score', 'mean'),
        mention_count=('mention_count', 'sum')
    )

    grouped = grouped.fillna(0.0)
    grouped = grouped.reset_index()
    return grouped


def run(argv=None):
    parser = argparse.ArgumentParser(description='Build social_sentiment.csv from Twitter search')
    parser.add_argument('--query', help='Twitter search query')
    parser.add_argument('--start', help='Start date yyyy-mm-dd')
    parser.add_argument('--end', help='End date yyyy-mm-dd')
    parser.add_argument('--interval', default='15m', help='Aggregation interval, e.g. 15m or 1h')
    parser.add_argument('--max-tweets', type=int, default=2000, help='Maximum number of tweets to scrape')
    parser.add_argument('--output', default='data/social_sentiment.csv', help='Output CSV path')
    parser.add_argument('--output-dir', default='data/sentiment', help='Output directory for batch asset sentiment files')
    parser.add_argument('--batch-file', help='CSV file with a symbol column for batch sentiment generation')
    parser.add_argument('--gmx-dir', help='Directory containing GMX OHLC files (auto-detect symbols)')
    parser.add_argument('--gmx-timeframe', default='15m', help='Timeframe suffix used in GMX filenames, e.g. 15m')
    parser.add_argument('--query-template', default='{symbol}', help='Query template for batch mode, e.g. "{symbol} OR {symbol}USD"')
    parser.add_argument('--symbols', help='Comma-separated subset of symbols to process when using --batch-file')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output file')
    parser.add_argument('--input-file', help='Use an existing CSV or JSON file with raw tweet text instead of scraping')
    parser.add_argument('--text-column', default='text', help='Column name for tweet text when using --input-file')
    parser.add_argument('--timestamp-column', default='timestamp', help='Column name for tweet timestamps when using --input-file')
    parser.add_argument('--source', default='snscrape', choices=['snscrape', 'twitter'],
                        help='Tweet source. Use twitter for API-based fetching or snscrape for scraping.')
    parser.add_argument('--bearer-token', help='Twitter bearer token. Also supported via TWITTER_BEARER_TOKEN env var.')
    parser.add_argument('--twitter-full-archive', action='store_true',
                        help='Use Twitter full-archive search (paid access required).')
    args = parser.parse_args(argv)

    interval = args.interval.lower().strip()
    if interval.endswith('m'):
        interval = interval[:-1] + 'min'
    elif interval.endswith('h'):
        interval = interval[:-1] + 'h'

    if not interval.endswith('min') and not interval.endswith('h'):
        raise ValueError('Interval must end with m or h, e.g. 15m or 1h')

    # If user requested GMX directory mode, auto-discover symbols from filenames
    if args.gmx_dir:
        gmx_path = Path(args.gmx_dir)
        if not gmx_path.exists() or not gmx_path.is_dir():
            parser.error(f'GMX dir not found: {args.gmx_dir}')

        tf = args.gmx_timeframe
        pattern = f"gmx_arbitrum_*_{tf}.csv"
        symbols = []
        for p in gmx_path.glob(pattern):
            stem = p.stem
            # remove prefix and suffix
            sym = stem.removeprefix('gmx_arbitrum_').removesuffix(f'_{tf}')
            if sym:
                symbols.append(sym)

        symbols = sorted(dict.fromkeys(symbols))
        if args.symbols:
            selected = {s.strip() for s in args.symbols.split(',') if s.strip()}
            symbols = [s for s in symbols if s in selected]

        if not symbols:
            parser.error('No symbols discovered in GMX dir')

        # proceed with batch processing below using discovered symbols
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        failures = []
        for symbol in symbols:
            query = build_symbol_query(symbol, args.query_template)
            output_path = output_dir / f"social_sentiment_{_safe_symbol_filename(symbol)}.csv"
            if output_path.exists() and not args.force:
                print(f'Skipping existing output: {output_path}')
                continue

            try:
                tweets = get_tweets(
                    query,
                    args.start,
                    args.end,
                    args.max_tweets,
                    source=args.source,
                    input_file=None,
                    text_column=args.text_column,
                    timestamp_column=args.timestamp_column,
                    bearer_token=args.bearer_token,
                    twitter_full_archive=args.twitter_full_archive,
                )
            except Exception as exc:
                failures.append({'symbol': symbol, 'error': str(exc), 'query': query})
                print(f'Failed sentiment collection for {symbol}: {exc}')
                continue

            if not tweets:
                print(f'No tweets collected for {symbol}; skipping {output_path}')
                continue

            result = aggregate_sentiment(tweets, interval=interval)
            if result.empty:
                print(f'No sentiment data aggregated for {symbol}; skipping {output_path}')
                continue

            result['symbol'] = symbol
            result['query'] = query
            result.to_csv(output_path, index=False)
            print(f'Wrote sentiment CSV: {output_path}')

        if failures:
            failure_path = output_dir / 'sentiment_failures.csv'
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            print(f'Wrote sentiment failure report: {failure_path}')

        return 0

    if args.batch_file:
        if args.input_file or args.query:
            parser.error('When using --batch-file, do not specify --input-file or --query.')
        if not (args.start and args.end):
            parser.error('Batch mode requires --start and --end.')
    elif args.input_file:
        if args.query or args.start or args.end:
            parser.error('When using --input-file, do not specify --query, --start, or --end.')
    else:
        if not (args.query and args.start and args.end):
            parser.error('Either --input-file or --query/--start/--end must be specified.')

    if args.batch_file:
        symbols = _read_asset_symbols(args.batch_file)
        if args.symbols:
            selected = {s.strip() for s in args.symbols.split(',') if s.strip()}
            symbols = [s for s in symbols if s in selected]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not symbols:
            raise ValueError('No symbols found in batch asset file.')

        failures = []
        for symbol in symbols:
            query = build_symbol_query(symbol, args.query_template)
            output_path = output_dir / f"social_sentiment_{_safe_symbol_filename(symbol)}.csv"
            if output_path.exists() and not args.force:
                print(f'Skipping existing output: {output_path}')
                continue

            try:
                tweets = get_tweets(
                    query,
                    args.start,
                    args.end,
                    args.max_tweets,
                    source=args.source,
                    input_file=None,
                    text_column=args.text_column,
                    timestamp_column=args.timestamp_column,
                    bearer_token=args.bearer_token,
                    twitter_full_archive=args.twitter_full_archive,
                )
            except Exception as exc:
                failures.append({'symbol': symbol, 'error': str(exc), 'query': query})
                print(f'Failed sentiment collection for {symbol}: {exc}')
                continue

            if not tweets:
                print(f'No tweets collected for {symbol}; skipping {output_path}')
                continue

            result = aggregate_sentiment(tweets, interval=interval)
            if result.empty:
                print(f'No sentiment data aggregated for {symbol}; skipping {output_path}')
                continue

            result['symbol'] = symbol
            result['query'] = query
            result.to_csv(output_path, index=False)
            print(f'Wrote sentiment CSV: {output_path}')

        if failures:
            failure_path = output_dir / 'sentiment_failures.csv'
            pd.DataFrame(failures).to_csv(failure_path, index=False)
            print(f'Wrote sentiment failure report: {failure_path}')

        return 0

    output_path = os.path.abspath(args.output)
    if os.path.exists(output_path) and not args.force:
        print(f'Output already exists: {output_path}')
        print('Use --force to overwrite')
        return 1

    tweets = get_tweets(
        args.query,
        args.start,
        args.end,
        args.max_tweets,
        source=args.source,
        input_file=args.input_file,
        text_column=args.text_column,
        timestamp_column=args.timestamp_column,
        bearer_token=args.bearer_token,
        twitter_full_archive=args.twitter_full_archive,
    )
    print(f'Collected {len(tweets)} tweets')

    result = aggregate_sentiment(tweets, interval=interval)
    if result.empty:
        print('No sentiment data aggregated. Check your query and date range.')
        return 1

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f'Wrote sentiment CSV: {output_path}')
    print(result.head())
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
