#!/usr/bin/env python3
"""
Bot de Trading LSTM v3.3 - CORREGIDO
🔥 Ahora usa TODOS los features correctamente
"""
from config import Config
from argparse import ArgumentParser
import os
import time
import logging
import warnings
import pickle
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import requests

Path('logs/matplotlib').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(Path('logs/matplotlib').resolve()))

import numpy as np
import pandas as pd
import krakenex
import schedule
from tensorflow.keras.models import load_model

from kraken_data import load_kraken_ohlc

warnings.filterwarnings('ignore')

Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def asset_label():
    return getattr(Config, 'GMX_SYMBOL', Config.SYMBOL).replace('-USD', '')


def gmx_ohlc_path(symbol=None, timeframe=None):
    symbol = symbol or getattr(Config, 'GMX_SYMBOL', 'GMX')
    timeframe = timeframe or Config.TIMEFRAME
    return Path(Config.GMX_OHLC_DIR) / f"gmx_arbitrum_{symbol.upper()}_{timeframe}.csv"


def list_gmx_symbols(timeframe=None):
    timeframe = timeframe or Config.TIMEFRAME
    suffix = f"_{timeframe}"
    blacklist = {symbol.upper() for symbol in getattr(Config, 'GMX_SYMBOL_BLACKLIST', set())}
    symbols = []
    for path in Path(Config.GMX_OHLC_DIR).glob(f"gmx_arbitrum_*{suffix}.csv"):
        symbol = path.stem.removeprefix("gmx_arbitrum_").removesuffix(suffix)
        if symbol.upper() in blacklist:
            continue
        symbols.append(symbol)
    return sorted(symbols)


def load_gmx_ohlc(symbol=None, timeframe=None):
    path = gmx_ohlc_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"GMX OHLC no encontrado: {path}")

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            'open': 'Open',
            'open_time.1': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'open_time': 'Date',
        }
    )

    for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            values = df.loc[:, df.columns == col].bfill(axis=1).iloc[:, 0]
            df = df.loc[:, df.columns != col]
            df[col] = values

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas en {path}: {missing_columns}")

    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required_columns)
    df = df.set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def using_gmx_data():
    return getattr(Config, 'DATA_SOURCE', '').upper() == 'GMX'

# ======================== Telegram with Commands ========================

class TelegramNotifier:
    """Telegram notifier with mobile monitoring"""
    
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        
    def send_message(self, text):
        """Send a text message"""
        if not Config.TELEGRAM_ENABLED:
            logger.info(f"Telegram disabled: {text[:80]}...")
            return
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ Telegram: {text[:50]}...")
            else:
                logger.error(f"❌ Telegram error {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error send_message: {e}")
    
    def send_status_report(self, bot_instance):
        """Sends a complete status report"""
        try:
            uptime = datetime.now() - bot_instance.start_time
            hours = int(uptime.total_seconds() // 3600)
            minutes = int((uptime.total_seconds() % 3600) // 60)
            
            balance_info = bot_instance.trader.update_balance()
            
            position = bot_instance.trader.current_position
            position_text = "❌ Sin posición"
            
            if position:
                pos_info = bot_instance.trader.get_position_info()
                if pos_info:
                    emoji = "🟢" if pos_info['type'] == 'LONG' else "🔴"
                    position_text = (
                        f"{emoji} {pos_info['symbol']} {pos_info['type']} | "
                        f"PnL: {pos_info['pnl']:.2f}% / ${pos_info['pnl_usd']:.2f}"
                    )
            
            pred_text = "No Recent Prediction"
            if bot_instance.last_prediction_prob is not None:
                pred_text = f"{bot_instance.last_prediction_prob*100:.1f}% prob. subida"
            
            total_trades = len(bot_instance.trader.trade_history)
            winning_trades = sum(1 for t in bot_instance.trader.trade_history if t['pnl_percent'] > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            text = (
                f"📊 *Status Report*\n"
                f"━━━━━━━━━━━━━━━━━━\n\n"
                f"⏰ *Uptime:* `{hours}h {minutes}m`\n"
                f"💰 *Balance:* `${balance_info['total_usd']:.2f}`\n"
                f"📍 *Position:* {position_text}\n"
                f"🔮 *Last Prediction:* `{pred_text}`\n\n"
                f"📈 *History:*\n"
                f"  • Trades: `{total_trades}`\n"
                f"  • Win Rate: `{win_rate:.1f}%`\n\n"
                f"✅ *Bot Operating Correctly*"
            )
            
            self.send_message(text)
            
        except Exception as e:
            logger.error(f"Error en status_report: {e}")
            self.send_message(f"⚠️ Error Generating Report: {str(e)}")
    
    def send_position_update(self, position_info):
        emoji = "🟢" if position_info['type'] == 'LONG' else "🔴"
        text = (
            f"{emoji} *Active Position*\n\n"
            f"Symbol: `{position_info['symbol']}`\n"
            f"Type: `{position_info['type']}`\n"
            f"Entry Price: `${position_info['entry']:.4f}`\n"
            f"Current: `${position_info['current']:.4f}`\n"
            f"PnL: `{position_info['pnl']:.2f}% / ${position_info['pnl_usd']:.2f}`\n"
            f"SL: `${position_info['sl']:.4f}`\n"
            f"TP: `${position_info['tp']:.4f}`\n"
            f"Trailing: `{'✅' if position_info['trailing'] else '❌'}`\n"
            f"Time: `{position_info['duration']}`"
        )
        self.send_message(text)
    
    def send_balance_report(self, balance_info):
        if not balance_info:
            return
        
        label = asset_label()
        text = (
            "💰 *BALANCE*\n\n"
            f"Total: `${balance_info['total_usd']:.2f}`\n"
            f"{label}: `{balance_info.get('asset', balance_info.get('ada', 0)):.2f}`\n"
            f"USD: `${balance_info['usd']:.2f}`\n"
            f"Max Trade: `${balance_info['max_trade_size']:.2f}`"
        )
        self.send_message(text)
    
    def get_updates(self):
        """Receives User Commands"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'offset': self.last_update_id + 1, 'timeout': 1}
            response = requests.get(url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data['ok'] and data['result']:
                    return data['result']
            return []
            
        except Exception as e:
            return []
    
    def process_commands(self, bot_instance):
        """Processes Commands from Telegram"""
        try:
            updates = self.get_updates()
            
            for update in updates:
                self.last_update_id = update['update_id']
                
                if 'message' in update and 'text' in update['message']:
                    text = update['message']['text'].lower().strip()
                    
                    if text == '/status':
                        self.send_status_report(bot_instance)
                    elif text == '/balance':
                        balance = bot_instance.trader.update_balance()
                        if balance:
                            self.send_balance_report(balance)
                    elif text == '/position':
                        if bot_instance.trader.current_position:
                            pos = bot_instance.trader.get_position_info()
                            if pos:
                                self.send_position_update(pos)
                        else:
                            self.send_message("❌ No Open Position")
                    elif text == '/help':
                        help_text = (
                            "🤖 *Available Commands*\n\n"
                            "`/status` - Complete Report\n"
                            "`/balance` - View Balance\n"
                            "`/position` - Current Position\n"
                            "`/help` - This Help"
                        )
                        self.send_message(help_text)
                        
        except Exception as e:
            logger.error(f"Error Processing Commands: {e}")

# ======================== Data Handler with Complete Features ========================

class DataHandler:
    def __init__(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"✅ Scaler Loaded: {scaler_path}")

    def fetch_data(self, symbol, period, interval=None, include_sentiment=True):
        try:
            # Default interval to the configured timeframe so sentiment/resampling align
            interval = interval or Config.TIMEFRAME

            if using_gmx_data():
                asset_symbol = symbol or Config.GMX_SYMBOL
                logger.info(f"Loading GMX OHLC {asset_symbol} {interval}...")
                df = load_gmx_ohlc(asset_symbol, interval)
                if df.empty:
                    raise ValueError("Sin datos GMX")
                logger.info(
                    f"✔️ GMX local OHLC: {len(df)} velas "
                    f"desde {gmx_ohlc_path(asset_symbol, interval)}"
                )
                if include_sentiment:
                    df = self.merge_sentiment_data(df, interval, symbol=asset_symbol)
                return df

            logger.info(f"Loading Kraken OHLC {Config.KRAKEN_PAIR}...")
            df = load_kraken_ohlc(Config.KRAKEN_PAIR, period, interval, prefer_live=True, logger=logger)
            if df.empty:
                raise ValueError("No Data")
            if include_sentiment:
                df = self.merge_sentiment_data(df, interval, symbol=Config.SYMBOL)
            return df
        except Exception as e:
            logger.error(f"Error Retrieving Data: {e}")
            return None
    
    def calculate_atr(self, df, period=14):
        """Calculates ATR"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        if isinstance(atr, pd.DataFrame):
            atr = atr.iloc[:, 0]
        
        return atr

    def _normalize_interval(self, interval):
        interval = str(interval).lower().strip()
        if interval.endswith('m'):
            return interval[:-1] + 'min'
        if interval.endswith('h'):
            return interval[:-1] + 'h'
        return interval

    def _normalize_sentiment_token(self, value):
        return ''.join(ch for ch in str(value).lower() if ch.isalnum())

    def _sentiment_aliases(self, symbol=None):
        if not symbol:
            return []

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

        alias_map = getattr(Config, 'SENTIMENT_SYMBOL_ALIASES', {})
        configured = []
        for key in (raw_symbol.upper(), clean_symbol.upper()):
            configured.extend(alias_map.get(key, []))
        aliases.extend(configured)
        aliases.extend(f"${alias}" for alias in configured)
        aliases.extend(f"#{alias}" for alias in configured)

        seen = set()
        unique_aliases = []
        for alias in aliases:
            key = str(alias).strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_aliases.append(str(alias))
        return unique_aliases

    def _safe_sentiment_filename_symbol(self, value):
        safe = ''.join(ch if ch.isalnum() else '_' for ch in str(value).strip())
        return '_'.join(part for part in safe.split('_') if part)

    def _get_sentiment_path(self, symbol=None):
        if not Config.SENTIMENT_ENABLED:
            return None

        sentiment_path = Config.SENTIMENT_CSV_PATH
        if sentiment_path.exists() and sentiment_path.is_file():
            return sentiment_path

        if symbol:
            # Try symbol-specific files in the configured directory.
            sentiment_dir = getattr(Config, 'SENTIMENT_CSV_DIR', None)
            filename_template = getattr(Config, 'SENTIMENT_CSV_FILE_TEMPLATE', 'social_sentiment_{symbol}.csv')
            if sentiment_dir is not None:
                for alias in self._sentiment_aliases(symbol):
                    candidate = Path(sentiment_dir) / filename_template.format(
                        symbol=self._safe_sentiment_filename_symbol(alias)
                    )
                    if candidate.exists():
                        return candidate

            if '{symbol}' in str(sentiment_path):
                for alias in self._sentiment_aliases(symbol):
                    candidate = Path(str(sentiment_path).format(
                        symbol=self._safe_sentiment_filename_symbol(alias)
                    ))
                    if candidate.exists():
                        return candidate

        if sentiment_path.exists() and sentiment_path.is_file():
            return sentiment_path

        return None

    def _load_sentiment_data(self, symbol=None):
        sentiment_path = self._get_sentiment_path(symbol)
        if sentiment_path is None:
            logger.warning(
                f"Sentiment file not found for symbol '{symbol}' using path {Config.SENTIMENT_CSV_PATH}"
            )
            return None

        df = pd.read_csv(sentiment_path)
        if 'timestamp' in df.columns:
            ts_col = 'timestamp'
        elif 'date' in df.columns:
            ts_col = 'date'
        else:
            logger.error("Sentiment file missing 'timestamp' or 'date' column")
            return None

        if 'sentiment_score' not in df.columns:
            logger.error("Sentiment file missing 'sentiment_score' column")
            return None

        if 'mention_count' not in df.columns:
            logger.warning("Sentiment file missing 'mention_count'; defaulting to 1 for each row")
            df['mention_count'] = 1

        symbol_columns = ['symbol', 'asset', 'ticker', 'tag', 'query']
        matching_symbol_columns = [col for col in symbol_columns if col in df.columns]
        if symbol and matching_symbol_columns:
            alias_keys = {
                self._normalize_sentiment_token(alias)
                for alias in self._sentiment_aliases(symbol)
            }
            mask = pd.Series(False, index=df.index)
            for col in matching_symbol_columns:
                mask = mask | df[col].map(self._normalize_sentiment_token).isin(alias_keys)

            df = df[mask]
            if df.empty:
                logger.warning(f"Sentiment file {sentiment_path} has no rows matching {symbol} aliases")
                return None

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_localize(None)
        df = df.set_index(ts_col).sort_index()
        return df[['sentiment_score', 'mention_count']]

    def merge_sentiment_data(self, df, interval, symbol=None):
        sentiment = self._load_sentiment_data(symbol)
        if sentiment is None:
            return df

        resample_interval = self._normalize_interval(
            getattr(Config, 'SENTIMENT_RESAMPLE_INTERVAL', interval) or interval
        )
        try:
            sentiment = sentiment.resample(resample_interval).agg({
                'sentiment_score': 'mean',
                'mention_count': 'sum'
            })
        except Exception as e:
            logger.error(f"Error resampling sentiment data: {e}")
            return df

        sentiment = sentiment.reindex(df.index, method='ffill')
        sentiment['sentiment_score'] = sentiment['sentiment_score'].fillna(0.0)
        sentiment['mention_count'] = sentiment['mention_count'].fillna(0.0)

        df = df.join(sentiment, how='left')
        return df
    
    def prepare_features(self, df):
        """
        🔥 CRITICAL: Must calculate EXACTLY the same features as train_model.py
        + INFINITE VALUE CLEANUP
        """
        df = df.copy()
        
        # Flatten MultiIndex if Present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure Series
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
        
        # 1. Returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 3. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # 🔥 Evitar división por 0
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi'] - 50) / 50
        
        # 4. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        macd_std = df['macd_diff'].rolling(50).std()
        df['macd_norm'] = (df['macd_diff'] - df['macd_diff'].rolling(50).mean()) / (macd_std + 1e-10)
        
        # 5. Momentum
        df['momentum'] = df['Close'].pct_change(periods=24)
        
        # 6. Relative Volume
        volume_ma = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / (volume_ma + 1e-10)
        
        # 7. ATR Normalized
        atr_values = self.calculate_atr(df, Config.ATR_PERIOD)
        if isinstance(atr_values, pd.DataFrame):
            atr_values = atr_values.iloc[:, 0]
        
        close_values = df['Close']
        if isinstance(close_values, pd.DataFrame):
            close_values = close_values.iloc[:, 0]
        
        df['atr_norm'] = atr_values.values / (close_values.values + 1e-10)
        
        # 8. Buying Pressure
        df['green_candles'] = (df['Close'] > df['Open']).astype(int).rolling(10).mean()
        
        # 9. Price Range
        df['price_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
        
        # 10. Directional Volume
        volume_directional = np.where(
            df['Close'].values > df['Open'].values, 
            df['Volume'].values, 
            -df['Volume'].values
        )
        if volume_directional.ndim > 1:
            volume_directional = volume_directional.flatten()
        
        df['volume_directional'] = pd.Series(volume_directional, index=df.index).rolling(10).mean()
        
        # 11. Bollinger Bands
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        bb_middle = close_series.rolling(20).mean()
        bb_std = close_series.rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_range = bb_upper - bb_lower
        
        df['bb_position'] = (close_series - bb_lower) / (bb_range + 1e-10)
        
        # 12. OBV
        obv_direction = np.sign(df['returns'].fillna(0).values)
        volume_values = df['Volume'].values if not isinstance(df['Volume'], pd.DataFrame) else df['Volume'].iloc[:, 0].values
        
        df['volume_obv'] = pd.Series(volume_values * obv_direction, index=df.index).rolling(10).sum()
        
        # ========== SENTIMENT FEATURES ==========
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
            df['sentiment_change'] = df['sentiment_score'].diff(1).fillna(0.0)
            df['sentiment_volatility'] = df['sentiment_score'].rolling(window=20).std().fillna(0.0)
            df['sentiment_mentions'] = df['mention_count'].fillna(0.0)
        
        # 🔥 FEATURES NUEVOS v3.3
        
        # 13. Price vs MA50
        ma50 = df['Close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['Close'] - ma50) / (ma50 + 1e-10)
        
        # 14. Volume Change
        df['volume_change'] = df['Volume'].pct_change(5).fillna(0)
        
        # 15. High-Low ratio
        df['hl_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
        
        # 16. Close position in range
        hl_range = df['High'] - df['Low']
        df['close_position'] = (df['Close'] - df['Low']) / (hl_range + 1e-10)
        
        # 17. Short-Term Trend
        ma5 = df['Close'].rolling(5).mean()
        ma20 = df['Close'].rolling(20).mean()
        df['trend_short'] = (ma5 / (ma20 + 1e-10)) - 1
        
        # 18. Long-Term Trend
        df['trend_long'] = (ma20 / (ma50 + 1e-10)) - 1
        
        # 🔥 CRITICAL CLEANUP: Replace Infinite Values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def normalize_data(self, data):
        """Normalizar con limpieza de infinitos"""
        # 🔥 Replace Infinite Values and NaN with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(data)

# ======================== KRAKEN TRADER ========================

class KrakenTrader:
    def __init__(self):
        self.api = krakenex.API(
            key=Config.KRAKEN_API_KEY,
            secret=Config.KRAKEN_PRIVATE_KEY
        )
        
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.sl_price = None
        self.tp_price = None
        self.trailing_activated = False
        self.position_size = None
        self.position_symbol = None
        
        self.dry_run_cash = Config.DRY_RUN_BALANCE_USD
        self.total_balance_usd = Config.DRY_RUN_BALANCE_USD if Config.DRY_RUN else 0
        self.available_ada = 0
        self.available_usd = Config.DRY_RUN_BALANCE_USD if Config.DRY_RUN else 0
        
        self.trade_history = []
        if Config.DRY_RUN:
            self.load_dry_run_state()

    def load_dry_run_state(self):
        state_path = Path(Config.DRY_RUN_STATE_PATH)
        trades_path = Path(Config.DRY_RUN_TRADES_PATH)

        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                self.dry_run_cash = float(state.get('dry_run_cash', self.dry_run_cash))
                position = state.get('position') or {}
                self.current_position = position.get('type')
                self.entry_price = position.get('entry_price')
                self.entry_time = datetime.fromisoformat(position['entry_time']) if position.get('entry_time') else None
                self.sl_price = position.get('sl_price')
                self.tp_price = position.get('tp_price')
                self.trailing_activated = bool(position.get('trailing_activated', False))
                self.position_size = position.get('position_size')
                self.position_symbol = position.get('position_symbol')
                logger.info(f"Loaded DRY_RUN state from {state_path}")
            except Exception as exc:
                logger.error(f"Could not load DRY_RUN state: {exc}")

        if trades_path.exists():
            try:
                trades_df = pd.read_csv(trades_path)
                self.trade_history = trades_df.to_dict('records')
                logger.info(f"Loaded DRY_RUN trade history from {trades_path}")
            except Exception as exc:
                logger.error(f"Could not load DRY_RUN trade history: {exc}")

    def save_dry_run_state(self):
        if not Config.DRY_RUN:
            return

        state_path = Path(Config.DRY_RUN_STATE_PATH)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        position = None
        if self.current_position:
            position = {
                'type': self.current_position,
                'entry_price': self.entry_price,
                'entry_time': self.entry_time.isoformat() if self.entry_time else None,
                'sl_price': self.sl_price,
                'tp_price': self.tp_price,
                'trailing_activated': self.trailing_activated,
                'position_size': self.position_size,
                'position_symbol': self.position_symbol,
            }

        state = {
            'dry_run_cash': self.dry_run_cash,
            'equity': self.get_dry_run_equity(),
            'position': position,
            'updated_at': datetime.now().isoformat(),
        }
        state_path.write_text(json.dumps(state, indent=2))

    def append_dry_run_trade(self, trade):
        trades_path = Path(Config.DRY_RUN_TRADES_PATH)
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        row = trade.copy()
        for key in ('entry_time', 'exit_time'):
            if hasattr(row.get(key), 'isoformat'):
                row[key] = row[key].isoformat()
        pd.DataFrame([row]).to_csv(
            trades_path,
            mode='a',
            header=not trades_path.exists(),
            index=False,
        )
        
    def update_balance(self):
        try:
            if Config.DRY_RUN:
                self.available_ada = 0.0
                self.available_usd = self.dry_run_cash
                self.total_balance_usd = self.get_dry_run_equity()
                logger.info(f"Simulated DRY_RUN Balance: ${self.total_balance_usd:.2f}")

                return {
                    'total_usd': self.total_balance_usd,
                    'ada': self.available_ada,
                    'asset': self.available_ada,
                    'usd': self.available_usd,
                    'max_trade_size': self.calculate_max_trade_size()
                }

            response = self.api.query_private('Balance')
            
            if 'error' in response and response['error']:
                raise Exception(f"Error Kraken: {response['error']}")
            
            balance = response.get('result', {})
            
            self.available_ada = float(balance.get('ADA', 0))
            self.available_usd = float(balance.get('ZUSD', 0))
            
            current_ada_price = self.get_current_price()
            
            if current_ada_price is None:
                logger.warning("Unable to Retrieve Price")
                return None
            
            self.total_balance_usd = (self.available_ada * current_ada_price) + self.available_usd
            
            logger.info(f"Balance: ${self.total_balance_usd:.2f}")
            
            return {
                'total_usd': self.total_balance_usd,
                'ada': self.available_ada,
                'asset': self.available_ada,
                'usd': self.available_usd,
                'max_trade_size': self.calculate_max_trade_size()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving balance: {e}")
            return None
    
    def calculate_max_trade_size(self):
        risk_amount = self.total_balance_usd * (Config.RISK_PERCENTAGE / 100)
        return risk_amount
    
    def calculate_position_size(self, current_price, sl_distance):
        try:
            risk_amount_usd = self.calculate_max_trade_size()
            sl_distance_pct = (sl_distance / current_price) * 100
            
            if sl_distance_pct == 0:
                return Config.MIN_TRADE_AMOUNT
            
            position_size_ada = (risk_amount_usd / current_price) / (sl_distance_pct / 100)
            max_affordable = self.available_usd / current_price
            position_size_ada = min(position_size_ada, max_affordable * 0.95)
            position_size_ada = max(position_size_ada, Config.MIN_TRADE_AMOUNT)
            
            logger.info(f"Position Size: {position_size_ada:.2f} {asset_label()}")
            return position_size_ada
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Config.MIN_TRADE_AMOUNT
    
    def get_current_price(self, symbol=None):
        try:
            if using_gmx_data():
                df = load_gmx_ohlc(symbol or Config.GMX_SYMBOL, Config.TIMEFRAME)
                return float(df['Close'].iloc[-1])

            response = self.api.query_public('Ticker', {'pair': Config.KRAKEN_PAIR})
            
            if 'error' in response and response['error']:
                raise Exception(f"Error: {response['error']}")
            
            ticker = response['result'][Config.KRAKEN_PAIR]
            price = float(ticker['c'][0])
            
            return price
            
        except Exception as e:
            logger.error(f"Error Retrieving Price: {e}")
            return None
    
    def open_position(self, direction, price, sl, tp, atr):
        try:
            balance_info = self.update_balance()
            if not balance_info:
                logger.error("Unable to update balance")
                return False
            
            sl_distance = abs(price - sl)
            self.position_size = self.calculate_position_size(price, sl_distance)
            
            if self.position_size < Config.MIN_TRADE_AMOUNT:
                logger.warning("Position too small")
                return False

            if Config.DRY_RUN:
                self.current_position = direction
                self.entry_price = price
                self.entry_time = datetime.now()
                self.sl_price = sl
                self.tp_price = tp
                self.trailing_activated = False
                self.position_symbol = Config.GMX_SYMBOL if using_gmx_data() else asset_label()
                logger.info(
                    f"DRY_RUN: Simulate Position Opening {direction}: "
                    f"symbol={self.position_symbol}, price={price:.4f}, "
                    f"size={self.position_size:.2f}, SL={sl:.4f}, TP={tp:.4f}"
                )
                self.save_dry_run_state()
                return True

            if using_gmx_data():
                logger.error("GMX CSV data source is local only. Set DRY_RUN=true or add an exchange adapter before live orders.")
                return False
            
            order_type = 'buy' if direction == 'LONG' else 'sell'
            
            order = self.api.query_private('AddOrder', {
                'pair': Config.KRAKEN_PAIR,
                'type': order_type,
                'ordertype': 'market',
                'volume': f'{self.position_size:.2f}',
                'leverage': '2'
            })
            
            if 'error' in order and order['error']:
                raise Exception(f"Error: {order['error']}")
            
            self.current_position = direction
            self.entry_price = price
            self.entry_time = datetime.now()
            self.sl_price = sl
            self.tp_price = tp
            self.trailing_activated = False
            self.position_symbol = Config.GMX_SYMBOL if using_gmx_data() else asset_label()
            
            logger.info(f"Position {direction} opened: {price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error Opening Position: {e}")
            return False
    
    def close_position(self, reason="Manual"):
        """
        SAFE Position Closure with Kraken Response Validation
        🔥 FIXED: Verifies That Kraken Accepted the Order Before Resetting State
        """
        try:
            if not self.current_position:
                logger.warning("No position to close")
                return False
            
            # Save Information BEFORE Attempting to Close (for Rollback if It Fails)
            position_backup = {
                'type': self.current_position,
                'entry': self.entry_price,
                'size': self.position_size,
                'sl': self.sl_price,
                'tp': self.tp_price,
                'entry_time': self.entry_time
            }
            
            order_type = 'sell' if self.current_position == 'LONG' else 'buy'
            
            close_symbol = self.position_symbol or asset_label()
            logger.info(f"🔄 Attempting to Close {self.current_position}: {self.position_size:.2f} {close_symbol}")
            
            # 🔥 PRE-CLOSURE VALIDATION
            if self.position_size < Config.MIN_TRADE_AMOUNT:
                logger.error(f"❌ Position too small: {self.position_size:.2f} {asset_label()} < {Config.MIN_TRADE_AMOUNT}")
                # DO NOT Reset State, Keep Position Open
                return False

            if Config.DRY_RUN:
                current_price = self.get_current_price(close_symbol) or self.entry_price
                pnl = self.calculate_pnl(current_price)
                pnl_usd = self.calculate_pnl_usd(current_price)
                self.dry_run_cash += pnl_usd
                duration = datetime.now() - self.entry_time
                trade = {
                    'entry_time': self.entry_time,
                    'exit_time': datetime.now(),
                    'symbol': close_symbol,
                    'type': self.current_position,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'pnl_percent': pnl,
                    'pnl_usd': pnl_usd,
                    'reason': reason,
                    'duration': duration,
                    'order_id': 'dry_run'
                }
                self.trade_history.append(trade)
                self.append_dry_run_trade(trade)
                logger.info(
                    f"DRY_RUN: Simulate Position Closure {self.current_position}: "
                    f"symbol={close_symbol}, PnL={pnl:.2f}%, PnL=${pnl_usd:.2f}, "
                    f"cash=${self.dry_run_cash:.2f}, Reason={reason}"
                )
                self.current_position = None
                self.entry_price = None
                self.entry_time = None
                self.sl_price = None
                self.tp_price = None
                self.position_size = None
                self.position_symbol = None
                self.trailing_activated = False
                self.save_dry_run_state()
                return True

            if using_gmx_data():
                logger.error("GMX CSV data source is local only. Set DRY_RUN=true or add an exchange adapter before live closes.")
                return False
            
            # 🔥 SEND ORDER TO KRAKEN
            order = self.api.query_private('AddOrder', {
                'pair': Config.KRAKEN_PAIR,
                'type': order_type,
                'ordertype': 'market',
                'volume': f'{self.position_size:.2f}'
            })
            
            # 🔥 CRITICAL VALIDATION: Verify Kraken Response
            if 'error' in order and order['error']:
                error_msg = ', '.join(order['error'])
                logger.error(f"❌ KRAKEN REJECTED THE CLOSURE: {error_msg}")
                logger.error(f"   Order: {order}")
                
                # 🔥 DO NOT RESET STATE - Maintain Internal Position
                logger.warning("⚠️ INTERNAL STATE UNCHANGED - Position Remains Active")
                
                # Notify Critical Error
                if hasattr(self, 'notifier'):
                    self.notifier.send_message(
                        f"🚨 *CRITICAL ERROR WHILE CLOSING*\n\n"
                        f"Position: `{self.current_position}`\n"
                        f"Size: `{self.position_size:.2f} {asset_label()}`\n"
                        f"Error Kraken: `{error_msg}`\n\n"
                        f"⚠️ Manually Verify on Kraken\n"
                        f"⚠️ Bot Maintains Internal State"
                    )
                
                return False
            
            # 🔥 VERIFY THAT A VALID RESULT EXISTS
            if 'result' not in order or not order['result']:
                logger.error(f"❌ Invalid response from Kraken: {order}")
                logger.warning("⚠️ INTERNAL STATE UNCHANGED")
                return False
            
            # ✅ ORDER ACCEPTED - Calculate P&L
            current_price = self.get_current_price()
            if current_price is None:
                current_price = self.entry_price
                logger.warning("⚠️ No current price available, using entry price")
            
            pnl = self.calculate_pnl(current_price)
            duration = datetime.now() - self.entry_time
            
            # Save to History
            trade_result = {
                'entry_time': self.entry_time,
                'exit_time': datetime.now(),
                'type': self.current_position,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pnl_percent': pnl,
                'reason': reason,
                'duration': duration,
                'order_id': order['result'].get('txid', ['unknown'])[0]
            }
            self.trade_history.append(trade_result)
            
            logger.info(f"✅ POSITION CLOSED SUCCESSFULLY")
            logger.info(f"   Type: {self.current_position}")
            logger.info(f"   PnL: {pnl:.2f}%")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Order ID: {trade_result['order_id']}")
            
            # 🔥 ONLY NOW Reset Internal State
            self.current_position = None
            self.entry_price = None
            self.entry_time = None
            self.sl_price = None
            self.tp_price = None
            self.position_size = None
            self.position_symbol = None
            self.trailing_activated = False
            
            # 🔥 UPDATE BALANCE
            self.update_balance()
            
            return True
            
        except requests.exceptions.Timeout:
            logger.error("❌ TIMEOUT While Closing Position")
            logger.warning("⚠️ INTERNAL STATE UNCHANGED - Verify on Kraken manually")
            return False
            
        except Exception as e:
            logger.error(f"❌ EXCEPTION While Closing Position: {e}")
            logger.error(f"   Traceback: ", exc_info=True)
            logger.warning("⚠️ INTERNAL STATE UNCHANGED")
            
            # Notify Error
            if hasattr(self, 'notifier'):
                self.notifier.send_message(
                    f"🚨 *EXCEPTION WHILE CLOSING POSITION*\n\n"
                    f"Error: `{str(e)}`\n\n"
                    f"⚠️ Verify on Kraken manually"
                )
            
            return False


    def verify_position_sync(self):
        """
        🔥 NEW METHOD: Verify Synchronization with Kraken
        Call this periodically to detect de-synchronization issues.
        """
        try:
            if using_gmx_data():
                return 'synced'

            # Query Open Positions on Kraken
            response = self.api.query_private('OpenPositions')
            
            if 'error' in response and response['error']:
                logger.error(f"Error Fetching Positions: {response['error']}")
                return None
            
            open_positions = response.get('result', {})

            # Verify Whether There Are Positions for the Configured Pair
            ada_positions = [p for p in open_positions.values() 
                            if Config.KRAKEN_PAIR in p.get('pair', '')]
            
            # Bot Internal State
            bot_has_position = self.current_position is not None
            kraken_has_position = len(ada_positions) > 0
            
            # 🚨 DETECT DESYNCHRONIZATION
            if bot_has_position and not kraken_has_position:
                logger.error("🚨 DESYNCHRONIZATION: Bot Believes a Position Is Open, Kraken Does NOT")
                logger.error(f"   Bot: {self.current_position} @ {self.entry_price}")
                logger.error("   Kraken: No Positions")
                
                if hasattr(self, 'notifier'):
                    self.notifier.send_message(
                        "🚨 *DE-SYNCHRONIZATION DETECTED*\n\n"
                        "Bot: Active Position\n"
                        "Kraken: No Positions\n\n"
                        "🔧 Resetting Internal State..."
                    )
                
                # Reset Internal State
                self.current_position = None
                self.entry_price = None
                self.entry_time = None
                self.sl_price = None
                self.tp_price = None
                self.position_size = None
                
                return 'desync_fixed'
                
            elif not bot_has_position and kraken_has_position:
                logger.error("🚨 DE-SYNCHRONIZATION: Kraken Has a Position, Bot Does NOT")
                logger.error(f"   Kraken: {len(ada_positions)} posiciones")
                logger.error("   Bot: No Internal Position")
                
                if hasattr(self, 'notifier'):
                    self.notifier.send_message(
                        "🚨 *DE-SYNCHRONIZATION DETECTED*\n\n"
                        "Bot: No Internal Position\n"
                        "Kraken: Active Positions\n\n"
                        "⚠️ CLOSE MANUALLY ON KRAKEN"
                    )
                
                return 'manual_action_needed'
            
            # ✅ Everything Synchronized
            return 'synced'
            
        except Exception as e:
            logger.error(f"Error Verifying Synchronization: {e}")
            return None

    def calculate_pnl(self, current_price):
        if not self.entry_price:
            return 0

        if self.current_position == 'LONG':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100

    def calculate_pnl_usd(self, current_price):
        if not self.entry_price or not self.position_size:
            return 0.0

        if self.current_position == 'LONG':
            return (current_price - self.entry_price) * self.position_size
        return (self.entry_price - current_price) * self.position_size

    def get_dry_run_equity(self):
        equity = self.dry_run_cash
        if self.current_position:
            current_price = self.get_current_price(self.position_symbol) or self.entry_price
            equity += self.calculate_pnl_usd(current_price)
        return equity
    
    def update_trailing_stop(self, current_price, atr):
        if not self.current_position or not atr:
            return
        
        pnl = self.calculate_pnl(current_price)
        activation_threshold = (Config.TRAILING_STOP_ACTIVATION * atr / self.entry_price) * 100
        
        if not self.trailing_activated and pnl > activation_threshold:
            self.trailing_activated = True
            logger.info(f"Trailing Stop Activated (PnL: {pnl:.2f}%)")
        
        if self.trailing_activated:
            trailing_distance = Config.TRAILING_STOP_DISTANCE * atr
            
            if self.current_position == 'LONG':
                new_sl = current_price - trailing_distance
                if new_sl > self.sl_price:
                    self.sl_price = new_sl
                    logger.info(f"SL updated: {self.sl_price:.4f}")
            else:
                new_sl = current_price + trailing_distance
                if new_sl < self.sl_price:
                    self.sl_price = new_sl
                    logger.info(f"SL updated: {self.sl_price:.4f}")
    
    def check_exit_conditions(self, current_price):
        if not self.current_position:
            return False
        
        if self.current_position == 'LONG':
            if current_price <= self.sl_price:
                self.close_position("Stop Loss")
                return True
            if current_price >= self.tp_price:
                self.close_position("Take Profit")
                return True
        else:
            if current_price >= self.sl_price:
                self.close_position("Stop Loss")
                return True
            if current_price <= self.tp_price:
                self.close_position("Take Profit")
                return True
        
        if self.entry_time:
            elapsed = datetime.now() - self.entry_time
            if elapsed.total_seconds() >= 3600:
                self.close_position("End of Window 1H")
                return True
        
        return False
    
    def get_position_info(self):
        if not self.current_position:
            return None
        
        current_price = self.get_current_price(self.position_symbol)
        if not current_price:
            return None
        
        duration = datetime.now() - self.entry_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        
        return {
            'type': self.current_position,
            'symbol': self.position_symbol or asset_label(),
            'entry': self.entry_price,
            'current': current_price,
            'pnl': self.calculate_pnl(current_price),
            'pnl_usd': self.calculate_pnl_usd(current_price),
            'sl': self.sl_price,
            'tp': self.tp_price,
            'trailing': self.trailing_activated,
            'duration': f'{hours}h {minutes}m'
        }

# ======================== TRADING BOT ========================

class TradingBot:
    def __init__(self):
        model_path = Config.MODEL_DIR / Config.MODEL_NAME
        scaler_path = Config.MODEL_DIR / Config.SCALER_NAME
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model Not Found: {model_path}\n"
                f"Run first: python train_model.py"
            )
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"❌ Scaler Not Found: {scaler_path}")
        
        self.model = load_model(model_path)
        logger.info(f"✅ Model Loaded: {model_path}")
        
        self.data_handler = DataHandler(scaler_path)
        self.trader = KrakenTrader()
        self.notifier = TelegramNotifier()
        
        self.current_atr = None
        self.last_prediction_prob = None
        self.start_time = datetime.now()
        
        logger.info("✅ Bot Initialized")
        self.notifier.send_message("🤖 Trading Bot v3.3 Started\n✅ Model with 18 features loaded")

    def refresh_gmx_data_cache(self, force=False):
        """Refresh GMX candle CSVs before signal evaluation while preserving the local cache."""
        if not using_gmx_data() or not getattr(Config, 'GMX_AUTO_REFRESH_ENABLED', True):
            return True

        now = time.time()
        last_refresh = getattr(self, '_last_gmx_refresh_ts', 0)
        refresh_seconds = int(getattr(Config, 'GMX_AUTO_REFRESH_SECONDS', 3300))
        if not force and last_refresh and (now - last_refresh) < refresh_seconds:
            logger.info("GMX data refresh skipped; cache was refreshed recently")
            return True

        update_script = Path(getattr(Config, 'GMX_UPDATE_SCRIPT'))
        update_config = Path(getattr(Config, 'GMX_UPDATE_CONFIG'))
        if not update_script.exists():
            logger.warning(f"GMX update script not found: {update_script}")
            return False
        if not update_config.exists():
            logger.warning(f"GMX update config not found: {update_config}")
            return False

        cmd = [
            sys.executable,
            str(update_script),
            '--config', str(update_config),
            '--period', Config.TIMEFRAME,
            '--chain', getattr(Config, 'GMX_UPDATE_CHAIN', 'arbitrum'),
        ]

        try:
            logger.info(f"Refreshing GMX {Config.TIMEFRAME} OHLC cache before scan")
            result = subprocess.run(
                cmd,
                cwd=update_script.parent,
                text=True,
                capture_output=True,
                timeout=int(getattr(Config, 'GMX_UPDATE_TIMEOUT_SECONDS', 900)),
            )
            if result.stdout:
                logger.info(result.stdout.strip())
            if result.stderr:
                logger.warning(result.stderr.strip())
            if result.returncode != 0:
                logger.warning(f"GMX data refresh failed with exit code {result.returncode}")
                return False

            self._last_gmx_refresh_ts = now
            logger.info("GMX OHLC cache refresh complete")
            return True
        except Exception as exc:
            logger.warning(f"GMX data refresh failed: {exc}")
            return False
    
    def predict_direction(self, symbol=None, include_sentiment=True):
        """
        🔥 FIXED: Uses ALL 18 Features
        Returns the Probability of an Upward Move [0-1]
        """
        try:
            # 🔥 Download 60 Days of Data to Ensure Sufficient Data After Feature Calculation
            symbol = symbol or (Config.GMX_SYMBOL if using_gmx_data() else Config.SYMBOL)
            df = self.data_handler.fetch_data(
                symbol,
                period='60d',  # Increased from 5d
                interval=Config.TIMEFRAME,
                include_sentiment=include_sentiment
            )
            
            if df is None or len(df) < 200:  # We Need Enough Data to MA50 + SEQUENCE_LENGTH
                raise ValueError("Insufficient Data")
            
            # Calculate ALL Features
            df = self.data_handler.prepare_features(df)
            
            # Clean NaNs
            df = df.dropna()
            
            if len(df) < Config.SEQUENCE_LENGTH:
                raise ValueError(f"Insufficient Data after cleaning NaNs: {len(df)}")
            
            # Extract the Last SEQUENCE_LENGTH Candles
            feature_data = df[Config.FEATURE_COLUMNS].values[-Config.SEQUENCE_LENGTH:]
            
            # 🔥 Validation: ensure no infinite values
            if not np.isfinite(feature_data).all():
                logger.warning("⚠️ Infinite Values Detected in Features, Cleaning...")
                feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            scaled_data = self.data_handler.normalize_data(feature_data)
            
            # Reshape for Model: (1, SEQUENCE_LENGTH, num_features)
            X = scaled_data.reshape(1, Config.SEQUENCE_LENGTH, len(Config.FEATURE_COLUMNS))
            
            # Prediction: Probability of an Upward Move
            prediction_prob = self.model.predict(X, verbose=0)[0][0]
            
            # Get Current Price and ATR
            current_price = df['Close'].iloc[-1]
            self.current_atr = self.data_handler.calculate_atr(df, Config.ATR_PERIOD).iloc[-1]
            
            self.last_prediction_prob = prediction_prob
            self.last_feature_row = df.iloc[-1]
            
            logger.info(f"{symbol} Prediction: {prediction_prob*100:.1f}% Upward Move Probability | Price: ${current_price:.4f}")
            logger.info(f"✅ Prediction saved in bot.last_prediction_prob")
            
            return prediction_prob, current_price, self.last_feature_row
            
        except Exception as e:
            logger.error(f"Error in Prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def execute_trading_logic(self):
        try:
            logger.info("🔔 Starting Signal Evaluation...")
            self.notifier.send_message("⏰ Evaluating Signal...")

            if using_gmx_data():
                self.refresh_gmx_data_cache()
            
            balance_info = self.trader.update_balance()
            if balance_info:
                self.notifier.send_balance_report(balance_info)
                logger.info("✅ Balance Reported")
            
            symbols = list_gmx_symbols(Config.TIMEFRAME) if using_gmx_data() else [Config.SYMBOL]
            if not symbols:
                self.notifier.send_message("❌ No symbols available")
                return

            threshold = Config.MIN_SIGNAL_THRESHOLD
            valid_signals = []
            rejected = []

            for symbol in symbols:
                prediction_prob, current_price, feature_row = self.predict_direction(symbol, include_sentiment=False)

                if prediction_prob is None:
                    rejected.append((symbol, "no prediction"))
                    continue

                if prediction_prob > threshold:
                    direction = 'LONG'
                elif prediction_prob < (1 - threshold):
                    direction = 'SELL'
                else:
                    rejected.append((symbol, f"no clear signal {prediction_prob*100:.1f}%"))
                    continue

                if Config.ENABLE_ENTRY_CONTEXT_FILTER and not self.is_trade_allowed(feature_row, prediction_prob):
                    rejected.append((symbol, f"entry filter {prediction_prob*100:.1f}%"))
                    continue

                if not self.current_atr:
                    rejected.append((symbol, "ATR unavailable"))
                    continue

                valid_signals.append({
                    'symbol': symbol,
                    'direction': direction,
                    'probability': float(prediction_prob),
                    'price': float(current_price),
                    'atr': float(self.current_atr),
                    'feature_row': feature_row,
                    'confidence': abs(float(prediction_prob) - 0.5),
                })

            logger.info(f"Scanned {len(symbols)} symbols; valid signals: {len(valid_signals)}; rejected: {len(rejected)}")

            if not valid_signals:
                top_rejections = ", ".join(f"{symbol}: {reason}" for symbol, reason in rejected[:10])
                msg = f"⸻ No valid symbols after scanning {len(symbols)} markets"
                if top_rejections:
                    msg += f"\nTop rejections: {top_rejections}"
                logger.info(msg)
                self.notifier.send_message(msg)
                return

            candidates = sorted(valid_signals, key=lambda item: item['confidence'], reverse=True)
            candidates = self.apply_on_demand_sentiment(candidates)
            if not candidates:
                msg = "⸻ No valid symbols after on-demand sentiment check"
                logger.info(msg)
                self.notifier.send_message(msg)
                return

            selected = candidates[0]
            selected_symbol = selected['symbol']
            direction = selected['direction']
            prediction_prob = selected['probability']
            current_price = selected['price']
            self.current_atr = selected['atr']

            # Calculate stops
            sl_distance = self.current_atr * Config.ATR_SL_MULTIPLIER
            tp_distance = self.current_atr * Config.ATR_TP_MULTIPLIER
            
            # Close Existing Position
            if self.trader.current_position:
                current_symbol = self.trader.position_symbol
                if current_symbol == selected_symbol and self.trader.current_position == direction:
                    logger.info(
                        f"Keeping existing {direction} position on {selected_symbol}; "
                        "selected signal is unchanged"
                    )
                    self.notifier.send_message(
                        f"✅ Keeping Existing Position\n"
                        f"Symbol: `{selected_symbol}`\n"
                        f"Direction: `{direction}`\n"
                        f"Probability: `{prediction_prob*100:.1f}%`"
                    )
                    return

                self.trader.close_position("Nueva hora")
                self.notifier.send_message("🔄 Posición cerrada por nueva señal")

            Config.GMX_SYMBOL = selected_symbol
            
            logger.info(
                f"🎯 Selected {selected_symbol}: direction={direction}, "
                f"prob={prediction_prob*100:.1f}%, threshold={threshold*100:.0f}%"
            )
            
            if direction == 'LONG':
                sl = current_price - sl_distance
                tp = current_price + tp_distance
                
                logger.info(f"🟢 Attempting to Open LONG Position on {selected_symbol}: Price=${current_price:.4f}, prob={prediction_prob*100:.1f}%")
                
                if self.trader.open_position(direction, current_price, sl, tp, self.current_atr):
                    msg = (
                        f"🟢 *LONG OPENED*\n\n"
                        f"Symbol: `{selected_symbol}`\n"
                        f"Price: `${current_price:.4f}`\n"
                        f"Probability: `{prediction_prob*100:.1f}%`\n"
                        f"Size: `{self.trader.position_size:.2f} {asset_label()}`\n"
                        f"SL: `${sl:.4f}` | TP: `${tp:.4f}`\n"
                        f"ATR: `${self.current_atr:.4f}`"
                    )
                    logger.info(f"📱 Sending LONG Notification to Telegram...")
                    self.notifier.send_message(msg)
                    logger.info(f"✅ LONG Notification Sent")
                else:
                    logger.error("❌ Failed to Open LONG Position")
                    self.notifier.send_message("❌ Error Opening LONG Position")
                    
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
                
                logger.info(f"🔴 Attempting to Open SHORT Position on {selected_symbol}: Price=${current_price:.4f}, prob={prediction_prob*100:.1f}%")
                
                if self.trader.open_position(direction, current_price, sl, tp, self.current_atr):
                    msg = (
                        f"🔴 *SHORT OPENED*\n\n"
                        f"Symbol: `{selected_symbol}`\n"
                        f"Price: `${current_price:.4f}`\n"
                        f"Probability: `{prediction_prob*100:.1f}%`\n"
                        f"Size: `{self.trader.position_size:.2f} {asset_label()}`\n"
                        f"SL: `${sl:.4f}` | TP: `${tp:.4f}`\n"
                        f"ATR: `${self.current_atr:.4f}`"
                    )
                    logger.info(f"📱 Sending SHORT Notification to Telegram...")
                    self.notifier.send_message(msg)
                    logger.info(f"✅ SHORT Notification Sent")
                else:
                    logger.error("❌ Failed to Open SHORT Position")
                    self.notifier.send_message("❌ Error Opening SHORT Position")
            
        except Exception as e:
            logger.error(f"Trading Error: {e}")
            import traceback
            traceback.print_exc()
            self.notifier.send_message(f"❌ Error: {str(e)}")
    
    def is_trade_allowed(self, feature_row, prediction_prob):
        """Additional gating filter before opening a trade."""
        if feature_row is None:
            return False

        if prediction_prob > (1 - Config.ENTRY_FILTER_MIN_PROBABILITY) and prediction_prob < Config.ENTRY_FILTER_MIN_PROBABILITY:
            logger.info("⛔ Prediction is too weak for the stricter entry filter")
            return False

        direction = 'LONG' if prediction_prob > 0.5 else 'SHORT'

        if Config.ENTRY_FILTER_REQUIRE_TREND:
            price_vs_ma50 = float(feature_row.get('price_vs_ma50', 0.0))
            trend_short = float(feature_row.get('trend_short', 0.0))

            if direction == 'LONG' and (price_vs_ma50 <= Config.ENTRY_FILTER_MIN_PRICE_VS_MA50 or trend_short <= Config.ENTRY_FILTER_MIN_TREND):
                logger.info("⛔ Long entry rejected: trend or price vs MA50 not aligned")
                return False

            if direction == 'SHORT' and (price_vs_ma50 >= -Config.ENTRY_FILTER_MIN_PRICE_VS_MA50 or trend_short >= -Config.ENTRY_FILTER_MIN_TREND):
                logger.info("⛔ Short entry rejected: trend or price vs MA50 not aligned")
                return False

        if Config.ENTRY_FILTER_MIN_VOLUME_RATIO is not None:
            volume_ratio = float(feature_row.get('volume_ratio', 0.0))
            if volume_ratio < Config.ENTRY_FILTER_MIN_VOLUME_RATIO:
                logger.info("⛔ Entry rejected: volume ratio below required threshold")
                return False

        if Config.SENTIMENT_ENABLED and 'sentiment_score' in feature_row:
            sentiment_score = float(feature_row.get('sentiment_score', 0.0))
            mention_count = float(feature_row.get('mention_count', 0.0))

            if mention_count < Config.SENTIMENT_MIN_MENTION_COUNT:
                logger.info("⛔ Entry rejected: sentiment mention volume too low")
                return False

            if direction == 'LONG' and sentiment_score < Config.SENTIMENT_MIN_SCORE_LONG:
                logger.info("⛔ Long entry rejected: sentiment is not bullish enough")
                return False

            if direction == 'SHORT' and sentiment_score > Config.SENTIMENT_MAX_SCORE_SHORT:
                logger.info("⛔ Short entry rejected: sentiment is not bearish enough")
                return False

        return True

    def apply_on_demand_sentiment(self, candidates):
        if not Config.SENTIMENT_ENABLED or not getattr(Config, 'SENTIMENT_ON_DEMAND_ENABLED', False):
            return candidates

        limit = min(len(candidates), int(getattr(Config, 'SENTIMENT_CANDIDATE_LIMIT', 5)))
        checked = []
        for candidate in candidates[:limit]:
            symbol = candidate['symbol']
            self.ensure_recent_sentiment(symbol)
            _, _, feature_row = self.predict_direction(symbol, include_sentiment=True)
            if feature_row is None or 'sentiment_score' not in feature_row:
                if getattr(Config, 'SENTIMENT_REQUIRE_FOR_TRADE', False):
                    logger.info(f"⛔ {symbol} rejected: sentiment unavailable")
                    continue
                checked.append(candidate)
                continue

            if self.is_sentiment_allowed(feature_row, candidate['direction']):
                candidate['feature_row'] = feature_row
                candidate['sentiment_score'] = float(feature_row.get('sentiment_score', 0.0))
                candidate['mention_count'] = float(feature_row.get('mention_count', 0.0))
                checked.append(candidate)

        if checked:
            return checked

        return [] if getattr(Config, 'SENTIMENT_REQUIRE_FOR_TRADE', False) else candidates

    def is_sentiment_allowed(self, feature_row, direction):
        sentiment_score = float(feature_row.get('sentiment_score', 0.0))
        mention_count = float(feature_row.get('mention_count', 0.0))

        if mention_count < Config.SENTIMENT_MIN_MENTION_COUNT:
            logger.info("⛔ Entry rejected: sentiment mention volume too low")
            return False

        if direction == 'LONG' and sentiment_score < Config.SENTIMENT_MIN_SCORE_LONG:
            logger.info("⛔ Long entry rejected: sentiment is not bullish enough")
            return False

        if direction == 'SELL' and sentiment_score > Config.SENTIMENT_MAX_SCORE_SHORT:
            logger.info("⛔ Short entry rejected: sentiment is not bearish enough")
            return False

        return True

    def ensure_recent_sentiment(self, symbol):
        if not Config.SENTIMENT_ENABLED:
            return False

        sentiment_path = self.data_handler._get_sentiment_path(symbol)
        refresh_hours = float(getattr(Config, 'SENTIMENT_REFRESH_HOURS', 4))
        if sentiment_path and sentiment_path.exists():
            age_hours = (time.time() - sentiment_path.stat().st_mtime) / 3600
            if age_hours < refresh_hours:
                return True

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=int(getattr(Config, 'SENTIMENT_QUERY_LOOKBACK_DAYS', 1)))
        query_template = "({aliases}) -is:retweet lang:en"
        cmd = [
            os.environ.get('PYTHON', 'python'),
            'sentiment_collector.py',
            '--gmx-dir', str(Config.GMX_OHLC_DIR),
            '--gmx-timeframe', Config.TIMEFRAME,
            '--symbols', symbol,
            '--source', 'twitter',
            '--start', start_date.isoformat(),
            '--end', end_date.isoformat(),
            '--interval', Config.TIMEFRAME,
            '--max-tweets', str(int(getattr(Config, 'SENTIMENT_MAX_TWEETS', 10))),
            '--query-template', query_template,
            '--force',
        ]

        try:
            logger.info(f"Fetching on-demand sentiment for {symbol}")
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, text=True, capture_output=True, timeout=90)
            if result.stdout:
                logger.info(result.stdout.strip())
            if result.returncode != 0:
                logger.warning(f"Sentiment fetch failed for {symbol}: {result.stderr.strip()}")
                return False
            return True
        except Exception as exc:
            logger.warning(f"Sentiment fetch failed for {symbol}: {exc}")
            return False

    def monitor_position(self):
        try:
            if not self.trader.current_position:
                return
            
            current_price = self.trader.get_current_price(self.trader.position_symbol)
            if current_price is None:
                return
            
            if self.current_atr:
                self.trader.update_trailing_stop(current_price, self.current_atr)
            
            position_info = self.trader.get_position_info()
            if position_info:
                self.notifier.send_position_update(position_info)
            
            pnl = self.trader.calculate_pnl(current_price)
            pnl_usd = self.trader.calculate_pnl_usd(current_price)
            if self.trader.check_exit_conditions(current_price):
                emoji = "✅" if pnl > 0 else "❌"
                self.notifier.send_message(f"{emoji} Position Closed: `{pnl:.2f}% / ${pnl_usd:.2f}`")
            
        except Exception as e:
            logger.error(f"Monitoring Error: {e}")
    
    def send_periodic_status(self):
        """Sends Automatic Status Reports"""
        self.notifier.send_status_report(self)
    
    def check_commands(self):
        """Checks Telegram Commands"""
        self.notifier.process_commands(self)
    
    def run(self):
        """Main Loop with Mobile Monitoring"""

        # Trading Signals Every Hour
        for hour in range(24):
            schedule.every().day.at(f"{hour:02d}:00").do(self.execute_trading_logic)
        
        # Position Monitoring Every 5 Minutes
        schedule.every(Config.MONITOR_INTERVAL_MINUTES).minutes.do(self.monitor_position)
        
        # 🔥 NEW: Verify Synchronization Every 10 Minutes
        schedule.every(10).minutes.do(lambda: self.trader.verify_position_sync())
        
        # Automatic Report Every 6 Hours
        schedule.every(6).hours.do(self.send_periodic_status)
        
        # Check Commands Every 30 Seconds
        schedule.every(30).seconds.do(self.check_commands)
            
        logger.info("🚀 Bot Running...")
        self.notifier.send_message(
            "✅ Bot Operating Automatically\n\n"
            "📱 *Available Commands:*\n"
            "`/status` - View Status\n"
            "`/balance` - View Balance\n"
            "`/position` - View Current Position\n"
            "`/help` - Help"
        )
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Bot Stopped by User")
                self.notifier.send_message("🛑 Bot detenido")
                break
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(60)


# ======================== BACKTEST ========================

def parse_args():
    parser = ArgumentParser(description="GMX 15m LSTM trading bot")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest instead of live scheduled bot")
    parser.add_argument("--all-assets", action="store_true", help="Backtest every GMX asset for the selected timeframe")
    parser.add_argument("--symbol", default=Config.GMX_SYMBOL, help="GMX symbol to backtest")
    parser.add_argument("--timeframe", default=Config.TIMEFRAME, help="GMX CSV timeframe, e.g. 5m or 15m")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting capital")
    return parser.parse_args()


def run_lstm_backtest_for_symbol(model, data_handler, symbol, timeframe, starting_capital):
    df_raw = load_gmx_ohlc(symbol, timeframe)
    df = data_handler.prepare_features(df_raw)
    df['ATR'] = data_handler.calculate_atr(df_raw, Config.ATR_PERIOD)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) <= Config.SEQUENCE_LENGTH + 1:
        raise ValueError(f"Insufficient Data for {symbol}: {len(df)} velas limpias")

    features = df[Config.FEATURE_COLUMNS].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = data_handler.normalize_data(features)

    seq_len = Config.SEQUENCE_LENGTH
    X = np.array([scaled[i - seq_len:i] for i in range(seq_len, len(scaled))])
    probs = model.predict(X, verbose=0).reshape(-1)
    signal_index = df.index[seq_len:]

    signals = pd.DataFrame(index=signal_index)
    signals['prob'] = probs
    signals['Close'] = df.loc[signal_index, 'Close']
    signals['High_next'] = df['High'].shift(-1).loc[signal_index]
    signals['Low_next'] = df['Low'].shift(-1).loc[signal_index]
    signals['Close_next'] = df['Close'].shift(-1).loc[signal_index]
    signals['ATR'] = df.loc[signal_index, 'ATR']
    signals = signals.dropna()

    capital = starting_capital
    trade_log = []
    equity = []
    threshold = Config.MIN_SIGNAL_THRESHOLD

    for timestamp, row in signals.iterrows():
        probability = float(row['prob'])
        entry_price = float(row['Close'])
        atr = float(row['ATR'])

        if probability > threshold:
            direction = 'LONG'
            stop_loss = entry_price - atr * Config.ATR_SL_MULTIPLIER
            take_profit = entry_price + atr * Config.ATR_TP_MULTIPLIER
        elif probability < (1 - threshold):
            direction = 'SHORT'
            stop_loss = entry_price + atr * Config.ATR_SL_MULTIPLIER
            take_profit = entry_price - atr * Config.ATR_TP_MULTIPLIER
        else:
            equity.append((timestamp, capital))
            continue

        stop_distance = abs(entry_price - stop_loss)
        risk_amount = capital * (Config.RISK_PERCENTAGE / 100)
        risk_size = risk_amount / stop_distance if stop_distance else 0
        max_affordable = capital / entry_price
        units = min(risk_size, max_affordable * 0.95)
        if units <= 0:
            equity.append((timestamp, capital))
            continue

        high_next = float(row['High_next'])
        low_next = float(row['Low_next'])
        close_next = float(row['Close_next'])

        if direction == 'LONG':
            if low_next <= stop_loss:
                exit_price = stop_loss
                reason = 'Stop Loss'
            elif high_next >= take_profit:
                exit_price = take_profit
                reason = 'Take Profit'
            else:
                exit_price = close_next
                reason = 'Next Candle Close'
            profit = (exit_price - entry_price) * units
        else:
            if high_next >= stop_loss:
                exit_price = stop_loss
                reason = 'Stop Loss'
            elif low_next <= take_profit:
                exit_price = take_profit
                reason = 'Take Profit'
            else:
                exit_price = close_next
                reason = 'Next Candle Close'
            profit = (entry_price - exit_price) * units

        capital += profit
        trade_log.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'probability': probability,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'units': units,
            'profit': profit,
            'capital': capital,
            'reason': reason,
        })
        equity.append((timestamp, capital))

    trades = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity, columns=['timestamp', 'equity']).set_index('timestamp')
    if trades.empty:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': len(df),
            'predictions': len(signals),
            'signals_traded': 0,
            'start': signals.index.min() if not signals.empty else None,
            'end': signals.index.max() if not signals.empty else None,
            'starting_capital': starting_capital,
            'final_capital': starting_capital,
            'profit': 0.0,
            'return_pct': 0.0,
            'win_rate_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'longs': 0,
            'shorts': 0,
            'stop_losses': 0,
            'take_profits': 0,
            'window_exits': 0,
        }, trades

    wins = trades[trades['profit'] > 0]
    drawdown = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min() * 100
    summary = {
        'symbol': symbol,
        'timeframe': timeframe,
        'candles': len(df),
        'predictions': len(signals),
        'signals_traded': len(trades),
        'start': signals.index.min(),
        'end': signals.index.max(),
        'starting_capital': starting_capital,
        'final_capital': capital,
        'profit': capital - starting_capital,
        'return_pct': (capital / starting_capital - 1) * 100,
        'win_rate_pct': len(wins) / len(trades) * 100,
        'max_drawdown_pct': drawdown,
        'longs': int((trades['direction'] == 'LONG').sum()),
        'shorts': int((trades['direction'] == 'SHORT').sum()),
        'stop_losses': int((trades['reason'] == 'Stop Loss').sum()),
        'take_profits': int((trades['reason'] == 'Take Profit').sum()),
        'window_exits': int((trades['reason'] == 'Next Candle Close').sum()),
    }
    return summary, trades


def run_backtest(args):
    symbols = list_gmx_symbols(args.timeframe) if args.all_assets else [args.symbol]
    if not symbols:
        raise FileNotFoundError(f"No GMX {args.timeframe} files found in {Config.GMX_OHLC_DIR}")

    summaries = []
    all_trades = []
    model_path = Config.MODEL_DIR / Config.MODEL_NAME
    scaler_path = Config.MODEL_DIR / Config.SCALER_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    for symbol in symbols:
        try:
            model = load_model(model_path)
            data_handler = DataHandler(scaler_path)

            summary, trades = run_lstm_backtest_for_symbol(
                model,
                data_handler,
                symbol,
                args.timeframe,
                args.capital,
            )
            summaries.append(summary)
            if not trades.empty:
                all_trades.append(trades)
        except Exception as exc:
            summaries.append({
                'symbol': symbol,
                'timeframe': args.timeframe,
                'error': str(exc),
            })

    summary_df = pd.DataFrame(summaries)
    if 'return_pct' in summary_df.columns:
        summary_df = summary_df.sort_values('return_pct', ascending=False, na_position='last')

    prefix = f"trading_bot_gmx_{args.timeframe}"
    if args.all_assets:
        prefix += "_all_assets"
    else:
        prefix += f"_{args.symbol}"

    summary_path = f"{prefix}_summary.csv"
    trades_path = f"{prefix}_trades.csv"
    summary_df.to_csv(summary_path, index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False)

    print(f"Backtested {len(symbols)} asset(s) for {args.timeframe}")
    print(f"Summary saved to {summary_path}")
    if all_trades:
        print(f"Trades saved to {trades_path}")

    display_columns = [
        'symbol',
        'return_pct',
        'final_capital',
        'profit',
        'signals_traded',
        'win_rate_pct',
        'max_drawdown_pct',
    ]
    available_columns = [col for col in display_columns if col in summary_df.columns]
    print(summary_df[available_columns].head(20).to_string(index=False))

# ======================== MAIN ========================

if __name__ == "__main__":
    try:
        args = parse_args()
        if args.backtest:
            run_backtest(args)
            exit(0)

        print("\n" + "="*60)
        print("🤖 GMX 15m LSTM TRADING BOT v3.3")
        print("   🔥 Uses All 18 Features Correctly")
        print("   🔥 Clasificador binario optimizado")
        print("="*60 + "\n")
        
        Config.validate()
        logger.info("✅ Credenciales validadas")
        
        bot = TradingBot()
        bot.run()
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n{e}\n")
        exit(1)
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        print(f"\n❌ Error: {e}\n")
        exit(1)
