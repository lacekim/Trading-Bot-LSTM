#!/usr/bin/env python3
"""
CNN-LSTM for Trading - FIXED BINARY CLASSIFICATION v3.3
Predicts direction (up/down) with a threshold
18 features plus infinite-value cleanup
Optimized for: ADA-USD 1h
"""

import sys
import os
import logging
import warnings
from datetime import datetime
from pathlib import Path
import pickle
import requests

import numpy as np
import pandas as pd

Path('logs/matplotlib').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(Path('logs/matplotlib').resolve()))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from argparse import ArgumentParser
from config import Config
from kraken_data import load_kraken_ohlc

warnings.filterwarnings('ignore')


def gmx_ohlc_path(symbol=None, timeframe=None):
    symbol = symbol or Config.GMX_SYMBOL
    timeframe = timeframe or Config.TIMEFRAME
    return Path(Config.GMX_OHLC_DIR) / f"gmx_arbitrum_{symbol.upper()}_{timeframe}.csv"


def list_gmx_symbols(timeframe=None):
    timeframe = timeframe or Config.TIMEFRAME
    suffix = f"_{timeframe}.csv"
    symbols = []
    for path in Path(Config.GMX_OHLC_DIR).glob(f"gmx_arbitrum_*{suffix}"):
        symbol = path.stem.removeprefix("gmx_arbitrum_").removesuffix(suffix)
        symbols.append(symbol)
    return sorted(symbols)


def load_gmx_ohlc(symbol=None, timeframe=None):
    path = gmx_ohlc_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"GMX OHLC not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'open_time': 'Date',
        }
    )

    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required_columns)
    df = df.set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

sns.set_style('darkgrid')

Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        
    def send_message(self, text):
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {'chat_id': self.chat_id, 'text': text, 'parse_mode': 'Markdown'}
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"✔️ Telegram: {text[:50]}...")
        except Exception as e:
            logger.error(f"❌ Error send_message: {e}")
    
    def send_photo(self, photo_path, caption=""):
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id, 'caption': caption}
                response = requests.post(url, files=files, data=data, timeout=30)
            if response.status_code == 200:
                logger.info(f"✔️ Image sent: {photo_path}")
        except Exception as e:
            logger.error(f"❌ Error send_photo: {e}")

class DataHandler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.last_price = None

    def fetch_data(self, symbol, period, interval='1h'):
        try:
            if getattr(Config, 'DATA_SOURCE', '').upper() == 'GMX':
                asset_symbol = symbol or Config.GMX_SYMBOL
                logger.info(f"Loading GMX OHLC {asset_symbol} - {period} - {interval}...")
                df = load_gmx_ohlc(asset_symbol, interval)
                if df.empty:
                    raise ValueError("No GMX data")
                if len(df) < Config.MIN_TRAIN_CANDLES:
                    raise ValueError(
                        f"Insufficient data to train: {len(df)} candles "
                        f"(< {Config.MIN_TRAIN_CANDLES}). Backfill the historical gap."
                    )
                logger.info(f"✔️ {len(df)} candles loaded for {asset_symbol}")
                return df

            logger.info(f"Loading Kraken OHLC {Config.KRAKEN_PAIR} - {period} - {interval}...")
            df = load_kraken_ohlc(
                Config.KRAKEN_PAIR,
                period,
                interval,
                prefer_live=False,
                require_contiguous=True,
                logger=logger
            )
            if df.empty:
                raise ValueError("No data")
            if len(df) < Config.MIN_TRAIN_CANDLES:
                raise ValueError(
                    f"Insufficient data to train: {len(df)} contiguous candles "
                    f"(< {Config.MIN_TRAIN_CANDLES}). Backfill the historical gap."
                )
            logger.info(f"✔️ {len(df)} candles loaded")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def calculate_atr(self, df, period=14):
        """Calculate ATR and return a pandas Series"""
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
    
    def prepare_features(self, df, prediction_horizon=1):
        """
        Improved features + threshold target + infinite-value cleanup
        
        v3.3: target = moves >1%, not >0%.
        """
        df = df.copy()
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure all columns are Series
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
        
        # ========== EXISTING FEATURES ==========
        
        # 1. Log returns
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Rolling volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 3. Normalized RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi'] - 50) / 50
        
        # 4. Normalized MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        macd_std = df['macd_diff'].rolling(50).std()
        df['macd_norm'] = (df['macd_diff'] - df['macd_diff'].rolling(50).mean()) / (macd_std + 1e-10)
        
        # 5. Momentum
        df['momentum'] = df['Close'].pct_change(periods=24)
        
        # 6. Relative volume
        volume_ma = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / (volume_ma + 1e-10)
        
        # 7. Normalized ATR
        atr_values = df['ATR']
        if isinstance(atr_values, pd.DataFrame):
            atr_values = atr_values.iloc[:, 0]
        
        close_values = df['Close']
        if isinstance(close_values, pd.DataFrame):
            close_values = close_values.iloc[:, 0]
        
        df['atr_norm'] = atr_values.values / (close_values.values + 1e-10)
        
        # 8. Buying pressure
        df['green_candles'] = (df['Close'] > df['Open']).astype(int).rolling(10).mean()
        
        # 9. Relative price range
        df['price_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
        
        # 10. Directional volume
        volume_directional = np.where(
            df['Close'].values > df['Open'].values, 
            df['Volume'].values, 
            -df['Volume'].values
        )
        if volume_directional.ndim > 1:
            volume_directional = volume_directional.flatten()
        
        df['volume_directional'] = pd.Series(volume_directional, index=df.index).rolling(10).mean()
        
        # 11. Bollinger Bands position
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        bb_middle = close_series.rolling(20).mean()
        bb_std = close_series.rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        bb_range = bb_upper - bb_lower
        df['bb_position'] = (close_series - bb_lower) / (bb_range + 1e-10)
        
        # 12. OBV simplificado
        obv_direction = np.sign(df['returns'].fillna(0).values)
        volume_values = df['Volume'].values if not isinstance(df['Volume'], pd.DataFrame) else df['Volume'].iloc[:, 0].values
        
        df['volume_obv'] = pd.Series(volume_values * obv_direction, index=df.index).rolling(10).sum()
        
        # ========== FEATURES NUEVOS v3.3 🔥 ==========
        
        # 13. Precio vs MA 50 (tendencia)
        ma50 = df['Close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['Close'] - ma50) / (ma50 + 1e-10)
        
        # 14. Cambio de volumen
        df['volume_change'] = df['Volume'].pct_change(5).fillna(0)
        
        # 15. High-Low ratio (volatilidad intraday)
        df['hl_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
        
        # 16. Close position en rango
        hl_range = df['High'] - df['Low']
        df['close_position'] = (df['Close'] - df['Low']) / (hl_range + 1e-10)
        
        # 17. Tendencia corto plazo
        ma5 = df['Close'].rolling(5).mean()
        ma20 = df['Close'].rolling(20).mean()
        df['trend_short'] = (ma5 / (ma20 + 1e-10)) - 1
        
        # 18. Tendencia largo plazo
        ma50_long = df['Close'].rolling(50).mean()
        df['trend_long'] = (ma20 / (ma50_long + 1e-10)) - 1
        
        # ========== IMPROVED TARGET ==========
        
        # Calculate future return
        df['future_return'] = df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        
        # Threshold target: only significant moves
        df['target'] = (df['future_return'] > Config.MOVEMENT_THRESHOLD).astype(int)
        
        # Critical cleanup: replace infinities and NaN
        # First replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Count NaN/inf before cleanup
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            logger.info(f"🔧 Cleaning {nan_counts.sum()} NaN/infinite values")
        
        # Clean NaNs
        df = df.dropna()
        
        # Log distribution
        n_up = df['target'].sum()
        n_down = len(df) - n_up
        pct_up = n_up / len(df) * 100
        
        logger.info(f"📊 Target distribution (>{Config.MOVEMENT_THRESHOLD*100:.1f}%):")
        logger.info(f"   Significant up moves: {n_up} ({pct_up:.1f}%)")
        logger.info(f"   Not up moves:         {n_down} ({100-pct_up:.1f}%)")
        
        # Warn if classes are heavily imbalanced
        if pct_up < 30 or pct_up > 70:
            logger.warning(f"⚠️ Imbalanced classes ({pct_up:.1f}% / {100-pct_up:.1f}%)")
            logger.warning("   Consider adjusting MOVEMENT_THRESHOLD")
        
        return df
    
    def prepare_sequences(self, data, target, seq_length):
        """Create feature sequences and binary targets"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            end_ix = i + seq_length
            if end_ix > len(data) - 1:
                break
            
            seq_x = data[i:end_ix]
            seq_y = target[end_ix]
            
            X.append(seq_x)
            y.append(seq_y)
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, data, fit=True):
        """Normalize with infinite-value validation"""
        # Validate that there are no infinities before normalizing
        if not np.isfinite(data).all():
            logger.error("❌ CRITICAL: Infinities found in data before normalization")
            # Replace infinities with NaN and then 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.warning("⚠️ Infinities replaced with 0")
        
        if fit:
            scaled = self.scaler.fit_transform(data)
        else:
            scaled = self.scaler.transform(data)
        
        return scaled

class CNNLSTMClassifier:
    """
    Fixed CNN-LSTM binary classification model.
    Output: probability that price rises [0-1].
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        model = Sequential([
            # Small CNN
            Conv1D(
                filters=32,
                kernel_size=3,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(0.01),
                input_shape=input_shape
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),

            # 2 LSTMs
            LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            
            LSTM(16, return_sequences=False, kernel_regularizer=l2(0.001)),
            Dropout(0.5),

            Dense(16, activation='relu'),
            Dropout(0.3),
            
            # Fixed binary output
            Dense(1, activation='sigmoid')  # 1 neuron + sigmoid for binary classification
        ])
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            clipnorm=1.0
        )
        
        # Binary crossentropy for binary classification
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        logger.info("✔️ CNN-LSTM Classifier model built")
        logger.info(f"   Total parameters: {model.count_params():,}")
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train with class weights for balancing"""
        
        # Calculate class weights
        n_samples = len(y_train)
        n_positive = np.sum(y_train)
        n_negative = n_samples - n_positive
        
        class_weight = {
            0: n_samples / (2 * n_negative) if n_negative > 0 else 1.0,
            1: n_samples / (2 * n_positive) if n_positive > 0 else 1.0
        }
        
        logger.info(f"Class weights: {class_weight}")
        
        early_stop = EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        logger.info("🚀 Starting training...")
        logger.info(f"   Train samples: {len(X_train)}")
        logger.info(f"   Val samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight,
            verbose=1
        )
        
        logger.info("✔️ Training completed")
        return self.history
    
    def predict(self, X):
        """Return probabilities [0, 1]"""
        return self.model.predict(X, verbose=0)
    
    def save_model(self, path):
        self.model.save(path)
        logger.info(f"✔️ Model saved: {path}")

class Visualizer:
    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Binary Crossentropy Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline')
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(history.history['auc'], label='Train AUC', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        axes[1, 0].set_title('AUC-ROC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Chart saved: {save_path}")
    
    @staticmethod
    def plot_predictions(y_true, y_pred_proba, save_path='predictions.png'):
        """Visualize binary classification"""
        
        # Convert probabilities to predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Probabilities vs time
        axes[0, 1].plot(y_pred_proba, label='Up Probability', alpha=0.7, linewidth=1)
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].scatter(np.where(y_true == 1)[0], [0.9]*np.sum(y_true), 
                          color='green', alpha=0.3, s=10, label='Actual: Up')
        axes[0, 1].scatter(np.where(y_true == 0)[0], [0.1]*np.sum(1-y_true), 
                          color='red', alpha=0.3, s=10, label='Actual: Down')
        axes[0, 1].set_title('Predicted Probabilities', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Probability distribution
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.6, 
                       color='green', label='Actual Up')
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.6, 
                       color='red', label='Actual Down')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Probability Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics by threshold
        thresholds = np.linspace(0.3, 0.7, 50)
        accuracies = []
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba > threshold).astype(int)
            acc = accuracy_score(y_true, y_pred_thresh)
            accuracies.append(acc)
        
        axes[1, 1].plot(thresholds, accuracies, linewidth=2)
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Probability Threshold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Chart saved: {save_path}")

def validate_classifier(y_true, y_pred_proba, threshold=0.5):
    """Robust validation for binary classifier"""
    
    logger.info("\n" + "="*60)
    logger.info("🔍 BINARY CLASSIFIER VALIDATION")
    logger.info("="*60)
    
    # Convert to binary predictions
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    
    # Prediction distribution
    pct_pred_up = np.mean(y_pred) * 100
    pct_true_up = np.mean(y_true) * 100
    
    logger.info(f"\n📊 Metrics (threshold={threshold}):")
    logger.info(f"   Accuracy:  {accuracy:.1f}%")
    logger.info(f"   Precision: {precision:.1f}%")
    logger.info(f"   Recall:    {recall:.1f}%")
    logger.info(f"   F1-Score:  {f1:.1f}%")
    logger.info(f"\n   Actual:    {pct_true_up:.1f}% up")
    logger.info(f"   Predicted: {pct_pred_up:.1f}% up")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"   TN: {tn} | FP: {fp}")
    logger.info(f"   FN: {fn} | TP: {tp}")
    
    # Evaluation
    if accuracy < 52:
        logger.warning("   ❌ CRITICAL: Accuracy < 52% (worse than random)")
        status = "CRITICAL"
    elif accuracy < 55:
        logger.warning("   ⚠️ ALERTA: Accuracy marginal")
        status = "WARNING"
    elif accuracy < 60:
        logger.info("   ✔️ Acceptable accuracy")
        status = "OK"
    else:
        logger.info("   ✔️✔️ Very good accuracy!")
        status = "EXCELLENT"
    
    # Simulated Sharpe
    strategy_returns = np.where(y_pred == 1, 
                               np.where(y_true == 1, 1, -1),
                               np.where(y_true == 0, 1, -1))
    
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        logger.info(f"\n💰 Simulated Sharpe Ratio: {sharpe:.2f}")
    else:
        sharpe = 0
    
    logger.info("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sharpe': sharpe,
        'status': status,
        'cm': cm
    }

def parse_args():
    parser = ArgumentParser(description="Train the fixed CNN-LSTM classifier")
    parser.add_argument("--timeframe", default=Config.TIMEFRAME, help="Timeframe for the training data")
    parser.add_argument("--lookback", default=Config.LOOKBACK_PERIOD, help="Historical lookback period")
    parser.add_argument("--no-telegram", action='store_true', help="Do not send Telegram notifications")
    return parser.parse_args()


def train_model(timeframe=None, lookback=None, send_telegram=True):
    try:
        Config.MODEL_DIR.mkdir(exist_ok=True)

        asset_symbol = Config.GMX_SYMBOL if Config.DATA_SOURCE.upper() == 'GMX' else Config.SYMBOL
        notifier = TelegramNotifier() if send_telegram else None
        if notifier:
            notifier.send_message(
                f"🔥 *CLASSIFIER TRAINING v3.3*\n\n"
                f"Asset: `{asset_symbol}`\n"
                f"🎯 Objective: Predict direction >1%\n"
                f"🔥 18 features + infinite cleanup\n"
                f"🔥 Fixed binary architecture\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

        data_handler = DataHandler()
        model = CNNLSTMClassifier()
        visualizer = Visualizer()

        timeframe = timeframe or Config.TIMEFRAME
        lookback = lookback or Config.LOOKBACK_PERIOD

        # Download data
        df = data_handler.fetch_data(
            asset_symbol,
            period=lookback,
            interval=timeframe
        )

        if df is None or df.empty:
            raise ValueError("No data was obtained")

        logger.info(f"📊 Data: {len(df)} candles ({lookback}) for {asset_symbol}")

        # Calculate ATR
        atr_result = data_handler.calculate_atr(df, Config.ATR_PERIOD)
        if isinstance(atr_result, pd.DataFrame):
            df['ATR'] = atr_result.iloc[:, 0]
        else:
            df['ATR'] = atr_result

        current_atr = df['ATR'].iloc[-1]

        # Prepare features and binary target
        df = data_handler.prepare_features(df, prediction_horizon=1)
        logger.info(f"✔️ Features calculated: {len(df)} valid rows")

        # Store last price
        data_handler.last_price = float(df['Close'].iloc[-1])

        # Select features (18 total)
        feature_cols = Config.FEATURE_COLUMNS
        logger.info(f"📊 Using {len(feature_cols)} features")

        features = df[feature_cols].values
        target = df['target'].values

        # Critical validation before normalization
        if not np.isfinite(features).all():
            logger.error("❌ CRITICAL: Features contain infinities or NaN")
            nan_mask = ~np.isfinite(features)
            logger.error(f"   Location: {np.where(nan_mask)}")
            raise ValueError("Features contain invalid values after prepare_features")

        # Split before normalizing
        split_idx = int(len(features) * Config.TRAIN_SPLIT)
        train_features = features[:split_idx]
        val_features = features[split_idx:]
        train_target = target[:split_idx]
        val_target = target[split_idx:]

        logger.info(f"📊 Split: Train={len(train_features)}, Val={len(val_features)}")

        # Normalize
        train_scaled = data_handler.normalize_data(train_features, fit=True)
        val_scaled = data_handler.normalize_data(val_features, fit=False)

        # Create sequences
        X_train, y_train = data_handler.prepare_sequences(
            train_scaled, train_target, Config.SEQUENCE_LENGTH
        )
        X_val, y_val = data_handler.prepare_sequences(
            val_scaled, val_target, Config.SEQUENCE_LENGTH
        )

        logger.info(f"📈 Sequences: Train={len(X_train)}, Val={len(X_val)}")

        # Build and train
        model.build_model(input_shape=(Config.SEQUENCE_LENGTH, len(feature_cols)))
        history = model.train(X_train, y_train, X_val, y_val)

        # Predictions
        y_pred_proba = model.predict(X_val).flatten()

        # Validation
        validation_results = validate_classifier(y_val, y_pred_proba, threshold=0.5)

        # Generate charts
        history_path = 'training_history.png'
        predictions_path = 'predictions.png'
        visualizer.plot_training_history(history, history_path)
        visualizer.plot_predictions(y_val, y_pred_proba, predictions_path)

        # Save model
        model_path = Config.MODEL_DIR / Config.MODEL_NAME
        scaler_path = Config.MODEL_DIR / Config.SCALER_NAME
        metadata_path = Config.MODEL_DIR / 'model_metadata.pkl'

        model.save_model(model_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(data_handler.scaler, f)

        metadata = {
            'symbol': asset_symbol,
            'last_price': data_handler.last_price,
            'feature_cols': feature_cols,
            'atr': float(current_atr),
            'model_type': 'binary_classifier',
            'version': 'v3.3'
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"✔️ Everything saved correctly for {asset_symbol}")

        # Telegram message
        epochs_trained = len(history.history['loss'])
        acc = validation_results['accuracy']
        prec = validation_results['precision']
        rec = validation_results['recall']
        sharpe = validation_results['sharpe']
        status = validation_results['status']

        if status == "CRITICAL":
            emoji = "🚫"
            text = "DO NOT USE"
        elif status == "WARNING":
            emoji = "⚠️"
            text = "REVIEW"
        elif status == "OK":
            emoji = "✔️"
            text = "APPROVED"
        else:
            emoji = "✔️✔️"
            text = "EXCELLENT"

        metrics_text = (
            f"{emoji} *{text}*\n\n"
            f"🎯 *Asset:* `{asset_symbol}`\n"
            f"📊 *Data:* `{len(df)} candles`\n"
            f"📍 *Samples:* Train `{len(X_train)}` | Val `{len(X_val)}`\n\n"
            f"📈 *CNN-LSTM Binary Classifier v3.3*\n"
            f"   • Conv1D: 32 + BN\n"
            f"   • LSTM: 32+16\n"
            f"   • Params: `{model.model.count_params():,}`\n"
            f"   • Features: {len(feature_cols)} 🔥\n"
            f"   • Output: Sigmoid [0,1]\n\n"
            f"🏋️ *Training:*\n"
            f"   • Epochs: `{epochs_trained}/{Config.EPOCHS}`\n"
            f"   • Loss: Binary Crossentropy\n"
            f"   • Target: >{Config.MOVEMENT_THRESHOLD*100:.0f}% move\n\n"
            f"📊 *Classification Metrics:*\n"
            f"   • Accuracy:  `{acc:.1f}%`\n"
            f"   • Precision: `{prec:.1f}%`\n"
            f"   • Recall:    `{rec:.1f}%`\n"
            f"   • F1-Score:  `{validation_results['f1']:.1f}%`\n"
            f"   • Sharpe:    `{sharpe:.2f}`\n\n"
            f"🎯 *Parameters:*\n"
            f"   • ATR: `{current_atr:.4f}`\n"
            f"   • Price: `${data_handler.last_price:.4f}`"
        )

        if status == "CRITICAL":
            metrics_text += "\n\n🚫 *DO NOT USE - Insufficient accuracy*"
        elif status == "WARNING":
            metrics_text += "\n\n⚠️ *2 months of paper trading required*"
        elif status == "OK":
            metrics_text += "\n\n✔️ *1 month of paper trading recommended*"
        else:
            metrics_text += "\n\n✔️✔️ *Ready for paper trading*"

        if notifier:
            notifier.send_message(metrics_text)
            import time
            time.sleep(2)
            notifier.send_photo(history_path, '📈 Classifier Training')
            time.sleep(2)
            notifier.send_photo(predictions_path, '🎯 Confusion Matrix')

        logger.info("✔️ Process completed")
        return True

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if notifier:
            try:
                notifier.send_message(f"❌ Error:\n```\n{str(e)}\n```")
            except:
                pass
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 FIXED CNN-LSTM BINARY CLASSIFIER v3.3")
    print("   🎯 Predicts: Will it rise >1%?")
    print("   🔥 18 features + infinite cleanup")
    print("   ✔️ Correct binary architecture")
    print("="*60 + "\n")
    
    try:
        args = parse_args()
        Config.validate(skip_telegram=args.no_telegram)
        logger.info("✔️ Credentials validated")

        success = train_model(
            timeframe=args.timeframe,
            lookback=args.lookback,
            send_telegram=not args.no_telegram
        )

        if success:
            print("\n✔️ Entrenamiento completado\n")
            sys.exit(0)
        else:
            print("\n❌ Entrenamiento fallido\n")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        print(f"\n❌ Error fatal: {e}\n")
        sys.exit(1)
