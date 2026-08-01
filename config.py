"""
Configuration for trading with GMX 15m data.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Optimized configuration for using GMX 15m as the data source"""
    
    # ==================== ASSET ====================
    DATA_SOURCE = os.getenv('DATA_SOURCE', 'GMX')
    SYMBOL = os.getenv('SYMBOL', 'GMX')
    GMX_SYMBOL = os.getenv('GMX_SYMBOL', 'GMX')
    KRAKEN_PAIR = os.getenv('KRAKEN_PAIR', 'ADAUSD')
    
    # ==================== DATA ====================
    # Default timeframe used by the bot for fetching OHLC and aligning sentiment
    TIMEFRAME = '1h'
    LOOKBACK_PERIOD = '2y'
    DATA_DIR = Path('./data')
    GMX_OHLC_DIR = Path('/Users/mike/Documents/GitHub/Algorithmic-Trading-with-Deep-Learning/data/GMX_OHLCVT')
    TRADINGVIEW_HISTORY_DIR = Path(os.getenv('TRADINGVIEW_HISTORY_DIR', './data/TradingView_OHLCVT'))
    GMX_AUTO_REFRESH_ENABLED = os.getenv('GMX_AUTO_REFRESH_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
    GMX_AUTO_REFRESH_SECONDS = int(os.getenv('GMX_AUTO_REFRESH_SECONDS', '3300'))
    GMX_UPDATE_SCRIPT = Path('/Users/mike/Documents/GitHub/Algorithmic-Trading-with-Deep-Learning/update_gmx_data.py')
    GMX_UPDATE_CONFIG = Path('/Users/mike/Documents/GitHub/Algorithmic-Trading-with-Deep-Learning/config_gmx.json')
    GMX_UPDATE_CHAIN = os.getenv('GMX_UPDATE_CHAIN', 'arbitrum')
    GMX_UPDATE_TIMEOUT_SECONDS = int(os.getenv('GMX_UPDATE_TIMEOUT_SECONDS', '900'))
    GMX_SYMBOL_BLACKLIST = {
        'DAI',
        'SUSD',
        'USDC',
        'USDC.E',
        'USDE',
        'USDT',
    }
    KRAKEN_OHLC_DIR = Path('/Users/mike/Documents/GitHub/Algorithmic-Trading-with-Deep-Learning/data/Kraken_OHLCVT')
    KRAKEN_MAX_GAP_HOURS = 2
    MIN_TRAIN_CANDLES = 2000
    
    # ==================== MODEL ====================
    MODEL_DIR = Path('./models')
    MODEL_NAME = 'lstm_ada_model.h5'
    SCALER_NAME = 'scaler_ada.pkl'
    
    # Shorter sequence for 1h
    # 36 candles = 1.5 days, balancing context and overfitting.
    SEQUENCE_LENGTH = 36
    
    # Optimized architecture
    LSTM_UNITS = [128, 64]
    DROPOUT_RATE = 0.35
    BATCH_SIZE = 32
    
    TRAIN_SPLIT = 0.8
    EPOCHS = 300
    
    # ==================== TARGET ====================
    # Only significant moves
    # For 1h, 1% is a meaningful move.
    MOVEMENT_THRESHOLD = 0.01  # 1%
    
    # ==================== RISK MANAGEMENT ====================
    RISK_PERCENTAGE = 1.0
    DRY_RUN = os.getenv('DRY_RUN', 'true').lower() in ('1', 'true', 'yes', 'on')
    DRY_RUN_BALANCE_USD = float(os.getenv('DRY_RUN_BALANCE_USD', '1000'))
    DRY_RUN_STATE_PATH = Path(os.getenv('DRY_RUN_STATE_PATH', './data/dry_run_state.json'))
    DRY_RUN_TRADES_PATH = Path(os.getenv('DRY_RUN_TRADES_PATH', './data/dry_run_trades.csv'))
    TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
    
    # ==================== STOPS Y TARGETS ====================
    ATR_PERIOD = 14
    
    # Stops for 1h, tighter than 4h.
    ATR_SL_MULTIPLIER = 0.75  # Compromise between 0.5 and 1.0
    ATR_TP_MULTIPLIER = 1.5   # Ratio 1:2
    
    # ==================== TRAILING STOP ====================
    TRAILING_STOP_ACTIVATION = 2.5
    TRAILING_STOP_DISTANCE = 1.5
    
    # ==================== SIGNAL FILTERS ====================
    # Stricter threshold
    MIN_SIGNAL_THRESHOLD = 0.58  # 58% minimum confidence
    MIN_DIRECTIONAL_CONFIDENCE = 0.58

    # Keep live entries aligned with the model signal path; sentiment runs after candidates are found.
    ENABLE_ENTRY_CONTEXT_FILTER = False
    ENTRY_FILTER_MIN_VOLUME_RATIO = 1.0
    ENTRY_FILTER_REQUIRE_TREND = True
    ENTRY_FILTER_MIN_TREND = 0.0
    ENTRY_FILTER_MIN_PRICE_VS_MA50 = 0.0
    ENTRY_FILTER_MIN_PROBABILITY = 0.60  # Optional stronger filter for extra discipline

    # Social sentiment gating
    SENTIMENT_ENABLED = True
    SENTIMENT_ON_DEMAND_ENABLED = True
    SENTIMENT_CANDIDATE_LIMIT = 5
    SENTIMENT_MAX_TWEETS = 10
    SENTIMENT_REFRESH_HOURS = 4
    SENTIMENT_QUERY_LOOKBACK_DAYS = 1
    SENTIMENT_REQUIRE_FOR_TRADE = False
    SENTIMENT_CSV_PATH = Path('./data/social_sentiment.csv')
    SENTIMENT_CSV_DIR = Path('./data/sentiment')
    SENTIMENT_CSV_FILE_TEMPLATE = 'social_sentiment_{symbol}.csv'
    SENTIMENT_SYMBOL_ALIASES = {
        '0G': ['Zero Gravity', '0G Labs'],
        'AAVE': ['Aave'],
        'ADA': ['Cardano'],
        'AERO': ['Aerodrome'],
        'AI16Z': ['ai16z'],
        'AIXBT': ['aixbt'],
        'ALGO': ['Algorand'],
        'ANIME': ['Animecoin'],
        'APE': ['ApeCoin'],
        'APE_DEPRECATED': ['APE', 'ApeCoin'],
        'APT': ['Aptos'],
        'ARB': ['Arbitrum'],
        'AR': ['Arweave'],
        'ASTER': ['Aster'],
        'ATOM': ['Cosmos'],
        'AVAX': ['Avalanche'],
        'AVNT': ['Avantis'],
        'BCH': ['Bitcoin Cash'],
        'BERA': ['Berachain'],
        'BNB': ['Binance Coin', 'BNB Chain'],
        'BOME': ['Book of Meme'],
        'BONK': ['Bonk'],
        'BRENTOIL': ['Brent Oil', 'Brent Crude'],
        'BRETT': ['Brett'],
        'BTC': ['Bitcoin', 'XBT'],
        'CAKE': ['PancakeSwap'],
        'CC': ['CC'],
        'CHZ': ['Chiliz'],
        'CRO': ['Cronos'],
        'CRV': ['Curve DAO', 'Curve'],
        'CVX': ['Convex Finance', 'Convex'],
        'DAI': ['Dai'],
        'DASH': ['Dash'],
        'DOGE': ['Dogecoin'],
        'DOLO': ['Dolomite'],
        'DOT': ['Polkadot'],
        'DYDX': ['dYdX'],
        'EIGEN': ['EigenLayer'],
        'ENA': ['Ethena'],
        'ETH': ['Ethereum', 'Ether'],
        'FARTCOIN': ['Fartcoin'],
        'FET': ['Fetch.ai', 'Artificial Superintelligence Alliance'],
        'FIL': ['Filecoin'],
        'FLOKI': ['Floki'],
        'GLV [ETH-USDC]': ['GLV ETH USDC', 'ETH USDC GLV'],
        'GMX': ['GMX_IO', 'GMX exchange', 'GMX protocol'],
        'GOLD': ['Gold', 'XAU'],
        'HBAR': ['Hedera'],
        'HYPE': ['Hyperliquid'],
        'ICP': ['Internet Computer'],
        'INJ': ['Injective'],
        'IP': ['Story Protocol'],
        'JTO': ['Jito'],
        'JUP': ['Jupiter'],
        'KAS': ['Kaspa'],
        'KTA': ['Keeta'],
        'LDO': ['Lido DAO', 'Lido'],
        'LINEA': ['Linea'],
        'LINK': ['Chainlink'],
        'LIT': ['Litentry'],
        'LTC': ['Litecoin'],
        'MEGA': ['MegaETH'],
        'MELANIA': ['Melania'],
        'MEME': ['Memecoin'],
        'MET': ['Metronome'],
        'MEW': ['cat in a dogs world'],
        'MKR': ['Maker'],
        'MNT': ['Mantle'],
        'MON': ['Monad'],
        'MOODENG': ['Moo Deng'],
        'MORPHO': ['Morpho'],
        'NATGAS': ['Natural Gas'],
        'NEAR': ['NEAR Protocol'],
        'OKB': ['OKB'],
        'OM': ['MANTRA'],
        'ONDO': ['Ondo'],
        'OP': ['Optimism'],
        'ORDI': ['ORDI'],
        'PENDLE': ['Pendle'],
        'PENGU': ['Pudgy Penguins'],
        'PEPE': ['Pepe'],
        'PI': ['Pi Network'],
        'POL': ['Polygon'],
        'PUMP': ['Pump.fun', 'Pump'],
        'RENDER': ['Render'],
        'SATS': ['1000SATS', 'Sats'],
        'SEI': ['Sei'],
        'SHIB': ['Shiba Inu'],
        'SILVER': ['Silver', 'XAG'],
        'SKY': ['Sky Protocol'],
        'SOL': ['Solana'],
        'SPCX': ['SPX6900'],
        'SPX6900': ['SPX6900'],
        'STX': ['Stacks'],
        'SUI': ['Sui'],
        'SYRUP': ['Maple Finance', 'Syrup'],
        'S': ['Sonic'],
        'TAO': ['Bittensor'],
        'TBTC': ['Threshold Bitcoin', 'tBTC'],
        'TIA': ['Celestia'],
        'TON': ['Toncoin', 'The Open Network'],
        'TRUMP': ['Official Trump', 'Trump'],
        'TRX': ['TRON'],
        'UNI': ['Uniswap'],
        'USDC.E': ['USDC.e', 'Bridged USDC'],
        'USDC': ['USD Coin'],
        'USDT': ['Tether'],
        'VIRTUAL': ['Virtuals Protocol'],
        'VVV': ['Venice Token'],
        'WELL': ['Moonwell'],
        'WIF': ['dogwifhat'],
        'WLD': ['Worldcoin'],
        'WLFI': ['World Liberty Financial'],
        'WSTETH': ['Wrapped stETH', 'Lido wstETH'],
        'WTIOIL': ['WTI Oil', 'West Texas Intermediate'],
        'XAUT.V2': ['Tether Gold', 'XAUT'],
        'XAUT': ['Tether Gold'],
        'XLM': ['Stellar'],
        'XMR': ['Monero'],
        'XPL': ['Plasma'],
        'XRP': ['Ripple'],
        'ZEC': ['Zcash'],
        'ZORA': ['Zora'],
        'ZRO': ['LayerZero'],
    }
    SENTIMENT_MIN_MENTION_COUNT = 10
    SENTIMENT_MIN_SCORE_LONG = 0.10
    SENTIMENT_MAX_SCORE_SHORT = -0.10
    # If set to a falsy value (None or ''), the bot will use the data fetch interval (e.g. '1h')
    SENTIMENT_RESAMPLE_INTERVAL = None

    MIN_TRADE_AMOUNT = 30.0
    
    # ==================== SCHEDULE ====================
    TRAIN_HOUR = 3
    MONITOR_INTERVAL_MINUTES = 5
    
    # ==================== API KEYS ====================
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', 'xxxxxxxxxxxxx')
    KRAKEN_PRIVATE_KEY = os.getenv('KRAKEN_PRIVATE_KEY', 'xxxxxxxxxxxxxxxxxxxxx')
    # TELEGRAM_BOT_TOKEN is the preferred name; TELEGRAM_TOKEN remains
    # backwards-compatible with the original bot.
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN', '')
    TELEGRAM_BOT_TOKEN = TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    TELEGRAM_ALLOWED_CHAT_IDS = {
        value.strip() for value in os.getenv('TELEGRAM_ALLOWED_CHAT_IDS', TELEGRAM_CHAT_ID).split(',')
        if value.strip()
    }
    
    # Feature list; must match train_model.py

    FEATURE_COLUMNS = [
        # Original features (12)
        'returns', 
        'volatility', 
        'rsi_norm', 
        'macd_norm', 
        'momentum', 
        'volume_ratio', 
        'atr_norm', 
        'green_candles',
        'price_range', 
        'volume_directional', 
        'bb_position', 
        'volume_obv',
        
        # New v3.3 features (6)
        'price_vs_ma50',
        'volume_change',
        'hl_ratio',
        'close_position',
        'trend_short',
        'trend_long'
    ]
    
    @classmethod
    def validate(cls, skip_telegram=False):
        """Validate credentials"""
        missing = []

        if cls.DATA_SOURCE.upper() != 'GMX':
            if not cls.KRAKEN_API_KEY:
                missing.append('KRAKEN_API_KEY')
            if not cls.KRAKEN_PRIVATE_KEY:
                missing.append('KRAKEN_PRIVATE_KEY')

        if not skip_telegram:
            if not cls.TELEGRAM_TOKEN:
                missing.append('TELEGRAM_TOKEN')
            if not cls.TELEGRAM_CHAT_ID:
                missing.append('TELEGRAM_CHAT_ID')

        if missing:
            raise EnvironmentError(
                f"❌ Missing variables: {', '.join(missing)}\n"
                f"Create them in the .env file"
            )

        return True
    
    @classmethod
    def get_summary(cls):
        """Configuration summary"""
        return f"""
╔══════════════════════════════════════════════════════════╗
║         ADA TRADING CONFIGURATION v3.3 - 1H             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📊 ASSET:          {cls.SYMBOL}                         ║
║  ⏱️  TIMEFRAME:     {cls.TIMEFRAME}                      ║
║  📄 LOOKBACK:       {cls.LOOKBACK_PERIOD}                ║
║                                                          ║
║  🧠 MODEL: Binary Classifier v3.3                        ║
║     • Sequence:     {cls.SEQUENCE_LENGTH} candles (1.5 days)║
║     • LSTM:         {cls.LSTM_UNITS}                     ║
║     • Features:     {len(cls.FEATURE_COLUMNS)} 🔥        ║
║                                                          ║
║  🎯 TARGET:                                              ║
║     • Threshold:    {cls.MOVEMENT_THRESHOLD*100:.1f}% move 🔥║
║     • Noise filter: Significant moves                   ║
║                                                          ║
║  💰 MANAGEMENT:                                          ║
║     • Risk:         {cls.RISK_PERCENTAGE}%               ║
║     • Min Signal:   {cls.MIN_SIGNAL_THRESHOLD*100:.0f}% 🔥║
║     • SL:           {cls.ATR_SL_MULTIPLIER}x ATR         ║
║     • TP:           {cls.ATR_TP_MULTIPLIER}x ATR         ║
║     • R:R Ratio:    1:{cls.ATR_TP_MULTIPLIER/cls.ATR_SL_MULTIPLIER:.1f}║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

KEY CHANGES v3.3:

   ✅ MOVEMENT_THRESHOLD: >1% (filters noise)
   ✅ FEATURES: 18 total (6 new)
   ✅ SEQUENCE: 36h (1.5 days)
   ✅ MIN_SIGNAL: 58% (more selective)
   
📊 TARGET: 56-60% accuracy on 1h
"""
