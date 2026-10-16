import asyncio
import logging
import os
import pickle
import time
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

import ccxt.async_support as ccxt_supp


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger



class Bot:
    def __init__(self, seq_len):
        self.logger = setup_logger('log', './trading_bot/log.log') # Setup logger
        self.seq_len = seq_len # Sequence length
        self.TIME_FRAME = '15m' # Timeframe
        self.coin = 'BTC' # Coin
        self.asset_usd = 'FDUSD' # USD asset

        self.pos_size_USD = 10 # Position size in USD
        self.slippage_percentage = 0.05 # Slippage %

        self.price_notional = 2 # Price notional
        self.size_notional = 5 # Size notional

        self.model_symbol = f'{self.coin}USDT' # The pair that the model was trained on
        self.symbol = f"{self.coin}/{self.asset_usd}" # The pair that we actually trade with

        self.position = "NONE" # Current position "LONG" / "SHORT"
        self.open_prices = [] # Open price list
        self.actions = [] # Action list
        self.equities = [] # Equity list
        self.total_fee = 0 # Total fee
        self.last_sec_wait = 20 # Seconds to check for new candle


    def write_to_log(self, msg):
        self.logger.info(msg)
        print(msg)


    async def create_exchange(self):
        # API keys -- read from the environment, never hardcoded here. Set
        # BINANCE_API_KEY / BINANCE_API_SECRET in the shell (or a local,
        # gitignored .env loaded before this runs) before starting the bot.
        apiKey = os.environ.get("BINANCE_API_KEY")
        secret = os.environ.get("BINANCE_API_SECRET")
        if not apiKey or not secret:
            raise RuntimeError(
                "BINANCE_API_KEY / BINANCE_API_SECRET are not set in the environment."
            )

        for _ in range(20):
            try:
                # Async client
                self.exchange = ccxt_supp.binance({'apiKey': apiKey, 
                                                    'secret': secret,
                                                    'options': {
                                                        'adjustForTimeDifference': True,  # Enable automatic time adjustment
                                                    },})
                # Async time sync
                await self.exchange.load_time_difference()  

                # Sync client
                self.client = ccxt.binance() 
                self.write_to_log(f"Exchange created.")
                break
            except Exception as err:
                time.sleep(0.4)
                self.write_to_log(f"Failed to get account with API key: {str(err)}")


    def get_candles(self, symbol):
        # Fetch OHLCV
        klines = self.client.fetch_ohlcv(symbol, self.TIME_FRAME, limit=self.seq_len+1)
        klines = np.array(klines, dtype=object) # Convert to NumPy array

        # Extract columns
        open_times = pd.to_datetime(klines[:, 0].astype(np.int64), unit='ms')
        open_prices = klines[:, 1].astype(float)
        high_prices = klines[:, 2].astype(float)
        low_prices = klines[:, 3].astype(float)
        close_prices = klines[:, 4].astype(float)
        volumes = klines[:, 5].astype(float)

        return open_times, open_prices, high_prices, low_prices, close_prices, volumes



    # async def get_candles_async(self, symbol):
    #     # Fetch OHLCV
    #     klines = await self.exchange.fetch_ohlcv(symbol, self.TIME_FRAME, limit=self.seq_len+1)
    #     klines = np.array(klines, dtype=object) # Convert to NumPy array

    #     # Extract columns
    #     open_times = pd.to_datetime(klines[:, 0].astype(np.int64), unit='ms')
    #     open_prices = klines[:, 1].astype(float)
    #     high_prices = klines[:, 2].astype(float)
    #     low_prices = klines[:, 3].astype(float)
    #     close_prices = klines[:, 4].astype(float)
    #     volumes = klines[:, 5].astype(float)

    #     return open_times, open_prices, high_prices, low_prices, close_prices, volumes


    async def update_new_data_for_model(self): 
        self.write_to_log("... Getting the new candle ...")

        # Get current time
        while True:
            try:
                open_times, opens, highs, lows, closes, volumes = self.get_candles(self.model_symbol)
                curr_time = open_times[-2] # -2 because we want fixed time
                break
            except Exception as err:
                self.write_to_log(f"Failed to get_candles: {err}")
                await asyncio.sleep(0.4)

        # Start loop check new candle
        while True:
            try:
                # Get new time
                open_times, opens, highs, lows, closes, volumes = self.get_candles(self.model_symbol)
                new_time = open_times[-2] # -2 because we want fixed time
                
                # Compare new time with current time
                if new_time == curr_time:
                    continue
                
                # Get market input data (for preprocessing)
                self.opens_USDT   = np.array(opens[:-1], dtype=np.float32)  # open (exclude unfinished candle)
                self.highs_USDT   = np.array(highs[:-1], dtype=np.float32)  # high (exclude unfinished candle)
                self.lows_USDT    = np.array(lows[:-1], dtype=np.float32)   # low (exclude unfinished candle)
                self.closes_USDT  = np.array(closes[:-1], dtype=np.float32) # close (exclude unfinished candle)
                self.volumes_USDT = np.array(volumes[:-1], dtype=np.float32) # volume (exclude unfinished candle)
                break

            except Exception as err: 
                self.write_to_log(f"Error in update_new_data_for_model: {str(err)}")
                await asyncio.sleep(0.4)












    def update_curr_prices(self):
        for _ in range(25):
            try:
                open_times, opens, highs, lows, closes, volumes = self.get_candles(self.symbol)

                self.curr_open = opens[-1] # Update current open price (from the latest candle)
                self.curr_close = closes[-1] # Update current close price (from the latest candle)
                self.write_to_log(f'curr close: {self.curr_close}')
                break

            except Exception as err: 
                self.write_to_log(f"Error in update_curr_prices: {str(err)}")



    async def place_limit_order(self, side, is_double_size=False):
        # Calculate slippage
        slippage_price = self.curr_close * self.slippage_percentage / 100
        
        # Calculate order price
        if side=="BUY":
            order_price = self.curr_close - slippage_price 

        elif side=="SELL":
            order_price = self.curr_close + slippage_price 

        # Calculate order size (in Coin value)
        order_size = self.pos_size_USD / order_price
        
        # Calculate size (if double)
        if is_double_size:
            order_size = order_size * 2 

        # Round the price & size with the required notional
        order_price = np.round(order_price, self.price_notional) # Price
        order_size = self.round_down(order_size, self.size_notional) # Size

        # Place order
        for _ in range(10):
            try:
                order = await self.exchange.create_order(
                    symbol=self.symbol,
                    side=side,
                    type='LIMIT',
                    amount=order_size, 
                    price = order_price,
                    params={'marginMode': 'cross'} # Specify cross margin
                )
                self.write_to_log(f"{side} Order is placed. Price: {order_price}. Quantity: {order_size} {self.coin}")
                self.write_to_log(f"Order details: {order}")
                
                # Define fee
                if order['fee']:
                    fee = order['fee']['cost']
                    self.write_to_log(f"Fee: ${fee}")
                    self.total_fee += fee # Update total fee
                break

            except Exception as e:
                self.write_to_log(f"FAILED to placed Future limit {side}: {str(e)}")
                await asyncio.sleep(0.4)




    async def check_balance(self):
        # Fetch balance
        for _ in range(5):
            try:
                balance = await self.exchange.fetch_balance({'type': 'margin',
                                                            'symbol': self.symbol})
                await asyncio.sleep(0.3)
                break
            
            except Exception as e:
                self.write_to_log(f"Failed to fetch_balance: {e}")
                await asyncio.sleep(0.3)

        # Define balance of USD & Coin
        self.usd_balance = balance['free'][self.asset_usd] # e.g. FDUSD
        self.coin_balance = balance['free'][self.coin] # e.g. BTC

        # Define borrowed coin
        assets = balance['info']['userAssets']
        for asset in assets:
            if asset['asset'] == self.coin:
                self.coin_borrowed = float(asset['borrowed'])
                break

        # Update current price
        self.update_curr_prices()

        # Calculate equity: Balance USD ($) + Balance Coin ($) - Borrowed Coin ($)
        self.equity = self.usd_balance + self.coin_balance * self.curr_close - self.coin_borrowed * self.curr_close
        self.equity = np.round(self.equity, 2) # Round equity

        self.write_to_log(f"Balance: {self.usd_balance} {self.asset_usd}  |  {self.coin_balance} {self.coin}")
        self.write_to_log(f"Borrowed: {self.coin_borrowed} {self.coin}")
        self.write_to_log(f"Equity: {self.equity}")



    async def check_positions(self):
        # Check balance
        await self.check_balance()

        # Define position
        min_usd = 5 # Minimum USD
        min_coin = min_usd / self.curr_close # Minimum coin amount

        if self.usd_balance < min_usd and self.coin_balance > min_coin:
            self.position = "LONG"
            self.write_to_log(f"Current position: {self.position}")

        elif self.usd_balance > min_usd and self.coin_balance < min_coin:
            self.position = "SHORT"
            self.write_to_log(f"Current position: {self.position}")

        else:
            self.position = "NONE"
            self.write_to_log("WARNING: No open positions found!")



    async def cancel_order(self):
        for _ in range(2):
            try:
                result = await self.exchange.cancel_all_orders(self.symbol, params={'marginMode': 'cross'})
                self.write_to_log(f"Canceled order: {result}")

            except Exception as err:
                self.write_to_log(f"Couldn't cancel orders: {err}")
                await asyncio.sleep(0.3)


    def save(self):
        self.saved_data = {
            "open_prices": self.open_prices,
            'actions': self.actions,
            'equities': self.equities,
            'total_fee': self.total_fee,
        }
        with open("./trading_bot/saved_data.pkl", "wb") as file:
            pickle.dump(self.saved_data, file)


    def load(self):
        try:
            with open("./trading_bot/saved_data.pkl", "rb") as file:
                self.saved_data = pickle.load(file)

            self.open_prices = self.saved_data["open_prices"]
            self.actions = self.saved_data['actions']
            self.equities = self.saved_data['equities']
            self.total_fee = self.saved_data['total_fee']
        except:
            print(f"No saved_data to load!")


    async def close(self):
        for _ in range(20):
            try:
                await self.exchange.close()
                await asyncio.sleep(0.1)
                break

            except Exception as err:
                await asyncio.sleep(0.5)
                print(f"Failed to close connection. Err: " + str(err))





    def plot_performance(self):
        try:
            scaler = MinMaxScaler(feature_range=(min(self.equities), max(self.equities)))
            scaled_open_prices = scaler.fit_transform(np.array(self.open_prices).reshape(-1, 1)).flatten() 

            chart_title = f"Equity: ${self.equity:.2f}"
            plt.figure(figsize=(12, 6))
            plt.plot(scaled_open_prices, color='green', label='BTC Price')
            plt.plot(self.equities, label="Equity Curve", color="blue")
            plt.title(chart_title)
            plt.xlabel("Time Step")
            plt.ylabel("Equity (USD)")
            plt.legend()
            plt.grid(True)
            plt.savefig('./trading_bot/performance.png', dpi=300)
            plt.close()

        except Exception as e:
            print(f"Couldn't plot the graph: {e}")



    def round_down(self, x, decimals):
        """
        We replace np.round with this round_down method.
        Before placing orders, we should **round down** round the order's size, not round up.
        Usage example: round_down(0.049, 2) -> Output: 0.04 
        """
        factor = 10 ** decimals
        return math.floor(x * factor) / factor


















































































