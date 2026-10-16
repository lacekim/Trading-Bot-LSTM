import asyncio
from datetime import datetime

from .load_scalers import train_scalers, seq_len
from .load_models import get_action, load_model
from .bot import Bot


async def main():

    # Load models
    load_model()

    # Init bot
    bot = Bot(seq_len)
    bot.load() 


    # # ========== TESTING ==========
    # await bot.check_balance()

    # exit()
    # # =============================


    while True:
        # Create exchange
        await bot.create_exchange()

        # Update new data for model
        await bot.update_new_data_for_model()

        # Get prediction from the model
        CURR_ACTION = get_action(bot.opens_USDT,
                                bot.highs_USDT,
                                bot.lows_USDT,
                                bot.closes_USDT,
                                bot.volumes_USDT,
                                train_scalers,)
        
        # Update market price
        bot.update_curr_prices() 

        # Check current position
        await bot.check_positions() 

        # BUY CASE:
        if CURR_ACTION == 1 and bot.position != "LONG":
            if bot.position == "SHORT":
                is_double_size = True
            else:
                is_double_size = False
            # Place limit buy order
            await bot.place_limit_order("BUY", is_double_size)

        # SELL CASE:
        elif CURR_ACTION == 0 and bot.position != "SHORT":
            if bot.position == "LONG":
                is_double_size = True
            else:
                is_double_size = False
            # Place limit sell order
            await bot.place_limit_order("SELL", is_double_size)


        bot.write_to_log(f"Waiting for new candle...")

        while True:
            # Convert timeframe str to int
            timeframe = float("".join(ch for ch in bot.TIME_FRAME if ch.isdigit()))

            # Convert timeframe (like '15m') into seconds
            fifteen_min_secs = timeframe * 60  

            # Current minute within this timeframe (e.g., 0–14 for 15m)
            minute_now = datetime.now().minute % timeframe

            # How many seconds have passed so far in this candle
            now_secs = minute_now * 60 + datetime.now().second  

            # How many seconds left until the candle closes
            wait_secs = fifteen_min_secs - now_secs  

            # Small sleep so the loop doesn’t burn CPU
            await asyncio.sleep(0.5)  

            # Once we’re in the last few seconds of this candle, break the loop
            if wait_secs < bot.last_sec_wait:
                break


        # Cancel any open order 
        await bot.cancel_order() 

        # Check equity
        await bot.check_balance()

        # Record results for later analysis
        bot.actions.append(CURR_ACTION)
        bot.open_prices.append(bot.curr_open)
        bot.equities.append(bot.equity)
        
        # Save the bot's state
        bot.save()

        # Plot performance
        bot.plot_performance()

        # Close connections
        await bot.close()




if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())

        except Exception as e:
            print(f"Error: {e}")
























































