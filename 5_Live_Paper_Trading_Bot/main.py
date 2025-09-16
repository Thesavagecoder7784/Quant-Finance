import asyncio
import logging
import time
from trader import Trader
from config import STRATEGIES, TRADE_CAPITAL_PERCENTAGE
import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

def setup_logging():
    """Sets up the logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

async def main():
    """
    Main asynchronous function to run the trading bot.
    """
    setup_logging()
    logging.info("--- Starting Paper Trading Bot ---")

    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
    trader = Trader(strategies=STRATEGIES, trade_capital_percentage=TRADE_CAPITAL_PERCENTAGE)

    while True:
        try:
            clock = api.get_clock()
            if clock.is_open:
                logging.info("Market is open. Running trading strategy...")
                await trader.run()
            else:
                logging.info("Market is closed. Waiting for the next open.")
                time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                if time_to_open > 0:
                    logging.info(f"Sleeping for {time_to_open / 60:.2f} minutes until market open.")
                    await asyncio.sleep(time_to_open)
                continue
        except Exception as e:
            logging.critical(f"An unhandled error occurred in the main loop: {e}", exc_info=True)

        logging.info("Waiting for 1 minute before next run...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())