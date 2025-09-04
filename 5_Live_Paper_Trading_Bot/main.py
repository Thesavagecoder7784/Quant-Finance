import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import time
import logging
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

# --- Logging Setup ---
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
    # Silence noisy logs from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

# --- Configuration ---
SYMBOLS = ["SPY", "QQQ", "DIA"]
SHORT_SMA_PERIOD = 20
LONG_SMA_PERIOD = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TRADE_CAPITAL_PERCENTAGE = 0.05 # Percentage of buying power to use per trade
DATA_FETCH_LIMIT = LONG_SMA_PERIOD + RSI_PERIOD + 5 # Number of data points to fetch

# --- Alpaca API Setup ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

def get_historical_data(symbol, timeframe, limit):
    """
    Fetches historical bar data from the Alpaca API.
    """
    try:
        bars = api.get_bars(symbol, timeframe, limit=limit).df
        # Alpaca data is timezone-aware, which is good.
        return bars
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def calculate_smas(data, short_period, long_period):
    """Calculates short and long Simple Moving Averages."""
    data['SMA_Short'] = data['close'].rolling(window=short_period).mean()
    data['SMA_Long'] = data['close'].rolling(window=long_period).mean()
    return data

def calculate_rsi(data, period):
    """Calculates the Relative Strength Index (RSI) robustly."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Handle division by zero for RSI calculation
    rs = gain / loss
    rs[loss == 0] = float('inf') # Where loss is 0, RSI is 100
    
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def get_current_position(symbol):
    """Gets the current position quantity for a symbol."""
    try:
        position = api.get_position(symbol)
        return int(position.qty)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404: # Position not found
            return 0
        else:
            logging.error(f"APIError getting position for {symbol}: {e}")
            return 0
    except Exception as e:
        logging.error(f"Error getting position for {symbol}: {e}")
        return 0

def place_order(symbol, qty, side):
    """
    Places a market order and logs the action.
    """
    if qty <= 0:
        logging.warning(f"Attempted to place an order with zero or negative quantity for {symbol}. Skipping.")
        return None
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        logging.info(f"Placed {side} order for {qty} shares of {symbol}. Order ID: {order.id}")
        return order
    except Exception as e:
        logging.error(f"Error placing {side} order for {symbol}: {e}")
        return None

def run_trading_strategy():
    """
    Main trading strategy logic loop.
    """
    try:
        account = api.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        logging.info(f"Account Equity: ${equity:,.2f}, Buying Power: ${buying_power:,.2f}")
    except Exception as e:
        logging.error(f"Failed to get account information: {e}")
        return

    for symbol in SYMBOLS:
        logging.info(f"--- Running strategy for {symbol} ---")

        # Fetch data using Alpaca API
        data = get_historical_data(symbol, TimeFrame.Minute, DATA_FETCH_LIMIT)

        if data.empty or len(data) < DATA_FETCH_LIMIT:
            logging.warning(f"Could not retrieve sufficient historical data for {symbol}. Skipping.")
            continue

        # Calculate indicators
        data = calculate_smas(data, SHORT_SMA_PERIOD, LONG_SMA_PERIOD)
        data = calculate_rsi(data, RSI_PERIOD)

        # Check for NaN values
        if data.iloc[-1].isnull().any():
            logging.warning(f"Not enough data to calculate indicators for {symbol}. Waiting for more data.")
            continue

        # Get latest values
        latest_data = data.iloc[-1]
        previous_data = data.iloc[-2]
        latest_close = latest_data['close']

        logging.info(f"[{symbol}] Close: {latest_close:.2f} | "
                     f"Short SMA: {latest_data['SMA_Short']:.2f} | "
                     f"Long SMA: {latest_data['SMA_Long']:.2f} | "
                     f"RSI: {latest_data['RSI']:.2f}")

        # Get current position and calculate trade size
        current_position = get_current_position(symbol)
        trade_value = buying_power * TRADE_CAPITAL_PERCENTAGE
        trade_quantity = int(trade_value / latest_close) if latest_close > 0 else 0

        logging.info(f"[{symbol}] Current Position: {current_position} shares | Calculated Trade Size: {trade_quantity} shares")

        if trade_quantity == 0:
            logging.warning(f"[{symbol}] Calculated trade quantity is 0. Skipping trade logic.")
            continue

        # --- Trading Logic ---
        # Buy Signal: Golden Cross (Short SMA crosses above Long SMA) + RSI confirmation
        is_buy_signal = (previous_data['SMA_Short'] < previous_data['SMA_Long'] and
                         latest_data['SMA_Short'] >= latest_data['SMA_Long'] and
                         latest_data['RSI'] < RSI_OVERBOUGHT)

        # Sell Signal: Death Cross (Short SMA crosses below Long SMA) + RSI confirmation
        is_sell_signal = (previous_data['SMA_Short'] > previous_data['SMA_Long'] and
                          latest_data['SMA_Short'] <= latest_data['SMA_Long'] and
                          latest_data['RSI'] > RSI_OVERSOLD)

        if is_buy_signal:
            logging.info(f"BUY signal detected for {symbol}.")
            target_position = trade_quantity
            qty_to_order = target_position - current_position
            if qty_to_order > 0:
                logging.info(f"Current position is {current_position}. Target is {target_position}. Placing BUY for {qty_to_order} shares.")
                place_order(symbol, qty_to_order, 'buy')
            else:
                logging.info(f"Already in a sufficient long position for {symbol}. No new BUY order needed.")

        elif is_sell_signal:
            logging.info(f"SELL signal detected for {symbol}.")
            target_position = -trade_quantity # Target a short position
            qty_to_order = target_position - current_position
            if qty_to_order < 0:
                logging.info(f"Current position is {current_position}. Target is {target_position}. Placing SELL for {abs(qty_to_order)} shares.")
                place_order(symbol, abs(qty_to_order), 'sell')
            else:
                logging.info(f"Already in a sufficient short position for {symbol}. No new SELL order needed.")
        else:
            logging.info(f"No trading signal for {symbol}. Holding current position.")

# --- Main Loop ---
if __name__ == "__main__":
    setup_logging()
    logging.info("--- Starting Paper Trading Bot ---")
    while True:
        try:
            # Check if market is open before running
            clock = api.get_clock()
            if clock.is_open:
                logging.info("Market is open. Running trading strategy...")
                run_trading_strategy()
            else:
                logging.info("Market is closed. Waiting for the next open.")
                # Calculate time until next open and sleep
                time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                if time_to_open > 0:
                    logging.info(f"Sleeping for {time_to_open / 60:.2f} minutes until market open.")
                    time.sleep(time_to_open)
                continue # Re-check clock immediately after sleep
        except Exception as e:
            logging.critical(f"An unhandled error occurred in the main loop: {e}", exc_info=True)

        # Wait before the next check/run
        logging.info("Waiting for 1 minute before next run...")
        time.sleep(60)