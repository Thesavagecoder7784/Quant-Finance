import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import time
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

# --- Configuration ---
SYMBOLS = ["SPY", "QQQ", "DIA"] 
SHORT_SMA_PERIOD = 20
LONG_SMA_PERIOD = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Risk Management
TRADE_CAPITAL_PERCENTAGE = 0.05 # Percentage of buying power to use per trade

# --- Alpaca API Setup ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

def get_historical_data(symbol, interval, limit):
    # Alpaca API for historical data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1mo", interval=interval)
    return data.tail(limit)

def calculate_smas(data, short_period, long_period):
    data['SMA_Short'] = data['Close'].rolling(window=short_period).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_period).mean()
    return data

def calculate_rsi(data, period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def get_current_position(symbol):
    try:
        position = api.get_position(symbol)
        return int(position.qty)
    except Exception as e:
        # No position found or other error
        return 0

def place_order(symbol, qty, side, type='market', time_in_force='gtc'):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        print(f"Placed {side} order for {qty} shares of {symbol}. Order ID: {order.id}")
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def run_trading_strategy():
    # Get account information once per strategy run for dynamic position sizing
    account = api.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    print(f"Account Equity: {equity:.2f}")
    print(f"Buying Power: {buying_power:.2f}")

    for symbol in SYMBOLS:
        print(f"\nRunning trading strategy for {symbol}...")

        data = get_historical_data(symbol, "5m", LONG_SMA_PERIOD + RSI_PERIOD + 5) # Get enough data for SMAs and RSI

        if data.empty:
            print("Could not retrieve historical data. Skipping strategy run for {symbol}.")
            continue

        data = calculate_smas(data, SHORT_SMA_PERIOD, LONG_SMA_PERIOD)
        data = calculate_rsi(data, RSI_PERIOD)

        # Ensure we have enough data for SMA and RSI calculation
        if pd.isna(data['SMA_Short'].iloc[-1]) or pd.isna(data['SMA_Long'].iloc[-1]) or pd.isna(data['RSI'].iloc[-1]):
            print(f"Not enough data to calculate SMAs or RSI for {symbol}. Waiting for more data...")
            continue

        current_position = get_current_position(symbol)
        latest_close = data['Close'].iloc[-1]
        latest_short_sma = data['SMA_Short'].iloc[-1]
        latest_long_sma = data['SMA_Long'].iloc[-1]
        latest_rsi = data['RSI'].iloc[-1]
        previous_short_sma = data['SMA_Short'].iloc[-2]
        previous_long_sma = data['SMA_Long'].iloc[-2]

        print(f"Current Close: {latest_close:.2f}")
        print(f"Short SMA ({SHORT_SMA_PERIOD}): {latest_short_sma:.2f}")
        print(f"Long SMA ({LONG_SMA_PERIOD}): {latest_long_sma:.2f}")
        print(f"RSI ({RSI_PERIOD}): {latest_rsi:.2f}")
        print(f"Current Position: {current_position} shares")

        # Calculate trade quantity based on a percentage of buying power
        trade_value = buying_power * TRADE_CAPITAL_PERCENTAGE
        trade_quantity = int(trade_value / latest_close) if latest_close > 0 else 0

        if trade_quantity == 0:
            print("Calculated trade quantity is 0. Skipping trade for {symbol}.")
            continue

        print(f"Calculated Trade Quantity: {trade_quantity} shares")

        # Trading Logic with RSI confirmation
        # Crossover Up (Buy Signal)
        if previous_short_sma < previous_long_sma and latest_short_sma >= latest_long_sma and latest_rsi < RSI_OVERBOUGHT:
            print("BUY signal detected (Short SMA crossed above Long SMA and RSI is not overbought)")
            if current_position <= 0: # Only buy if not already long or if short
                if current_position < 0: # If currently short, close short position first
                    print(f"Closing short position of {abs(current_position)} shares for {symbol}.")
                    place_order(symbol, abs(current_position), 'buy')
                    time.sleep(2) # Wait for order to process
                print(f"Placing BUY order for {trade_quantity} shares of {symbol}")
                place_order(symbol, trade_quantity, 'buy')
            else:
                print(f"Already in a long position or flat for {symbol}. No new BUY order.")

        # Crossover Down (Sell Signal)
        elif previous_short_sma > previous_long_sma and latest_short_sma <= latest_long_sma and latest_rsi > RSI_OVERSOLD:
            print("SELL signal detected (Short SMA crossed below Long SMA and RSI is not oversold)")
            if current_position >= 0: # Only sell if not already short or if long
                if current_position > 0: # If currently long, close long position first
                    print(f"Closing long position of {current_position} shares for {symbol}.")
                    place_order(symbol, current_position, 'sell')
                    time.sleep(2) # Wait for order to process
                print(f"Placing SELL order for {trade_quantity} shares of {symbol}")
                place_order(symbol, trade_quantity, 'sell')
            else:
                print(f"Already in a short position or flat for {symbol}. No new SELL order.")
        else:
            print(f"No SMA crossover or RSI confirmation signal for {symbol}. Holding current position.")

# --- Main Loop ---
if __name__ == "__main__":
    print("Starting paper trading bot...")
    while True:
        try:
            run_trading_strategy()
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
        print("Waiting for 1 minute before next run...")
        time.sleep(60) # Wait for 1 minute (60 seconds)
