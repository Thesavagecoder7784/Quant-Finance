import alpaca_trade_api as tradeapi
import logging
import asyncio
from alpaca_trade_api.rest import TimeFrame
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

class Trader:
    """
    The Trader class that manages the trading loop, account information, and positions.
    """
    def __init__(self, strategies, trade_capital_percentage):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
        self.strategies = strategies
        self.trade_capital_percentage = trade_capital_percentage

    async def get_historical_data(self, symbol, timeframe, limit):
        """
        Fetches historical bar data from the Alpaca API asynchronously.
        """
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            return bars
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    async def get_current_position(self, symbol):
        """
        Gets the current position quantity for a symbol asynchronously.
        """
        try:
            position = self.api.get_position(symbol)
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

    async def place_order(self, symbol, qty, side):
        """
        Places a market order and logs the action asynchronously.
        """
        if qty <= 0:
            logging.warning(f"Attempted to place an order with zero or negative quantity for {symbol}. Skipping.")
            return None
        try:
            order = self.api.submit_order(
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

    async def run(self):
        """
        Main trading loop.
        """
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            logging.info(f"Account Equity: ${equity:,.2f}, Buying Power: ${buying_power:,.2f}")
        except Exception as e:
            logging.error(f"Failed to get account information: {e}")
            return

        for strategy in self.strategies:
            logging.info(f"--- Running strategy: {strategy.name} ---")
            data = {}
            tasks = [self.get_historical_data(symbol, TimeFrame.Minute, 200) for symbol in strategy.symbols]
            results = await asyncio.gather(*tasks)
            for i, symbol in enumerate(strategy.symbols):
                data[symbol] = results[i]

            signals = strategy.generate_signals(data)

            for symbol, signal_data in signals.items():
                signal = signal_data['signal']
                if signal == 'hold':
                    logging.info(f"[{symbol}] No trading signal. Holding current position.")
                    continue

                latest_close = data[symbol].iloc[-1]['close']
                current_position = await self.get_current_position(symbol)
                trade_value = buying_power * self.trade_capital_percentage
                trade_quantity = int(trade_value / latest_close) if latest_close > 0 else 0

                logging.info(f"[{symbol}] Current Position: {current_position} shares | Calculated Trade Size: {trade_quantity} shares")

                if trade_quantity == 0:
                    logging.warning(f"[{symbol}] Calculated trade quantity is 0. Skipping trade logic.")
                    continue

                if signal == 'buy':
                    logging.info(f"BUY signal detected for {symbol}.")
                    target_position = trade_quantity
                    qty_to_order = target_position - current_position
                    if qty_to_order > 0:
                        logging.info(f"Current position is {current_position}. Target is {target_position}. Placing BUY for {qty_to_order} shares.")
                        await self.place_order(symbol, qty_to_order, 'buy')
                    else:
                        logging.info(f"Already in a sufficient long position for {symbol}. No new BUY order needed.")

                elif signal == 'sell':
                    logging.info(f"SELL signal detected for {symbol}.")
                    target_position = -trade_quantity # Target a short position
                    qty_to_order = target_position - current_position
                    if qty_to_order < 0:
                        logging.info(f"Current position is {current_position}. Target is {target_position}. Placing SELL for {abs(qty_to_order)} shares.")
                        await self.place_order(symbol, abs(qty_to_order), 'sell')
                    else:
                        logging.info(f"Already in a sufficient short position for {symbol}. No new SELL order needed.")
