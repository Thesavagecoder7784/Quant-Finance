import pandas as pd
from .base_strategy import Strategy

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

    rs = gain / loss
    rs[loss == 0] = float('inf')
    
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

class SmaRsiStrategy(Strategy):
    """
    A strategy that uses a Simple Moving Average (SMA) crossover with a
    Relative Strength Index (RSI) confirmation to generate trading signals.
    """
    def __init__(self, symbols, parameters):
        super().__init__("SMA/RSI Crossover", symbols, parameters)

    def generate_signals(self, data):
        """
        Generates trading signals for each symbol based on the SMA/RSI strategy.
        """
        signals = {}
        for symbol in self.symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None or symbol_data.empty:
                signals[symbol] = {'signal': 'hold'}
                continue

            short_sma_period = self.parameters.get('short_sma_period', 20)
            long_sma_period = self.parameters.get('long_sma_period', 50)
            rsi_period = self.parameters.get('rsi_period', 14)
            rsi_overbought = self.parameters.get('rsi_overbought', 70)
            rsi_oversold = self.parameters.get('rsi_oversold', 30)

            symbol_data = calculate_smas(symbol_data, short_sma_period, long_sma_period)
            symbol_data = calculate_rsi(symbol_data, rsi_period)

            if symbol_data.iloc[-1].isnull().any():
                signals[symbol] = {'signal': 'hold'}
                continue

            latest_data = symbol_data.iloc[-1]
            previous_data = symbol_data.iloc[-2]

            is_buy_signal = (previous_data['SMA_Short'] < previous_data['SMA_Long'] and
                             latest_data['SMA_Short'] >= latest_data['SMA_Long'] and
                             latest_data['RSI'] < rsi_overbought)

            is_sell_signal = (previous_data['SMA_Short'] > previous_data['SMA_Long'] and
                              latest_data['SMA_Short'] <= latest_data['SMA_Long'] and
                              latest_data['RSI'] > rsi_oversold)

            if is_buy_signal:
                signals[symbol] = {'signal': 'buy'}
            elif is_sell_signal:
                signals[symbol] = {'signal': 'sell'}
            else:
                signals[symbol] = {'signal': 'hold'}
        
        return signals
