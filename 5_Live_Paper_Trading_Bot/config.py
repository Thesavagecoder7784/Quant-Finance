import os
from dotenv import load_dotenv
from strategies.sma_rsi_strategy import SmaRsiStrategy

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

TRADE_CAPITAL_PERCENTAGE = 0.05

STRATEGIES = [
    SmaRsiStrategy(
        symbols=["SPY", "QQQ", "DIA"],
        parameters={
            "short_sma_period": 20,
            "long_sma_period": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
        },
    ),
]
