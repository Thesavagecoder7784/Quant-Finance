# Live Paper Trading Bot

This directory contains a live paper trading bot that uses the Alpaca API to trade based on a defined strategy.

## Strategy

The bot currently uses an SMA/RSI crossover strategy. The strategy is defined in `strategies/sma_rsi_strategy.py`.

The bot trades the following symbols:
- SPY
- QQQ
- DIA
- TSLA
- NVDA
- AMD

## Capital Allocation

The bot allocates a percentage of its capital to each trade. The capital is distributed evenly among all the symbols in the strategy. This is to avoid exhausting the buying power on a few trades.

## How to Run

1.  Install the required packages: `pip install -r requirements.txt`
2.  Set up your Alpaca API keys in a `.env` file. See `.env.example` for an example.
3.  Run the bot: `python main.py`
