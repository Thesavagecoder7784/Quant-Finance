import logging
import os
import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
# --- Database Configuration ---
DATABASE_NAME = "news_sentiment.db"

# --- Ticker Configuration ---
# Example tickers for sentiment analysis and backtesting
TARGET_TICKERS = ["AAPL", "MSFT", "GOOG"]

# Simple keyword mapping for Reddit ticker inference
TICKER_KEYWORDS_MAP = {
    "AAPL": ["Apple", "iPhone", "Mac", "Tim Cook"],
    "MSFT": ["Microsoft", "Windows", "Azure", "Satya Nadella"],
    "GOOG": ["Google", "Alphabet", "Android", "Sundar Pichai"]
}
REDDIT_SUBREDDITS = ["stocks", "wallstreetbets", "investing"]

# --- Date Range Configuration ---
TODAY = datetime.date.today()
# Fetch news sentiment for the last 7 days by default
NEWS_START_DATE = (TODAY - datetime.timedelta(days=7)).isoformat()
NEWS_END_DATE = TODAY.isoformat()

# Backtesting period (e.g., 2 years for more meaningful results)
BACKTEST_START_DATE = (TODAY - datetime.timedelta(days=365*2)).isoformat()
BACKTEST_END_DATE = TODAY.isoformat()

# --- Backtesting Strategy Parameters ---
INITIAL_CAPITAL = 100000
BUY_THRESHOLD = 0.1       # VADER compound score threshold to generate a buy signal
SELL_THRESHOLD = -0.05    # VADER compound score threshold to generate a sell signal
SENTIMENT_MODEL_FOR_STRATEGY = 'avg_vader_compound' # Can be 'avg_vader_compound' or 'finbert_sentiment_diff'
TRADING_AMOUNT_PER_TRADE_PERCENTAGE = 0.1 # Trade 10% of current available capital/holdings per signal
TRANSACTION_COST_PERCENTAGE = 0.001 # 0.1% transaction cost per trade (buy and sell)

# --- Sharpe Ratio Parameters ---
RISK_FREE_RATE_ANNUAL = 0.0001 # 0.01% annual risk-free rate for Sharpe calculation
TRADING_DAYS_PER_YEAR = 252 # Typical number of trading days in a year