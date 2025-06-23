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
TARGET_TICKERS = ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN"]

# Simple keyword mapping for Reddit ticker inference
TICKER_KEYWORDS_MAP = {
    "AAPL": ["Apple", "iPhone", "Mac", "Tim Cook", "Apple Watch", "iPad", "iOS", "App Store"],
    "MSFT": ["Microsoft", "Windows", "Azure", "Satya Nadella", "Xbox", "Office", "LinkedIn"],
    "GOOG": ["Google", "Alphabet", "Android", "Sundar Pichai", "YouTube", "Chrome", "Google Cloud"],
    "NVDA": ["NVIDIA", "Nvidia", "GPU", "AI", "Jensen Huang", "Graphics cards"],
    "AMZN": ["Amazon", "AWS", "Jeff Bezos", "Andy Jassy", "Prime", "Kindle", "Whole Foods"],
    "META": ["Meta", "Facebook", "Instagram", "WhatsApp", "Mark Zuckerberg", "Quest", "Reality Labs"],
    "TSLA": ["Tesla", "Elon Musk", "EV", "Cybertruck", "Model 3", "Model Y", "Gigafactory", "Autopilot"],
    "JPM": ["JPMorgan", "JP Morgan", "Jamie Dimon", "Chase", "Banking", "Financial services"],
    "V": ["Visa", "Payments", "Credit card", "Debit card"],
    "UNH": ["UnitedHealth", "Optum", "Healthcare", "Insurance"],
    "XOM": ["Exxon", "Mobil", "Oil", "Gas", "Energy"],
    "WMT": ["Walmart", "Retail", "Sam's Club", "e-commerce"],
    "AVGO": ["Broadcom", "Semiconductor", "Software", "Chips"]
}
REDDIT_SUBREDDITS = [
    "stocks",
    "wallstreetbets",
    "investing",
    "StockMarket",       
    "PersonalFinance",    
    "quant",               
    "algotrading",        
    "dividends",           
    "Daytrading",          
    "economy"              
]

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
BUY_THRESHOLD = 0.1       # Threshold to generate a buy signal
SELL_THRESHOLD = -0.05    # Threshold to generate a sell signal
SENTIMENT_MODEL_FOR_STRATEGY = 'avg_vader_compound' # Can be 'avg_vader_compound' or 'finbert_sentiment_diff'
TRADING_AMOUNT_PER_TRADE_PERCENTAGE = 0.1 # Trade 10% of current available capital/holdings per signal
TRANSACTION_COST_PERCENTAGE = 0.001

# --- Sharpe Ratio Parameters ---
RISK_FREE_RATE_ANNUAL = 0.0001 # 0.01% annual risk-free rate for Sharpe calculation
TRADING_DAYS_PER_YEAR = 252

STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.10 # 10% take profit