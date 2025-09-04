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
# Set the overall period for data ingestion to the last two months
NEWS_START_DATE = "2025-08-01"
NEWS_END_DATE = "2025-08-12"

# Define the split for training and testing (out-of-sample)
# Training period: First ~70% of the data
TRAIN_START_DATE = "2025-08-01"
TRAIN_END_DATE = "2025-08-10"

# Testing period: Last ~30% of the data, unseen by the strategy during training
TEST_START_DATE = "2025-08-11"
TEST_END_DATE = "2025-08-12"

# --- Backtesting Strategy Parameters ---
INITIAL_CAPITAL = 100000
BUY_THRESHOLD = 0.1       # Threshold to generate a buy signal
SELL_THRESHOLD = -0.05    # Threshold to generate a sell signal
SENTIMENT_MODEL_FOR_STRATEGY = 'finbert_net_sentiment' # Use FinBERT net sentiment for strategy
TRADING_AMOUNT_PER_TRADE_PERCENTAGE = 0.25 # Trade 25% of current available capital/holdings per signal
TRANSACTION_COST_PERCENTAGE = 0.001

# --- Sharpe Ratio Parameters ---
RISK_FREE_RATE_ANNUAL = 0.0001 # 0.01% annual risk-free rate for Sharpe calculation
TRADING_DAYS_PER_YEAR = 252

STOP_LOSS_PERCENTAGE = 0.07  # 7% stop loss
TAKE_PROFIT_PERCENTAGE = 0.25 # 25% take profit