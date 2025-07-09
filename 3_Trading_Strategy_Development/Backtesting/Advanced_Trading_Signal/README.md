# Advanced Financial News Sentiment Pipeline with Algorithmic Backtesting

This repository contains a comprehensive pipeline designed to ingest financial news, analyze its sentiment, store the results, and, most importantly, backtest advanced trading strategies based on these sentiment signals. This tool is useful for exploring the potential of news sentiment in quantitative finance and systematic trading strategies.

## Features
1. Multi-Source Data Ingestion: Gathers financial news headlines from:
   - NewsAPI: For broad news coverage related to specific tickers.
   - Reddit: Fetches discussions from a wide array of financial subreddits, intelligently associating general news with relevant tickers using keyword mapping.
   - Finviz (Basic Web Scraping): Extracts headlines for specific tickers directly from Finviz.
2. Dual Sentiment Analysis: Applies two distinct sentiment models for robust analysis:
   - VADER (Valence Aware Dictionary and sEntiment Reasoner): A lexicon and rule-based sentiment analysis tool.
   - FinBERT: A transformer-based model specifically fine-tuned for the financial domain, offering more domain-specific accuracy.
3. Data Storage: Persists all raw headlines and their associated sentiment scores in an SQLite database (news_sentiment.db) for easy access and historical analysis.
4. Daily Aggregates: Computes daily average sentiment scores for each ticker, providing a concise summary of sentiment trends over time.
5. Historical Price Fetching: Integrates yfinance to reliably fetch historical stock price data for target tickers, essential for backtesting.

Algorithmic Backtesting Engine:
- Simulates trading a sentiment-driven strategy against historical data.
- Includes realistic transaction costs for both buy and sell orders.
- Features a robust trade pairing mechanism for accurate profit/loss and win rate calculation on closed positions.
- Calculates key performance metrics: Cumulative Returns, Maximum Drawdown, Sharpe Ratio, and Win Rate.
- Benchmarks the strategy against a simple "Buy & Hold" approach.

Visualization: Provides basic plots to visualize sentiment trends over time for various tickers and strategy performance compared to the benchmark.

## Why it Matters
Quantitative funds and algorithmic traders increasingly leverage alternative data sources like news sentiment to identify market inefficiencies and generate alpha. This pipeline demonstrates how sentiment signals can be quantified and evaluated, offering insights into market dynamics and potentially informing data-driven trading decisions. Rigorous backtesting is critical to validate if a strategy offers a statistical edge before risking real capital.

## Project Structure
```
3_Trading_Strategy_Development/Backtesting/Advanced_Trading_Signal/
├── .gitignore
├── config.py                 # Centralized configuration settings (API keys, parameters, thresholds)
├── data_ingestion.py         # Functions for fetching headlines (NewsAPI, Reddit, Finviz) and historical prices (yfinance)
├── database_manager.py       # Functions for creating database schema and storing/retrieving data
├── pipeline_orchestrator.py  # Orchestrates news ingestion and sentiment analysis
├── sentiment_analysis.py     # Functions for VADER and FinBERT sentiment analysis
├── backtesting.py            # Core backtesting logic, strategy simulation, and performance metrics
├── visualization.py          # Contains functions for visualizing aggregated sentiment data and backtest performance
└── main.py                   # The main entry point to run the entire pipeline (orchestrates sentiment + backtest)
└── requirements.txt          # Lists all necessary Python dependencies
```

## Setup and Installation
1. Clone the Repository (or save the files)
Ensure all the .py files and requirements.txt are in their respective locations within your project folder.

2. Install Dependencies
It is highly recommended to use a virtual environment to manage dependencies:
```bash
# Install required packages
pip install -r requirements.txt
```
```
requirements.txt:

requests
pandas
nltk
transformers
torch
praw
beautifulsoup4
yfinance
matplotlib
seaborn
numpy
```

3. Obtain and Configure API Keys
Update config.py with your API keys:

- NewsAPI: Sign up at NewsAPI.org for your API key. Update NEWSAPI_KEY.

- Reddit (PRAW):
1. Log in to Reddit. Go to Reddit App Preferences.
2. Click "create an app" (choose script).
3. Fill in name, redirect uri (e.g., http://localhost:8080).
4. Copy your client_id and secret.
5. Update REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and set a unique REDDIT_USER_AGENT in config.py.

- Finviz: Uses basic web scraping; no API key needed.

4. Download NLTK Data
The VADER sentiment analyzer requires specific lexicon data. The script will attempt to download it automatically, but you can do it manually if preferred:
```code
import nltk
nltk.download('vader_lexicon')
```
5. Configure Strategy Parameters (config.py)
Before running, customize the following in config.py:
```code
TARGET_TICKERS: Your list of stock symbols (e.g., ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN", "META", "TSLA", "JPM", "V", "UNH", "XOM", "WMT", "AVGO"]).

NEWS_START_DATE, NEWS_END_DATE: Period for fetching news data (e.g., last 7 days).

BACKTEST_START_DATE, BACKTEST_END_DATE: Crucial - The historical period for backtesting (e.g., 2+ years for reliable results).

INITIAL_CAPITAL: Starting funds for the simulation.

BUY_THRESHOLD, SELL_THRESHOLD: Sentiment scores that trigger buy/sell signals.

SENTIMENT_MODEL_FOR_STRATEGY: Choose avg_vader_compound or finbert_net_sentiment (if calculated).

TRADING_AMOUNT_PER_TRADE_PERCENTAGE: Percentage of capital/holdings to trade per signal.

TRANSACTION_COST_PERCENTAGE: Realistic costs per trade.

STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE: (If implemented) Percentages to trigger automatic exits.
```

## The Trading Strategy (Current Implementation in backtesting.py)
The strategy implemented in the run_backtest function is a simple sentiment-based momentum strategy:
- Sentiment Signal: Uses the avg_vader_compound score (from VADER analysis, averaged daily).
- Buy Rule: Initiates a buy if the stock's sentiment score is ≥ BUY_THRESHOLD.
- Sell Rule: Initiates a sell if the stock's sentiment score is ≤ SELL_THRESHOLD.
- Position Sizing: Trades a fixed percentage of available capital (for buys) or current holdings (for sells) as defined by TRADING_AMOUNT_PER_TRADE_PERCENTAGE.
- Transaction Costs: A TRANSACTION_COST_PERCENTAGE is applied to the value of every buy and sell.
- Trade Closure: Positions are closed based on SELL_THRESHOLD being met. Unclosed positions at the end of the backtest are liquidated at the final price.

(Future - Not yet in code but discussed): Implementation of Stop-Loss (e.g., sell if price drops 5% from entry) and Take-Profit (e.g., sell if price gains 10% from entry) will be added for more sophisticated risk management.

##  How to Run the Pipeline
1. Ensure all setup steps are complete.
2. Open your terminal or command prompt.
3. Navigate to the root directory of your project (the folder containing main.py).

Execute the main script:
```bash
python main.py
```

4. The script will sequentially:
   - Initialize the sentiment models.
   - Fetch and process Reddit headlines once.
   - For each target ticker, fetch NewsAPI and Finviz headlines.
5. Analyze sentiment for all collected headlines (including inferred Reddit posts) and store them in the news_sentiment.db database.
6. Aggregate daily sentiment scores.
7. Fetch historical stock prices for all target tickers over the backtesting period.
8. Run the backtesting simulation, applying the defined trading strategy.
9. Print a detailed summary of backtest results and performance metrics.
10. Generate and display sentiment trend and strategy performance visualizations.

## Interpreting the Output
The console output provides real-time progress and final results:

- News Ingestion: You'll see logs confirming fetches from NewsAPI, Finviz, and critically, a single fetch for all Reddit subreddits, followed by the total number of Reddit posts collected. This confirms the performance optimization for Reddit data.
- Sentiment Storage: Stored X sentiment results into news_sentiment.db. indicates the total number of sentiment records analyzed and saved. This number will be significantly higher with more tickers and subreddits.
- Historical Prices: Logs show successful fetching (or occasional timeouts if network issues occur) and the shape of the price data.
- Backtest Results: This is the core of the output. Key metrics include:
- Cumulative Returns: Directly compare Strategy Cumulative Returns vs. Benchmark (Buy & Hold) Cumulative Returns.
- Max Drawdown: Compare Strategy Max Drawdown vs. Benchmark Max Drawdown. Lower is better for your strategy.
- Sharpe Ratio: Compare Strategy Sharpe Ratio vs. Benchmark Sharpe Ratio. A higher positive Sharpe is better.
- Total Trades Executed: The total number of buy/sell actions.
- Win Rate: Winning Trades / Total Closed Trades. This is the percentage of profitable completed trades.

## Observations from Recent Runs:

Performance is Highly Dependent on Ticker Universe: Recent tests show that while the strategy might underperform the benchmark when applied to a broad set of 13 tickers (e.g., Strategy Cumulative Returns: -1.25%, Benchmark: 0.63%), it demonstrated outperformance on a smaller, more focused subset of 5 tickers (e.g., Strategy Cumulative Returns: 0.08%, Benchmark: -0.85%). This indicates that the current strategy parameters and sentiment signals might be more effective for specific types of stocks (e.g., highly liquid tech giants) than across diverse sectors.

Transaction Costs Matter: The introduction of transaction costs clearly showed a reduction in net returns, highlighting their real-world impact.

Win Rate vs. Profitability: A win rate around 50% or even slightly below (like 33.33% in the 5-ticker run) can still lead to positive returns if the average profit on winning trades significantly outweighs the average loss on losing trades. This underscores the importance of Average Win/Loss per trade, which is implicitly reflected in the overall cumulative returns.
