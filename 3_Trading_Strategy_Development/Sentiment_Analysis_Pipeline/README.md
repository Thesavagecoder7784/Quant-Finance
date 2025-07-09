# Financial News Sentiment Pipeline
This project implements a comprehensive pipeline to ingest financial headlines from various sources, analyze their sentiment using dual models (VADER and FinBERT), store the results, and provide tools for daily aggregation and visualization. This kind of sentiment analysis is crucial for quantitative funds that rely on sentiment signals in their systematic trading strategies.

## Features
- Multi-Source Data Ingestion: Gathers financial news headlines from:
    - NewsAPI: For broad news coverage related to specific tickers.
    - Reddit: Fetches discussions from financial subreddits (e.g., r/stocks, r/wallstreetbets), attempting to associate general news with relevant tickers.
    - Finviz (Basic Web Scraping): Extracts headlines for specific tickers directly from Finviz.
    - yfinance: Fetches historical stock data for tickers.
- Dual Sentiment Analysis: Applies two distinct sentiment models for robust analysis:
    - VADER (Valence Aware Dictionary and sEntiment Reasoner): A lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
    - FinBERT: A transformer-based model (RoBERTa fine-tuned on financial text) designed for better accuracy in the financial domain.
- Data Storage: Persists all raw headlines and their associated sentiment scores in an SQLite database (news_sentiment.db) for easy access and historical analysis. The architecture supports easy migration to PostgreSQL.
- Daily Aggregates: Computes daily average sentiment scores for each ticker, providing a concise summary of sentiment trends.
- Visualization Guidance: Visualizes the sentiment with different tickers using matplotlib and seaborn
- Batch Processing: Efficiently processes headlines in batches for improved performance.
- Modular Architecture: The project is structured into modular Python files for better organization and maintainability.

## Why it Matters
Quantitative funds heavily rely on sentiment signals in systematic strategies. Understanding the collective mood expressed in financial news can provide valuable insights into market movements, enabling data-driven trading decisions.

## Project Structure
The project is structured into modular Python files for better organization and maintainability:

```
3_Trading_Strategy_Development/Sentiment_Analysis_Pipeline/
├── .gitignore
├── config.py                 # Configuration settings (API keys, database name)
├── data_ingestion.py         # Functions for fetching headlines from various sources (NewsAPI, Reddit, Finviz)
├── data_processor.py         # Functions for processing raw data before sentiment analysis
├── database_manager.py       # Functions for creating database schema and storing/retrieving data
├── main.py                   # The main entry point to run the entire pipeline
├── pipeline_orchestrator.py  # Orchestrates the data flow through ingestion, analysis, and storage
├── requirements.txt          # Lists all necessary Python dependencies
├── sentiment_analysis.py     # Functions for VADER and FinBERT sentiment analysis
└── visualization.py          # Contains functions for visualizing aggregated sentiment data
```
## Setup and Installation
1. Clone the Repository (or save the files)
Assuming you have these files saved locally.

2. Install Dependencies
It is highly recommended to use a virtual environment.

Install required packages
```code
pip install -r requirements.txt
```

3. Obtain API Keys
This pipeline requires API keys for data ingestion. You'll need to set these in config.py (or directly in main.py if you prefer a single file structure, as in the current state of the provided code snippet).

NewsAPI:
1. Go to NewsAPI.org.
2. Sign up for a free developer account.
3. Your API key will be displayed on your dashboard.
4. Update NEWSAPI_KEY in config.py.

Reddit (PRAW):
1. Log in to your Reddit account.
2. Go to Reddit App Preferences.
3. Click "create an app" or "create another app".
4. Choose script as the application type.
5. Fill in the name (e.g., "MySentimentPipeline"), description (optional), about url (optional, can be http://localhost), and redirect uri (e.g., http://localhost:8080).
6. After creation, your client_id (below the app name, e.g., ipKNG8aB9hgodf64vt4Bew) and secret (next to "secret") will be displayed.
7. Update REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in config.py.
8. Set REDDIT_USER_AGENT to a unique, descriptive string (e.g., "my_financial_sentiment_app_v1.0").

Finviz: No API key is strictly required as the current implementation uses basic web scraping. However, note that web scraping can be fragile and dependent on Finviz's website structure. Always review robots.txt and terms of service.

4. Download NLTK Data
The VADER sentiment analyzer requires specific lexicon data. The script will attempt to download it automatically, but you can do it manually if preferred:

```code
import nltk
nltk.download('vader_lexicon')
```

## How to Run the Pipeline
The core logic is contained within the main.py file.

Ensure all dependencies are installed and API keys are configured.

Run the main.py script:
```code
python main.py
```

The script will:
1. Create an SQLite database file (news_sentiment.db) if it doesn't exist.
2. Fetch news headlines for the specified target_tickers (e.g., AAPL, MSFT, GOOG) for the last 7 days.
3. Run VADER and FinBERT sentiment analysis on each headline.
4. Store the results in the database.
5. Compute and print daily aggregated sentiment scores.
6. Generate and display basic line plots for average VADER and FinBERT positive sentiment (if matplotlib and seaborn are installed and the visualize_sentiment_data function is uncommented in main.py's __name__ == "__main__" block).

Interpreting the Output
The console output will display INFO and ERROR messages indicating the pipeline's progress and any issues (e.g., Reddit API errors due to incorrect credentials).

The most important part of the output will be the DataFrame containing daily aggregates, similar to this:

        date  ticker  avg_vader_compound  avg_finbert_positive  avg_finbert_negative  avg_finbert_neutral
0  2025-06-15    AAPL            0.226167              0.649718              0.266490             0.083792
1  2025-06-15    MSFT            0.145217              0.857913              0.046976             0.095111
...

date: The aggregation date.

ticker: The stock symbol (AAPL, MSFT, GOOG) or MARKET for general news where a specific ticker couldn't be inferred (e.g., from some Reddit posts).

avg_vader_compound: Average VADER sentiment score (-1 to +1). Positive indicates overall positive news, negative for negative news.

avg_finbert_positive, avg_finbert_negative, avg_finbert_neutral: Average probabilities (0 to 1) for a headline being positive, negative, or neutral according to FinBERT.

These aggregated scores represent the overall sentiment of news for a given ticker on a given day. They can be used to track sentiment trends and potentially correlate them with stock price movements.
