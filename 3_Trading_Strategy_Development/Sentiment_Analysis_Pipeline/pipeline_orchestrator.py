import datetime
import logging

from data_ingestion import fetch_finviz, fetch_newsapi, fetch_reddit
from database_manager import create_database_schema, store_sentiment_data
from sentiment_analysis import analyze_finbert_sentiment, analyze_vader_sentiment
from data_processor import assign_ticker_to_reddit_headlines, process_headlines_in_batch
from config import get_finbert_models, vader_analyzer

def run_sentiment_pipeline(tickers: list[str], start_date: str, end_date: str):
    create_database_schema()
    all_processed_headlines = []

    reddit_subreddits = ["stocks", "wallstreetbets", "investing"]
    ticker_keywords_map = {
        "AAPL": ["Apple", "iPhone", "Mac", "Tim Cook"],
        "MSFT": ["Microsoft", "Windows", "Azure", "Satya Nadella"],
        "GOOG": ["Google", "Alphabet", "Android", "Sundar Pichai"]
    }

    # --- Reddit Processing (once) ---
    logging.info("Fetching and processing Reddit headlines.")
    reddit_headlines = []
    for sub in reddit_subreddits:
        reddit_headlines.extend(fetch_reddit(sub))
    
    reddit_headlines = assign_ticker_to_reddit_headlines(reddit_headlines, ticker_keywords_map)
    
    finbert_tokenizer, finbert_model = get_finbert_models()
    sentiment_analyzers = {
        'vader': vader_analyzer,
        'finbert': lambda texts: analyze_finbert_sentiment(texts, finbert_tokenizer, finbert_model)  # Passing models
    }

    processed_reddit_headlines = process_headlines_in_batch(reddit_headlines, sentiment_analyzers)
    all_processed_headlines.extend(processed_reddit_headlines)
    logging.info(f"Finished processing {len(reddit_headlines)} Reddit headlines.")


    # --- Ticker-specific Processing ---
    for ticker in tickers:
        logging.info(f"Processing headlines for ticker: {ticker}")

        newsapi_headlines = fetch_newsapi(ticker, start_date, end_date)
        finviz_headlines = fetch_finviz(ticker)
        
        raw_headlines = newsapi_headlines + finviz_headlines

        if not raw_headlines:
            logging.warning(f"No ticker-specific headlines found for {ticker} from any active sources in the specified period.")
            continue

        processed_ticker_headlines = process_headlines_in_batch(raw_headlines, sentiment_analyzers)
        for headline in processed_ticker_headlines:
            if 'ticker' not in headline or not headline['ticker']:
                headline['ticker'] = ticker.upper()
        all_processed_headlines.extend(processed_ticker_headlines)

    if all_processed_headlines:
        store_sentiment_data(all_processed_headlines)
    else:
        logging.info("No headlines processed to store.")

    logging.info("Sentiment pipeline run completed.")