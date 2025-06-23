import datetime
import logging

# Import functions from other modules
import config
import data_ingestion
import sentiment_analysis
import database_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_sentiment_pipeline(tickers: list[str], start_date: str, end_date: str):
    """
    Orchestrates the news sentiment ingestion and analysis pipeline.
    Fetches news, analyzes sentiment, and stores results in the database.
    """
    database_manager.create_database_schema()
    all_processed_headlines = []

    for ticker in tickers:
        logging.info(f"Processing headlines for ticker: {ticker}")

        # Ingest headlines from various sources
        newsapi_headlines = data_ingestion.fetch_newsapi(ticker, start_date, end_date)
        # Yahoo Finance news is a conceptual placeholder
        # yahoo_headlines = data_ingestion.fetch_yahoo_finance(ticker, start_date, end_date)
        finviz_headlines = data_ingestion.fetch_finviz(ticker)

        # Fetch Reddit headlines from general subreddits
        current_reddit_headlines = []
        for sub in config.REDDIT_SUBREDDITS:
            current_reddit_headlines.extend(data_ingestion.fetch_reddit(sub))

        raw_headlines = newsapi_headlines + finviz_headlines + current_reddit_headlines

        if not raw_headlines:
            logging.warning(f"No headlines found for {ticker} from any active sources in the specified period.")
            continue

        for headline in raw_headlines:
            text = headline.get("title", "")
            if not text:
                continue

            # Attempt to infer ticker for Reddit/general news if not already present
            headline_ticker = headline.get("ticker")
            if headline_ticker is None and headline.get("source", "").startswith("Reddit"):
                for k_ticker, keywords in config.TICKER_KEYWORDS_MAP.items():
                    if any(keyword.lower() in text.lower() for keyword in keywords):
                        headline_ticker = k_ticker
                        break
                if headline_ticker is None:
                    headline_ticker = 'MARKET' # Assign 'MARKET' if no specific ticker inferred

            vader_sentiment = sentiment_analysis.analyze_vader_sentiment(text)
            finbert_sentiment = sentiment_analysis.analyze_finbert_sentiment(text)

            processed_headline = {
                "timestamp": headline.get("timestamp", datetime.datetime.now().isoformat()),
                "ticker": headline_ticker,
                "headline": text,
                "source": headline.get("source", "Unknown"),
                "vader_compound": vader_sentiment["compound"],
                "vader_classification": vader_sentiment["classification"],
                "finbert_positive": finbert_sentiment["positive"],
                "finbert_negative": finbert_sentiment["negative"],
                "finbert_neutral": finbert_sentiment["neutral"],
                "finbert_classification": finbert_sentiment["classification"]
            }
            all_processed_headlines.append(processed_headline)

    if all_processed_headlines:
        database_manager.store_sentiment_data(all_processed_headlines)
    else:
        logging.info("No headlines processed to store.")

    logging.info("News sentiment pipeline run completed.")

# Ensure NLTK VADER lexicon is downloaded
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon')
    logging.info("NLTK VADER lexicon downloaded.")