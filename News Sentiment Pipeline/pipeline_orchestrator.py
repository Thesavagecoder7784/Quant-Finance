import datetime
import logging

from data_ingestion import fetch_finviz, fetch_newsapi, fetch_reddit
from database_manager import create_database_schema, store_sentiment_data
from sentiment_analysis import analyze_finbert_sentiment, analyze_vader_sentiment

def run_sentiment_pipeline(tickers: list[str], start_date: str, end_date: str):
    create_database_schema()
    all_processed_headlines = []

    reddit_subreddits = ["stocks", "wallstreetbets", "investing"]
    ticker_keywords_map = {
        "AAPL": ["Apple", "iPhone", "Mac", "Tim Cook"],
        "MSFT": ["Microsoft", "Windows", "Azure", "Satya Nadella"],
        "GOOG": ["Google", "Alphabet", "Android", "Sundar Pichai"]
    }

    for ticker in tickers:
        logging.info(f"Processing headlines for ticker: {ticker}")

        newsapi_headlines = fetch_newsapi(ticker, start_date, end_date)
        finviz_headlines = fetch_finviz(ticker)

        current_reddit_headlines = []
        for sub in reddit_subreddits:
            current_reddit_headlines.extend(fetch_reddit(sub))

        raw_headlines = newsapi_headlines + finviz_headlines + current_reddit_headlines

        if not raw_headlines:
            logging.warning(f"No headlines found for {ticker} from any active sources in the specified period.")
            continue

        for headline in raw_headlines:
            text = headline.get("title", "")
            if not text:
                continue

            headline_ticker = headline.get("ticker")
            if headline_ticker is None and headline.get("source", "").startswith("Reddit"):
                for k_ticker, keywords in ticker_keywords_map.items():
                    if any(keyword.lower() in text.lower() for keyword in keywords):
                        headline_ticker = k_ticker
                        break
                if headline_ticker is None:
                    headline_ticker = 'MARKET'

            vader_sentiment = analyze_vader_sentiment(text)
            finbert_sentiment = analyze_finbert_sentiment(text)

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
        store_sentiment_data(all_processed_headlines)
    else:
        logging.info("No headlines processed to store.")

    logging.info("Sentiment pipeline run completed.")