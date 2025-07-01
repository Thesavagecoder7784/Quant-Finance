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

    # --- Step 1: Fetch ALL general Reddit headlines ONCE ---
    # These are not ticker-specific at this stage, and will be inferred later.
    logging.info("Fetching general Reddit headlines from all configured subreddits...")
    all_reddit_headlines_fetched = []
    for sub in config.REDDIT_SUBREDDITS:
        all_reddit_headlines_fetched.extend(data_ingestion.fetch_reddit(sub))
    logging.info(f"Finished fetching all general Reddit headlines. Total: {len(all_reddit_headlines_fetched)} posts.")


    # --- Step 2: Process headlines for each individual ticker ---
    for ticker in tickers:
        logging.info(f"Processing headlines for ticker: {ticker}")

        # Ingest headlines from ticker-specific sources
        newsapi_headlines = data_ingestion.fetch_newsapi(ticker, start_date, end_date)
        finviz_headlines = data_ingestion.fetch_finviz(ticker)

        # Combine all raw headlines for the current ticker
        # Initialize with ticker-specific headlines
        raw_headlines_for_ticker = newsapi_headlines + finviz_headlines

        # Add Reddit headlines that are relevant to this ticker (or general market)
        # We iterate through all fetched Reddit headlines and associate them
        for reddit_headline in all_reddit_headlines_fetched:
            # Create a copy to avoid modifying the original list item during inference
            headline_copy = reddit_headline.copy()
            text = headline_copy.get("title", "")
            
            # Attempt to infer ticker for this Reddit headline
            inferred_ticker = None
            if text:
                # Prioritize direct ticker match if the ticker is in the headline text
                if ticker.lower() in text.lower():
                    inferred_ticker = ticker
                else:
                    # Fallback to general keyword mapping
                    for k_ticker, keywords in config.TICKER_KEYWORDS_MAP.items():
                        if any(keyword.lower() in text.lower() for keyword in keywords):
                            inferred_ticker = k_ticker
                            break
            
            # Assign 'MARKET' if no specific ticker inferred, or the actual ticker
            headline_copy["ticker"] = inferred_ticker if inferred_ticker else 'MARKET'
            
            # Only add to this ticker's processing if it's directly relevant or general market
            # If you want ONLY ticker-specific Reddit posts, you'd change 'MARKET' below
            if headline_copy["ticker"] == ticker or headline_copy["ticker"] == 'MARKET':
                raw_headlines_for_ticker.append(headline_copy)


        if not raw_headlines_for_ticker:
            logging.warning(f"No relevant headlines found for {ticker} from any active sources in the specified period.")
            continue

        for headline in raw_headlines_for_ticker:
            text = headline.get("title", "")
            if not text:
                continue

            # Ensure the ticker is correctly assigned after all inference attempts
            # If it came from NewsAPI or Finviz, it already has the correct ticker.
            # If it's a Reddit post, it has the inferred ticker or 'MARKET'.
            final_headline_ticker = headline.get("ticker")

            vader_sentiment = sentiment_analysis.analyze_vader_sentiment(text)
            finbert_sentiment = sentiment_analysis.analyze_finbert_sentiment(text)

            processed_headline = {
                "timestamp": headline.get("timestamp", datetime.datetime.now().isoformat()),
                "ticker": final_headline_ticker,
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