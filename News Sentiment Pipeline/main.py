import logging

from config import ensure_nltk_vader, get_default_dates
from database_manager import get_daily_aggregates
from pipeline_orchestrator import run_sentiment_pipeline
from visualization import visualize_sentiment_data

if __name__ == "__main__":
    ensure_nltk_vader()

    target_tickers = ["AAPL", "MSFT", "GOOG"]
    start_date_str, end_date_str = get_default_dates()

    logging.info(f"Starting pipeline for tickers: {target_tickers} from {start_date_str} to {end_date_str}")
    run_sentiment_pipeline(target_tickers, start_date_str, end_date_str)

    daily_sentiment_df = get_daily_aggregates()
    visualize_sentiment_data(daily_sentiment_df)
    logging.info("Pipeline execution finished.")