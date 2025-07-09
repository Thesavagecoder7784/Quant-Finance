import logging

from config import ensure_nltk_vader, get_default_dates
from database_manager import get_daily_aggregates
from pipeline_orchestrator import run_sentiment_pipeline
from visualization import visualize_sentiment_data
from data_ingestion import fetch_yfinance_data
import numpy as np

def handle_outliers(df, column='Close', threshold=3):
    df['z_score'] = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[df['z_score'] > threshold]
    if not outliers.empty:
        logging.warning(f"Outliers detected in {column} column:\n{outliers}")
    return df[df['z_score'] <= threshold].drop(columns=['z_score'])

if __name__ == "__main__":
    ensure_nltk_vader()

    target_tickers = ["AAPL", "MSFT", "GOOG"]
    start_date_str, end_date_str = get_default_dates()

    logging.info(f"Starting pipeline for tickers: {target_tickers} from {start_date_str} to {end_date_str}")
    run_sentiment_pipeline(target_tickers, start_date_str, end_date_str)

    daily_sentiment_df = get_daily_aggregates()
    visualize_sentiment_data(daily_sentiment_df)

    # Fetch and process stock data
    for ticker in target_tickers:
        stock_data = fetch_yfinance_data(ticker, start_date_str, end_date_str)
        if not stock_data.empty:
            stock_data_no_outliers = handle_outliers(stock_data.copy())
            logging.info(f"Original vs. outlier-removed data points for {ticker}: {len(stock_data)} vs. {len(stock_data_no_outliers)}")
            # Further processing or visualization of stock_data_no_outliers can be done here.

    logging.info("Pipeline execution finished.")