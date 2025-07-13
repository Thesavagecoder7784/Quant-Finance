import datetime
import logging
import pandas as pd
import numpy as np

# Import functions from other modules
import config
import data_ingestion
import sentiment_analysis
import database_manager
import backtesting
import visualization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_sentiment_pipeline(tickers: list[str], start_date: str, end_date: str):
    """
    Orchestrates the news sentiment ingestion and analysis pipeline.
    Fetches news, analyzes sentiment, and stores results in the database.
    """
    database_manager.create_database_schema()
    all_processed_headlines = []

    # --- Step 1: Fetch ALL general Reddit headlines ONCE ---
    logging.info("Fetching general Reddit headlines from all configured subreddits...")
    all_reddit_headlines_fetched = []
    for sub in config.REDDIT_SUBREDDITS:
        all_reddit_headlines_fetched.extend(data_ingestion.fetch_reddit(sub))
    logging.info(f"Finished fetching all general Reddit headlines. Total: {len(all_reddit_headlines_fetched)} posts.")

    # --- Step 2: Process headlines for each individual ticker ---
    for ticker in tickers:
        logging.info(f"Processing headlines for ticker: {ticker}")

        newsapi_headlines = data_ingestion.fetch_newsapi(ticker, start_date, end_date)
        finviz_headlines = data_ingestion.fetch_finviz(ticker)
        raw_headlines_for_ticker = newsapi_headlines + finviz_headlines

        for reddit_headline in all_reddit_headlines_fetched:
            headline_copy = reddit_headline.copy()
            text = headline_copy.get("title", "")
            inferred_ticker = None
            if text:
                if ticker.lower() in text.lower():
                    inferred_ticker = ticker
                else:
                    for k_ticker, keywords in config.TICKER_KEYWORDS_MAP.items():
                        if any(keyword.lower() in text.lower() for keyword in keywords):
                            inferred_ticker = k_ticker
                            break
            headline_copy["ticker"] = inferred_ticker if inferred_ticker else 'MARKET'
            if headline_copy["ticker"] == ticker or headline_copy["ticker"] == 'MARKET':
                raw_headlines_for_ticker.append(headline_copy)

        if not raw_headlines_for_ticker:
            logging.warning(f"No relevant headlines found for {ticker} from any active sources.")
            continue

        for headline in raw_headlines_for_ticker:
            text = headline.get("title", "")
            if not text:
                continue
            
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

def optimize_parameters(sentiment_df_train: pd.DataFrame, price_df_train: pd.DataFrame):
    logging.info("--- Starting Parameter Optimization (Grid Search) ---")
    best_sharpe = -np.inf
    best_params = {'buy_threshold': config.BUY_THRESHOLD, 'sell_threshold': config.SELL_THRESHOLD}

    # Define ranges for thresholds
    buy_threshold_range = np.arange(0.0, 0.5, 0.05)  # Example range
    sell_threshold_range = np.arange(-0.5, 0.0, 0.05) # Example range

    for buy_t in buy_threshold_range:
        for sell_t in sell_threshold_range:
            # Ensure buy_t is always greater than sell_t
            if buy_t <= sell_t:
                continue
            
            logging.debug(f"Testing params: Buy={buy_t:.2f}, Sell={sell_t:.2f}")
            _, daily_returns, _ = backtesting.run_backtest(
                sentiment_df=sentiment_df_train,
                price_df=price_df_train,
                initial_capital=config.INITIAL_CAPITAL,
                buy_threshold=buy_t,
                sell_threshold=sell_t,
                sentiment_model=config.SENTIMENT_MODEL_FOR_STRATEGY,
                trading_amount_per_trade_percentage=config.TRADING_AMOUNT_PER_TRADE_PERCENTAGE,
                transaction_cost_percentage=config.TRANSACTION_COST_PERCENTAGE
            )
            
            current_sharpe = backtesting.calculate_sharpe_ratio(daily_returns)
            logging.debug(f"  -> Sharpe: {current_sharpe:.2f}")

            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
                best_params['buy_threshold'] = buy_t
                best_params['sell_threshold'] = sell_t

    logging.info(f"--- Parameter Optimization Complete. Best Sharpe: {best_sharpe:.2f} with params: {best_params} ---")
    return best_params

def run_backtesting_pipeline(start_date: str, end_date: str, description: str, is_optimization_run: bool = False, optimized_params: dict = None):
    """
    Orchestrates the backtesting pipeline using data within a specified date range.
    """
    logging.info(f"--- Starting {description} Backtesting Pipeline ---")
    logging.info(f"Period: {start_date} to {end_date}")

    # 1. Get aggregated sentiment data
    sentiment_df = database_manager.get_daily_aggregates()
    if sentiment_df.empty:
        logging.error("No sentiment data available from the database. Cannot run backtest.")
        return

    # 2. Get historical price data
    price_df = data_ingestion.fetch_historical_prices(
        tickers=config.TARGET_TICKERS,
        start_date=start_date,
        end_date=end_date
    )
    if price_df.empty:
        logging.error("No price data available for the specified tickers and dates. Cannot run backtest.")
        return

    # Filter both dataframes to the specified date range for this run
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    sentiment_df = sentiment_df[(sentiment_df['date'] >= start_date_dt) & (sentiment_df['date'] <= end_date_dt)]
    price_df = price_df[(price_df.index >= start_date_dt) & (price_df.index <= end_date_dt)]

    if sentiment_df.empty or price_df.empty:
        logging.warning(f"No overlapping sentiment or price data in the period {start_date} to {end_date}. Skipping backtest.")
        return

    # --- Step 3.1: Run the backtest on the training data for optimization ---
    # This is where we find the best parameters
    optimized_params = optimize_parameters(sentiment_df_train=sentiment_df, price_df_train=price_df)

    # 3.2. Run the backtest on the testing data with optimized parameters
    portfolio_value, strategy_returns, benchmark_returns = backtesting.run_backtest(
        sentiment_df=sentiment_df,
        price_df=price_df,
        initial_capital=config.INITIAL_CAPITAL,
        buy_threshold=optimized_params['buy_threshold'],
        sell_threshold=optimized_params['sell_threshold'],
        sentiment_model=config.SENTIMENT_MODEL_FOR_STRATEGY,
        trading_amount_per_trade_percentage=config.TRADING_AMOUNT_PER_TRADE_PERCENTAGE,
        transaction_cost_percentage=config.TRANSACTION_COST_PERCENTAGE
    )

    # 4. Visualize the results
    if not portfolio_value.empty:
        visualization.plot_backtest_results(
            portfolio_value,
            strategy_returns,
            benchmark_returns,
            title=f"Sentiment Strategy Performance ({description})"
        )
    else:
        logging.warning("Backtest produced no results to visualize.")

    logging.info(f"--- Completed {description} Backtesting Pipeline ---")

# Ensure NLTK VADER lexicon is downloaded
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon')
    logging.info("NLTK VADER lexicon downloaded.")
''
''