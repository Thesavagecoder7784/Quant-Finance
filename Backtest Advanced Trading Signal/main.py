import logging
import config
import pipeline_orchestrator
import database_manager
import data_ingestion
import backtesting
import visualization
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the entire sentiment analysis and backtesting pipeline.
    """
    logging.info("--- Starting Full Pipeline ---")

    # --- Step 1: Run the News Sentiment Pipeline ---
    # This step can be commented out if the database is already populated with recent data.
    # logging.info(f"Fetching news sentiment for tickers: {config.TARGET_TICKERS} from {config.NEWS_START_DATE} to {config.NEWS_END_DATE}")
    # pipeline_orchestrator.run_sentiment_pipeline(
    #     config.TARGET_TICKERS,
    #     config.NEWS_START_DATE,
    #     config.NEWS_END_DATE
    # )
    # logging.info("--- News Sentiment Pipeline Finished ---")

    # --- Step 2: Retrieve Aggregated Sentiment Data ---
    full_sentiment_df = database_manager.get_daily_aggregates(db_name=config.DATABASE_NAME)
    backtest_sentiment_df = full_sentiment_df[full_sentiment_df['ticker'] != 'MARKET']

    if backtest_sentiment_df.empty:
        logging.error("No relevant sentiment data for backtesting. Exiting.")
        return

    # --- Step 3: Fetch Historical Stock Prices ---
    historical_prices_df = data_ingestion.fetch_historical_prices(
        config.TARGET_TICKERS,
        config.BACKTEST_START_DATE,
        config.BACKTEST_END_DATE
    )

    if historical_prices_df.empty:
        logging.error("No historical price data for backtesting. Exiting.")
        return

    # Clean and align data
    historical_prices_df = historical_prices_df[config.TARGET_TICKERS].dropna(how='all')
    backtest_sentiment_df['date'] = pd.to_datetime(backtest_sentiment_df['date'])


    # --- Step 4: Run Standard Backtest ---
    logging.info("\n--- Running Standard Backtest ---")
    portfolio_value, strategy_returns, benchmark_returns = backtesting.run_backtest(
        sentiment_df=backtest_sentiment_df,
        price_df=historical_prices_df,
    )

    # --- Step 5: Run Walk-Forward Optimization ---
    logging.info("\n--- Running Walk-Forward Optimization ---")
    walk_forward_returns = backtesting.run_walk_forward_optimization(
        sentiment_df=backtest_sentiment_df,
        price_df=historical_prices_df,
    )

    # --- Step 6: Visualize Results ---
    logging.info("\n--- Visualizing Results ---")

    # Plot standard backtest results
    if not portfolio_value.empty:
        plt.figure(figsize=(14, 7))
        portfolio_value.plot(label='Sentiment Strategy (Standard Backtest)', color='blue')
        (config.INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()).plot(label='Buy & Hold Benchmark', color='red', linestyle='--')
        plt.title('Portfolio Value: Standard Backtest vs. Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        logging.warning("No portfolio value history from standard backtest to plot.")

    # Plot walk-forward optimization results
    if not walk_forward_returns.empty:
        plt.figure(figsize=(14, 7))
        (config.INITIAL_CAPITAL * (1 + walk_forward_returns).cumprod()).plot(label='Sentiment Strategy (Walk-Forward)', color='green')
        (config.INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()).plot(label='Buy & Hold Benchmark', color='red', linestyle='--')
        plt.title('Portfolio Value: Walk-Forward Optimization vs. Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        logging.warning("No returns from walk-forward optimization to plot.")

    # Visualize sentiment data
    visualization.visualize_sentiment_data(full_sentiment_df)

    logging.info("--- Full Pipeline Finished ---")


if __name__ == "__main__":
    main()
