import logging
import pandas as pd

# Import functions/configurations from other modules
import config
import pipeline_orchestrator
import database_manager
import data_ingestion
import backtesting
import visualization
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- Step 1: Run the News Sentiment Pipeline ---
    logging.info(f"--- Starting News Sentiment Pipeline ---")
    logging.info(f"Fetching news sentiment for tickers: {config.TARGET_TICKERS} from {config.NEWS_START_DATE} to {config.NEWS_END_DATE}")
    
    # This will fetch news, analyze sentiment, and store in the database
    pipeline_orchestrator.run_sentiment_pipeline(
        config.TARGET_TICKERS,
        config.NEWS_START_DATE,
        config.NEWS_END_DATE
    )
    logging.info(f"--- News Sentiment Pipeline Finished ---")

    # --- Step 2: Retrieve Aggregated Sentiment Data for Backtesting/Visualization ---
    # Fetch sentiment data for the entire backtest period (which might be longer than news collection period)
    # Ensure this fetch covers the historical range needed for backtesting
    full_sentiment_df = database_manager.get_daily_aggregates(db_name=config.DATABASE_NAME)
    
    # Filter out 'MARKET' sentiment for ticker-specific backtest, or adjust logic if 'MARKET' is part of strategy
    # For backtesting individual tickers, we often exclude general market sentiment.
    backtest_sentiment_df = full_sentiment_df[full_sentiment_df['ticker'] != 'MARKET']

    if backtest_sentiment_df.empty:
        logging.error("No relevant sentiment data available for backtesting. Exiting backtest and visualization.")
    else:
        # --- Step 3: Fetch Historical Stock Prices for Backtesting Period ---
        historical_prices_df = data_ingestion.fetch_historical_prices(
            config.TARGET_TICKERS,
            config.BACKTEST_START_DATE,
            config.BACKTEST_END_DATE
        )

        if historical_prices_df.empty:
            logging.error("No historical price data available for backtesting. Exiting backtest and visualization.")
        else:
            # --- Step 4: Run Backtest ---
            logging.info(f"--- Starting Backtest ---")
            
            # You can choose which sentiment feature to use for the strategy
            # For FinBERT, you might derive a "net" sentiment (positive - negative)
            # backtest_sentiment_df['finbert_sentiment_diff'] = backtest_sentiment_df['avg_finbert_positive'] - backtest_sentiment_df['avg_finbert_negative']

            # Ensure historical_prices_df is aligned and clean for the target tickers
            historical_prices_df = historical_prices_df[config.TARGET_TICKERS].dropna(how='all')

            portfolio_value_history, strategy_returns, benchmark_returns = backtesting.run_backtest(
                sentiment_df=backtest_sentiment_df, # Use filtered sentiment
                price_df=historical_prices_df,
                initial_capital=config.INITIAL_CAPITAL,
                buy_threshold=config.BUY_THRESHOLD,
                sell_threshold=config.SELL_THRESHOLD,
                sentiment_model=config.SENTIMENT_MODEL_FOR_STRATEGY,
                trading_amount_per_trade_percentage=config.TRADING_AMOUNT_PER_TRADE_PERCENTAGE
            )
            logging.info(f"--- Backtest Finished ---")

            # --- Step 5: Visualize Performance (Optional, can be separate in a notebook) ---
            if not portfolio_value_history.empty:
                plt.figure(figsize=(14, 7))
                portfolio_value_history.plot(label='Sentiment Strategy', color='blue')
                (config.INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()).plot(label='Buy & Hold Benchmark', color='red', linestyle='--')
                plt.title('Portfolio Value Over Time: Sentiment Strategy vs. Buy & Hold')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                logging.warning("No portfolio value history to plot.")

        # --- Step 6: Visualize Sentiment Data ---
        logging.info(f"--- Starting Sentiment Visualization ---")
        visualization.visualize_sentiment_data(full_sentiment_df) # Visualize all sentiment data, including MARKET
        logging.info(f"--- Sentiment Visualization Finished ---")

    logging.info("Full Pipeline (Sentiment + Backtest + Visualization) execution finished.")