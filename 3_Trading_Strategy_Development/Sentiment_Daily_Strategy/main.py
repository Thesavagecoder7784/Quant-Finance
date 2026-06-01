import logging
import config
import pipeline_orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- Step 1: Run the News Sentiment Pipeline for the full date range ---
    # This ensures your database has all the necessary data.
    # Uncomment and run this if your database needs to be populated or updated.
    logging.info("--- Starting News Sentiment Ingestion ---")
    pipeline_orchestrator.run_sentiment_pipeline(
        tickers=config.TARGET_TICKERS,
        start_date=config.NEWS_START_DATE,
        end_date=config.NEWS_END_DATE
    )
    logging.info("--- News Sentiment Ingestion Finished ---")

    # --- Step 2: Run the Backtest on the Training Period to optimize parameters ---
    # This is your "in-sample" test where you tune parameters.
    logging.info("--- Starting Training (In-Sample) Backtest and Parameter Optimization ---")
    optimized_params = pipeline_orchestrator.run_backtesting_pipeline(
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        description="Training (In-Sample)",
        is_optimization_run=True # Indicate this is for optimization
    )
    logging.info("--- Training (In-Sample) Backtest and Parameter Optimization Finished ---")

    if optimized_params:
        # --- Step 3: Run the Backtest on the Testing Period with optimized parameters ---
        # This is your "out-of-sample" test. The results here are the most honest measure of performance.
        logging.info("--- Starting Testing (Out-of-Sample) Backtest with Optimized Parameters ---")
        pipeline_orchestrator.run_backtesting_pipeline(
            start_date=config.TEST_START_DATE,
            end_date=config.TEST_END_DATE,
            description="Testing (Out-of-Sample)",
            optimized_params=optimized_params # Pass the optimized parameters
        )
        logging.info("--- Testing (Out-of-Sample) Backtest Finished ---")
    else:
        logging.warning("Parameter optimization failed or returned no parameters. Skipping testing phase.")

    logging.info("--- Full Train/Test Pipeline Finished ---")
''