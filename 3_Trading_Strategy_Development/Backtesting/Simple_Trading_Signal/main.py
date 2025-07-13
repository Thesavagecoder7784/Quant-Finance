import logging
import config
import pipeline_orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- Step 1: Run the News Sentiment Pipeline for the full date range ---
    # This ensures your database has all the necessary data.
    # You can comment this out if your database is already populated for the required dates.
    #logging.info("--- Starting News Sentiment Ingestion ---")
    #pipeline_orchestrator.run_sentiment_pipeline(
    #    tickers=config.TARGET_TICKERS,
    #    start_date=config.NEWS_START_DATE,
    #    end_date=config.NEWS_END_DATE
    #)
    #logging.info("--- News Sentiment Ingestion Finished ---")

    # --- Step 2: Run the Backtest on the Training Period ---
    # This is your "in-sample" test where you can tune parameters.
    pipeline_orchestrator.run_backtesting_pipeline(
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        description="Training (In-Sample)"
    )

    # --- Step 3: Run the Backtest on the Testing Period ---
    # This is your "out-of-sample" test. The results here are the most honest measure of performance.
    # Use the same parameters you finalized during the training phase.
    pipeline_orchestrator.run_backtesting_pipeline(
        start_date=config.TEST_START_DATE,
        end_date=config.TEST_END_DATE,
        description="Testing (Out-of-Sample)"
    )

    logging.info("--- Full Train/Test Pipeline Finished ---")
''