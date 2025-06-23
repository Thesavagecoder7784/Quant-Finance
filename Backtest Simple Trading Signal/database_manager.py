import sqlite3
import pandas as pd
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_database_schema(db_name: str = config.DATABASE_NAME):
    """
    Creates the necessary tables in the SQLite database.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT,
            headline TEXT NOT NULL,
            source TEXT,
            vader_compound REAL,
            vader_classification TEXT,
            finbert_positive REAL,
            finbert_negative REAL,
            finbert_neutral REAL,
            finbert_classification TEXT
        )
    """)
    conn.commit()
    conn.close()
    logging.info(f"Database schema created/verified for {db_name}.")

def store_sentiment_data(data: list[dict], db_name: str = config.DATABASE_NAME):
    """
    Stores a list of sentiment analysis results into the SQLite database.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    for item in data:
        if not item.get("timestamp") or not item.get("headline"):
            logging.warning(f"Skipping record due to missing timestamp or headline: {item}")
            continue

        cursor.execute("""
            INSERT INTO sentiment_results (
                timestamp, ticker, headline, source,
                vader_compound, vader_classification,
                finbert_positive, finbert_negative, finbert_neutral, finbert_classification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.get("timestamp"),
            item.get("ticker"),
            item.get("headline"),
            item.get("source"),
            item.get("vader_compound"),
            item.get("vader_classification"),
            item.get("finbert_positive"),
            item.get("finbert_negative"),
            item.get("finbert_neutral"),
            item.get("finbert_classification")
        ))
    conn.commit()
    conn.close()
    logging.info(f"Stored {len(data)} sentiment results into {db_name}.")

def get_daily_aggregates(db_name: str = config.DATABASE_NAME) -> pd.DataFrame:
    """
    Computes daily average sentiment scores from the database.
    """
    conn = sqlite3.connect(db_name)
    query = """
        SELECT
            strftime('%Y-%m-%d', timestamp) as date,
            ticker,
            AVG(vader_compound) as avg_vader_compound,
            AVG(finbert_positive) as avg_finbert_positive,
            AVG(finbert_negative) as avg_finbert_negative,
            AVG(finbert_neutral) as avg_finbert_neutral
        FROM sentiment_results
        WHERE ticker IS NOT NULL AND ticker != '' -- Exclude records without proper ticker
        GROUP BY date, ticker
        ORDER BY date, ticker;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    logging.info("Generated daily sentiment aggregates.")
    return df