import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Import configuration
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_sentiment_data(df: pd.DataFrame):
    """
    Visualizes daily sentiment aggregates.
    """
    if df.empty:
        logging.warning("No data available for visualization.")
        return

    df['date'] = pd.to_datetime(df['date'])

    # Filter out 'MARKET' sentiment if desired for ticker-specific charts, or plot all
    # df_filtered = df[df['ticker'] != 'MARKET']

    # Plot VADER Compound Sentiment
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='date', y='avg_vader_compound', hue='ticker', marker='o')
    plt.title('Daily Average VADER Compound Sentiment by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Score')
    plt.grid(True)
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot FinBERT Positive Sentiment
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='date', y='avg_finbert_positive', hue='ticker', marker='o')
    plt.title('Daily Average FinBERT Positive Sentiment by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Average Positive Score')
    plt.grid(True)
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Optional: Plot FinBERT Net Sentiment (Positive - Negative)
    plt.figure(figsize=(14, 7))
    df['finbert_net_sentiment'] = df['avg_finbert_positive'] - df['avg_finbert_negative']
    sns.lineplot(data=df, x='date', y='finbert_net_sentiment', hue='ticker', marker='o')
    plt.title('Daily Average FinBERT Net Sentiment (Positive - Negative) by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Net Sentiment Score')
    plt.grid(True)
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    logging.info("Visualizations generated.")
    print("Sample of daily aggregates used for visualization:")
    print(df.head())