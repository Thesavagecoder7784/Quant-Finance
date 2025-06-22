import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATABASE_NAME

def visualize_sentiment_data(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'])
    df_filtered = df[df['ticker'] != 'MARKET']

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='date', y='avg_vader_compound', hue='ticker', marker='o')
    plt.title('Daily Average VADER Compound Sentiment by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Score')
    plt.grid(True)
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='date', y='avg_finbert_positive', hue='ticker', marker='o')
    plt.title('Daily Average FinBERT Positive Sentiment by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Average Positive Score')
    plt.grid(True)
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    logging.info("Visualization guidance provided. Use libraries like Matplotlib/Seaborn.")
    if not df.empty:
        logging.info("DataFrame for visualization is not empty. Ready for plotting.")
        print("Sample of daily aggregates for visualization:")
        print(df.head())
    else:
        logging.warning("No data available for visualization.")