import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_backtest_results(portfolio_value: pd.Series, strategy_returns: pd.Series, benchmark_returns: pd.Series, title: str):
    """
    Visualizes the backtesting results, comparing the strategy's portfolio value
    against a Buy & Hold benchmark.
    """
    if portfolio_value.empty:
        logging.warning("Portfolio value history is empty. Cannot generate plot.")
        return

    plt.figure(figsize=(14, 7))
    
    # Plot Strategy Portfolio Value
    portfolio_value.plot(label='Sentiment Strategy', color='blue')
    
    # Plot Buy & Hold Benchmark
    # Calculate the cumulative return of the benchmark and multiply by initial capital
    benchmark_value = config.INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()
    benchmark_value.plot(label='Buy & Hold Benchmark', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    logging.info(f"Generated plot: {title}")

def visualize_sentiment_data(df: pd.DataFrame):
    """
    Visualizes daily sentiment aggregates.
    """
    if df.empty:
        logging.warning("No data available for sentiment visualization.")
        return

    df['date'] = pd.to_datetime(df['date'])
    
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

    # Plot FinBERT Net Sentiment (Positive - Negative)
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

    logging.info("Sentiment visualizations generated.")
