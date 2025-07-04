import yfinance as yf # Added for self-contained main.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the algorithms (functions only, as they are now self-contained for their __main__ blocks)
from pca_algorithm import run_pca
from svd_algorithm import run_svd

def fetch_financial_data_main(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for a list of tickers from Yahoo Finance.
    This is a local helper function for main.py.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame where columns are tickers and rows are dates,
                      containing adjusted close prices.
    """
    print(f"Main: Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    if isinstance(data, pd.Series): # Handle case where only one ticker is fetched
        data = data.to_frame(name=tickers[0])
    print("Main: Data fetch complete.")
    return data

def preprocess_data_main(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the financial data by calculating daily returns and handling NaNs.
    This is a local helper function for main.py.

    Args:
        data_df (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pd.DataFrame: DataFrame of daily returns, with NaNs dropped.
    """
    print("Main: Preprocessing data: calculating daily returns and handling NaNs...")
    returns_df = data_df.pct_change().dropna()
    print(f"Main: Original data shape: {data_df.shape}")
    print(f"Main: Returns data shape after dropping NaNs: {returns_df.shape}")
    if returns_df.empty:
        print("Main: Warning: Returns DataFrame is empty after preprocessing. This might indicate insufficient data.")
    return returns_df

def main():
    """
    Main function to run the financial linear algebra application.
    Fetches data, applies PCA, and applies SVD.
    """
    print("\n--- Running main.py orchestrator ---")
    # Define the financial assets and date range
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'GS']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d') # Last 5 years

    # 1. Fetch financial data (local to main.py)
    adj_close_prices = fetch_financial_data_main(tickers, start_date, end_date)
    if adj_close_prices.empty:
        print("Main: No data fetched. Exiting.")
        return

    print("\n--- Main: Raw Adjusted Close Prices (first 5 rows) ---")
    print(adj_close_prices.head())
    print(f"Main: Number of missing values before dropping: {adj_close_prices.isnull().sum().sum()}")
    adj_close_prices.dropna(inplace=True)
    if adj_close_prices.empty:
        print("Main: Adjusted close prices DataFrame became empty after dropping NaNs. Exiting.")
        return
    print(f"Main: Number of missing values after dropping: {adj_close_prices.isnull().sum().sum()}")


    # 2. Preprocess data (calculate daily returns, local to main.py)
    daily_returns = preprocess_data_main(adj_close_prices)
    if daily_returns.empty:
        print("Main: Daily returns DataFrame is empty. Exiting.")
        return

    print("\n--- Main: Daily Returns (first 5 rows) ---")
    print(daily_returns.head())
    print(f"Main: Daily Returns Shape: {daily_returns.shape}")

    # 3. Apply PCA for Factor Analysis using the imported function
    print("\n--- Main: Applying Principal Component Analysis (PCA) ---")
    num_pca_components = min(3, daily_returns.shape[1])
    if num_pca_components == 0:
        print("Main: Not enough features to perform PCA. Skipping PCA.")
    else:
        principal_components_df, explained_variance_ratio, pca_loadings = run_pca(daily_returns, n_components=num_pca_components)

        print(f"\nMain: PCA Results: {num_pca_components} Principal Components (first 5 rows):")
        print(principal_components_df.head())

        print("\nMain: PCA Explained Variance Ratio:")
        print(explained_variance_ratio)
        print(f"Main: Total variance explained by {num_pca_components} principal components: {explained_variance_ratio.sum():.4f}")

        print("\nMain: PCA Component Loadings (Weights of original assets on each PC):")
        print(pca_loadings)

        print("\n--- Main: Interpretation of PCA ---")
        print("The principal components represent orthogonal (uncorrelated) factors driving the stock returns.")
        print("PC1 typically captures the largest common variance, often interpreted as the overall market factor.")
        print("Subsequent PCs capture remaining variances, which could be industry-specific or other latent factors.")
        print("The 'loadings' indicate how each original stock contributes to these principal components.")


    # 4. Apply SVD for Dimensionality Reduction using the imported function
    print("\n\n--- Main: Applying Singular Value Decomposition (SVD) for Dimensionality Reduction ---")
    num_svd_components = min(2, daily_returns.shape[1] - 1)
    if num_svd_components <= 0:
        print("Main: Not enough features to perform SVD dimensionality reduction. Skipping SVD.")
    else:
        svd_reduced_data = run_svd(daily_returns, n_components=num_svd_components)

        print(f"\nMain: SVD Results: Data Reduced to {num_svd_components} Dimensions (first 5 rows):")
        print(svd_reduced_data.head())

        print("\n--- Main: Interpretation of SVD ---")
        print("SVD provides a way to decompose a matrix and can be used for dimensionality reduction.")
        print("By selecting a smaller number of singular values/vectors, we can project the data")
        print("into a lower-dimensional space while retaining most of the original variance.")
        print(f"The SVD reduced data now has {svd_reduced_data.shape[1]} columns instead of {daily_returns.shape[1]},")
        print("simplifying the dataset for further analysis or visualization.")


if __name__ == '__main__':
    main()
