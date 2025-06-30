import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings("ignore", message="The algorithm terminated successfully and looks like the chosen method found a solution.", category=UserWarning)

def fetch_historical_prices(tickers, start_date, end_date):
    """
    Fetches historical adjusted closing prices for a list of tickers
    from Yahoo Finance using yfinance.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame of adjusted closing prices. Index is Date, columns are tickers.
                      Returns an empty DataFrame if no data is found or an error occurs.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    try:
        # Download data for multiple tickers
        # progress=False hides the download progress bar for cleaner output
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data found for tickers: {tickers} in the specified date range.")
            return pd.DataFrame()

        # Check if the downloaded data has a MultiIndex column (typical for multiple tickers)
        # or a simple Index (typical for a single ticker).
        if isinstance(data.columns, pd.MultiIndex):
            # For multiple tickers, data['Close'] directly extracts a DataFrame
            # with tickers as columns (e.g., Close | AAPL, Close | MSFT) -> AAPL, MSFT
            prices_df = data['Close']
        else:
            # For a single ticker, data['Close'] returns a Series.
            # Convert it to a DataFrame by selecting it using double brackets,
            # then explicitly set its column name to the ticker.
            prices_df = data[['Close']]
            prices_df.columns = tickers # Set the single column name to the ticker

        # Drop any rows with NaN values that might result from missing data for some tickers.
        # This ensures all assets have complete data for the period.
        prices_df = prices_df.dropna()

        if prices_df.empty:
            print(f"No complete data available for all tickers after dropping NaNs.")
            return pd.DataFrame()

        print("Data fetched successfully.")
        return prices_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Calculates the portfolio's expected return, standard deviation (volatility),
    and Sharpe Ratio.

    Args:
        weights (np.array): A numpy array of asset weights in the portfolio.
        expected_returns (np.array or pd.Series): Annualized expected returns for each asset.
        cov_matrix (np.array or pd.DataFrame): Annualized covariance matrix of asset returns.
        risk_free_rate (float): The annualized risk-free rate.

    Returns:
        tuple: (portfolio_return, portfolio_std_dev, sharpe_ratio)
    """
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev != 0 else 0.0
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def get_annualized_metrics(daily_prices):
    """
    Calculates annualized expected returns and the annualized covariance matrix
    from historical daily prices.

    Args:
        daily_prices (pd.DataFrame): DataFrame of daily closing prices.
                                    Columns are asset tickers, index is date.

    Returns:
        tuple: (expected_returns, cov_matrix, asset_names)
        - expected_returns (pd.Series): Annualized expected returns for each asset.
        - cov_matrix (pd.DataFrame): Annualized covariance matrix of asset returns.
        - asset_names (list): List of asset tickers.
    """
    if not isinstance(daily_prices, pd.DataFrame) or daily_prices.empty:
        raise ValueError("daily_prices must be a non-empty pandas DataFrame.")
    
    asset_names = daily_prices.columns.tolist()

    # Calculate daily returns
    daily_returns = daily_prices.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError("Insufficient data to calculate returns after dropping NaNs.")

    # Define periods per year for annualization (e.g., 252 for daily trading days)
    periods_per_year = 252

    # Annualize expected returns (mean daily return * periods_per_year)
    expected_returns = daily_returns.mean() * periods_per_year

    # Annualize covariance matrix (daily covariance * periods_per_year)
    cov_matrix = daily_returns.cov() * periods_per_year

    return expected_returns, cov_matrix, asset_names

def minimize_volatility_objective(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Objective function to minimize portfolio standard deviation.
    Used for finding portfolios on the efficient frontier and the Minimum Volatility Portfolio.
    """
    return calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)[1]

def neg_sharpe_ratio_objective(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Objective function to minimize the negative Sharpe Ratio, which is equivalent
    to maximizing the Sharpe Ratio.
    """
    return -calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolios(expected_returns, cov_matrix, risk_free_rate, num_portfolios=10000):
    """
    Performs Monte Carlo simulations to generate random portfolios and then
    uses optimization to find the Minimum Volatility and Maximum Sharpe Ratio portfolios.

    Args:
        expected_returns (pd.Series): Annualized expected returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix of asset returns.
        risk_free_rate (float): The annualized risk-free rate.
        num_portfolios (int): Number of random portfolios to generate for visualization.

    Returns:
        tuple: (max_sharpe_portfolio, min_vol_portfolio, all_portfolio_results)
        - max_sharpe_portfolio (dict): Optimal weights and metrics for Max Sharpe Ratio portfolio.
        - min_vol_portfolio (dict): Optimal weights and metrics for Minimum Volatility portfolio.
        - all_portfolio_results (pd.DataFrame): Returns, std devs, and Sharpe Ratios of random portfolios.
    """
    num_assets = len(expected_returns)
    results = np.zeros((3, num_portfolios)) # 0: return, 1: std_dev, 2: sharpe
    all_weights = np.zeros((num_assets, num_portfolios))

    # Constraints for optimization: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds for optimization: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # --- Monte Carlo Simulation (for initial visual spread) ---
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # Normalize weights to sum to 1
        
        portfolio_return, portfolio_std_dev, sharpe = calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix, risk_free_rate
        )
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe
        all_weights[:, i] = weights

    all_portfolio_results = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

    # --- Optimization for Maximum Sharpe Ratio Portfolio ---
    # Initial guess: equal weights
    initial_weights = np.array([1./num_assets] * num_assets)
    
    max_sharpe_opt = minimize(
        neg_sharpe_ratio_objective,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP', # Sequential Least SQuares Programming
        bounds=bounds,
        constraints=constraints
    )
    max_sharpe_weights = max_sharpe_opt.x
    max_sharpe_return, max_sharpe_std_dev, max_sharpe_sharpe = calculate_portfolio_metrics(
        max_sharpe_weights, expected_returns, cov_matrix, risk_free_rate
    )
    max_sharpe_portfolio = {
        'weights': max_sharpe_weights,
        'return': max_sharpe_return,
        'std_dev': max_sharpe_std_dev,
        'sharpe_ratio': max_sharpe_sharpe
    }

    # --- Optimization for Minimum Volatility Portfolio ---
    min_vol_opt = minimize(
        minimize_volatility_objective,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    min_vol_weights = min_vol_opt.x
    min_vol_return, min_vol_std_dev, min_vol_sharpe = calculate_portfolio_metrics(
        min_vol_weights, expected_returns, cov_matrix, risk_free_rate
    )
    min_vol_portfolio = {
        'weights': min_vol_weights,
        'return': min_vol_return,
        'std_dev': min_vol_std_dev,
        'sharpe_ratio': min_vol_sharpe
    }

    return max_sharpe_portfolio, min_vol_portfolio, all_portfolio_results

def generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate, num_points=50):
    """
    Generates points for the efficient frontier by optimizing for minimum volatility
    at various target returns.

    Args:
        expected_returns (pd.Series): Annualized expected returns for each asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix of asset returns.
        risk_free_rate (float): The annualized risk-free rate.
        num_points (int): Number of points to plot on the efficient frontier.

    Returns:
        tuple: (frontier_returns, frontier_std_devs)
    """
    num_assets = len(expected_returns)
    
    # Ensure min_return is not greater than max_return, which can happen with short historical data or negative returns
    min_return_val = expected_returns.min()
    max_return_val = expected_returns.max()

    # Define a range of target returns to optimize for. Extend the range slightly beyond max_return_val.
    # Add a small epsilon to max_return_val to ensure linspace works correctly if min and max are very close.
    target_returns = np.linspace(min_return_val, max_return_val * 1.2 + 0.001, num_points)

    frontier_std_devs = []
    frontier_returns = []

    # Constraints for optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    for target_ret in target_returns:
        # Constraint to hit the target return
        target_return_constraint = {
            'type': 'eq',
            'fun': lambda weights: calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)[0] - target_ret
        }
        
        # Initial guess for weights
        initial_weights = np.array([1./num_assets] * num_assets)

        opt = minimize(
            minimize_volatility_objective,
            initial_weights,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=[constraints, target_return_constraint]
        )
        
        if opt.success:
            frontier_returns.append(target_ret)
            frontier_std_devs.append(calculate_portfolio_metrics(opt.x, expected_returns, cov_matrix, risk_free_rate)[1])
    
    return np.array(frontier_returns), np.array(frontier_std_devs)

# --- Main execution and example usage ---
if __name__ == "__main__":
    # --- User Input for Tickers and Date Range ---
    # Example: 'AAPL,MSFT,GOOG,AMZN,TSLA'
    tickers_input = input("Enter comma-separated stock tickers (e.g., AAPL,MSFT,GOOG): ").upper()
    stock_tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]

    # You can adjust these dates or prompt the user for them
    start_date = input("Enter start date (YYYY-MM-DD, e.g., 2020-01-01): ")
    end_date = input("Enter end date (YYYY-MM-DD, e.g., 2023-01-01): ")

    if not stock_tickers:
        print("No valid tickers entered. Exiting.")
        exit()

    # 1. Fetch Actual Historical Price Data
    prices_df = fetch_historical_prices(stock_tickers, start_date, end_date)

    if prices_df.empty:
        print("Cannot proceed with optimization due to no data. Please check tickers and date range.")
        exit()

    print("\n--- Fetched Daily Prices (first 5 rows) ---")
    print(prices_df.head())
    print("\n" + "="*50 + "\n")

    # 2. Calculate Annualized Expected Returns and Covariance Matrix
    try:
        expected_returns, cov_matrix, asset_names = get_annualized_metrics(prices_df)
    except ValueError as e:
        print(f"Error calculating annualized metrics: {e}")
        print("Please ensure enough historical data is available for the selected tickers and date range.")
        exit()

    print("--- Annualized Expected Returns ---")
    print(expected_returns.round(4))
    print("\n--- Annualized Covariance Matrix ---")
    print(cov_matrix.round(4))
    print("\n" + "="*50 + "\n")

    # 3. Define Risk-Free Rate
    risk_free_rate = 0.02 # 2% annualized - this could also be user input or fetched from a source

    # 4. Optimize Portfolios (Max Sharpe and Min Volatility)
    print("--- Optimizing Portfolios ---")
    max_sharpe_portfolio, min_vol_portfolio, all_portfolio_results = optimize_portfolios(
        expected_returns, cov_matrix, risk_free_rate
    )

    print("\n--- Maximum Sharpe Ratio Portfolio ---")
    print(f"  Return: {max_sharpe_portfolio['return']:.4f}")
    print(f"  Volatility: {max_sharpe_portfolio['std_dev']:.4f}")
    print(f"  Sharpe Ratio: {max_sharpe_portfolio['sharpe_ratio']:.4f}")
    print("  Weights:")
    for i, weight in enumerate(max_sharpe_portfolio['weights']):
        print(f"    {asset_names[i]}: {weight:.4f}")

    print("\n--- Minimum Volatility Portfolio ---")
    print(f"  Return: {min_vol_portfolio['return']:.4f}")
    print(f"  Volatility: {min_vol_portfolio['std_dev']:.4f}")
    print(f"  Sharpe Ratio: {min_vol_portfolio['sharpe_ratio']:.4f}")
    print("  Weights:")
    for i, weight in enumerate(min_vol_portfolio['weights']):
        print(f"    {asset_names[i]}: {weight:.4f}")
    print("\n" + "="*50 + "\n")

    # 5. Generate Efficient Frontier Points
    print("--- Generating Efficient Frontier ---")
    frontier_returns, frontier_std_devs = generate_efficient_frontier(
        expected_returns, cov_matrix, risk_free_rate
    )

    # 6. Plotting the Efficient Frontier
    plt.figure(figsize=(12, 8))
    
    # Plot all random portfolios
    plt.scatter(
        all_portfolio_results['Volatility'],
        all_portfolio_results['Return'],
        c=all_portfolio_results['Sharpe Ratio'],
        cmap='viridis',
        marker='o',
        alpha=0.3,
        s=10,
        label='Random Portfolios'
    )
    plt.colorbar(label='Sharpe Ratio')

    # Plot the Efficient Frontier
    plt.plot(frontier_std_devs, frontier_returns, color='red', linestyle='--', linewidth=2, label='Efficient Frontier')

    # Plot the Minimum Volatility Portfolio
    plt.scatter(
        min_vol_portfolio['std_dev'],
        min_vol_portfolio['return'],
        marker='*',
        color='blue',
        s=500,
        label='Minimum Volatility Portfolio'
    )

    # Plot the Maximum Sharpe Ratio Portfolio
    plt.scatter(
        max_sharpe_portfolio['std_dev'],
        max_sharpe_portfolio['return'],
        marker='*',
        color='green',
        s=500,
        label='Maximum Sharpe Ratio Portfolio'
    )
    
    plt.title('Markowitz Efficient Frontier', fontsize=16)
    plt.xlabel('Portfolio Volatility (Annualized Standard Deviation)', fontsize=12)
    plt.ylabel('Portfolio Expected Return (Annualized)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(labelspacing=0.8)
    plt.tight_layout()
    plt.show()

    print("\nPlot generated showing Efficient Frontier, Minimum Volatility Portfolio, and Maximum Sharpe Ratio Portfolio.")
