import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings

# Suppress specific UserWarnings from scipy.optimize if too verbose
warnings.filterwarnings("ignore", message="The algorithm terminated successfully and looks like the chosen method found a solution.", category=UserWarning)

# --- Re-used Helper Functions from Markowitz Optimization ---

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
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data found for tickers: {tickers} in the specified date range.")
            return pd.DataFrame()

        prices_df = pd.DataFrame()

        # Handle different yfinance output structures
        if isinstance(data.columns, pd.MultiIndex):
            # Attempt to get 'Adj Close' from the first level of a MultiIndex
            if 'Adj Close' in data.columns.levels[0]:
                prices_df = data['Adj Close']
            else:
                # If 'Adj Close' is not a primary level, iterate through columns to find it
                # This handles cases like ('Price', 'Adj Close') or if 'Adj Close' is deeper
                adj_close_cols = []
                for col_tuple in data.columns:
                    if 'Adj Close' in col_tuple: # Check if 'Adj Close' is anywhere in the tuple
                        adj_close_cols.append(col_tuple)
                
                if adj_close_cols:
                    prices_df = data[adj_close_cols]
                    # Flatten the MultiIndex columns if they are not just (Metric, Ticker)
                    # We want the final column names to be just the tickers.
                    # This assumes the ticker symbol is always the last element of the column tuple.
                    if len(adj_close_cols[0]) > 1:
                        prices_df.columns = [col[-1] for col in prices_df.columns]
                else:
                    # Fallback to 'Close' if 'Adj Close' is not found in any MultiIndex level
                    close_cols = []
                    for col_tuple in data.columns:
                        if 'Close' in col_tuple:
                            close_cols.append(col_tuple)
                    if close_cols:
                        prices_df = data[close_cols]
                        if len(close_cols[0]) > 1:
                            prices_df.columns = [col[-1] for col in prices_df.columns]
                        print("Warning: 'Adj Close' not found in MultiIndex. Using 'Close' prices instead.")
                    else:
                        print("Error: Neither 'Adj Close' nor 'Close' found in MultiIndex columns. Cannot proceed.")
                        return pd.DataFrame()
        else:
            # For single ticker downloads, data.columns is a flat Index
            if 'Adj Close' in data.columns:
                prices_df = data[['Adj Close']] # Select as DataFrame
                prices_df.columns = tickers # Rename the column to the ticker symbol
            elif 'Close' in data.columns:
                prices_df = data[['Close']] # Select as DataFrame
                prices_df.columns = tickers # Rename the column to the ticker symbol
                print("Warning: 'Adj Close' not found for single ticker. Using 'Close' prices instead.")
            else:
                print("Error: Neither 'Adj Close' nor 'Close' column found for single ticker. Cannot proceed.")
                return pd.DataFrame()

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
    """
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev != 0 else 0.0
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def get_annualized_metrics(daily_prices):
    """
    Calculates annualized expected returns and the annualized covariance matrix
    from historical daily prices.
    """
    if not isinstance(daily_prices, pd.DataFrame) or daily_prices.empty:
        raise ValueError("daily_prices must be a non-empty pandas DataFrame.")
    
    asset_names = daily_prices.columns.tolist()
    daily_returns = daily_prices.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError("Insufficient data to calculate returns after dropping NaNs.")

    periods_per_year = 252 # Assuming daily returns for equities. Adjust if weekly (52) or monthly (12).
    expected_returns = daily_returns.mean() * periods_per_year
    cov_matrix = daily_returns.cov() * periods_per_year

    return expected_returns, cov_matrix, asset_names

def minimize_volatility_objective(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Objective function to minimize portfolio standard deviation.
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
    """
    num_assets = len(expected_returns)
    results = np.zeros((3, num_portfolios))
    all_weights = np.zeros((num_assets, num_portfolios))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return, portfolio_std_dev, sharpe = calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix, risk_free_rate
        )
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe
        all_weights[:, i] = weights

    all_portfolio_results = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

    initial_weights = np.array([1./num_assets] * num_assets)
    
    max_sharpe_opt = minimize(
        neg_sharpe_ratio_objective, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    max_sharpe_weights = max_sharpe_opt.x
    max_sharpe_return, max_sharpe_std_dev, max_sharpe_sharpe = calculate_portfolio_metrics(
        max_sharpe_weights, expected_returns, cov_matrix, risk_free_rate
    )
    max_sharpe_portfolio = {
        'weights': max_sharpe_weights, 'return': max_sharpe_return,
        'std_dev': max_sharpe_std_dev, 'sharpe_ratio': max_sharpe_sharpe
    }

    min_vol_opt = minimize(
        minimize_volatility_objective, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    min_vol_weights = min_vol_opt.x
    min_vol_return, min_vol_std_dev, min_vol_sharpe = calculate_portfolio_metrics(
        min_vol_weights, expected_returns, cov_matrix, risk_free_rate
    )
    min_vol_portfolio = {
        'weights': min_vol_weights, 'return': min_vol_return,
        'std_dev': min_vol_std_dev, 'sharpe_ratio': min_vol_sharpe
    }

    return max_sharpe_portfolio, min_vol_portfolio, all_portfolio_results

def generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate, num_points=50):
    """
    Generates points for the efficient frontier by optimizing for minimum volatility
    at various target returns.
    """
    num_assets = len(expected_returns)
    min_return_val = expected_returns.min()
    max_return_val = expected_returns.max()

    target_returns = np.linspace(min_return_val, max_return_val * 1.2 + 0.001, num_points)

    frontier_std_devs = []
    frontier_returns = []

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    for target_ret in target_returns:
        target_return_constraint = {
            'type': 'eq',
            'fun': lambda weights: calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)[0] - target_ret
        }
        
        initial_weights = np.array([1./num_assets] * num_assets)

        opt = minimize(
            minimize_volatility_objective, initial_weights,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method='SLSQP', bounds=bounds, constraints=[constraints, target_return_constraint]
        )
        
        if opt.success:
            frontier_returns.append(target_ret)
            frontier_std_devs.append(calculate_portfolio_metrics(opt.x, expected_returns, cov_matrix, risk_free_rate)[1])
    
    return np.array(frontier_returns), np.array(frontier_std_devs)

# --- Black-Litterman Specific Functions ---

def get_market_caps(tickers):
    """
    Fetches market capitalization for a list of tickers using yfinance.
    """
    market_caps = {}
    print(f"Fetching market caps for {tickers}...")
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            # 'marketCap' can sometimes be missing or None
            if 'marketCap' in info and info['marketCap'] is not None:
                market_caps[ticker_symbol] = info['marketCap']
            else:
                print(f"Warning: Market cap not found for {ticker_symbol}. Skipping.")
        except Exception as e:
            print(f"Error fetching market cap for {ticker_symbol}: {e}. Skipping.")
    
    # Convert to Pandas Series, handling cases where all market caps might be missing
    if not market_caps:
        print("Error: No market capitalization data could be fetched for any ticker.")
        return pd.Series(dtype=float)
        
    mc_series = pd.Series(market_caps)
    # Filter out any tickers that couldn't fetch market caps
    valid_tickers = [t for t in mc_series.index if t in tickers]
    if len(valid_tickers) < len(tickers):
        print(f"Warning: Only fetched market caps for {len(valid_tickers)} out of {len(tickers)} tickers.")
    return mc_series[valid_tickers]


def calculate_implied_equilibrium_returns(market_caps, cov_matrix, risk_aversion, risk_free_rate):
    """
    Calculates the implied equilibrium returns (Pi) based on market capitalization weights.

    Args:
        market_caps (pd.Series): Market capitalizations for the assets.
        cov_matrix (pd.DataFrame): Annualized covariance matrix of asset returns.
        risk_aversion (float): The risk aversion parameter (lambda).
        risk_free_rate (float): The annualized risk-free rate.

    Returns:
        pd.Series: Implied equilibrium excess returns (Pi).
    """
    if market_caps.empty or cov_matrix.empty:
        raise ValueError("Market caps or covariance matrix are empty, cannot calculate implied equilibrium returns.")
    
    # Align market caps and cov_matrix to common tickers
    common_tickers = list(set(market_caps.index) & set(cov_matrix.index))
    if not common_tickers:
        raise ValueError("No common tickers between market caps and covariance matrix indices.")

    market_caps = market_caps[common_tickers]
    cov_matrix = cov_matrix.loc[common_tickers, common_tickers]

    # Calculate market-value weights
    market_weights = market_caps / market_caps.sum()
    market_weights = market_weights.reindex(cov_matrix.columns).fillna(0) # Ensure weights are in same order as cov_matrix

    # Calculate implied equilibrium excess returns (Pi = lambda * Sigma * Market_Weights)
    pi = risk_aversion * np.dot(cov_matrix, market_weights)
    
    # Convert to Series with asset names
    pi_series = pd.Series(pi, index=cov_matrix.columns)
    
    # Implied total equilibrium returns = implied excess returns + risk-free rate
    implied_equilibrium_total_returns = pi_series + risk_free_rate
    
    return implied_equilibrium_total_returns

def black_litterman_posterior_returns(tau, cov_matrix, pi, P, Q, Omega=None):
    """
    Calculates the Black-Litterman posterior expected returns.

    Args:
        tau (float): Scalar representing the uncertainty in the prior (implied equilibrium)
                     and confidence in the views. Often a small value (e.g., 0.025 - 0.05).
        cov_matrix (pd.DataFrame): Annualized covariance matrix of asset returns.
        pi (pd.Series): Implied equilibrium excess returns (prior returns).
        P (np.array): A KxN matrix where K is the number of views and N is the number of assets.
                      Each row defines a view.
        Q (np.array): A Kx1 vector of expected returns for each view.
        Omega (np.array, optional): KxK diagonal matrix representing the uncertainty
                                    (variance) of each view. If None, it's calculated
                                    as a proportion of P * cov_matrix * P.T.

    Returns:
        pd.Series: Black-Litterman posterior expected returns.
    """
    # Ensure pi is aligned with cov_matrix columns
    pi = pi.reindex(cov_matrix.columns).fillna(0)
    
    # Prior covariance matrix for returns (Sigma_prior = tau * Sigma)
    sigma_prior = tau * cov_matrix

    # Calculate Omega if not provided
    if Omega is None:
        # A common heuristic for Omega:
        # Diagonal matrix with values based on the variance of the views implied by the prior
        # Omega = diag(P * (tau * Sigma) * P.T)
        Omega = np.diag(np.diag(np.dot(np.dot(P, sigma_prior), P.T)))
        if np.any(np.diag(Omega) <= 0):
            print("Warning: Non-positive diagonal elements in auto-calculated Omega. Using a small positive value.")
            Omega = Omega + np.identity(len(Omega)) * 1e-6 # Add small value to avoid singular matrix if zero

    # Black-Litterman Formula (using matrix algebra)
    # The posterior (view-adjusted) expected returns:
    # E[R] = inv( (tau*Sigma_prior)^-1 + P.T * Omega^-1 * P ) * ( (tau*Sigma_prior)^-1 * Pi + P.T * Omega^-1 * Q )
    
    # Calculate components
    inv_sigma_prior = np.linalg.inv(sigma_prior)
    inv_omega = np.linalg.inv(Omega)

    # First term of the inverse
    term1_inv = inv_sigma_prior + np.dot(np.dot(P.T, inv_omega), P)

    # Second term
    term2 = np.dot(inv_sigma_prior, pi) + np.dot(np.dot(P.T, inv_omega), Q)

    # Final posterior returns
    posterior_returns = np.dot(np.linalg.inv(term1_inv), term2)
    
    return pd.Series(posterior_returns, index=cov_matrix.columns)


# --- Main execution and example usage ---
if __name__ == "__main__":
    # --- User Input for Tickers and Date Range ---
    # Example: 'AAPL,MSFT,GOOG,AMZN,TSLA'
    tickers_input = input("Enter comma-separated stock tickers (e.g., AAPL,MSFT,GOOG,AMZN): ").upper()
    stock_tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]

    if not stock_tickers:
        print("No valid tickers entered. Exiting.")
        exit()

    # Use a reasonable historical period for covariance calculation
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d') # Last 5 years

    print(f"\nUsing historical data from {start_date} to {end_date} for calculations.")

    # 1. Fetch Actual Historical Price Data
    prices_df = fetch_historical_prices(stock_tickers, start_date, end_date)

    if prices_df.empty or len(prices_df.columns) < len(stock_tickers):
        print("Cannot proceed. Not all ticker data found or insufficient data after cleaning.")
        # Filter stock_tickers to only include those for which data was successfully fetched
        stock_tickers = prices_df.columns.tolist()
        if not stock_tickers:
            print("No valid tickers with data to analyze. Exiting.")
            exit()
        print(f"Proceeding with available tickers: {stock_tickers}")

    # 2. Calculate Annualized Covariance Matrix (and historical returns for reference)
    try:
        historical_expected_returns, cov_matrix, asset_names = get_annualized_metrics(prices_df)
    except ValueError as e:
        print(f"Error calculating annualized metrics: {e}")
        print("Please ensure enough historical data is available for the selected tickers and date range.")
        exit()

    print("\n" + "="*50 + "\n")
    print("--- Historical Annualized Expected Returns (for reference) ---")
    print(historical_expected_returns.round(4))
    print("\n--- Annualized Covariance Matrix ---")
    print(cov_matrix.round(4))
    print("\n" + "="*50 + "\n")

    # 3. Define Black-Litterman Parameters
    risk_free_rate = 0.02 # 2% annualized
    risk_aversion = 2.5   # Common value for equity markets (lambda)
    tau = 0.025           # Factor representing uncertainty in prior estimate / confidence in views

    # 4. Get Market Capitalizations
    # This can be time-consuming for many tickers
    market_caps = get_market_caps(stock_tickers)
    
    if market_caps.empty:
        print("Could not fetch market caps. Cannot calculate implied equilibrium returns. Exiting.")
        exit()

    # Filter `cov_matrix` to only include assets for which market caps were found
    common_tickers_for_bl = list(set(market_caps.index) & set(cov_matrix.index))
    if not common_tickers_for_bl:
        print("No common tickers for Market Caps and Covariance matrix. Exiting.")
        exit()
    
    market_caps = market_caps[common_tickers_for_bl]
    cov_matrix_bl = cov_matrix.loc[common_tickers_for_bl, common_tickers_for_bl]
    asset_names_bl = common_tickers_for_bl
    historical_expected_returns_bl = historical_expected_returns.loc[common_tickers_for_bl]

    # 5. Calculate Implied Equilibrium Returns (Prior)
    implied_equilibrium_returns = calculate_implied_equilibrium_returns(
        market_caps, cov_matrix_bl, risk_aversion, risk_free_rate
    )
    print("--- Implied Equilibrium Returns (Prior) ---")
    print(implied_equilibrium_returns.round(4))
    print("\n" + "="*50 + "\n")

    # 6. Define Investor Views (P and Q matrices, and optionally Omega)
    # This is where you would typically input your convictions.
    # For demonstration, we'll hardcode a few example views.
    # In a real system, these would come from an ML model or user input.

    num_assets_bl = len(asset_names_bl)
    
    # --- Example Views ---
    # View 1: Absolute view on AAPL - "AAPL will return 15% annually"
    # View 2: Relative view - "MSFT will outperform GOOG by 3% annually"

    # Important: P and Q must correspond to assets in `asset_names_bl`
    # Map ticker names to their index for constructing P matrix
    ticker_to_idx = {ticker: i for i, ticker in enumerate(asset_names_bl)}
    
    # Let's create views programmatically based on user input or a simple structure
    views_data = [] # List of tuples: (type, target_ticker, value) or (type, ticker1, ticker2, value)

    print("\n--- Defining Investor Views ---")
    print("Enter your views. Type 'done' when finished.")
    print("Example Absolute View: 'AAPL,0.15' (Ticker, Expected Annual Return)")
    print("Example Relative View: 'MSFT,GOOG,0.03' (Ticker Outperforming, Ticker Underperforming, Outperformance Percentage)")
    
    view_count = 0
    while True:
        view_input = input(f"View {view_count + 1} (or 'done'): ").strip()
        if view_input.lower() == 'done':
            break
        
        parts = view_input.split(',')
        if len(parts) == 2: # Absolute view
            ticker = parts[0].strip().upper()
            try:
                value = float(parts[1].strip())
                if ticker in asset_names_bl:
                    views_data.append(('absolute', ticker, value))
                    view_count += 1
                else:
                    print(f"Warning: Ticker '{ticker}' not in selected assets. Skipping view.")
            except ValueError:
                print("Invalid value for absolute view. Please use a number.")
        elif len(parts) == 3: # Relative view
            ticker1 = parts[0].strip().upper()
            ticker2 = parts[1].strip().upper()
            try:
                value = float(parts[2].strip())
                if ticker1 in asset_names_bl and ticker2 in asset_names_bl:
                    views_data.append(('relative', ticker1, ticker2, value))
                    view_count += 1
                else:
                    print(f"Warning: One or both tickers ('{ticker1}', '{ticker2}') not in selected assets. Skipping view.")
            except ValueError:
                print("Invalid value for relative view. Please use a number.")
        else:
            print("Invalid view format. Please try again.")

    if not views_data:
        print("No valid views entered. Proceeding with pure implied equilibrium returns (Markowitz with market prior).")
        P = np.array([]).reshape(0, num_assets_bl)
        Q = np.array([])
    else:
        P_list = []
        Q_list = []
        for view_type, *view_args in views_data:
            p_row = np.zeros(num_assets_bl)
            if view_type == 'absolute':
                ticker, value = view_args
                p_row[ticker_to_idx[ticker]] = 1
                Q_list.append(value)
            elif view_type == 'relative':
                ticker1, ticker2, value = view_args
                p_row[ticker_to_idx[ticker1]] = 1
                p_row[ticker_to_idx[ticker2]] = -1
                Q_list.append(value)
            P_list.append(p_row)
        
        P = np.array(P_list)
        Q = np.array(Q_list)

    if P.shape[0] == 0: # If no views were actually added
        print("No effective views were formulated. Black-Litterman will effectively revert to using implied equilibrium returns.")
        bl_posterior_returns = implied_equilibrium_returns
    else:
        print(f"\nConstructed P matrix:\n{P}")
        print(f"\nConstructed Q vector:\n{Q}")
        # 7. Calculate Black-Litterman Posterior Expected Returns
        bl_posterior_returns = black_litterman_posterior_returns(
            tau, cov_matrix_bl, implied_equilibrium_returns, P, Q
        )

    print("\n" + "="*50 + "\n")
    print("--- Black-Litterman Posterior Expected Returns ---")
    print(bl_posterior_returns.round(4))
    print("\n" + "="*50 + "\n")

    # 8. Optimize Portfolios using Black-Litterman Posterior Returns
    print("--- Optimizing Portfolios with Black-Litterman Returns ---")
    max_sharpe_portfolio_bl, min_vol_portfolio_bl, all_portfolio_results_bl = optimize_portfolios(
        bl_posterior_returns, cov_matrix_bl, risk_free_rate
    )

    print("\n--- Black-Litterman Maximum Sharpe Ratio Portfolio ---")
    print(f"  Return: {max_sharpe_portfolio_bl['return']:.4f}")
    print(f"  Volatility: {max_sharpe_portfolio_bl['std_dev']:.4f}")
    print(f"  Sharpe Ratio: {max_sharpe_portfolio_bl['sharpe_ratio']:.4f}")
    print("  Weights:")
    for i, weight in enumerate(max_sharpe_portfolio_bl['weights']):
        print(f"    {asset_names_bl[i]}: {weight:.4f}")

    print("\n--- Black-Litterman Minimum Volatility Portfolio ---")
    print(f"  Return: {min_vol_portfolio_bl['return']:.4f}")
    print(f"  Volatility: {min_vol_portfolio_bl['std_dev']:.4f}")
    print(f"  Sharpe Ratio: {min_vol_portfolio_bl['sharpe_ratio']:.4f}")
    print("  Weights:")
    for i, weight in enumerate(min_vol_portfolio_bl['weights']):
        print(f"    {asset_names_bl[i]}: {weight:.4f}")
    print("\n" + "="*50 + "\n")

    # 9. Generate and Plot Efficient Frontiers (Historical vs. Black-Litterman)
    print("--- Generating Efficient Frontiers ---")
    frontier_returns_hist, frontier_std_devs_hist = generate_efficient_frontier(
        historical_expected_returns_bl, cov_matrix_bl, risk_free_rate
    )
    frontier_returns_bl, frontier_std_devs_bl = generate_efficient_frontier(
        bl_posterior_returns, cov_matrix_bl, risk_free_rate
    )

    plt.figure(figsize=(14, 9))
    
    # Plot Historical Efficient Frontier
    plt.plot(frontier_std_devs_hist, frontier_returns_hist, color='blue', linestyle='-', linewidth=2, label='Historical Efficient Frontier')
    plt.scatter(min_vol_portfolio_bl['std_dev'], min_vol_portfolio_bl['return'], marker='o', color='darkblue', s=200, label='BL Min Vol Portfolio')
    plt.scatter(max_sharpe_portfolio_bl['std_dev'], max_sharpe_portfolio_bl['return'], marker='o', color='darkgreen', s=200, label='BL Max Sharpe Portfolio')


    # Plot Black-Litterman Efficient Frontier
    plt.plot(frontier_std_devs_bl, frontier_returns_bl, color='green', linestyle='--', linewidth=2, label='Black-Litterman Efficient Frontier')
    
    plt.title('Efficient Frontiers: Historical vs. Black-Litterman', fontsize=16)
    plt.xlabel('Portfolio Volatility (Annualized Standard Deviation)', fontsize=12)
    plt.ylabel('Portfolio Expected Return (Annualized)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(labelspacing=0.8)
    plt.tight_layout()
    plt.show()

    print("\nPlot generated comparing Historical and Black-Litterman Efficient Frontiers.")
