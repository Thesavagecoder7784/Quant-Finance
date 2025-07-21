import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- VaR Calculation Functions ---

def historical_var(returns, confidence_level=0.95):
    """Calculates Historical VaR."""
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[index]
    return var

def parametric_var(returns, confidence_level=0.95):
    """Calculates Parametric VaR (Variance-Covariance)."""
    mean = np.mean(returns)
    std = np.std(returns)
    z = norm.ppf(1 - confidence_level)
    var = -(mean + z * std)
    return var

def monte_carlo_var(returns, num_simulations=10000, confidence_level=0.95):
    """Calculates VaR using Monte Carlo simulation."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    simulated_returns = np.random.normal(mu, sigma, num_simulations)
    sorted_returns = np.sort(simulated_returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var_percent = -sorted_returns[index]
    return var_percent

# --- Visualization Functions ---

def visualize_portfolio_performance(portfolio_returns, initial_capital, var_param_pct):
    """
    Visualizes the growth of the portfolio over time with a dynamic VaR band.
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Calculate portfolio value over time
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value_over_time = cumulative_returns * initial_capital
    
    # Calculate the dynamic VaR band based on the portfolio's value each day
    var_dollar_band = var_param_pct * portfolio_value_over_time

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot portfolio value
    ax.plot(portfolio_value_over_time.index, portfolio_value_over_time.values, color='#005A9C', linewidth=2, label='Portfolio Value')
    
    # Plot VaR band
    ax.fill_between(portfolio_value_over_time.index, 
                     portfolio_value_over_time - var_dollar_band, 
                     portfolio_value_over_time, 
                     color='#FF5733', alpha=0.3, label='95% VaR Zone (Parametric)')

    ax.set_title('Portfolio Performance and Value at Risk (VaR)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Annotate final value
    final_value = portfolio_value_over_time.iloc[-1]
    ax.text(portfolio_value_over_time.index[-1], final_value, f' Final Value: ${final_value:,.2f}', 
             verticalalignment='center', horizontalalignment='right', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6))

    plt.tight_layout(pad=2.0)
    plt.show()

def visualize_returns_distribution(portfolio_returns, var_hist_pct, var_param_pct, var_mc_pct):
    """
    Visualizes the daily returns distribution with VaR levels.
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot histogram and KDE
    sns.histplot(portfolio_returns, bins=50, kde=True, stat="density", ax=ax,
                 color='#005A9C', alpha=0.6, label='Daily Returns Distribution')
    
    # Plot VaR lines
    ax.axvline(x=-var_hist_pct, color='#FFC300', linestyle='--', linewidth=2, label=f'Historical VaR: {-var_hist_pct:.2%}')
    ax.axvline(x=-var_param_pct, color='#FF5733', linestyle='--', linewidth=2, label=f'Parametric VaR: {-var_param_pct:.2%}')
    ax.axvline(x=-var_mc_pct, color='#C70039', linestyle='--', linewidth=2, label=f'Monte Carlo VaR: {-var_mc_pct:.2%}')
    
    ax.set_title('Daily Returns Distribution with 95% VaR Levels', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Daily Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=2.0)
    plt.show()


if __name__ == "__main__":
    # 1. Define a diversified portfolio and weights
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'V'] # Tech, Finance
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # Equal weighting
    portfolio_value = 100_000

    # 2. Fetch historical data
    print(f"Fetching historical data for portfolio: {', '.join(tickers)}...")
    data = yf.download(tickers, start='2020-01-01', end='2025-01-01')['Close']

    if data.empty:
        print("Could not download data. Please check tickers and network connection.")
    else:
        # 3. Calculate daily returns for each asset
        daily_returns = data.pct_change().dropna()

        # 4. Calculate daily portfolio returns
        portfolio_returns = daily_returns.dot(weights)

        # 5. Calculate Portfolio VaR as a percentage
        var_hist_pct = historical_var(portfolio_returns)
        var_param_pct = parametric_var(portfolio_returns)
        var_mc_pct = monte_carlo_var(portfolio_returns)

        # Calculate VaR in dollar terms for the initial portfolio value
        var_hist_dollar = var_hist_pct * portfolio_value
        var_param_dollar = var_param_pct * portfolio_value
        var_mc_dollar = var_mc_pct * portfolio_value

        print("\n--- Portfolio VaR Analysis ---")
        print(f"Initial Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Historical VaR (95%): ${var_hist_dollar:,.2f} ({-var_hist_pct:.2%})")
        print(f"Parametric VaR (95%): ${var_param_dollar:,.2f} ({-var_param_pct:.2%})")
        print(f"Monte Carlo VaR (95%): ${var_mc_dollar:,.2f} ({-var_mc_pct:.2%})")
        print("\nThis means that, with 95% confidence, the portfolio is not expected to lose more than the calculated VaR amounts in a single day.")

        # 6. Visualize the results
        visualize_portfolio_performance(portfolio_returns, portfolio_value, var_param_pct)
        visualize_returns_distribution(portfolio_returns, var_hist_pct, var_param_pct, var_mc_pct)

