import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch historical data and run Monte Carlo simulation
def monte_carlo_with_real_data(ticker, days_to_simulate=252, simulation_paths=50):
    # Get historical data for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Calculate annualized volatility from historical data
    historical_volatility = stock_data['Returns'].std() * np.sqrt(252)
    
    # Get the latest closing price
    S0 = stock_data['Close'].iloc[-1].item()
    
    # Calculate the drift (using historical mean return)
    mean_daily_return = stock_data['Returns'].mean()
    
    # Set up parameters for simulation
    dt = 1/252  # Daily time steps
    T = days_to_simulate / 252  # Time horizon in years
    n_steps = days_to_simulate
    
    # Initialize the stock price paths
    paths = np.zeros((simulation_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate random stock price paths
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(simulation_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mean_daily_return - 0.5 * historical_volatility**2) * dt 
                                            + historical_volatility * np.sqrt(dt) * Z)
    
    # Create time axis for plotting
    time_axis = pd.date_range(start=end_date, periods=n_steps+1, freq='B')
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    
    # Plot historical data
    historical_dates = stock_data.index[-30:]  # Last 30 days
    historical_prices = stock_data['Close'][-30:]
    plt.plot(historical_dates, historical_prices, 'k-', linewidth=2, label=f'{ticker} Historical')
    
    # Plot Monte Carlo paths
    for i in range(simulation_paths):
        plt.plot(time_axis, paths[i], 'b-', alpha=0.1)
    
    # Calculate statistics
    mean_path = np.mean(paths, axis=0)
    plt.plot(time_axis, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    # Add percentiles
    upper_95 = np.percentile(paths, 95, axis=0)
    lower_5 = np.percentile(paths, 5, axis=0)
    plt.plot(time_axis, upper_95, 'g--', linewidth=1.5, label='95th Percentile')
    plt.plot(time_axis, lower_5, 'g--', linewidth=1.5, label='5th Percentile')
    
    # Calculate final price statistics
    final_prices = paths[:, -1]
    expected_price = np.mean(final_prices)
    price_std = np.std(final_prices)
    
    # Add chart details
    plt.title(f'Monte Carlo Simulation for {ticker}\nExpected Price: ${expected_price:.2f} ± ${price_std:.2f} (1σ)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return {
        'current_price': S0,
        'expected_price': expected_price,
        'volatility': historical_volatility,
        'price_std': price_std,
        'upper_95': upper_95[-1],
        'lower_5': lower_5[-1]
    }

# Example usage
ticker = 'AAPL'  # Example: Apple Inc.
results = monte_carlo_with_real_data(ticker, days_to_simulate=252, simulation_paths=100)

print(f"Monte Carlo Results for {ticker}:")
print(f"Current Price: ${results['current_price']:.2f}")
print(f"Expected Price (1 year): ${results['expected_price']:.2f}")
print(f"Annual Volatility: {results['volatility']*100:.2f}%")
print(f"95% Confidence Interval: ${results['lower_5']:.2f} to ${results['upper_95']:.2f}")

plt.show()
