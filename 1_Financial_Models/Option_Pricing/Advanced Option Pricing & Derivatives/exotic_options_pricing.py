
import numpy as np

def monte_carlo_asian_option(S0, K, T, r, sigma, num_simulations, num_steps, option_type='call'):
    """
    Prices an Asian option using Monte Carlo simulation.

    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    num_simulations: Number of Monte Carlo paths
    num_steps: Number of steps in each simulation path
    option_type: 'call' or 'put'
    """
    dt = T / num_steps
    prices = np.zeros((num_simulations, num_steps + 1))
    prices[:, 0] = S0

    for i in range(num_simulations):
        for j in range(1, num_steps + 1):
            z = np.random.standard_normal()
            prices[i, j] = prices[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate average price for each path (arithmetic average)
    average_prices = np.mean(prices[:, 1:], axis=1) # Exclude S0 from average

    if option_type == 'call':
        payoffs = np.maximum(average_prices - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - average_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    price = np.exp(-r * T) * np.mean(payoffs)
    return price

# Example Usage (will be added later for all options)
if __name__ == "__main__":
    # Parameters for Asian Option
    S0_asian = 100
    K_asian = 100
    T_asian = 1
    r_asian = 0.05
    sigma_asian = 0.2
    num_simulations_asian = 10000
    num_steps_asian = 252 # Daily steps for a year

    print("Pricing Asian Options:")
    asian_call_price = monte_carlo_asian_option(S0_asian, K_asian, T_asian, r_asian, sigma_asian, num_simulations_asian, num_steps_asian, 'call')
    print(f"Asian Call Option Price: {asian_call_price:.4f}")

    asian_put_price = monte_carlo_asian_option(S0_asian, K_asian, T_asian, r_asian, sigma_asian, num_simulations_asian, num_steps_asian, 'put')
    print(f"Asian Put Option Price: {asian_put_price:.4f}")

def monte_carlo_barrier_option(S0, K, T, r, sigma, num_simulations, num_steps, barrier, option_type='down_and_out_call'):
    """
    Prices a Barrier option (Down-and-Out Call) using Monte Carlo simulation.

    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    num_simulations: Number of Monte Carlo paths
    num_steps: Number of steps in each simulation path
    barrier: Barrier level
    option_type: Currently only 'down_and_out_call' is implemented
    """
    dt = T / num_steps
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = np.zeros(num_steps + 1)
        path[0] = S0
        knocked_out = False
        for j in range(1, num_steps + 1):
            z = np.random.standard_normal()
            path[j] = path[j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            if path[j] <= barrier:
                knocked_out = True
                break
        
        if not knocked_out and option_type == 'down_and_out_call':
            payoffs[i] = max(path[-1] - K, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    return price

    # Parameters for Barrier Option (Down-and-Out Call)
    S0_barrier = 100
    K_barrier = 100
    T_barrier = 1
    r_barrier = 0.05
    sigma_barrier = 0.2
    num_simulations_barrier = 10000
    num_steps_barrier = 252
    barrier_level = 90

    print("\nPricing Barrier Options (Down-and-Out Call):")
    barrier_call_price = monte_carlo_barrier_option(S0_barrier, K_barrier, T_barrier, r_barrier, sigma_barrier, num_simulations_barrier, num_steps_barrier, barrier_level, 'down_and_out_call')
    print(f"Down-and-Out Call Option Price: {barrier_call_price:.4f}")

def monte_carlo_lookback_option(S0, T, r, sigma, num_simulations, num_steps, option_type='floating_lookback_call'):
    """
    Prices a Lookback option (Floating Strike Call) using Monte Carlo simulation.

    S0: Initial stock price
    T: Time to maturity (in years)
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    num_simulations: Number of Monte Carlo paths
    num_steps: Number of steps in each simulation path
    option_type: Currently only 'floating_lookback_call' is implemented
    """
    dt = T / num_steps
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = np.zeros(num_steps + 1)
        path[0] = S0
        for j in range(1, num_steps + 1):
            z = np.random.standard_normal()
            path[j] = path[j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        if option_type == 'floating_lookback_call':
            min_price = np.min(path)
            payoffs[i] = max(path[-1] - min_price, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    return price

    # Parameters for Lookback Option (Floating Strike Call)
    S0_lookback = 100
    T_lookback = 1
    r_lookback = 0.05
    sigma_lookback = 0.2
    num_simulations_lookback = 10000
    num_steps_lookback = 252

    print("\nPricing Lookback Options (Floating Strike Call):")
    lookback_call_price = monte_carlo_lookback_option(S0_lookback, T_lookback, r_lookback, sigma_lookback, num_simulations_lookback, num_steps_lookback, 'floating_lookback_call')
    print(f"Floating Lookback Call Option Price: {lookback_call_price:.4f})")

def monte_carlo_rainbow_option(S1_0, S2_0, K, T, r, sigma1, sigma2, rho, num_simulations, num_steps, option_type='call_on_max'):
    """
    Prices a Rainbow option (Call on Max of two assets) using Monte Carlo simulation.

    S1_0: Initial price of asset 1
    S2_0: Initial price of asset 2
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (annual)
    sigma1: Volatility of asset 1
    sigma2: Volatility of asset 2
    rho: Correlation between asset 1 and asset 2
    num_simulations: Number of Monte Carlo paths
    num_steps: Number of steps in each simulation path
    option_type: Currently only 'call_on_max' is implemented
    """
    dt = T / num_steps
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        S1_path = np.zeros(num_steps + 1)
        S2_path = np.zeros(num_steps + 1)
        S1_path[0] = S1_0
        S2_path[0] = S2_0

        for j in range(1, num_steps + 1):
            z1 = np.random.standard_normal()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal()

            S1_path[j] = S1_path[j-1] * np.exp((r - 0.5 * sigma1**2) * dt + sigma1 * np.sqrt(dt) * z1)
            S2_path[j] = S2_path[j-1] * np.exp((r - 0.5 * sigma2**2) * dt + sigma2 * np.sqrt(dt) * z2)
        
        if option_type == 'call_on_max':
            payoffs[i] = max(max(S1_path[-1], S2_path[-1]) - K, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    return price

    # Parameters for Rainbow Option (Call on Max)
    S1_0_rainbow = 100
    S2_0_rainbow = 100
    K_rainbow = 100
    T_rainbow = 1
    r_rainbow = 0.05
    sigma1_rainbow = 0.2
    sigma2_rainbow = 0.25
    rho_rainbow = 0.5
    num_simulations_rainbow = 10000
    num_steps_rainbow = 252

    print("\nPricing Rainbow Options (Call on Max of two assets):")
    rainbow_call_price = monte_carlo_rainbow_option(S1_0_rainbow, S2_0_rainbow, K_rainbow, T_rainbow, r_rainbow, sigma1_rainbow, sigma2_rainbow, rho_rainbow, num_simulations_rainbow, num_steps_rainbow, 'call_on_max')
    print(f"Rainbow Call on Max Option Price: {rainbow_call_price:.4f})")
