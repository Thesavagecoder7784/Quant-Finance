import numpy as np
import matplotlib.pyplot as plt

def simulate_geometric_random_walk(num_steps, initial_price, mu, sigma):
    """
    Simulates a Geometric Random Walk, often used for asset prices.

    The price at each step is given by:
    S(t) = S(t-1) * exp( (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z )
    where Z is a standard normal random variable.
    Here, we assume dt = 1 for discrete steps.

    Args:
        num_steps (int): The number of steps in the walk.
        initial_price (float): The starting price of the asset.
        mu (float): The drift (average return) of the asset.
        sigma (float): The volatility (standard deviation of returns) of the asset.

    Returns:
        numpy.ndarray: An array containing the price of the asset at each step.
                       The length of the array will be `num_steps + 1`.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    if initial_price <= 0:
        raise ValueError("initial_price must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    prices = np.zeros(num_steps + 1)
    prices[0] = initial_price

    # Simulate random shocks from a standard normal distribution
    # dt is implicitly 1 for discrete steps
    random_shocks = np.random.normal(loc=0, scale=1, size=num_steps)

    # Calculate the multiplicative factors for each step
    # This is derived from the Ito's Lemma for geometric Brownian motion
    # with dt=1
    multiplicative_factors = np.exp((mu - 0.5 * sigma**2) + sigma * random_shocks)

    # Apply the multiplicative factors to get the price path
    for i in range(1, num_steps + 1):
        prices[i] = prices[i-1] * multiplicative_factors[i-1]

    return prices

if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_steps = 252 # Typical trading days in a year
    initial_price = 100.0
    mu = 0.05       # Annual drift (5%)
    sigma = 0.20    # Annual volatility (20%)

    # --- Simulate the Geometric Random Walk ---
    price_path = simulate_geometric_random_walk(num_steps, initial_price, mu, sigma)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(price_path, label=f'Geometric Random Walk (μ={mu}, σ={sigma})')
    plt.title('Simulation of a Geometric Random Walk (Asset Price Path)')
    plt.xlabel('Time Step (Days)')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Simulate multiple paths to show variability
    num_paths = 5
    plt.figure(figsize=(12, 6))
    for _ in range(num_paths):
        path = simulate_geometric_random_walk(num_steps, initial_price, mu, sigma)
        plt.plot(path, alpha=0.7)
    plt.title(f'Multiple Simulations of a Geometric Random Walk ({num_paths} Paths)')
    plt.xlabel('Time Step (Days)')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
