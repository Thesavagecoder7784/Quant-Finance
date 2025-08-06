"""
Feynman-Kac Theorem: Connecting PDEs and Stochastic Processes

This script demonstrates the Feynman-Kac theorem, which provides a powerful link
between parabolic partial differential equations (PDEs) and the expected value of
stochastic processes. In finance, it establishes that the solution to the
Black-Scholes PDE is the expected discounted payoff of a derivative under the
risk-neutral measure.

We will demonstrate this by:
1.  Pricing a European call option by solving the Black-Scholes PDE numerically
    using a finite difference method.
2.  Pricing the same option by simulating the underlying asset's price path using a
    Monte Carlo simulation (Geometric Brownian Motion) and calculating the
    discounted expected payoff.
3.  Comparing the results from both methods to show they converge, as predicted
    by the theorem.
"""

import numpy as np
from scipy.stats import norm

# --- 1. Monte Carlo Simulation (Expectation of Stochastic Process) ---

def monte_carlo_call_price(S, K, T, r, sigma, n_simulations):
    """
    Prices a European call option using Monte Carlo simulation.
    This represents the E[e^(-rT) * f(S_T)] part of Feynman-Kac.
    """
    # Generate random asset paths
    Z = np.random.standard_normal(n_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoff for each path
    payoff = np.maximum(ST - K, 0)

    # Discount the average payoff
    price = np.exp(-r * T) * np.mean(payoff)
    return price

# --- 2. Finite Difference Method (Solving the PDE) ---

def finite_difference_call_price(S, K, T, r, sigma, n_steps=1000, n_asset_steps=100):
    """
    Prices a European call option by solving the Black-Scholes PDE
    using an explicit finite difference method.
    """
    dt = T / n_steps
    # Set up the asset price grid
    S_max = S * 4
    S_min = S / 4
    dS = (S_max - S_min) / n_asset_steps
    S_grid = np.linspace(S_min, S_max, n_asset_steps + 1)

    # Set up the option value grid at maturity (time T)
    V = np.maximum(S_grid - K, 0)

    # Iterate backwards in time from T to 0
    for j in range(n_steps):
        V_old = V.copy()
        for i in range(1, n_asset_steps):
            # Black-Scholes PDE coefficients
            a = 0.5 * sigma**2 * i**2 * dt
            b = 0.5 * r * i * dt
            c = 1 - r * dt - 2 * a

            # Explicit finite difference scheme
            V[i] = a * V_old[i-1] + c * V_old[i] + (2*a + 2*b) * V_old[i+1] # Simplified from original BS formula for explicit scheme

    # Find the price at the initial stock price S
    # We need to interpolate because S might not be exactly on the grid
    price = np.interp(S, S_grid, V)
    return price

# --- 3. Analytical Black-Scholes Formula (For Verification) ---

def black_scholes_analytical(S, K, T, r, sigma):
    """Calculates the exact price of a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return price


if __name__ == '__main__':
    # Parameters
    S0 = 100.0      # Initial stock price
    K = 105.0       # Strike price
    T = 1.0         # Time to maturity (1 year)
    r = 0.05        # Risk-free rate (5%)
    sigma = 0.2     # Volatility (20%)
    n_simulations = 500000 # Number of simulations for Monte Carlo

    # --- Calculation ---
    print("Calculating option prices using different methods...")

    # Feynman-Kac Part 1: Expectation of Stochastic Process
    mc_price = monte_carlo_call_price(S0, K, T, r, sigma, n_simulations)

    # Feynman-Kac Part 2: PDE Solution
    # Note: Finite difference can be slow and less accurate without fine-tuning.
    # For demonstration, we use a moderate number of steps.
    fd_price = finite_difference_call_price(S0, K, T, r, sigma, n_steps=5000, n_asset_steps=500)

    # Analytical solution for a baseline comparison
    analytical_price = black_scholes_analytical(S0, K, T, r, sigma)

    # --- Results ---
    print("\n--- Feynman-Kac Theorem Demonstration ---")
    print(f"Initial Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print("-" * 45)
    print(f"1. Price from Monte Carlo (Stochastic Process Expectation): {mc_price:.4f}")
    print(f"2. Price from Finite Difference (PDE Solution):             {fd_price:.4f}")
    print(f"3. Price from Analytical Formula (Benchmark):               {analytical_price:.4f}")
    print("-" * 45)
    print("The closeness of the prices from methods 1 and 2 demonstrates the Feynman-Kac theorem.")
