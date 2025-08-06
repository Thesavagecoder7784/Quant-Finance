
"""
Girsanov's Theorem and Change of Measure in Finance

This script provides a conceptual and numerical demonstration of Girsanov's Theorem,
a cornerstone of quantitative finance for pricing derivatives. The theorem allows us
to switch from the real-world probability measure (P) to the risk-neutral
measure (Q), which is essential for arbitrage-free pricing.

Under the real-world measure P, the expected return of a stock is its actual
expected return, mu. Under the risk-neutral measure Q, the expected return of
any asset is the risk-free rate, r. Girsanov's theorem provides the Radon-Nikodym
derivative that connects these two measures.
"""

import numpy as np

def girsanov_theorem_simulation(S0, mu, sigma, r, T, n_steps, n_simulations):
    """
    Simulates stock price paths under both real-world (P) and risk-neutral (Q) measures.

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected return under the real-world measure P.
        sigma (float): Volatility of the stock.
        r (float): Risk-free interest rate.
        T (float): Time to maturity (in years).
        n_steps (int): Number of time steps in the simulation.
        n_simulations (int): Number of simulated price paths.

    Returns:
        tuple: A tuple containing:
            - S_p (np.ndarray): Simulated stock prices under measure P.
            - S_q (np.ndarray): Simulated stock prices under measure Q.
            - radon_nikodym (np.ndarray): The Radon-Nikodym derivative process.
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Generate standard Brownian motion paths
    W = np.random.standard_normal(size=(n_simulations, n_steps)) * np.sqrt(dt)
    W = np.insert(W, 0, 0, axis=1).cumsum(axis=1)

    # --- Real-world measure (P) ---
    # dS_t = mu * S_t * dt + sigma * S_t * dW_t
    S_p = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    # --- Risk-neutral measure (Q) ---
    # The market price of risk (lambda)
    lambda_ = (mu - r) / sigma

    # Girsanov's Theorem: dW_q = dW_p + lambda_ * dt
    # The Brownian motion under Q is W_q(t) = W_p(t) + lambda_ * t
    W_q = W + lambda_ * t

    # dS_t = r * S_t * dt + sigma * S_t * dW_q_t
    S_q = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W_q)

    # --- Radon-Nikodym Derivative (dQ/dP) ---
    # This process connects the two measures.
    # M_t = exp(-lambda * W_p(t) - 0.5 * lambda^2 * t)
    radon_nikodym = np.exp(-lambda_ * W - 0.5 * lambda_**2 * t)

    return S_p, S_q, radon_nikodym

if __name__ == '__main__':
    # Parameters
    S0 = 100      # Initial stock price
    mu = 0.10     # Expected return (10%)
    sigma = 0.20  # Volatility (20%)
    r = 0.05      # Risk-free rate (5%)
    T = 1.0       # Time horizon (1 year)
    n_steps = 252 # Number of trading days in a year
    n_simulations = 1000 # Number of simulations

    S_p, S_q, M = girsanov_theorem_simulation(S0, mu, sigma, r, T, n_steps, n_simulations)

    # --- Verification ---
    # The expected value of the terminal stock price under P, discounted at mu, should be S0.
    expected_S_p_terminal = np.mean(S_p[:, -1])
    print(f"Initial Price (S0): {S0}")
    print(f"Average Terminal Price under P: {expected_S_p_terminal:.4f}")
    print(f"Discounted at mu, E_p[S_T * exp(-mu*T)]: {np.mean(S_p[:, -1] * np.exp(-mu * T)):.4f}")


    # The expected value of the terminal stock price under Q, discounted at r, should be S0.
    # This is the fundamental asset pricing formula.
    expected_S_q_terminal = np.mean(S_q[:, -1])
    print(f"\nAverage Terminal Price under Q: {expected_S_q_terminal:.4f}")
    print(f"Discounted at r, E_q[S_T * exp(-r*T)]: {np.mean(S_q[:, -1] * np.exp(-r * T)):.4f}")

    # The change of measure in action: E_p[S_T * M_T] = E_q[S_T]
    # We can price under Q by taking the expectation under P and using the Radon-Nikodym derivative.
    expected_S_p_with_M = np.mean(S_p[:, -1] * M[:, -1])
    print(f"\nExpectation under P with Radon-Nikodym derivative (E_p[S_T * M_T]): {expected_S_p_with_M:.4f}")
    print(f"This should be close to the undiscounted terminal price under Q: {expected_S_q_terminal:.4f}")

