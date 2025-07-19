import numpy as np
from scipy.stats import norm

def merton_model(asset_value, debt, time_to_maturity, asset_volatility, risk_free_rate):
    """
    Calculates the probability of default and distance to default using the Merton model.

    Args:
        asset_value (float): Current market value of the firm's assets.
        debt (float): Face value of the firm's debt (default barrier).
        time_to_maturity (float): Time to maturity of the debt in years.
        asset_volatility (float): Volatility of the firm's asset value.
        risk_free_rate (float): Risk-free interest rate (annualized).

    Returns:
        tuple: A tuple containing:
            - default_probability (float): Probability of default.
            - distance_to_default (float): Distance to default.
    """
    if asset_value <= 0 or debt <= 0 or time_to_maturity <= 0 or asset_volatility <= 0:
        raise ValueError("All input parameters (asset_value, debt, time_to_maturity, asset_volatility) must be positive.")

    d1 = (np.log(asset_value / debt) + (risk_free_rate + 0.5 * asset_volatility**2) * time_to_maturity) / (asset_volatility * np.sqrt(time_to_maturity))
    d2 = d1 - asset_volatility * np.sqrt(time_to_maturity)

    # Probability of default is N(-d2)
    default_probability = norm.cdf(-d2)

    # Distance to default is d2
    distance_to_default = d2

    return default_probability, distance_to_default

if __name__ == '__main__':
    # Example Usage:
    V = 1000  # Asset Value
    D = 700   # Debt
    T = 1     # Time to Maturity (1 year)
    sigma_V = 0.2  # Asset Volatility (20%)
    r = 0.05  # Risk-Free Rate (5%)

    prob_default, dist_default = merton_model(V, D, T, sigma_V, r)

    print(f"Merton Model Results:")
    print(f"  Asset Value (V): {V}")
    print(f"  Debt (D): {D}")
    print(f"  Time to Maturity (T): {T}")
    print(f"  Asset Volatility (sigma_V): {sigma_V}")
    print(f"  Risk-Free Rate (r): {r}")
    print(f"  Probability of Default: {prob_default:.4f}")
    print(f"  Distance to Default: {dist_default:.4f}")

    # Example with higher debt
    D_high = 950
    prob_default_high, dist_default_high = merton_model(V, D_high, T, sigma_V, r)
    print(f"  Example with higher debt (D={D_high}):")
    print(f"  Probability of Default: {prob_default_high:.4f}")
    print(f"  Distance to Default: {dist_default_high:.4f}")
