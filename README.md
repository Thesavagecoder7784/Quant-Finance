# Quant-Finance
Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Black-Scholes Algorithm
Helps you calculate the fair price of a stock option (basically tells you how much a stock option should cost)
Traders use this formula to find the ideal price of a European Call or Put option

## Monte Carlo Algorithm
Randomized method to estimate complex things (for example, the value of an option) by simulating a lot of future possible scenarios (thousands and millions)
This is done by calculating the payoff and then using that to get the average value

## Value-at-Risk (VaR) methods
Estimates how much money you could potentially lose over a specific period of time, under normal market conditions, with a specific confidence interval (usually 95%)
1. Historical Simulation - uses past data to simulate the amount that could be lost
2. Variance-Covariance (Parametric VaR) - uses mathematical formula VaR = Z⋅σ⋅sqrt(t) to find the VaR
3. Monte Carlo Simulation - Use a stochastic model (e.g., lognormal with drift and volatility) to simulate asset returns, generate portfolio value outcomes, and identify the 5th percentile loss as Value at Risk (VaR).

