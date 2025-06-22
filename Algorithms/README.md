# Quant-Finance
Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Algorithms
### Black-Scholes Algorithm
Helps you calculate the fair price of a stock option (basically tells you how much a stock option should cost)
Traders use this formula to find the ideal price of a European Call or Put option

### Monte Carlo Algorithm
Randomized method to estimate complex things (for example, the value of an option) by simulating a lot of future possible scenarios (thousands and millions)
This is done by calculating the payoff and then using that to get the average value

### Value-at-Risk (VaR) Methods
Estimates how much money you could potentially lose over a specific period, under normal market conditions, with a specific confidence interval (usually 95%)
1. Historical Simulation - uses past data to simulate the amount that could be lost
2. Variance-Covariance (Parametric VaR) - uses the mathematical formula VaR = Z⋅σ⋅sqrt(t) to find the VaR
3. Monte Carlo Simulation - Use a stochastic model (e.g., lognormal with drift and volatility) to simulate asset returns, generate portfolio value outcomes, and identify the 5th percentile loss as Value at Risk (VaR).

### Multi-Factor Models
### Fama-French 3-Factor Model
Calculates your investment’s return based on market trends, company size, and value vs growth style.

### Fama-French 5-Factor Model
Adds to the 3-factor model by also measuring profitability and the extent to which companies reinvest.

### Carhart 4-Factor Model
Builds on the Fama-French 3-Factor model by adding momentum to explain returns from stocks that have recently performed well.

### Multi-Style (AQR-like)
Combines all the above and adds momentum, quality, and risk-taking style to explain returns more completely.
