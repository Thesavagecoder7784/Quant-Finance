# Quant-Finance
Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Algorithms
### Black-Scholes Algorithm
- Helps you calculate the fair price of a stock option (basically tells you how much a stock option should cost)
- Traders use this formula to find the ideal price of a European Call or Put option
Uses math (like probability and time value) to figure out what an option is worth today, assuming no surprises in the market.

### Monte Carlo Algorithm
- Randomized method to estimate complex things (for example, the value of an option) by simulating a lot of future possible scenarios (thousands and millions)
- This is done by calculating the payoff and then using that to get the average value
Runs tons of "what-if" simulations to guess future outcomes and averages them to get a fair estimate.

### Value-at-Risk (VaR) Methods
Estimates how much money you could potentially lose over a specific period, under normal market conditions, with a particular confidence interval (usually 95%)
1. Historical Simulation - uses past data to simulate the amount that could be lost. Consider past losses to see how bad things could get again.
2. Variance-Covariance (Parametric VaR) - uses the mathematical formula VaR = Z⋅σ⋅sqrt(t) to find the VaR. Assumes returns are normally distributed and calculates loss using just volatility and time.
3. Monte Carlo Simulation - Use a stochastic model (e.g., lognormal with drift and volatility) to simulate asset returns, generate portfolio value outcomes, and identify the 5th percentile loss as Value at Risk (VaR). Makes up thousands of possible future outcomes using randomness, then looks at the worst ones to estimate potential loss.

### Multi-Factor Models
### Fama-French 3-Factor Model
- Calculates your investment’s return based on market trends, company size, and value vs growth style.
Breaks your returns into 3 parts: how the market moves, whether the company is small or big, and if it’s cheap or expensive.

### Fama-French 5-Factor Model
- Adds to the 3-factor model by also measuring profitability and the extent to which companies reinvest.
Adds two more ways to explain returns: how profitable a company is, and how aggressively it reinvests.

### Carhart 4-Factor Model
- Builds on the Fama-French 3-Factor model by adding momentum to explain returns from stocks that have recently performed well.
Includes a “winners keep winning” factor to capture how recent good performance can continue short term.

### Multi-Style (AQR-like)
- Combines all the above and adds momentum, quality, and risk-taking style to explain returns more completely.
Adds extra factors like momentum, strong financial health, and low-risk preference — it’s what big quant funds use to fully explain returns.


