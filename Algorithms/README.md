# Quant-Finance
Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Algorithms
### Black-Scholes Algorithm
- Helps you calculate the fair price of a stock option (basically tells you how much a stock option should cost)
- Traders use this formula to find the ideal price of a European Call or Put option
Uses math (like probability and time value) to figure out what an option is worth today, assuming no surprises in the market.

#### Assumptions
- Constant volatility (doesn't change over time), No dividends (or needs dividend-adjusted version), Lognormal distribution of asset prices, Risk-free interest rate is constant, European options only (exercisable only at maturity), Markets are frictionless (no transaction costs or taxes), No arbitrage opportunities

#### Impacts & Adjustments:
- In real markets, volatility is not constant → use implied volatility or stochastic volatility models (e.g., Heston), Dividend-paying stocks require modified formulae, Use binomial models or Monte Carlo for American-style options

### Monte Carlo Algorithm
- Randomized method to estimate complex things (for example, the value of an option) by simulating a lot of future possible scenarios (thousands and millions)
- This is done by calculating the payoff and then using that to get the average value
Runs tons of "what-if" simulations to guess future outcomes and averages them to get a fair estimate.

#### Assumptions:
- Model for asset return is correct (e.g., Geometric Brownian Motion), Volatility and drift are known and constant (unless explicitly modeled to change), Sufficient simulations yield accurate estimates (Law of Large Numbers)
  
#### Impacts & Adjustments:

- Can be inaccurate if the underlying model is misspecified, Computationally expensive → can be slow, Use variance reduction techniques or quasi-random sequences for efficiency, Use dynamic volatility models (e.g., GARCH) to improve realism

### Value-at-Risk (VaR) Methods
Estimates how much money you could potentially lose over a specific period, under normal market conditions, with a particular confidence interval (usually 95%)
1. Historical Simulation - uses past data to simulate the amount that could be lost. Consider past losses to see how bad things could get again.
#### Assumptions:
- The past reflects future risks, Historical returns are sufficient and relevant, No distributional assumptions
#### Limitations:
- Misses black swan events not seen in history, May understate or overstate risk depending on recent volatility

2. Variance-Covariance (Parametric VaR) - uses the mathematical formula VaR = Z⋅σ⋅sqrt(t) to find the VaR. Assumes returns are normally distributed and calculates loss using just volatility and time.
#### Assumptions:
- Returns are normally distributed, Linear relationships between portfolio components, Constant volatility and correlations
#### Limitations:
- Underestimates risk in fat-tailed distributions, Not suitable for options or portfolios with non-linear payoffs
#### Fixes:
- Use Cornish-Fisher expansion to adjust for skewness/kurtosis, Switch to non-parametric methods when distribution is uncertain

3. Monte Carlo Simulation - Use a stochastic model (e.g., lognormal with drift and volatility) to simulate asset returns, generate portfolio value outcomes, and identify the 5th percentile loss as Value at Risk (VaR). Makes up thousands of possible future outcomes using randomness, then looks at the worst ones to estimate potential loss.
#### Assumptions:
- Correct specification of the stochastic process, Simulations cover all risk scenarios
#### Limitations:
- Garbage in, garbage out: If the model or inputs are flawed, so is the VaR, Very resource-intensive
#### Fixes:
- Model fat tails or jumps in asset prices, Use scenario analysis alongside for stress-testing

## Multi-Factor Models
### Fama-French 3-Factor Model
- Calculates your investment’s return based on market trends, company size, and value vs growth style.
Breaks your returns into 3 parts: how the market moves, whether the company is small or big, and if it’s cheap or expensive.
#### Assumptions:
- Market, size (SMB), and value (HML) are the main drivers of stock returns, Linear relationship between factors and returns, Factors are stable over time
#### Limitations:
- Misses momentum, profitability, and investment effects, Factors can be unstable in different regimes or markets

### Fama-French 5-Factor Model
- Adds to the 3-factor model by also measuring profitability and the extent to which companies reinvest.
Adds two more ways to explain returns: how profitable a company is, and how aggressively it reinvests.
#### Assumptions:
- Adds profitability (RMW) and investment (CMA) as key return drivers, Same linear, stable-factor assumptions
#### Limitations:
- May still fail to explain momentum, Overfitting risk with more factors, Assumes clean, accurate factor construction (which can vary by provider)

### Carhart 4-Factor Model
- Builds on the Fama-French 3-Factor model by adding momentum to explain returns from stocks that have recently performed well.
Includes a “winners keep winning” factor to capture how recent good performance can continue short term.
#### Assumptions:
Adds momentum to Fama-French 3, Factors are orthogonal and time-invariant
#### Limitations:
Momentum factor can reverse quickly in crashes, Still doesn’t account for quality, low-volatility, or liquidity

### Multi-Style (AQR-like)
- Combines all the above and adds momentum, quality, and risk-taking style to explain returns more completely.
Adds extra factors like momentum, strong financial health, and low-risk preference — it’s what big quant funds use to fully explain returns.
#### Assumptions:
- All factors (market, size, value, momentum, quality, low-risk, etc.) contribute linearly, No interaction effects between factors, Historical patterns will persist

#### Limitations:
- Data mining bias: too many factors = overfitting, Regime changes can render factors ineffective, Assumes consistent definition and persistence of styles across time

#### Fixes:
- Use regularization or Bayesian shrinkage in factor models, Combine with machine learning to detect nonlinearities or time-varying relationships



