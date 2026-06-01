

import yfinance as yf
import numpy as np
import pandas as pd

# --- 1. Data Retrieval ---
# We'll use yfinance to download historical stock data.
# Let's analyze Apple Inc. (AAPL) as our example.
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

# --- 2. Calculate Daily Returns ---
# We are interested in the daily price changes, not the absolute price.
# Returns are calculated as (today's price - yesterday's price) / yesterday's price.
data['Daily_Return'] = data['Close'].pct_change()

# Drop the first day's return as it's NaN
data = data.dropna()

# --- 3. Variance Calculation ---
# Variance measures the average squared deviation of each data point from the mean.
# A high variance indicates that the data points are very spread out from the mean.
# In finance, it quantifies the dispersion of returns, giving us a measure of risk.
variance = data['Daily_Return'].var()

# --- 4. Standard Deviation Calculation ---
# Standard deviation is the square root of the variance.
# It's often preferred over variance as it's in the same unit as the original data (in this case, daily returns).
# A higher standard deviation implies greater volatility and, therefore, greater unpredictability of returns.
std_deviation = data['Daily_Return'].std()

# --- 5. Interpretation and Consequences for Risk-Adjusted Strategies ---

print(f"--- Analysis for {ticker} from 2020-01-01 to 2023-01-01 ---")
print(f"Mean Daily Return: {data['Daily_Return'].mean():.5f}")
print(f"Variance of Daily Returns: {variance:.5f}")
print(f"Standard Deviation of Daily Returns: {std_deviation:.5f}")
print("\n" + "="*60 + "\n")

print("Understanding Variance and Standard Deviation in Finance:\n")
print("1. Not Just 'Volatility':")
print("   - While often used interchangeably with 'volatility', standard deviation is the precise statistical measure of it.")
print("   - It quantifies the *unpredictability* of an asset's returns. A low standard deviation means returns are close to the average, making them more predictable. A high standard deviation means returns are spread out over a wider range, indicating that future returns are harder to predict.")
print("   - Variance, being the squared value, gives more weight to larger deviations. This is useful in certain financial models but less intuitive for direct interpretation than standard deviation.\n")

print("2. Consequences for Risk-Adjusted Strategies:")
print("   - Sharpe Ratio: This is a cornerstone of modern portfolio theory. It's calculated as (Return of Portfolio - Risk-Free Rate) / Standard Deviation of Portfolio's Excess Return. A higher Sharpe Ratio indicates better performance for the amount of risk taken.")
print("     -> A strategy might yield high returns, but if it comes with an exceptionally high standard deviation, its Sharpe Ratio could be lower than a more stable, lower-return strategy.\n")
print("   - Portfolio Construction: When combining assets, an investor doesn't just consider the individual standard deviation of each asset but also how their returns move in relation to each other (covariance). The goal is often to find assets that, when combined, lower the overall portfolio's standard deviation without sacrificing too much return.")
print("     -> An asset with high individual standard deviation might still be a valuable addition to a portfolio if its returns are negatively correlated with the other assets, as it can act as a hedge and lower the total portfolio risk.\n")
print("   - Options Pricing: The Black-Scholes model, a fundamental equation in options pricing, uses standard deviation ('volatility') as a key input. Higher volatility leads to higher option premiums because of the increased probability of the option finishing in-the-money.\n")

print("In conclusion, while daily news might talk about 'market volatility' in a general sense, for a quantitative analyst, variance and standard deviation are the real, actionable metrics that measure the true unpredictability of returns. Understanding and using them is fundamental to building and evaluating robust, risk-adjusted investment strategies.")

# --- 6. (Optional) Visualization ---
# A histogram can visually show the distribution of returns.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(data['Daily_Return'], bins=50, alpha=0.7, color='blue', label='Daily Returns')
plt.axvline(data['Daily_Return'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {data['Daily_Return'].mean():.4f}")
plt.axvline(data['Daily_Return'].mean() + std_deviation, color='green', linestyle='dashed', linewidth=2, label=f"1 Std Dev: {std_deviation:.4f}")
plt.axvline(data['Daily_Return'].mean() - std_deviation, color='green', linestyle='dashed', linewidth=2)
plt.title(f'Distribution of Daily Returns for {ticker}')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.show()
