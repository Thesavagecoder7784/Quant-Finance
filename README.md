# Quant Finance Repository

Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Contents
- Algorithms
  - Foundational quantitative finance algorithms - Black Scholes Algorithm, CAPM factor modeling, Value-at-Risk (VaR), and Monte Carlo simulations
  - Multi-Factor Models - Carhart 4-factor model, Fama French 3 Factor model, Fama French 5 Factor model, AQR style Premia Multi Style Models
  - Portfolio & Risk Analysis - Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, Beta (vs Market), Jensen's Alpha (annualized), Treynor Ratio, CAGR
  - Advanced Portfolio Optimization - Markowitz Portfolio Optimization, Black-Litterman Optimization, Reinforcement Learning
  - Stochastic Processes - Brownian Motion Simulation, Poisson Motion Simulation, Jump Diffusion Simulation, Geometric Brownian Motion, Ornstein-Uhlenbeck, Cox-Ingersoll-Ross (CIR), Heston Model, Merton Jump-Diffusion
  - Advanced Probability - Conditional Expectation in Financial Markets, Martingales in Financial Markets
  - Linear Algebra Applications - PCA and SVD algorithms
- News Sentiment Pipeline – Ingests financial headlines from multiple sources (NewsAPI, Reddit, Finviz), analyzes sentiment using VADER and FinBERT, and stores daily-aggregated scores per ticker in a database to surface market mood signals.
- Backtest Simple Trading Signal – Converts those sentiment scores into buy/sell rules, runs a backtesting engine (with realistic costs) over historical prices, and evaluates portfolio performance via metrics like Sharpe, drawdown, cumulative returns, and win rate.
- Time-Series Analysis - Core framework for analyzing sequential market data. Involves identifying trends, seasonality, and autocorrelation within historical price or return series. Tools include ARIMA, GARCH, Cointegration, and stationarity tests (ADF, KPSS). Enables forecasting, anomaly detection, and modeling dependencies for financial instruments across time.
- Option Pricing Tool - A React-based tool for pricing options using the Black-Scholes model.
