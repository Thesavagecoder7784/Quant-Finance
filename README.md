# Quant Finance Repository

Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Contents
- Algorithms
  - Foundational quantitative finance algorithms - Black Scholes Algorithm, CAPM factor modeling, Value-at-Risk (VaR), and Monte Carlo simulations
  - Multi-Factor Models - Carhart 4-factor model, Fama French 3 Factor model, Fama French 5 Factor model, AQR style Premia Multi Style Models
  - Portfolio & Risk Analysis - Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, Beta (vs Market), Jensen's Alpha (annualized), Treynor Ratio, CAGR
  - Advanced Portfolio Optimization - Markowitz Portfolio Optimization, Black-Litterman Optimization
- News Sentiment Pipeline – Ingests financial headlines from multiple sources (NewsAPI, Reddit, Finviz), analyzes sentiment using VADER and FinBERT, and stores daily-aggregated scores per ticker in a database to surface market mood signals.
- Backtest Simple Trading Signal – Converts those sentiment scores into buy/sell rules, runs a backtesting engine (with realistic costs) over historical prices, and evaluates portfolio performance via metrics like Sharpe, drawdown, cumulative returns, and win rate.
