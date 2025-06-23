# Quant Finance Repository

Repository containing everything I've learned about Quantitative Finance over the summer of 25

## Contents
- Algorithms - Implements foundational quantitative finance algorithms like Black Scholes Algorithm, CAPM factor modeling, Value-at-Risk (VaR), and Monte Carlo simulations, enabling rigorous performance evaluation and risk analysis within your sentiment-driven backtesting pipeline
- News Sentiment Pipeline – Ingests financial headlines from multiple sources (NewsAPI, Reddit, Finviz), analyzes sentiment using VADER and FinBERT, and stores daily-aggregated scores per ticker in a database to surface market mood signals.
- Backtest Simple Trading Signal – Converts those sentiment scores into buy/sell rules, runs a backtesting engine (with realistic costs) over historical prices, and evaluates portfolio performance via metrics like Sharpe, drawdown, cumulative returns, and win rate.
