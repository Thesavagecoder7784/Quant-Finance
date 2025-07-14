# Quant Finance Repository

This repository contains a collection of projects and implementations related to Quantitative Finance, covering various models, trading strategies, and analytical tools.

## Contents

### 1. Financial Models

This section includes implementations of various financial models and quantitative algorithms.

#### Advanced Probability
- `Conditional Expectation in Financial Markets.py`: Implementation of conditional expectation concepts in financial markets.
- `Martingales in Financial Markets.py`: Implementation of martingales and their applications in finance.
- `README.md`: Overview of the advanced probability models.

#### Asset Pricing
- `capital-asset-pricing-model-capm.ipynb`: Jupyter Notebook demonstrating the Capital Asset Pricing Model (CAPM).
- **Multi-factor Models**
  - `aqr-style-premia-multi-style-models.ipynb`: Jupyter Notebook on AQR style premia multi-style models.
  - `carhart-4-factor-model.ipynb`: Jupyter Notebook on the Carhart 4-factor model.
  - `fama-french-3-factor-model.ipynb`: Jupyter Notebook on the Fama-French 3-factor model.
  - `fama-french-5-factor-model.ipynb`: Jupyter Notebook on the Fama-French 5-factor model.

#### Credit Risk Modeling
- `pd_estimation.py`: Python script for probability of default (PD) estimation.
- `README.md`: Overview of credit risk modeling.
- `requirements.txt`: Python dependencies for credit risk modeling.
- **Credit Derivatives**
  - `credit_default_swap.py`: Implementation of Credit Default Swap (CDS) pricing.
- **Reduced Form Models**
  - `jarrow_turnbull_model.py`: Implementation of the Jarrow-Turnbull model.
- **Structural Models**
  - `black_cox_model.py`: Implementation of the Black-Cox model.
  - `merton_model.py`: Implementation of the Merton model.

#### Linear Algebra Applications
- `main.py`: Main script for linear algebra applications.
- `pca_algorithm.py`: Implementation of Principal Component Analysis (PCA) algorithm.
- `README.md`: Overview of linear algebra applications.
- `svd_algorithm.py`: Implementation of Singular Value Decomposition (SVD) algorithm.

#### Monte Carlo Simulations
- `monte-carlo-algorithm.ipynb`: Jupyter Notebook demonstrating Monte Carlo simulations.

#### Option Pricing
- `black-scholes-algorithm.ipynb`: Jupyter Notebook demonstrating the Black-Scholes option pricing model.
- **Advanced Option Pricing & Derivatives**
  - `exotic_options_pricing.py`: Python script for pricing exotic options.
- **Option Pricing Tool (React Application)**

#### Stochastic Processes
- `branching_process.py`: Python script for simulating branching processes.
- `brownian_motion_simulation.py`: Python script for simulating Brownian motion.
- `continuous_time_markov_chain.py`: Python script for continuous-time Markov chains.
- `discrete_time_markov_chain.py`: Python script for discrete-time Markov chains.
- `gaussian_process.py`: Python script for Gaussian processes.
- `geometric_random_walk.py`: Python script for geometric random walks.
- `jump_diffusion.py`: Python script for jump diffusion models.
- `levy_process.py`: Python script for Levy processes.
- `martingale_process.py`: Python script for martingale processes.
- `poisson_process_simulation.py`: Python script for Poisson process simulations.
- `README.md`: Overview of stochastic processes.
- `renewal_process.py`: Python script for renewal processes.
- `simple_random_walk.py`: Python script for simple random walks.

### 2. Portfolio Management

This section focuses on portfolio optimization and risk analysis techniques.

#### Portfolio Optimization
- `Black-Litterman.py`: Implementation of the Black-Litterman model.
- `Markowitz.py`: Implementation of Markowitz Portfolio Optimization.
- `README.md`: Overview of portfolio optimization models.
- `Reinforcement Learning.py`: Exploration of reinforcement learning in portfolio optimization.

#### Risk Analysis
- `value-at-risk-var-methods.ipynb`: Jupyter Notebook on Value-at-Risk (VaR) methods.
- **Portfolio and Risk Analysis**
  - `portfolio-and-risk-analysis`: Directory containing additional portfolio and risk analysis files (details to be added if content is available).
  - `README.md`: Overview of portfolio and risk analysis.

### 3. Trading Strategy Development

This section covers the development of trading strategies, including sentiment analysis and backtesting.

#### Backtesting
- **Simple Trading Signal**
  - `.env`: Environment variables for the backtesting system.
  - `.gitignore`: Git ignore file for the backtesting system.
  - `backtesting.py`: Core backtesting engine.
  - `config.py`: Configuration for the backtesting system.
  - `data_ingestion.py`: Script for ingesting historical data.
  - `database_manager.py`: Manages the news sentiment database.
  - `main.py`: Main entry point for running backtests.
  - `news_sentiment.db`: SQLite database for news sentiment.
  - `pipeline_orchestrator.py`: Orchestrates the data pipeline.
  - `README.md`: Overview of the simple trading signal backtesting.
  - `sentiment_analysis.py`: Script for sentiment analysis.
  - `visualization.py`: Scripts for visualizing backtesting results.

#### Machine Learning Trading Strategy
- `config.py`: Configuration for the ML trading strategy.
- `feature_engineering.py`: Script for feature engineering.
- `main.py`: Main entry point for the ML trading strategy.
- `model_training.py`: Script for training ML models.
- `prediction.py`: Script for generating predictions.
- `README.md`: Overview of the ML trading strategy.
- `requirements.txt`: Python dependencies for the ML trading strategy.

#### News Sentiment Pipeline
- `.env`: Environment variables for the sentiment analysis pipeline.
- `.gitignore`: Git ignore file for the sentiment analysis pipeline.
- `config.py`: Configuration for the sentiment analysis pipeline.
- `data_ingestion.py`: Script for ingesting financial headlines.
- `data_processor.py`: Processes raw news data.
- `database_manager.py`: Manages the news sentiment database.
- `main.py`: Main entry point for the sentiment analysis pipeline.
- `pipeline_orchestrator.py`: Orchestrates the sentiment analysis pipeline.
- `README.md`: Overview of the news sentiment pipeline.
- `requirements.txt`: Python dependencies for the sentiment analysis pipeline.
- `sentiment_analysis.py`: Script for sentiment analysis using VADER and FinBERT.
- `visualization.py`: Scripts for visualizing sentiment data.

### 4. Time Series Analysis

This section provides tools and implementations for time series analysis in finance.

- `arima_model.py`: Implementation of ARIMA models.
- `cointegration.py`: Python script for cointegration analysis.
- `garch_model.py`: Implementation of GARCH models.
- `README.md`: Overview of time series analysis.
- `stationary_tests.py`: Python scripts for stationarity tests (ADF, KPSS).

### 5. Live Paper Trading Bot

This section contains the setup for a live paper trading bot.

- `config.py`: Configuration for the trading bot.
- `main.py`: Main entry point for the trading bot.
- `requirements.txt`: Python dependencies for the trading bot.

## Other Files

- `README.md`: This file.
- `xgboost_model.pkl`: A pickled XGBoost model.
