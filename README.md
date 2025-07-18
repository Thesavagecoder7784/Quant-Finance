# Quant Finance Repository

This repository serves as a comprehensive portfolio showcasing practical implementations of quantitative finance models, advanced analytical techniques, and algorithmic trading strategies. Developed with a focus on applying theoretical concepts to real-world financial data, this project demonstrates proficiency in quantitative modeling, statistical analysis, machine learning, and software development crucial for roles in quantitative finance.

## Key Highlights & Skills Demonstrated

*   **Quantitative Modeling:** Deep understanding and implementation of foundational and advanced financial models, including option pricing (Black-Scholes, Monte Carlo for exotic options), credit risk (Merton, Black-Cox), and asset pricing (CAPM, Fama-French, Carhart, AQR).
*   **Stochastic Processes:** Simulation and application of various stochastic processes (Brownian Motion, Jump Diffusion, Lévy, Martingales, Poisson) fundamental for modeling asset price dynamics and risk.
*   **Portfolio Optimization:** Expertise in classical (Markowitz Mean-Variance, Black-Litterman) and modern (Reinforcement Learning) portfolio construction and risk management techniques.
*   **Time Series Analysis:** Application of advanced time series models (ARIMA, GARCH) for forecasting and volatility modeling, alongside stationarity and cointegration tests for robust data analysis.
*   **Machine Learning & Data Science:** Implementation of ML algorithms (XGBoost) for predictive modeling in trading, and natural language processing (VADER, FinBERT) for sentiment analysis on financial news.
*   **Algorithmic Trading & Backtesting:** Development of end-to-end pipelines for data ingestion, sentiment analysis, strategy backtesting, and live paper trading, emphasizing performance metrics and realistic transaction costs.
*   **Risk Management:** Calculation and interpretation of key risk metrics such as Value-at-Risk (VaR), Maximum Drawdown, Sharpe Ratio, Sortino Ratio, Beta, and Alpha.
*   **Programming & Tools:** Strong programming skills in Python for quantitative development, data manipulation, and visualization, complemented by experience with React for interactive financial tools.

## Contents

### 1. Financial Models

This section includes implementations of various financial models and quantitative algorithms, forming the bedrock of quantitative finance.

#### Advanced Probability
-   `Conditional Expectation in Financial Markets.py`: Demonstrates the application of conditional expectation, a core concept in stochastic calculus, for understanding information flow and pricing in financial markets.
-   `Martingales in Financial Markets.py`: Implements martingale processes, crucial for arbitrage-free pricing theory and risk-neutral valuation.

#### Asset Pricing
-   `capital-asset-pricing-model-capm.ipynb`: Jupyter Notebook demonstrating the Capital Asset Pricing Model (CAPM) for assessing expected returns and systematic risk.
-   **Multi-factor Models**: Jupyter Notebooks implementing advanced asset pricing models:
    -   `aqr-style-premia-multi-style-models.ipynb`: Explores AQR's style premia models.
    -   `carhart-4-factor-model.ipynb`: Implements the Carhart 4-factor model, extending Fama-French with a momentum factor.
    -   `fama-french-3-factor-model.ipynb`: Implements the foundational Fama-French 3-factor model for explaining stock returns.
    -   `fama-french-5-factor-model.ipynb`: Implements the extended Fama-French 5-factor model.

#### Credit Risk Modeling
-   `pd_estimation.py`: Python script for Probability of Default (PD) estimation using logistic regression, a key component in credit risk assessment.
-   **Credit Derivatives**:
    -   `credit_default_swap.py`: Implementation of Credit Default Swap (CDS) pricing, a fundamental credit derivative.
-   **Reduced Form Models**:
    -   `jarrow_turnbull_model.py`: Implementation of the Jarrow-Turnbull model, a reduced-form model for default intensity.
-   **Structural Models**:
    -   `black_cox_model.py`: Implementation of the Black-Cox model, an extension of Merton's model incorporating a default barrier.
    -   `merton_model.py`: Implementation of the Merton model for corporate default, linking equity to an option on firm assets.

#### Linear Algebra Applications
-   `main.py`: Orchestrates the application of PCA and SVD to financial data.
-   `pca_algorithm.py`: Implementation of Principal Component Analysis (PCA) for dimensionality reduction and identifying latent risk factors in financial datasets.
-   `svd_algorithm.py`: Implementation of Singular Value Decomposition (SVD) for data compression and noise reduction in financial time series.

#### Monte Carlo Simulations
-   `monte-carlo-algorithm.ipynb`: Jupyter Notebook demonstrating Monte Carlo simulations, a versatile numerical method for pricing complex derivatives and risk assessment.

#### Option Pricing
-   `black-scholes-algorithm.ipynb`: Jupyter Notebook demonstrating the Black-Scholes option pricing model, a cornerstone of modern financial theory.
-   **Advanced Option Pricing & Derivatives**:
    -   `exotic_options_pricing.py`: Python script for pricing various exotic options (Asian, Barrier, Lookback, Rainbow) using Monte Carlo simulation.
-   **Option Pricing Tool (React Application)**: An interactive web application for European option pricing and Greeks calculation using the Black-Scholes model, built with React.

#### Stochastic Processes
-   Python scripts simulating various stochastic processes essential for financial modeling:
    -   `branching_process.py`: Modeling population growth or spread of defaults.
    -   `brownian_motion_simulation.py`: Fundamental for continuous-time financial models.
    -   `continuous_time_markov_chain.py`: For modeling state transitions (e.g., credit ratings).
    -   `discrete_time_markov_chain.py`: For discrete state changes.
    -   `gaussian_process.py`: For non-parametric regression and time series forecasting.
    -   `geometric_random_walk.py`: A common model for asset prices.
    -   `jump_diffusion.py`: Incorporates sudden, large price movements.
    -   `levy_process.py`: Generalization of Brownian motion with jumps.
    -   `martingale_process.py`: Demonstrates fair games and arbitrage-free pricing.
    -   `poisson_process_simulation.py`: Models discrete event occurrences (e.g., defaults).
    -   `renewal_process.py`: Models events with i.i.d. inter-arrival times.
    -   `simple_random_walk.py`: A basic discrete-time model.

### 2. Portfolio Management

This section focuses on advanced portfolio optimization and comprehensive risk analysis techniques.

#### Portfolio Optimization
-   `Black-Litterman.py`: Implementation of the Black-Litterman model, combining market equilibrium with investor views for robust portfolio allocation.
-   `Markowitz.py`: Implementation of Markowitz Mean-Variance Optimization (MVO) for constructing efficient portfolios by balancing risk and return.
-   `Reinforcement Learning.py`: Explores the application of Reinforcement Learning (RL) to dynamic portfolio optimization, training an agent to make optimal trading decisions.

#### Risk Analysis
-   `value-at-risk-var-methods.ipynb`: Jupyter Notebook on Value-at-Risk (VaR) methods for quantifying potential financial losses.
-   **Portfolio and Risk Analysis**:
    -   `portfolio-and-risk-analysis`: Python script implementing key portfolio performance and risk metrics: Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), Calmar Ratio, Beta (β), Jensen's Alpha (α), Treynor Ratio, and Compound Annual Growth Rate (CAGR).

### 3. Trading Strategy Development

This section covers the development and evaluation of data-driven trading strategies, including sentiment analysis and rigorous backtesting.

#### Backtesting
-   **Simple Trading Signal**: An end-to-end pipeline for backtesting a sentiment-driven trading strategy.
    -   `backtesting.py`: Core backtesting engine, simulating trades with realistic costs and calculating performance metrics (Cumulative Returns, Max Drawdown, Sharpe Ratio, Win Rate).
    -   `config.py`: Centralized configuration for strategy parameters.
    -   `data_ingestion.py`: Handles fetching historical data from various sources (NewsAPI, Reddit, Finviz, yfinance).
    -   `database_manager.py`: Manages SQLite database for news sentiment.
    -   `main.py`: Main entry point to orchestrate the sentiment and backtesting pipeline.
    -   `pipeline_orchestrator.py`: Orchestrates data flow and parameter optimization.
    -   `sentiment_analysis.py`: Implements VADER and FinBERT for sentiment analysis.
    -   `visualization.py`: Provides tools for visualizing sentiment trends and backtest performance.

#### News Sentiment Pipeline
-   A comprehensive pipeline for ingesting financial news, analyzing sentiment, and storing results.
    -   `config.py`: Configuration for the sentiment pipeline.
    -   `data_ingestion.py`: Functions for fetching headlines from NewsAPI, Reddit, and Finviz.
    -   `data_processor.py`: Processes raw news data, including ticker assignment for Reddit posts.
    -   `database_manager.py`: Manages the news sentiment database.
    -   `main.py`: Main entry point for the sentiment analysis pipeline.
    -   `pipeline_orchestrator.py`: Orchestrates the sentiment analysis workflow.
    -   `sentiment_analysis.py`: Functions for VADER and FinBERT sentiment analysis.
    -   `visualization.py`: Scripts for visualizing sentiment data.

### 4. Time Series Analysis

This section provides tools and implementations for advanced time series analysis in finance, crucial for forecasting and understanding dynamic financial data.

-   `arima_model.py`: Implementation of Autoregressive Integrated Moving Average (ARIMA) models for time series forecasting.
-   `cointegration.py`: Python script for performing cointegration tests, essential for identifying long-term relationships between non-stationary time series (e.g., for pair trading).
-   `garch_model.py`: Implementation of Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models for modeling and forecasting financial volatility.
-   `stationary_tests.py`: Python scripts for performing various stationarity tests (Augmented Dickey-Fuller (ADF) test, KPSS test), critical for validating time series assumptions.

### 5. Live Paper Trading Bot

This section contains the setup for a live paper trading bot, demonstrating real-time data integration and automated order execution.

- `config.py`: Configuration for the trading bot.
- `main.py`: Main entry point for the trading bot.
- `requirements.txt`: Python dependencies for the trading bot.

## Other Files

- `README.md`: This file.
- `xgboost_model.pkl`: A pickled XGBoost model.
