# Time Series Analysis in Quantitative Finance

This directory contains Python implementations for various time series analysis models and techniques commonly used in quantitative finance. These tools are essential for understanding, modeling, and forecasting financial data that evolves over time.

## Contents

- **arima_model.py:**
  Implementation of the Autoregressive Integrated Moving Average (ARIMA) model. ARIMA models are widely used for forecasting time series data, capturing linear relationships and trends.

- **cointegration.py:**
  Scripts for performing cointegration tests. Cointegration is a statistical property of time series that indicates whether two or more non-stationary time series have a long-term, stable relationship. This is crucial for pair trading strategies.

- **garch_model.py:**
  Implementation of the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model. GARCH models are used to model and forecast volatility in financial time series, which is a key component in risk management and option pricing.

- **stationary_tests.py:**
  Scripts for performing various stationarity tests (e.g., Augmented Dickey-Fuller (ADF) test, KPSS test). Stationarity is a critical assumption for many time series models, and these tests help determine if a series is stationary or requires differencing.
