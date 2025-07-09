# arima_garch_model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- ARIMA Model Implementation ---
def fit_and_forecast_arima(data: pd.Series, order: tuple, forecast_steps: int = 5):
    """
    Fits an ARIMA model to the data and generates a forecast.

    Args:
        data (pd.Series): The time series data.
        order (tuple): The (p, d, q) order of the ARIMA model.
        forecast_steps (int): Number of steps to forecast into the future.

    Returns:
        tuple: A tuple containing:
            - model_fit (statsmodels.tsa.arima.model.ARIMAResultsWrapper): Fitted ARIMA model.
            - forecast (pd.Series): Forecasted values.
            - conf_int (pd.DataFrame): Confidence intervals for the forecast.
    """
    print(f"\n--- Fitting ARIMA({order}) Model ---")
    try:
        # Fit the ARIMA model
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        print(model_fit.summary())

        # Generate forecast
        forecast_results = model_fit.get_forecast(steps=forecast_steps)
        forecast = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int(alpha=0.05) # 95% confidence interval

        print(f"\nARIMA Forecast for next {forecast_steps} steps:\n{forecast}")
        print(f"\nARIMA Forecast Confidence Intervals:\n{conf_int}")

        return model_fit, forecast, conf_int
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None, None, None


import yfinance as yf

# --- Example Usage ---
if __name__ == "__main__":
    # Fetch stock data from yfinance
    ticker = "AAPL"
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")['Close']
    data.name = f'{ticker}_Close'

    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original Data')
    plt.title('AAPL Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example ARIMA usage
    # For ARIMA, data should ideally be stationary. If not, differencing (d > 0) is needed.
    # Let's use (1,1,1) as an example order, meaning 1st order differencing.
    arima_order = (1, 1, 1)
    arima_fit, arima_forecast, arima_conf_int = fit_and_forecast_arima(data, arima_order, forecast_steps=10)

    if arima_forecast is not None:
        # Plot ARIMA forecast
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Original Data')
        plt.plot(arima_forecast.index, arima_forecast, color='red', label='ARIMA Forecast')
        plt.fill_between(arima_conf_int.index, arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
        plt.title(f'ARIMA({arima_order}) Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- ARIMA Model Script Finished ---")
