# cointegration.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

# --- Engle-Granger Two-Step Cointegration Test ---
def engle_granger_cointegration_test(series1: pd.Series, series2: pd.Series):
    """
    Performs the Engle-Granger two-step cointegration test.

    Steps:
    1. Regress one series on the other (e.g., Y = a + bX + e).
    2. Test the stationarity of the residuals (e). If residuals are stationary,
       the series are cointegrated.

    Args:
        series1 (pd.Series): The first time series.
        series2 (pd.Series): The second time series.

    Returns:
        dict: A dictionary containing test results:
            - 'regression_summary': Summary of the OLS regression.
            - 'adf_test_residuals': ADF test results for residuals.
            - 'is_cointegrated': Boolean indicating if series are cointegrated
                                 at 5% significance level.
    """
    print("\n--- Performing Engle-Granger Cointegration Test ---")

    # Ensure series are aligned
    common_index = series1.index.intersection(series2.index)
    s1 = series1.loc[common_index]
    s2 = series2.loc[common_index]

    if len(s1) == 0:
        print("Error: No common index between the two series.")
        return None

    # Step 1: Regress series1 on series2 (or vice versa)
    # Add a constant to the independent variable for the intercept
    X = sm.add_constant(s2)
    model = OLS(s1, X)
    results = model.fit()
    print("\n--- OLS Regression Summary (Series1 vs Series2) ---")
    print(results.summary())

    # Get the residuals
    residuals = results.resid
    residuals.name = 'Regression_Residuals'

    # Step 2: Test for stationarity of the residuals using ADF test
    print("\n--- ADF Test on Regression Residuals ---")
    adf_result = adfuller(residuals, autolag='AIC')

    adf_output = pd.Series(adf_result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    print(adf_output)

    # Check for cointegration (p-value < 0.05 for residuals implies stationarity)
    is_cointegrated = adf_result[1] < 0.05 # p-value at 5% significance
    print(f"\nAre the series cointegrated (based on ADF p-value < 0.05)? {is_cointegrated}")

    return {
        'regression_summary': results,
        'adf_test_residuals': adf_output,
        'is_cointegrated': is_cointegrated
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Generate two synthetic non-stationary (random walk) series that are cointegrated
    # They should share a common stochastic trend
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Common stochastic trend
    common_trend = np.cumsum(np.random.randn(n_samples) * 0.5)

    # Series 1: Common trend + stationary noise
    series1_noise = np.random.randn(n_samples) * 2
    series1 = pd.Series(common_trend + series1_noise, index=dates, name='Series1')

    # Series 2: Common trend * factor + different stationary noise
    series2_noise = np.random.randn(n_samples) * 1.5
    series2 = pd.Series(common_trend * 0.8 + 10 + series2_noise, index=dates, name='Series2')

    # Plot the synthetic series
    plt.figure(figsize=(12, 6))
    plt.plot(series1, label='Series 1')
    plt.plot(series2, label='Series 2')
    plt.title('Synthetic Cointegrated Time Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Perform cointegration test
    cointegration_results = engle_granger_cointegration_test(series1, series2)

    if cointegration_results:
        # Plot residuals if the test was successful
        residuals = cointegration_results['regression_summary'].resid
        plt.figure(figsize=(12, 6))
        plt.plot(residuals, label='Regression Residuals')
        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.title('Residuals from Cointegration Regression')
        plt.xlabel('Date')
        plt.ylabel('Residual Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Example of non-cointegrated series (two independent random walks)
    print("\n--- Testing Non-Cointegrated Series ---")
    series_a = pd.Series(np.cumsum(np.random.randn(n_samples)), index=dates, name='SeriesA')
    series_b = pd.Series(np.cumsum(np.random.randn(n_samples)), index=dates, name='SeriesB')

    plt.figure(figsize=(12, 6))
    plt.plot(series_a, label='Series A (Random Walk)')
    plt.plot(series_b, label='Series B (Random Walk)')
    plt.title('Synthetic Non-Cointegrated Time Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    engle_granger_cointegration_test(series_a, series_b)

    print("\n--- Cointegration Test Script Finished ---")
