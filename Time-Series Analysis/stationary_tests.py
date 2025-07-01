# stationarity_tests.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# --- Augmented Dickey-Fuller (ADF) Test ---
def run_adf_test(series: pd.Series):
    """
    Performs the Augmented Dickey-Fuller test for stationarity.
    Null Hypothesis (H0): The time series has a unit root (is non-stationary).
    Alternative Hypothesis (H1): The time series is stationary.

    Args:
        series (pd.Series): The time series data.

    Returns:
        pd.Series: A Series containing the ADF test results.
    """
    print(f"\n--- Running Augmented Dickey-Fuller (ADF) Test for '{series.name}' ---")
    result = adfuller(series, autolag='AIC')

    adf_output = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in result[4].items():
        adf_output[f'Critical Value ({key})'] = value

    print(adf_output)

    if adf_output['p-value'] <= 0.05:
        print(f"Conclusion: Reject H0. The series '{series.name}' is likely stationary (p-value = {adf_output['p-value']:.4f}).")
    else:
        print(f"Conclusion: Fail to reject H0. The series '{series.name}' is likely non-stationary (p-value = {adf_output['p-value']:.4f}).")
    return adf_output

# --- Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test ---
def run_kpss_test(series: pd.Series, regression: str = 'c'):
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
    Null Hypothesis (H0): The time series is trend-stationary or level-stationary.
    Alternative Hypothesis (H1): The time series is non-stationary (has a unit root).

    Args:
        series (pd.Series): The time series data.
        regression (str): The type of regression to use.
                          'c' for level stationarity (default).
                          'ct' for trend stationarity.

    Returns:
        pd.Series: A Series containing the KPSS test results.
    """
    print(f"\n--- Running KPSS Test for '{series.name}' (regression='{regression}') ---")
    result = kpss(series, regression=regression, nlags='auto')

    kpss_output = pd.Series(result[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
    for key, value in result[3].items():
        kpss_output[f'Critical Value ({key})'] = value

    print(kpss_output)

    if kpss_output['p-value'] <= 0.05:
        print(f"Conclusion: Reject H0. The series '{series.name}' is likely non-stationary (p-value = {kpss_output['p-value']:.4f}).")
    else:
        print(f"Conclusion: Fail to reject H0. The series '{series.name}' is likely stationary (p-value = {kpss_output['p-value']:.4f}).")
    return kpss_output

# --- Example Usage ---
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 150
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # 1. Stationary Series (White Noise)
    stationary_series = pd.Series(np.random.randn(n_samples), index=dates, name='Stationary_Series')

    # 2. Non-Stationary Series (Random Walk)
    non_stationary_series = pd.Series(np.cumsum(np.random.randn(n_samples) * 0.5), index=dates, name='Non_Stationary_Series')

    # 3. Trend Stationary Series (Random Walk + Deterministic Trend)
    trend_stationary_series = pd.Series(np.cumsum(np.random.randn(n_samples) * 0.1) + np.linspace(0, 10, n_samples),
                                        index=dates, name='Trend_Stationary_Series')

    # Plot the synthetic series
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(stationary_series)
    plt.title('Stationary Series (White Noise)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(non_stationary_series)
    plt.title('Non-Stationary Series (Random Walk)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(trend_stationary_series)
    plt.title('Trend-Stationary Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Run tests on Stationary Series
    print("\n--- Testing Stationary Series ---")
    run_adf_test(stationary_series)
    run_kpss_test(stationary_series, regression='c') # Test for level stationarity
    run_kpss_test(stationary_series, regression='ct') # Test for trend stationarity

    # Run tests on Non-Stationary Series
    print("\n--- Testing Non-Stationary Series ---")
    run_adf_test(non_stationary_series)
    run_kpss_test(non_stationary_series, regression='c')
    run_kpss_test(non_stationary_series, regression='ct')

    # Run tests on Trend Stationary Series
    print("\n--- Testing Trend Stationary Series ---")
    run_adf_test(trend_stationary_series)
    run_kpss_test(trend_stationary_series, regression='c') # Should indicate non-stationary if only level is considered
    run_kpss_test(trend_stationary_series, regression='ct') # Should indicate stationary if trend is considered

    # Example of differencing a non-stationary series to make it stationary
    print("\n--- Testing Differenced Non-Stationary Series ---")
    differenced_series = non_stationary_series.diff().dropna()
    differenced_series.name = 'Differenced_Non_Stationary_Series'

    plt.figure(figsize=(10, 5))
    plt.plot(differenced_series)
    plt.title('Differenced Non-Stationary Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    run_adf_test(differenced_series)
    run_kpss_test(differenced_series, regression='c')

    print("\n--- Stationarity Tests Script Finished ---")
