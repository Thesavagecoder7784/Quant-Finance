import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# --- GARCH Model Implementation ---
def fit_and_forecast_garch(returns: pd.Series, p: int = 1, q: int = 1, forecast_steps: int = 5):
    """
    Fits a GARCH(p, q) model to financial returns and forecasts volatility.

    Args:
        returns (pd.Series): The time series of financial returns.
        p (int): The order of the ARCH term (lagged squared residuals).
        q (int): The order of the GARCH term (lagged conditional variances).
        forecast_steps (int): Number of steps to forecast into the future.

    Returns:
        tuple: A tuple containing:
            - model_fit (arch.univariate.base.ARCHModelResult): Fitted GARCH model.
            - forecast_variance (pd.Series): Forecasted conditional variances.
            - forecast_volatility (pd.Series): Forecasted conditional volatilities (standard deviations).
    """
    print(f"\n--- Fitting GARCH({p},{q}) Model ---")
    try:
        # Fit the GARCH model
        # The 'mean' model is typically 'Constant' for returns
        # The 'vol' model is 'GARCH'
        model = arch_model(returns, mean='Constant', vol='GARCH', p=p, q=q, dist='normal')
        model_fit = model.fit(disp='off') # disp='off' to suppress iteration output
        print(model_fit.summary())

        # Generate forecast for conditional variance
        forecast_results = model_fit.forecast(horizon=forecast_steps, reindex=False)
        # The forecast object contains multiple horizons, we are interested in the first one
        forecast_variance = forecast_results.variance.iloc[-1]
        forecast_volatility = np.sqrt(forecast_variance)

        print(f"\nGARCH Forecasted Conditional Variances for next {forecast_steps} steps:\n{forecast_variance}")
        print(f"\nGARCH Forecasted Conditional Volatilities for next {forecast_steps} steps:\n{forecast_volatility}")

        return model_fit, forecast_variance, forecast_volatility
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        return None, None, None
    
# --- Example Usage ---
if __name__ == "__main__":
    # Generate some synthetic time series data for ARIMA
    np.random.seed(42)
    n_samples = 100

    # Generate some synthetic financial returns data for GARCH
    # GARCH models volatility of returns, so data should be centered around zero.
    # We create a series with changing volatility to demonstrate GARCH.
    returns = pd.Series(np.random.normal(loc=0, scale=0.1, size=n_samples),
                        index=pd.date_range(start='2020-01-01', periods=n_samples, freq='D'))
    # Introduce some volatility clustering
    returns.iloc[50:70] = np.random.normal(loc=0, scale=0.5, size=20)
    returns.name = 'Synthetic_Returns'

    # Plot original returns
    plt.figure(figsize=(12, 6))
    plt.plot(returns, label='Synthetic Returns')
    plt.title('Synthetic Financial Returns with Volatility Clustering')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example GARCH usage
    garch_p = 1
    garch_q = 1
    garch_fit, garch_var_forecast, garch_vol_forecast = fit_and_forecast_garch(returns, garch_p, garch_q, forecast_steps=10)

    if garch_vol_forecast is not None:
        # Plot GARCH forecast
        plt.figure(figsize=(12, 6))
        plt.plot(returns.index, returns.abs(), alpha=0.7, label='Absolute Returns (Proxy for Volatility)')
        # GARCH forecasts conditional standard deviation, which is volatility
        # The index for forecast needs to be created based on the last date of returns
        last_date = returns.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(garch_vol_forecast), freq='D')
        plt.plot(forecast_index, garch_vol_forecast, color='green', linestyle='--', marker='o', label='GARCH Forecasted Volatility')
        plt.title(f'GARCH({garch_p},{garch_q}) Forecasted Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Standard Deviation)')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\n--- GARCH Model Script Finished ---")