# Comprehensive Portfolio Performance and Risk Analytics
Explains key metrics used to evaluate investment portfolio performance and risk, providing a more comprehensive view than just raw returns.

## Metrics
1. Sharpe Ratio
Metric used to assess the performance of an investment by measuring its risk-adjusted return. Basically, indicates the extra return given based on the unit risk taken

Sharpe Ratio = (Rx - Rf)/StdDev Rx

- Rx: Expected Portfolio Return
- Rf: Risk Free Return
- StdDev Rx: Standard Deviation of portfolio return/volatility
  
2. Sortino Ratio
Risk-adjusted performance measure that focuses on downside risk, indicating how much excess return an investment generates for each unit of downside risk. It's a modification of the Sharpe ratio, but unlike the Sharpe ratio, it only considers the standard deviation of negative returns when calculating risk, making it particularly useful for risk-averse investors.

3. Maximum Drawdown (MDD)
Risk management metric that quantifies the largest peak-to-trough decline in the value of an investment portfolio or trading account over a specific period. It is the worst-case scenario of losses until the portfolio reaches a new high.

Maximum Drawdown = (Trough Value - Peak Value) / Peak Value 

4. Calmar Ratio
Risk-adjusted performance measure that assesses the potential return of an investment relative to its maximum drawdown, which is the largest peak-to-trough decline during a specific period. Used to evaluate the performance of investments, particularly in hedge funds and commodity trading advisors.

Calmar Ratio = Average Annual Rate of Return / Maximum Drawdown

5. Beta (β)

6. Jensen's Alpha (α)

7. Treynor Ratio

8. Value at Risk (VaR)

9. Compound Annual Growth Rate (CAGR)

## Example output
--- Portfolio Performance & Risk Analytics ---
Period: 756 trading days (~3.0 years)
Annual Risk-Free Rate: 2.00%

1. Sharpe Ratio: -0.4522
2. Sortino Ratio: -0.7171
3. Maximum Drawdown: -0.2918 (-29.18%)
4. Calmar Ratio: -0.2335
5. Beta (vs Market): 0.0238
6. Jensen's Alpha (annualized): -0.1966
7. Treynor Ratio: -2.8578
8. Value at Risk (95% Daily VaR): 0.0140 (1.40%)
9. CAGR (Initial: 100000.00, Final: 86240.56): -0.0481 (-4.81%)

--- Interpretation Hints ---
Sharpe Ratio: Higher is better, indicates better risk-adjusted return (total risk).
Sortino Ratio: Higher is better, focuses on downside risk.
Maximum Drawdown: Closer to 0 is better, indicates less severe losses from peak.
Calmar Ratio: Higher is better, measures return relative to max drawdown.
Beta: >1 more volatile than market, <1 less volatile, =1 moves with market.
Jensen's Alpha: Positive means portfolio outperformed CAPM's expected return.
Treynor Ratio: Higher is better, measures return relative to systematic risk.
Value at Risk (VaR): Estimates worst expected loss over a time frame at a confidence level.
CAGR: Smoothed average annual growth rate over the investment period.
