import numpy as np
import pandas as pd

class Backtester:
    """
    A class for backtesting trading strategies.
    """

    def __init__(self,  data: pd.DataFrame, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        """
        Initializes the Backtester.

        Args:
            strategy: The trading strategy to backtest.
            data: A pandas DataFrame containing the market data.
            initial_capital: The initial capital for the backtest.
            transaction_cost: The transaction cost for each trade.
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.signals = self.strategy.generate_signals(self.data)
        self.portfolio = self._create_portfolio()

    def run_backtest(self):
        """
        Runs the backtest.
        """
        positions = self.signals['signal'].diff()

        for i, (index, row) in enumerate(self.portfolio.iterrows()):
            if i == 0:
                self.portfolio.loc[index, 'holdings'] = self.initial_capital
                self.portfolio.loc[index, 'cash'] = self.initial_capital
                continue

            self.portfolio.loc[index, 'holdings'] = self.portfolio.iloc[i-1]['holdings']
            self.portfolio.loc[index, 'cash'] = self.portfolio.iloc[i-1]['cash']

            if positions.loc[index] == 1:
                # Buy signal
                self.portfolio.loc[index, 'cash'] -= row['close'] * 100 * (1 + self.transaction_cost)
                self.portfolio.loc[index, 'holdings'] += row['close'] * 100
            elif positions.loc[index] == -1:
                # Sell signal
                self.portfolio.loc[index, 'cash'] += row['close'] * 100 * (1 - self.transaction_cost)
                self.portfolio.loc[index, 'holdings'] -= row['close'] * 100

            self.portfolio.loc[index, 'total'] = self.portfolio.loc[index, 'cash'] + self.portfolio.loc[index, 'holdings']
            self.portfolio.loc[index, 'returns'] = self.portfolio.loc[index, 'total'] / self.portfolio.iloc[i-1]['total'] - 1

    def get_performance_metrics(self) -> dict:
        """
        Calculates the performance metrics for the backtest.

        Returns:
            A dictionary containing the performance metrics.
        """
        return {
            'cumulative_returns': self._calculate_cumulative_returns(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }

    def _create_portfolio(self) -> pd.DataFrame:
        """
        Creates the portfolio DataFrame.

        Returns:
            A pandas DataFrame representing the portfolio.
        """
        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = 0.0
        portfolio['total'] = 0.0
        portfolio['returns'] = 0.0
        return portfolio

    def _calculate_cumulative_returns(self) -> float:
        """
        Calculates the cumulative returns.

        Returns:
            The cumulative returns.
        """
        return (self.portfolio['total'][-1] / self.portfolio['total'][0]) - 1

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the Sharpe ratio.

        Args:
            risk_free_rate: The risk-free rate.

        Returns:
            The Sharpe ratio.
        """
        return (self.portfolio['returns'].mean() - risk_free_rate) / self.portfolio['returns'].std()

    def _calculate_max_drawdown(self) -> float:
        """
        Calculates the maximum drawdown.

        Returns:
            The maximum drawdown.
        """
        roll_max = self.portfolio['total'].cummax()
        drawdown = self.portfolio['total'] / roll_max - 1.0
        return drawdown.min()