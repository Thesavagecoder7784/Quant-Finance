import pandas as pd
import config
import logging


import pandas as pd
import numpy as np
import logging
import config

import pandas as pd
import numpy as np
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_drawdowns(returns: pd.Series) -> tuple[float, pd.Series]:
    """Calculates max drawdown and drawdown series from a series of returns."""
    if returns.empty:
        return 0.0, pd.Series(dtype=float)
    cumulative_returns = (1 + returns.fillna(0)).cumprod() # Fillna 0 to handle initial 0 returns
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown, drawdown

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float = config.RISK_FREE_RATE_ANNUAL, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """Calculates the annualized Sharpe Ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate_annual / periods_per_year)
    sharpe_ratio = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    return sharpe_ratio

def run_backtest(sentiment_df: pd.DataFrame, price_df: pd.DataFrame, initial_capital: float = config.INITIAL_CAPITAL,
                 buy_threshold: float = config.BUY_THRESHOLD, sell_threshold: float = config.SELL_THRESHOLD,
                 sentiment_model: str = config.SENTIMENT_MODEL_FOR_STRATEGY,
                 trading_amount_per_trade_percentage: float = config.TRADING_AMOUNT_PER_TRADE_PERCENTAGE,
                 transaction_cost_percentage: float = config.TRANSACTION_COST_PERCENTAGE
                ):
    """
    Runs a simple backtest of a sentiment-based trading strategy.
    Assumes sentiment is available at the start of the day (or prior day) to inform today's trade.
    """
    logging.info(f"Starting backtest with initial capital: ${initial_capital:,.2f}")
    logging.info(f"Strategy: Buy if {sentiment_model} > {buy_threshold}, Sell if {sentiment_model} < {sell_threshold}")
    logging.info(f"Transaction cost per trade (buy/sell): {transaction_cost_percentage:.2%}")

    price_df.index = pd.to_datetime(price_df.index).normalize()

    # Create FinBERT Net Sentiment if it's the chosen model
    if sentiment_model == 'finbert_net_sentiment':
        sentiment_df['finbert_net_sentiment'] = sentiment_df['avg_finbert_positive'] - sentiment_df['avg_finbert_negative']

    aligned_sentiment = sentiment_df.set_index(['date', 'ticker'])[sentiment_model].unstack().ffill()
    aligned_sentiment = aligned_sentiment.shift(1).reindex(price_df.index, method='ffill')
    aligned_sentiment.dropna(how='all', inplace=True)
    
    price_df = price_df.reindex(aligned_sentiment.index).dropna(how='all')

    if price_df.empty or aligned_sentiment.empty:
        logging.warning("Not enough aligned price or sentiment data for backtesting. Skipping backtest.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    portfolio_value = pd.Series(index=price_df.index, dtype=float)
    current_cash = initial_capital
    current_shares = {ticker: 0.0 for ticker in price_df.columns} # Shares held per ticker

    # Track open and closed trades for win rate calculation
    open_trades = {ticker: [] for ticker in price_df.columns} # List of {'shares': s, 'buy_price': p, 'buy_date': d}
    closed_trades_results = [] # List of {'profit_loss': pl, 'status': 'win'/'loss'}
    
    trade_count = 0 # Total buy/sell actions

    # Initialize daily returns for strategy and benchmark
    strategy_daily_returns = pd.Series(index=price_df.index, dtype=float).fillna(0)
    benchmark_daily_returns = pd.Series(index=price_df.index, dtype=float).fillna(0)

    # Calculate Buy & Hold benchmark return
    if not price_df.empty:
        first_valid_prices = price_df.iloc[0]
        tradable_tickers_benchmark = [t for t in price_df.columns if not pd.isna(first_valid_prices[t]) and first_valid_prices[t] > 0]
        
        if tradable_tickers_benchmark:
            initial_investment_per_ticker = initial_capital / len(tradable_tickers_benchmark)
            benchmark_shares = {ticker: initial_investment_per_ticker / first_valid_prices[ticker]
                                for ticker in tradable_tickers_benchmark}
        else:
            benchmark_shares = {}
            logging.warning("No tradable tickers found at the start for benchmark. Benchmark will be 0 returns.")

        for i in range(1, len(price_df.index)):
            date = price_df.index[i]
            prev_date = price_df.index[i-1]
            
            current_benchmark_value = sum(benchmark_shares.get(ticker, 0) * price_df.loc[date, ticker]
                                          for ticker in benchmark_shares if not pd.isna(price_df.loc[date, ticker]))
            prev_benchmark_value = sum(benchmark_shares.get(ticker, 0) * price_df.loc[prev_date, ticker]
                                       for ticker in benchmark_shares if not pd.isna(price_df.loc[prev_date, ticker]))

            if prev_benchmark_value > 0:
                benchmark_daily_returns.loc[date] = (current_benchmark_value / prev_benchmark_value) - 1
            else:
                benchmark_daily_returns.loc[date] = 0

    # Main backtest loop
    for i in range(len(price_df.index)):
        current_date = price_df.index[i]
        
        current_prices = price_df.loc[current_date]
        
        prev_portfolio_value = portfolio_value.iloc[i-1] if i > 0 else initial_capital

        # Determine trades for each ticker based on aligned sentiment
        for ticker in price_df.columns:
            if ticker in aligned_sentiment.columns and not pd.isna(current_prices[ticker]):
                current_price = current_prices[ticker]

                # --- Check for Stop Loss / Take Profit --- 
                # Create a temporary list to hold positions that are not closed
                temp_open_trades_for_ticker = []
                for pos in open_trades[ticker]:
                    if pos['shares'] > 0: # Only process active positions
                        action = None
                        if current_price <= pos['stop_loss_price']:
                            action = "STOP_LOSS"
                        elif current_price >= pos['take_profit_price']:
                            action = "TAKE_PROFIT"
                        
                        if action:
                            shares_to_close = pos['shares']
                            revenue_from_sale = shares_to_close * current_price
                            transaction_cost = revenue_from_sale * transaction_cost_percentage
                            
                            profit_loss_on_closing = (current_price - pos['buy_price']) * shares_to_close
                            closed_trades_results.append({
                                'profit_loss': profit_loss_on_closing,
                                'status': 'win' if profit_loss_on_closing > 0 else ('loss' if profit_loss_on_closing < 0 else 'neutral')
                            })
                            current_cash += (revenue_from_sale - transaction_cost)
                            current_shares[ticker] -= shares_to_close
                            trade_count += 1 # Count as a forced sell trade
                            logging.debug(f"{current_date} - {action} SELL {shares_to_close:.2f} {ticker} @ {current_price:.2f} (P/L: {profit_loss_on_closing:.2f})")
                        else:
                            temp_open_trades_for_ticker.append(pos) # Keep this position open
                open_trades[ticker] = temp_open_trades_for_ticker # Update open_trades for this ticker

                # --- Sentiment-based Buy/Sell Logic ---
                sentiment_score = aligned_sentiment.loc[current_date, ticker]

                if pd.isna(sentiment_score):
                    continue

                # current_price is already defined above
                
                # Buy Logic
                if sentiment_score >= buy_threshold and current_cash > 0:
                    amount_to_invest = trading_amount_per_trade_percentage * current_cash
                    shares_to_buy = amount_to_invest / current_price
                    
                    if current_cash >= shares_to_buy * current_price:
                        cost_of_shares = shares_to_buy * current_price
                        transaction_cost = cost_of_shares * transaction_cost_percentage
                        
                        if current_cash >= cost_of_shares + transaction_cost: # Ensure cash for shares + cost
                            current_shares[ticker] += shares_to_buy
                            current_cash -= (cost_of_shares + transaction_cost)
                            trade_count += 1
                            
                            # Calculate stop loss and take profit prices
                            stop_loss_price = current_price * (1 - config.STOP_LOSS_PERCENTAGE)
                            take_profit_price = current_price * (1 + config.TAKE_PROFIT_PERCENTAGE)

                            open_trades[ticker].append({
                                'shares': shares_to_buy, 
                                'buy_price': current_price, 
                                'buy_date': current_date,
                                'stop_loss_price': stop_loss_price,
                                'take_profit_price': take_profit_price
                            })
                            logging.debug(f"{current_date} - BUY {shares_to_buy:.2f} {ticker} @ {current_price:.2f} (Sentiment: {sentiment_score:.3f}, Cost: {transaction_cost:.2f})")
                        else:
                            logging.debug(f"{current_date} - Not enough cash for {ticker} shares + transaction cost.")
                    else:
                        logging.debug(f"{current_date} - Not enough cash to buy {ticker}.")
                
                # Sell Logic
                elif sentiment_score <= sell_threshold and current_shares[ticker] > 0:
                    shares_to_sell_total = trading_amount_per_trade_percentage * current_shares[ticker]
                    shares_sold_this_trade = 0.0
                    revenue_from_sale = 0.0

                    # Process open positions (FIFO)
                    temp_open_trades = []
                    while open_trades[ticker] and shares_to_sell_total > 0:
                        oldest_trade = open_trades[ticker].pop(0)
                        shares_to_sell_from_this_position = min(shares_to_sell_total, oldest_trade['shares'])
                        
                        # Calculate profit/loss for this chunk
                        profit_loss_on_chunk = (current_price - oldest_trade['buy_price']) * shares_to_sell_from_this_position
                        closed_trades_results.append({
                            'profit_loss': profit_loss_on_chunk,
                            'status': 'win' if profit_loss_on_chunk > 0 else ('loss' if profit_loss_on_chunk < 0 else 'neutral')
                        })

                        shares_sold_this_trade += shares_to_sell_from_this_position
                        revenue_from_sale += shares_to_sell_from_this_position * current_price
                        current_shares[ticker] -= shares_to_sell_from_this_position
                        shares_to_sell_total -= shares_to_sell_from_this_position

                        # If the oldest trade is not fully closed, add remaining back
                        if oldest_trade['shares'] - shares_to_sell_from_this_position > 0:
                            temp_open_trades.append({
                                'shares': oldest_trade['shares'] - shares_to_sell_from_this_position,
                                'buy_price': oldest_trade['buy_price'],
                                'buy_date': oldest_trade['buy_date'],
                                'stop_loss_price': oldest_trade['stop_loss_price'],
                                'take_profit_price': oldest_trade['take_profit_price']
                            })
                    
                    open_trades[ticker].extend(temp_open_trades) # Add back any remaining partial positions

                    if shares_sold_this_trade > 0:
                        transaction_cost = revenue_from_sale * transaction_cost_percentage
                        current_cash += (revenue_from_sale - transaction_cost)
                        trade_count += 1
                        logging.debug(f"{current_date} - SELL {shares_sold_this_trade:.2f} {ticker} @ {current_price:.2f} (Sentiment: {sentiment_score:.3f}, Cost: {transaction_cost:.2f})")
                    else:
                        logging.debug(f"{current_date} - No shares to sell for {ticker}.")


        # Update portfolio value for the day
        current_portfolio_total = current_cash + sum(current_shares[t] * current_prices[t]
                                                      for t in current_shares if t in current_prices.index and not pd.isna(current_prices[t]))
        portfolio_value.loc[current_date] = current_portfolio_total

        # Calculate daily return for strategy
        if i > 0 and prev_portfolio_value > 0:
            strategy_daily_returns.loc[current_date] = (current_portfolio_total / prev_portfolio_value) - 1
        else:
            strategy_daily_returns.loc[current_date] = 0

    # --- Close any remaining open positions at the end of the backtest ---
    final_date = price_df.index[-1]
    final_prices = price_df.loc[final_date]

    for ticker, positions in open_trades.items():
        for pos in positions:
            if pos['shares'] > 0 and not pd.isna(final_prices.get(ticker)):
                profit_loss_on_closing = (final_prices[ticker] - pos['buy_price']) * pos['shares']
                closed_trades_results.append({
                    'profit_loss': profit_loss_on_closing,
                    'status': 'win' if profit_loss_on_closing > 0 else ('loss' if profit_loss_on_closing < 0 else 'neutral')
                })
                # Adjust cash for closing remaining positions, deducting final transaction cost
                current_cash += (pos['shares'] * final_prices[ticker]) * (1 - transaction_cost_percentage)
                current_shares[ticker] = 0.0 # Clear shares
                logging.debug(f"{final_date} - CLOSED REMAINING {pos['shares']:.2f} {ticker} @ {final_prices[ticker]:.2f} (P/L: {profit_loss_on_closing:.2f})")


    # --- Performance Metrics Calculation ---
    final_portfolio_value = portfolio_value.iloc[-1] if not portfolio_value.empty else initial_capital
    cumulative_returns_strategy = (final_portfolio_value / initial_capital) - 1

    cumulative_returns_benchmark = (1 + benchmark_daily_returns).prod() - 1 if not benchmark_daily_returns.empty else 0.0

    max_drawdown_strategy, _ = calculate_drawdowns(strategy_daily_returns)
    max_drawdown_benchmark, _ = calculate_drawdowns(benchmark_daily_returns)

    sharpe_strategy = calculate_sharpe_ratio(strategy_daily_returns)
    sharpe_benchmark = calculate_sharpe_ratio(benchmark_daily_returns)

    # Calculate win rate from closed_trades_results
    winning_trades_count = sum(1 for trade in closed_trades_results if trade['status'] == 'win')
    losing_trades_count = sum(1 for trade in closed_trades_results if trade['status'] == 'loss')
    total_closed_trades = winning_trades_count + losing_trades_count # Exclude neutral trades if desired
    win_rate = (winning_trades_count / total_closed_trades) * 100 if total_closed_trades > 0 else 0.0


    logging.info("\n--- Backtest Results ---")
    logging.info(f"Initial Capital: ${initial_capital:,.2f}")
    logging.info(f"Final Strategy Value: ${final_portfolio_value:,.2f}")
    logging.info(f"Strategy Cumulative Returns: {cumulative_returns_strategy:.2%}")
    logging.info(f"Benchmark (Buy & Hold) Cumulative Returns: {cumulative_returns_benchmark:.2%}")
    logging.info(f"Strategy Max Drawdown: {max_drawdown_strategy:.2%}")
    logging.info(f"Benchmark Max Drawdown: {max_drawdown_benchmark:.2%}")
    logging.info(f"Strategy Sharpe Ratio (Annualized): {sharpe_strategy:.2f}")
    logging.info(f"Benchmark Sharpe Ratio (Annualized): {sharpe_benchmark:.2f}")
    logging.info(f"Total Trades Executed (Buy/Sell Actions): {trade_count}")
    logging.info(f"Total Closed Trades for Win Rate: {total_closed_trades}")
    logging.info(f"Winning Trades: {winning_trades_count}")
    logging.info(f"Losing Trades: {losing_trades_count}")
    logging.info(f"Strategy Win Rate: {win_rate:.2f}%")


    if cumulative_returns_strategy > cumulative_returns_benchmark and sharpe_strategy > sharpe_benchmark:
        logging.info("Inference: The sentiment-based strategy *outperformed* the Buy & Hold benchmark in returns and risk-adjusted returns!")
    elif cumulative_returns_strategy > cumulative_returns_benchmark:
        logging.info("Inference: The sentiment-based strategy *outperformed* the Buy & Hold benchmark in returns.")
    elif sharpe_strategy > sharpe_benchmark:
        logging.info("Inference: The sentiment-based strategy showed *better risk-adjusted returns* than Buy & Hold.")
    else:
        logging.info("Inference: The sentiment-based strategy *underperformed* the Buy & Hold benchmark for this period and parameters.")

    return portfolio_value, strategy_daily_returns, benchmark_daily_returns
