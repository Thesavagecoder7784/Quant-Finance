
import numpy as np
import matplotlib.pyplot as plt
from ..Limit_Order_Book.limit_order_book import LimitOrderBook

class LOBBehaviorModel:
    def __init__(self, initial_price, lambda_limit, lambda_market, lambda_cancel, order_size, num_steps):
        self.lob = LimitOrderBook()
        self.initial_price = initial_price
        self.lambda_limit = lambda_limit  # Arrival rate for limit orders
        self.lambda_market = lambda_market  # Arrival rate for market orders
        self.lambda_cancel = lambda_cancel  # Arrival rate for cancellations
        self.order_size = order_size
        self.num_steps = num_steps
        self.history = []

    def simulate(self):
        """Runs the LOB simulation."""
        # Initialize the book with some orders
        self.lob.add_order('buy', self.initial_price - 0.01, self.order_size)
        self.lob.add_order('sell', self.initial_price + 0.01, self.order_size)

        for _ in range(self.num_steps):
            self.step()
            self.record_state()

    def step(self):
        """A single step in the simulation, representing one event."""
        total_lambda = self.lambda_limit + self.lambda_market + self.lambda_cancel
        event_type = np.random.choice(['limit', 'market', 'cancel'], p=[
            self.lambda_limit / total_lambda,
            self.lambda_market / total_lambda,
            self.lambda_cancel / total_lambda
        ])

        if event_type == 'limit':
            self.place_limit_order()
        elif event_type == 'market':
            self.place_market_order()
        elif event_type == 'cancel':
            self.cancel_random_order()

    def place_limit_order(self):
        """Places a new limit order at a random price near the mid-price."""
        side = np.random.choice(['buy', 'sell'])
        mid_price = (self.lob.get_best_bid() + self.lob.get_best_ask()) / 2 if self.lob.get_best_bid() and self.lob.get_best_ask() else self.initial_price
        price_offset = np.random.uniform(-0.05, 0.05)
        price = mid_price + price_offset
        self.lob.add_order(side, price, self.order_size)

    def place_market_order(self):
        """Places a market order that consumes liquidity."""
        side = np.random.choice(['buy', 'sell'])
        if side == 'buy' and self.lob.get_best_ask():
            self.lob.add_order('buy', self.lob.get_best_ask(), self.order_size)
        elif side == 'sell' and self.lob.get_best_bid():
            self.lob.add_order('sell', self.lob.get_best_bid(), self.order_size)

    def cancel_random_order(self):
        """Cancels a random existing order."""
        if not self.lob.orders:
            return
        order_id_to_cancel = np.random.choice(list(self.lob.orders.keys()))
        self.lob.cancel_order(order_id_to_cancel)

    def record_state(self):
        """Records the current state of the LOB."""
        best_bid = self.lob.get_best_bid()
        best_ask = self.lob.get_best_ask()
        self.history.append({
            'mid_price': (best_bid + best_ask) / 2 if best_bid and best_ask else None,
            'spread': best_ask - best_bid if best_bid and best_ask else None
        })

    def plot_simulation(self):
        """Plots the simulated mid-price and spread over time."""
        mid_prices = [s['mid_price'] for s in self.history if s['mid_price'] is not None]
        spreads = [s['spread'] for s in self.history if s['spread'] is not None]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Mid-Price', color='tab:blue')
        ax1.plot(mid_prices, color='tab:blue', label='Mid-Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Spread', color='tab:red')
        ax2.plot(spreads, color='tab:red', label='Spread', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title('LOB Simulation: Mid-Price and Spread')
        plt.show()

