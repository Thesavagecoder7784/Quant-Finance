import numpy as np

class OrderBookDynamics:
    def __init__(self, lob):
        self.lob = lob
        self.history = []

    def record_state(self):
        """Records the current state of the LOB for historical analysis."""
        best_bid = self.lob.get_best_bid()
        best_ask = self.lob.get_best_ask()
        
        state = {
            'timestamp': self.lob.orders[next(reversed(self.lob.orders))]['timestamp'] if self.lob.orders else None,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': self.calculate_mid_price(best_bid, best_ask),
            'spread': self.calculate_spread(best_bid, best_ask),
            'order_flow_imbalance': self.calculate_order_flow_imbalance(),
            'bid_volume': self.get_total_bid_volume(),
            'ask_volume': self.get_total_ask_volume(),
            'bid_depth': len(self.lob.bids),
            'ask_depth': len(self.lob.asks)
        }
        self.history.append(state)

    def calculate_mid_price(self, best_bid, best_ask):
        """Calculates the mid-price from the best bid and ask."""
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def calculate_spread(self, best_bid, best_ask):
        """Calculates the spread between the best bid and ask."""
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_total_bid_volume(self):
        """Calculates the total volume of all bid orders."""
        return sum(order['quantity'] for price_level in self.lob.bids.values() for order in price_level)

    def get_total_ask_volume(self):
        """Calculates the total volume of all ask orders."""
        return sum(order['quantity'] for price_level in self.lob.asks.values() for order in price_level)

    def calculate_order_flow_imbalance(self):
        """
        Calculates the order flow imbalance.
        Formula: (BuyVolume - SellVolume) / (BuyVolume + SellVolume)
        """
        buy_volume = self.get_total_bid_volume()
        sell_volume = self.get_total_ask_volume()
        
        if (buy_volume + sell_volume) == 0:
            return 0
        
        return (buy_volume - sell_volume) / (buy_volume + sell_volume)

    def get_historical_data(self):
        """Returns the recorded history of LOB states."""
        return self.history

