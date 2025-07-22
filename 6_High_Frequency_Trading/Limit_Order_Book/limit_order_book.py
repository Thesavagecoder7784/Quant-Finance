from collections import deque
import itertools
from datetime import datetime

class LimitOrderBook:
    def __init__(self):
        self.bids = {}  # Price -> Deque of orders
        self.asks = {}  # Price -> Deque of orders
        self.orders = {}  # Order ID -> Order
        self.order_id_generator = itertools.count()
        self.sorted_bid_prices = []
        self.sorted_ask_prices = []

    def _add_order_to_book(self, order):
        """Adds a resting order to the book."""
        book = self.bids if order['side'] == 'buy' else self.asks
        price = order['price']
        if price not in book:
            book[price] = deque()
        book[price].append(order)
        self.orders[order['id']] = order

        # Update sorted price lists
        if order['side'] == 'buy' and price not in self.sorted_bid_prices:
            self.sorted_bid_prices.append(price)
            self.sorted_bid_prices.sort(reverse=True)
        elif order['side'] == 'sell' and price not in self.sorted_ask_prices:
            self.sorted_ask_prices.append(price)
            self.sorted_ask_prices.sort()

    def add_order(self, side, price, quantity):
        """Adds a new order and attempts to match it."""
        order_id = next(self.order_id_generator)
        order = {
            'id': order_id,
            'side': side,
            'price': price,
            'quantity': quantity,
            'timestamp': datetime.now()
        }
        
        trades = self._match_order(order)
        
        if order['quantity'] > 0:
            self._add_order_to_book(order)
            
        return order_id, trades

    def _match_order(self, order):
        """The core matching engine logic."""
        trades = []
        if order['side'] == 'buy':
            while self.sorted_ask_prices and order['price'] >= self.sorted_ask_prices[0] and order['quantity'] > 0:
                best_ask_price = self.sorted_ask_prices[0]
                ask_queue = self.asks[best_ask_price]
                
                while ask_queue and order['quantity'] > 0:
                    resting_order = ask_queue[0]
                    trade_quantity = min(order['quantity'], resting_order['quantity'])
                    
                    trades.append({
                        'price': best_ask_price,
                        'quantity': trade_quantity,
                        'time': datetime.now()
                    })
                    
                    order['quantity'] -= trade_quantity
                    resting_order['quantity'] -= trade_quantity
                    
                    if resting_order['quantity'] == 0:
                        ask_queue.popleft()
                        del self.orders[resting_order['id']]
                
                if not ask_queue:
                    del self.asks[best_ask_price]
                    self.sorted_ask_prices.pop(0)

        elif order['side'] == 'sell':
            while self.sorted_bid_prices and order['price'] <= self.sorted_bid_prices[0] and order['quantity'] > 0:
                best_bid_price = self.sorted_bid_prices[0]
                bid_queue = self.bids[best_bid_price]

                while bid_queue and order['quantity'] > 0:
                    resting_order = bid_queue[0]
                    trade_quantity = min(order['quantity'], resting_order['quantity'])

                    trades.append({
                        'price': best_bid_price,
                        'quantity': trade_quantity,
                        'time': datetime.now()
                    })

                    order['quantity'] -= trade_quantity
                    resting_order['quantity'] -= trade_quantity

                    if resting_order['quantity'] == 0:
                        bid_queue.popleft()
                        del self.orders[resting_order['id']]

                if not bid_queue:
                    del self.bids[best_bid_price]
                    self.sorted_bid_prices.pop(0)
                    
        return trades

    def cancel_order(self, order_id):
        """Cancels an existing order by its ID."""
        if order_id not in self.orders:
            return False
        
        order_to_cancel = self.orders[order_id]
        price = order_to_cancel['price']
        side = order_to_cancel['side']
        
        book = self.bids if side == 'buy' else self.asks
        
        if price in book:
            order_queue = book[price]
            for i, order in enumerate(order_queue):
                if order['id'] == order_id:
                    del order_queue[i]
                    if not order_queue:
                        del book[price]
                        if side == 'buy':
                            self.sorted_bid_prices.remove(price)
                        else:
                            self.sorted_ask_prices.remove(price)
                    del self.orders[order_id]
                    return True
        return False

    def get_best_bid(self):
        return self.sorted_bid_prices[0] if self.sorted_bid_prices else None

    def get_best_ask(self):
        return self.sorted_ask_prices[0] if self.sorted_ask_prices else None

    def __str__(self):
        book_str = "--- Limit Order Book ---\n"
        book_str += "ASKS:\n"
        # Sort asks from high to low for display
        for price in sorted(self.sorted_ask_prices, reverse=True):
            quantity = sum(order['quantity'] for order in self.asks[price])
            book_str += f"  {price}: {quantity}\n"
        
        book_str += "------------------------\n"
        
        # Bids are already sorted high to low
        for price in self.sorted_bid_prices:
            quantity = sum(order['quantity'] for order in self.bids[price])
            book_str += f"  {price}: {quantity}\n"
        book_str += "BIDS:\n"
        book_str += "------------------------\n"
        return book_str