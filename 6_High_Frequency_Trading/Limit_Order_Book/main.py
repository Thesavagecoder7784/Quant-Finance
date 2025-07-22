from limit_order_book import LimitOrderBook
import time

def print_trades(trades):
    if trades:
        print("--- TRADES EXECUTED ---")
        for trade in trades:
            print(f"  - Price: {trade['price']}, Quantity: {trade['quantity']}")
        print("-----------------------")

def main():
    lob = LimitOrderBook()
    print("Initial Order Book:")
    print(lob)

    print("\n--- Placing initial resting orders ---")
    lob.add_order('buy', 99, 10)
    lob.add_order('buy', 98, 20)
    lob.add_order('sell', 101, 15)
    order_id_to_cancel, _ = lob.add_order('sell', 102, 5)
    print(lob)

    print(f"\n--- Cancelling order {order_id_to_cancel} ---")
    cancelled = lob.cancel_order(order_id_to_cancel)
    if cancelled:
        print(f"Order {order_id_to_cancel} cancelled successfully.")
    print(lob)

    print("\n--- A buyer crosses the spread ---")
    print("Placing a buy order for 20 units at price 101...")
    _, trades = lob.add_order('buy', 101, 20)
    print_trades(trades)
    print(lob)

    print("\n--- A seller crosses the spread ---")
    print("Placing a sell order for 35 units at price 98...")
    _, trades = lob.add_order('sell', 98, 35)
    print_trades(trades)
    print(lob)

if __name__ == "__main__":
    main()