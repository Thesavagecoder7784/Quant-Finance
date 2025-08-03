
from ..Limit_Order_Book.limit_order_book import LimitOrderBook
from .order_book_dynamics import OrderBookDynamics
from .lob_behavior_model import LOBBehaviorModel

def run_market_microstructure_analysis():
    """Demonstrates the market microstructure analysis tools."""
    # --- 1. Order Book Dynamics Analysis ---
    print("--- Running Order Book Dynamics Analysis ---")
    lob = LimitOrderBook()
    dynamics = OrderBookDynamics(lob)

    # Add some orders to the book
    lob.add_order('buy', 99, 100)
    dynamics.record_state()
    lob.add_order('buy', 98, 200)
    dynamics.record_state()
    lob.add_order('sell', 101, 100)
    dynamics.record_state()
    lob.add_order('sell', 102, 150)
    dynamics.record_state()

    # A market order comes in
    lob.add_order('buy', 101, 50) # Market buy order
    dynamics.record_state()

    # Print historical data
    history = dynamics.get_historical_data()
    for state in history:
        mid_price_str = f"{state['mid_price']:.2f}" if state['mid_price'] is not None else "N/A"
        spread_str = f"{state['spread']:.2f}" if state['spread'] is not None else "N/A"
        print(f"Timestamp: {state['timestamp']}, Mid-Price: {mid_price_str}, "
              f"Spread: {spread_str}, OFI: {state['order_flow_imbalance']:.2f}, "
              f"Bid Volume: {state['bid_volume']}, Ask Volume: {state['ask_volume']}")

    # --- 2. LOB Behavior Modeling ---
    print("\n--- Running LOB Behavior Simulation ---")
    model = LOBBehaviorModel(
        initial_price=100.0,
        lambda_limit=5.0,      # Avg 5 limit orders per step
        lambda_market=1.0,     # Avg 1 market order per step
        lambda_cancel=2.0,     # Avg 2 cancellations per step
        order_size=10,
        num_steps=1000
    )
    model.simulate()
    model.plot_simulation()

if __name__ == "__main__":
    run_market_microstructure_analysis()


