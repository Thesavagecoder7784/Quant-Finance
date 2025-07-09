import random

def simulate_stock_price_conditional_expectation(initial_price: float, num_days: int):
    """
    Simulates a stock price path and demonstrates conditional expectation.

    Args:
        initial_price (float): The starting price of the stock.
        num_days (int): The number of days to simulate.

    Returns:
        None: Prints the simulation steps and conditional expectations.
    """
    print("--- Conditional Expectation in Financial Markets Simulation ---")
    print(f"Initial Stock Price: ${initial_price:.2f}")
    print(f"Simulation Duration: {num_days} days\n")

    # Define possible daily price movements and their probabilities
    # This is a simplified model. In reality, movements would be derived from
    # historical data and volatility models.
    price_movements = [-0.05, 0.00, 0.05]  # -5%, 0%, +5% change
    probabilities = [0.3, 0.4, 0.3]       # Probabilities for each movement

    current_price = initial_price
    price_history = [current_price]

    print(f"Day 0: Current Price = ${current_price:.2f}")

    for day in range(1, num_days + 1):
        # Calculate the conditional expectation of the next day's price
        # given the current day's price (E[S_{t+1} | S_t])
        expected_next_price = 0
        possible_next_prices = []
        for i, movement in enumerate(price_movements):
            next_possible_val = current_price * (1 + movement)
            expected_next_price += next_possible_val * probabilities[i]
            possible_next_prices.append(f"${next_possible_val:.2f} ({movement*100:+.0f}%)")

        print(f"\n--- Day {day} ---")
        print(f"Current Price (S_t): ${current_price:.2f}")
        print(f"Possible Next Day Prices: {', '.join(possible_next_prices)}")
        print(f"Conditional Expectation (E[S_{day} | S_{day-1}]): ${expected_next_price:.2f}")

        # Simulate the actual price movement for the current day
        # Choose a movement based on probabilities
        rand_val = random.random()
        cumulative_prob = 0
        chosen_movement = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                chosen_movement = price_movements[i]
                break

        price_change = current_price * chosen_movement
        current_price += price_change
        current_price = max(0, current_price) # Ensure price doesn't go below zero

        price_history.append(current_price)
        print(f"Actual Price Movement: {chosen_movement*100:+.0f}%")
        print(f"New Current Price (S_t+1): ${current_price:.2f}")

    print("\n--- Simulation Complete ---")
    print("Price History:", [f"${p:.2f}" for p in price_history])

# Example Usage:
if __name__ == "__main__":
    simulate_stock_price_conditional_expectation(initial_price=100.0, num_days=5)

