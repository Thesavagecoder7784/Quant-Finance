import random

def simulate_martingale_betting_game(initial_wealth: float, bet_amount: float, num_rounds: int):
    """
    Simulates a simple "fair game" to demonstrate a discrete-time martingale.
    In a fair game, the expected value of your wealth in the next round,
    given your current wealth, is equal to your current wealth.

    Args:
        initial_wealth (float): The starting wealth of the player.
        bet_amount (float): The amount bet in each round.
        num_rounds (int): The number of rounds to play.

    Returns:
        None: Prints the simulation steps and wealth history.
    """
    print("--- Martingale (Fair Game) Simulation ---")
    print(f"Initial Wealth: ${initial_wealth:.2f}")
    print(f"Bet Amount per Round: ${bet_amount:.2f}")
    print(f"Number of Rounds: {num_rounds}\n")

    current_wealth = initial_wealth
    wealth_history = [current_wealth]

    print(f"Round 0: Current Wealth = ${current_wealth:.2f}")

    for round_num in range(1, num_rounds + 1):
        # Simulate a coin toss: 0 for tails (lose), 1 for heads (win)
        # Assuming a fair coin (P(Heads) = P(Tails) = 0.5)
        coin_toss = random.choice([0, 1]) # 0 for loss, 1 for win

        # Calculate the expected wealth for the next round (E[W_{t+1} | W_t])
        # For a fair game: E[W_{t+1} | W_t] = 0.5 * (W_t + bet_amount) + 0.5 * (W_t - bet_amount)
        # Which simplifies to E[W_{t+1} | W_t] = W_t
        expected_next_wealth = 0.5 * (current_wealth + bet_amount) + 0.5 * (current_wealth - bet_amount)

        print(f"\n--- Round {round_num} ---")
        print(f"Current Wealth (W_t): ${current_wealth:.2f}")
        print(f"Expected Wealth for Next Round (E[W_{round_num} | W_{round_num-1}]): ${expected_next_wealth:.2f}")
        # Verify the martingale property: E[W_{t+1} | W_t] == W_t
        print(f"Martingale Property Check: {expected_next_wealth:.2f} == {current_wealth:.2f} (True for fair game)")

        # Apply the outcome of the coin toss
        if coin_toss == 1:
            current_wealth += bet_amount
            outcome = "Win"
        else:
            current_wealth -= bet_amount
            outcome = "Loss"

        # Ensure wealth doesn't go below zero (or a practical limit)
        current_wealth = max(0, current_wealth)

        wealth_history.append(current_wealth)
        print(f"Coin Toss: {'Heads (Win)' if coin_toss == 1 else 'Tails (Loss)'}")
        print(f"Outcome: {outcome} of ${bet_amount:.2f}")
        print(f"New Current Wealth (W_t+1): ${current_wealth:.2f}")

    print("\n--- Simulation Complete ---")
    print("Wealth History:", [f"${w:.2f}" for w in wealth_history])

# Example Usage:
if __name__ == "__main__":
    simulate_martingale_betting_game(initial_wealth=1000.0, bet_amount=50.0, num_rounds=10)

