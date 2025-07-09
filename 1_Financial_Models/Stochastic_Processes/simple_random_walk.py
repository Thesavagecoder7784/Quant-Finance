import numpy as np
import matplotlib.pyplot as plt

def simulate_simple_random_walk(num_steps, start_position=0, prob_up=0.5):
    """
    Simulates a simple random walk.

    A simple random walk moves +1 with probability `prob_up` and -1 with
    probability `1 - prob_up` at each step.

    Args:
        num_steps (int): The number of steps in the random walk.
        start_position (float): The initial position of the walk. Defaults to 0.
        prob_up (float): The probability of moving up (+1). Must be between 0 and 1.

    Returns:
        numpy.ndarray: An array containing the position of the walk at each step.
                       The length of the array will be `num_steps + 1` (including
                       the starting position).
    """
    if not (0 <= prob_up <= 1):
        raise ValueError("prob_up must be between 0 and 1.")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    # Initialize the walk with the starting position
    positions = np.zeros(num_steps + 1)
    positions[0] = start_position

    # Generate random steps: +1 with prob_up, -1 with (1 - prob_up)
    # np.random.choice([1, -1], size=num_steps, p=[prob_up, 1 - prob_up])
    # A more common way for Bernoulli trials:
    steps = np.where(np.random.rand(num_steps) < prob_up, 1, -1)

    # Accumulate the steps to get the positions
    positions[1:] = start_position + np.cumsum(steps)

    return positions

if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_steps = 1000
    start_pos = 0
    prob_up = 0.5 # Fair coin

    # --- Simulate the Simple Random Walk ---
    walk_path = simulate_simple_random_walk(num_steps, start_pos, prob_up)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(walk_path, label=f'Simple Random Walk (p={prob_up})')
    plt.title('Simulation of a Simple Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Example with a biased walk
    prob_up_biased = 0.6
    walk_path_biased = simulate_simple_random_walk(num_steps, start_pos, prob_up_biased)

    plt.figure(figsize=(12, 6))
    plt.plot(walk_path_biased, color='red', label=f'Biased Simple Random Walk (p={prob_up_biased})')
    plt.title('Simulation of a Biased Simple Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
