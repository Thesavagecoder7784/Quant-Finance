import numpy as np
import matplotlib.pyplot as plt

def simulate_fair_random_walk(num_steps, start_position=0):
    """
    Simulates a fair simple random walk, which is a classic example of a martingale.
    A fair random walk has an expected step of zero (e.g., +1 with p=0.5, -1 with p=0.5).

    Args:
        num_steps (int): The number of steps in the random walk.
        start_position (float): The initial position of the walk. Defaults to 0.

    Returns:
        numpy.ndarray: An array containing the position of the walk at each step.
                       The length of the array will be `num_steps + 1`.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    positions = np.zeros(num_steps + 1)
    positions[0] = start_position

    # Steps are +1 or -1 with equal probability (p=0.5)
    steps = np.where(np.random.rand(num_steps) < 0.5, 1, -1)

    positions[1:] = start_position + np.cumsum(steps)

    return positions

if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_steps = 1000
    start_pos = 0

    # --- Simulate the Fair Random Walk (Martingale) ---
    walk_path = simulate_fair_random_walk(num_steps, start_pos)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(walk_path, label='Fair Simple Random Walk (Martingale)')
    plt.title('Simulation of a Martingale (Fair Simple Random Walk)')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Simulate multiple paths to illustrate the martingale property
    # The expected value at any future time, given the present, is the present value.
    # So, the average of many paths should hover around the starting value.
    num_paths = 100
    all_paths = np.zeros((num_paths, num_steps + 1))
    for i in range(num_paths):
        all_paths[i, :] = simulate_fair_random_walk(num_steps, start_pos)

    # Plot individual paths
    plt.figure(figsize=(12, 6))
    for i in range(num_paths):
        plt.plot(all_paths[i, :], alpha=0.1, color='blue') # Faint individual paths

    # Plot the average path
    average_path = np.mean(all_paths, axis=0)
    plt.plot(average_path, color='red', linewidth=2, label='Average Path')
    plt.axhline(y=start_pos, color='green', linestyle='--', label='Starting Position / Expected Value')

    plt.title(f'Multiple Martingale Paths and Their Average ({num_paths} Paths)')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
