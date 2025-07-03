import numpy as np
import matplotlib.pyplot as plt

# --- Brownian Motion (Wiener Process) Simulation ---

def simulate_brownian_motion(num_steps, time_horizon, initial_value=0.0):
    """
    Simulates a standard Brownian Motion (Wiener Process) from scratch.

    A Brownian Motion B(t) is characterized by:
    1. B(0) = 0 (or some initial_value).
    2. Independent increments: B(t) - B(s) is independent of B(u) - B(v) for non-overlapping intervals.
    3. Stationary increments: B(t) - B(s) has the same distribution as B(t-s) - B(0).
    4. Normal increments: B(t) - B(s) follows a normal distribution with mean 0 and variance (t-s).
       Specifically, B(t) - B(s) ~ N(0, t-s).

    Args:
        num_steps (int): The number of discrete steps in the simulation.
        time_horizon (float): The total time period over which to simulate (e.g., 1.0 for one year).
        initial_value (float): The starting value of the Brownian Motion. Default is 0.

    Returns:
        tuple: A tuple containing:
            - time_points (numpy.ndarray): Array of time points.
            - brownian_path (numpy.ndarray): Array of the simulated Brownian Motion path.
    """
    if num_steps <= 0:
        raise ValueError("Number of steps must be positive.")
    
    if time_horizon <= 0:
        raise ValueError("Time horizon must be positive.")

    dt = time_horizon / num_steps
    time_points = np.linspace(0, time_horizon, num_steps + 1)
    brownian_path = np.zeros(num_steps + 1)
    brownian_path[0] = initial_value

    # Simulate increments using standard normal random variables
    # dWt = Z * sqrt(dt), where Z ~ N(0, 1)
    # W(t+dt) = W(t) + dWt
    for i in range(1, num_steps + 1):
        # Generate a random sample from a standard normal distribution (mean=0, std_dev=1)
        # This is a core component of Brownian Motion, representing the random "shock".
        dW = np.random.normal(0, 1) * np.sqrt(dt)
        brownian_path[i] = brownian_path[i-1] + dW

    return time_points, brownian_path

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate a single path
    num_steps_single = 1000
    time_horizon_single = 1.0 # 1 year
    time_points_single, path_single = simulate_brownian_motion(num_steps_single, time_horizon_single)

    plt.figure(figsize=(10, 6))
    plt.plot(time_points_single, path_single, label='Single Brownian Motion Path')
    plt.title('Simulated Brownian Motion Path')
    plt.xlabel('Time (t)')
    plt.ylabel('B(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Simulate multiple paths to observe properties (e.g., spread over time)
    num_paths_multiple = 10
    num_steps_multiple = 500
    time_horizon_multiple = 1.0

    plt.figure(figsize=(10, 6))
    for _ in range(num_paths_multiple):
        time_points_multiple, path_multiple = simulate_brownian_motion(num_steps_multiple, time_horizon_multiple)
        plt.plot(time_points_multiple, path_multiple, alpha=0.7) # alpha for transparency

    plt.title(f'{num_paths_multiple} Simulated Brownian Motion Paths')
    plt.xlabel('Time (t)')
    plt.ylabel('B(t)')
    plt.grid(True)
    plt.show()

    # Demonstrate the normal distribution of B(t) at a fixed time T
    num_simulations_dist = 10000
    fixed_time_T = 1.0 # At t=1, B(1) should be N(0, 1)
    end_values = []
    for _ in range(num_simulations_dist):
        _, path = simulate_brownian_motion(100, fixed_time_T) # 100 steps for each path
        end_values.append(path[-1])

    plt.figure(figsize=(10, 6))
    plt.hist(end_values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black',
             label=f'Distribution of B({fixed_time_T})')
    # Overlay a theoretical normal distribution curve
    from scipy.stats import norm
    x = np.linspace(min(end_values), max(end_values), 100)
    plt.plot(x, norm.pdf(x, loc=0, scale=np.sqrt(fixed_time_T)), 'r--', linewidth=2,
             label=f'Theoretical N(0, {fixed_time_T}) PDF')
    plt.title(f'Distribution of Brownian Motion at Time t={fixed_time_T}')
    plt.xlabel(f'B({fixed_time_T}) Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()
