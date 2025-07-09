import numpy as np
import matplotlib.pyplot as plt

def simulate_levy_process(num_steps, T, initial_value=0.0,
                          brownian_mu=0.0, brownian_sigma=1.0,
                          jump_lambda=0.1, jump_mu=0.0, jump_sigma=0.5):
    """
    Simulates a general Lévy process with a Brownian motion component
    and a compound Poisson jump component.

    The process X(t) is defined as:
    X(t) = (brownian_mu * t + brownian_sigma * W(t)) + sum_{i=1}^{N(t)} Y_i
    where:
    - W(t) is a standard Brownian motion.
    - N(t) is a Poisson process with rate `jump_lambda`.
    - Y_i are i.i.d. jump sizes (e.g., normally distributed with mean `jump_mu` and std `jump_sigma`).

    Args:
        num_steps (int): The number of discrete time steps.
        T (float): The total time duration for the simulation.
        initial_value (float): The starting value of the process. Defaults to 0.0.
        brownian_mu (float): Drift parameter for the Brownian motion component.
        brownian_sigma (float): Volatility parameter for the Brownian motion component.
        jump_lambda (float): Rate (intensity) of the Poisson process for jumps.
        jump_mu (float): Mean of the jump size distribution (e.g., Gaussian).
        jump_sigma (float): Standard deviation of the jump size distribution.

    Returns:
        numpy.ndarray: An array containing the values of the Lévy process at each time step.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if T <= 0:
        raise ValueError("T (total time) must be positive.")
    if brownian_sigma < 0 or jump_sigma < 0 or jump_lambda < 0:
        raise ValueError("brownian_sigma, jump_sigma, and jump_lambda must be non-negative.")

    dt = T / num_steps
    time_points = np.linspace(0, T, num_steps + 1)
    process_values = np.zeros(num_steps + 1)
    process_values[0] = initial_value

    # Simulate Brownian motion increments
    # dW = sqrt(dt) * Z, where Z ~ N(0,1)
    brownian_increments = np.random.normal(loc=0, scale=np.sqrt(dt), size=num_steps)
    brownian_component = brownian_mu * dt + brownian_sigma * brownian_increments

    # Simulate Poisson jumps
    # For each time step, determine if a jump occurs based on Poisson probability
    # The number of jumps in a small interval dt follows Poisson(lambda * dt)
    # For simplicity, we can approximate by checking if a random number is less than lambda * dt
    # or more accurately, simulate jump times and then add them.
    # Here, we'll use a simplified approach for discrete steps:
    # For each step, draw a random number of jumps from Poisson(lambda * dt)
    num_jumps_per_step = np.random.poisson(jump_lambda * dt, size=num_steps)
    jump_component = np.zeros(num_steps)

    for i in range(num_steps):
        if num_jumps_per_step[i] > 0:
            # If jumps occur, draw jump sizes
            jump_sizes = np.random.normal(loc=jump_mu, scale=jump_sigma, size=num_jumps_per_step[i])
            jump_component[i] = np.sum(jump_sizes)

    # Combine components
    for i in range(1, num_steps + 1):
        process_values[i] = process_values[i-1] + brownian_component[i-1] + jump_component[i-1]

    return time_points, process_values

if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_steps = 500
    total_time = 10.0
    initial_val = 0.0

    # Brownian motion parameters
    bm_mu = 0.1
    bm_sigma = 0.5

    # Jump parameters (Compound Poisson)
    jump_lambda = 0.5 # Average 0.5 jumps per unit time
    jump_mu = 0.2     # Mean jump size
    jump_sigma = 0.1  # Std dev of jump size

    # --- Simulate the Lévy Process ---
    time_points, levy_path = simulate_levy_process(
        num_steps, total_time, initial_val,
        bm_mu, bm_sigma,
        jump_lambda, jump_mu, jump_sigma
    )

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, levy_path, label='Lévy Process (BM + Jumps)')
    plt.title('Simulation of a Lévy Process')
    plt.xlabel('Time')
    plt.ylabel('Process Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Simulate multiple paths to show variability
    num_paths = 3
    plt.figure(figsize=(12, 6))
    for _ in range(num_paths):
        _, path = simulate_levy_process(
            num_steps, total_time, initial_val,
            bm_mu, bm_sigma,
            jump_lambda, jump_mu, jump_sigma
        )
        plt.plot(time_points, path, alpha=0.7)
    plt.title(f'Multiple Simulations of a Lévy Process ({num_paths} Paths)')
    plt.xlabel('Time')
    plt.ylabel('Process Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
