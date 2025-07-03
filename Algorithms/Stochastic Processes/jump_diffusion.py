import numpy as np
import matplotlib.pyplot as plt

# --- Jump Diffusion Process Simulation ---

def simulate_jump_diffusion(num_steps, time_horizon, initial_value,
                            drift, volatility,
                            jump_rate, jump_mean, jump_std):
    """
    Simulates a Jump Diffusion Process from scratch.

    A Jump Diffusion process combines a continuous diffusion component (like
    Brownian Motion) with a discrete jump component (like a Compound Poisson Process).
    It's often used to model asset prices that exhibit sudden, large movements.

    The process X(t) is typically modeled as:
    dX(t) = (drift - jump_rate * jump_mean) * dt + volatility * dW(t) + dJ(t)
    where:
    - drift (mu): The constant drift rate of the continuous component.
    - volatility (sigma): The volatility of the continuous component.
    - dW(t): Increment of a standard Brownian Motion.
    - dJ(t): Increment of a Compound Poisson Process.
      - The number of jumps follows a Poisson Process with rate 'jump_rate' (lambda_J).
      - The size of each jump follows a normal distribution with 'jump_mean' (mu_J)
        and 'jump_std' (sigma_J).

    For simplicity, this implementation will simulate the continuous part and
    then add jumps at Poisson-distributed times, with jump sizes drawn from a normal distribution.
    The drift term is adjusted to account for the mean of the jumps.

    Args:
        num_steps (int): The number of discrete steps for the continuous part.
        time_horizon (float): The total time period.
        initial_value (float): The starting value of the process.
        drift (float): The drift coefficient (mu) for the continuous part.
        volatility (float): The volatility coefficient (sigma) for the continuous part.
        jump_rate (float): The average rate (lambda_J) of jumps per unit of time.
        jump_mean (float): The mean (mu_J) of the jump size distribution.
        jump_std (float): The standard deviation (sigma_J) of the jump size distribution.

    Returns:
        tuple: A tuple containing:
            - time_points (numpy.ndarray): Array of time points.
            - jump_diffusion_path (numpy.ndarray): Array of the simulated path.
            - jump_times (list): List of times when jumps occurred.
            - jump_sizes (list): List of sizes of the jumps.
    """
    if num_steps <= 0 or time_horizon <= 0:
        raise ValueError("num_steps and time_horizon must be positive.")
    if volatility < 0 or jump_rate < 0 or jump_std < 0:
        raise ValueError("Volatility, jump rate, and jump standard deviation cannot be negative.")

    dt = time_horizon / num_steps
    time_points = np.linspace(0, time_horizon, num_steps + 1)
    jump_diffusion_path = np.zeros(num_steps + 1)
    jump_diffusion_path[0] = initial_value

    # Adjust drift for the mean effect of jumps, as per Merton's model for asset prices
    # This ensures the expected return is primarily driven by 'drift'
    # If modeling a general process, this adjustment might not be needed depending on the SDE form.
    adjusted_drift = drift - jump_rate * jump_mean

    jump_times = []
    jump_sizes = []

    # Simulate the continuous part and add jumps
    for i in range(1, num_steps + 1):
        # Continuous part (Euler-Maruyama approximation for SDE)
        # dX = (drift - lambda_J * mu_J) * dt + sigma * dW
        dW = np.random.normal(0, 1) * np.sqrt(dt)
        continuous_increment = adjusted_drift * dt + volatility * dW
        current_value = jump_diffusion_path[i-1] + continuous_increment

        # Jump part (Poisson process for jump occurrences)
        # Check if a jump occurs in this small time interval dt
        # The probability of a jump in dt is approximately jump_rate * dt
        if np.random.rand() < (jump_rate * dt): # Using uniform random number for Bernoulli trial
            jump_size = np.random.normal(loc=jump_mean, scale=jump_std)
            current_value += jump_size
            jump_times.append(time_points[i])
            jump_sizes.append(jump_size)

        jump_diffusion_path[i] = current_value

    return time_points, jump_diffusion_path, jump_times, jump_sizes

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters for simulation
    num_steps_jd = 1000
    time_horizon_jd = 1.0 # 1 year
    initial_value_jd = 100.0 # Initial asset price
    drift_jd = 0.05 # 5% annual drift
    volatility_jd = 0.20 # 20% annual volatility

    # Jump parameters
    jump_rate_jd = 0.5 # Average of 0.5 jumps per year
    jump_mean_jd = 0.10 # Average jump size is +10%
    jump_std_jd = 0.05 # Standard deviation of jump sizes is 5%

    time_points_jd, path_jd, jump_times_jd, jump_sizes_jd = \
        simulate_jump_diffusion(num_steps_jd, time_horizon_jd, initial_value_jd,
                                drift_jd, volatility_jd,
                                jump_rate_jd, jump_mean_jd, jump_std_jd)

    plt.figure(figsize=(12, 7))
    plt.plot(time_points_jd, path_jd, label='Jump Diffusion Path', color='blue')

    # Plot jumps as markers
    # Scale jump sizes for visual representation if needed, or just mark the jump times
    for j_time, j_size in zip(jump_times_jd, jump_sizes_jd):
        # Find the index in time_points closest to j_time
        idx = np.argmin(np.abs(time_points_jd - j_time))
        plt.plot(j_time, path_jd[idx], 'ro', markersize=8, alpha=0.7) # Red circle at jump point
        # Optional: Add an arrow or line to show jump magnitude
        # plt.annotate(f'{j_size:.2f}', (j_time, path_jd[idx]), textcoords="offset points", xytext=(0,10), ha='center', color='red')


    plt.title('Simulated Jump Diffusion Process')
    plt.xlabel('Time (t)')
    plt.ylabel('X(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Total jumps observed: {len(jump_times_jd)}")
    if jump_times_jd:
        print(f"Jump times: {[f'{t:.2f}' for t in jump_times_jd]}")
        print(f"Jump sizes: {[f'{s:.2f}' for s in jump_sizes_jd]}")

    # Simulate multiple paths to observe the variability
    num_paths_jd_multiple = 5
    plt.figure(figsize=(12, 7))
    for _ in range(num_paths_jd_multiple):
        _, path_jd_multi, _, _ = simulate_jump_diffusion(num_steps_jd, time_horizon_jd, initial_value_jd,
                                                         drift_jd, volatility_jd,
                                                         jump_rate_jd, jump_mean_jd, jump_std_jd)
        plt.plot(time_points_jd, path_jd_multi, alpha=0.7)

    plt.title(f'{num_paths_jd_multiple} Simulated Jump Diffusion Paths')
    plt.xlabel('Time (t)')
    plt.ylabel('X(t)')
    plt.grid(True)
    plt.show()
