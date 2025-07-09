import numpy as np
import matplotlib.pyplot as plt

def simulate_renewal_process(max_time, inter_arrival_dist_func, *dist_params):
    """
    Simulates a Renewal Process.

    A renewal process counts the number of events that occur over time,
    where the times between consecutive events (inter-arrival times) are
    independent and identically distributed (i.i.d.) random variables.

    Args:
        max_time (float): The total simulation time.
        inter_arrival_dist_func (callable): A function that generates a single
                                            random sample from the inter-arrival
                                            time distribution (e.g., np.random.exponential).
        *dist_params: Positional arguments to pass to `inter_arrival_dist_func`.
                      For `np.random.exponential`, this would be `scale`.
                      For `np.random.normal`, this would be `loc, scale`.

    Returns:
        tuple: A tuple containing:
            - list: A list of renewal (event) times.
            - list: A list of the number of renewals at each recorded time.
    """
    if max_time <= 0:
        raise ValueError("max_time must be positive.")

    renewal_times = [0.0] # The first event (start of process) is at time 0
    num_renewals = [0]
    current_time = 0.0
    event_count = 0

    while True:
        # Generate the next inter-arrival time
        try:
            next_inter_arrival_time = inter_arrival_dist_func(*dist_params)
        except TypeError:
            raise TypeError(f"inter_arrival_dist_func must be a callable that accepts {len(dist_params)} arguments.")
        except Exception as e:
            raise RuntimeError(f"Error generating inter-arrival time: {e}")

        if next_inter_arrival_time < 0:
            # Handle cases where distribution might return negative (e.g., badly parameterized normal)
            # For inter-arrival times, they must be non-negative.
            # For common distributions like exponential, this isn't an issue.
            print(f"Warning: Generated negative inter-arrival time ({next_inter_arrival_time}). Skipping.")
            continue

        current_time += next_inter_arrival_time

        if current_time > max_time:
            # If the next event occurs after max_time, stop
            break
        else:
            event_count += 1
            renewal_times.append(current_time)
            num_renewals.append(event_count)

    # Ensure the plot extends to max_time, even if no event occurs exactly at max_time
    if renewal_times[-1] < max_time:
        renewal_times.append(max_time)
        num_renewals.append(event_count) # The count remains the same until max_time

    return renewal_times, num_renewals

if __name__ == "__main__":
    # --- Simulation Parameters ---
    max_simulation_time = 100.0 # units of time

    # Example 1: Exponential inter-arrival times (Poisson process is a special case)
    # Mean inter-arrival time = 1/rate. Here, mean=5.0, so rate=0.2
    exp_scale = 5.0 # Mean of the exponential distribution (1/lambda)
    renewal_times_exp, num_renewals_exp = simulate_renewal_process(
        max_simulation_time, np.random.exponential, exp_scale
    )

    # --- Plotting Exponential Inter-Arrival ---
    plt.figure(figsize=(12, 6))
    plt.step(renewal_times_exp, num_renewals_exp, where='post', marker='o', linestyle='-', markersize=4)
    plt.title(f'Simulation of a Renewal Process (Exponential Inter-Arrivals, Mean={exp_scale})')
    plt.xlabel('Time')
    plt.ylabel('Number of Renewals (Events)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max_simulation_time)
    plt.ylim(bottom=0)
    plt.show()

    # Example 2: Normal inter-arrival times (less common for pure renewal, but demonstrates flexibility)
    # Note: Normal distribution can produce negative values, which are not valid inter-arrival times.
    # The `simulate_renewal_process` function includes a basic check for this.
    normal_loc = 5.0   # Mean inter-arrival time
    normal_scale = 1.0 # Standard deviation
    renewal_times_norm, num_renewals_norm = simulate_renewal_process(
        max_simulation_time, np.random.normal, normal_loc, normal_scale
    )

    # --- Plotting Normal Inter-Arrival ---
    plt.figure(figsize=(12, 6))
    plt.step(renewal_times_norm, num_renewals_norm, where='post', marker='o', linestyle='-', markersize=4, color='red')
    plt.title(f'Simulation of a Renewal Process (Normal Inter-Arrivals, Mean={normal_loc}, Std={normal_scale})')
    plt.xlabel('Time')
    plt.ylabel('Number of Renewals (Events)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max_simulation_time)
    plt.ylim(bottom=0)
    plt.show()
