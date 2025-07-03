import numpy as np
import matplotlib.pyplot as plt

# --- Poisson Process Simulation ---

def simulate_poisson_process(rate, time_horizon):
    """
    Simulates a homogeneous Poisson Process from scratch using inter-arrival times.

    A homogeneous Poisson Process N(t) counts the number of events that occur
    up to time t, given a constant average rate (lambda).

    Properties:
    1. N(0) = 0.
    2. Independent increments: The number of events in non-overlapping intervals are independent.
    3. Stationary increments: The distribution of the number of events in an interval
       depends only on the length of the interval, not its starting point.
    4. The number of events in any interval of length 'tau' follows a Poisson distribution
       with parameter (rate * tau), i.e., N(t+tau) - N(t) ~ Poisson(rate * tau).
    5. The inter-arrival times (time between consecutive events) are independent
       and identically distributed exponential random variables with parameter 'rate'.

    Args:
        rate (float): The constant average rate (lambda) of events per unit of time.
                      Must be positive.
        time_horizon (float): The total time period over which to simulate.
                              Must be positive.

    Returns:
        tuple: A tuple containing:
            - event_times (numpy.ndarray): Array of times at which events occur.
            - num_events_path (numpy.ndarray): Array of the cumulative number of events at each time point.
            - time_points_plot (numpy.ndarray): Time points for plotting the step function.
    """
    if rate <= 0:
        raise ValueError("Rate (lambda) must be positive.")
    if time_horizon <= 0:
        raise ValueError("Time horizon must be positive.")

    # Simulate inter-arrival times using the exponential distribution
    # The exponential distribution is given by: f(x; lambda) = lambda * exp(-lambda * x)
    # Its inverse CDF is used for sampling: x = -ln(1 - U) / lambda, where U ~ Uniform(0, 1)
    # In numpy, np.random.exponential(scale=1/rate) directly samples from Exp(rate)
    inter_arrival_times = []
    current_time = 0.0
    while current_time < time_horizon:
        # Generate a random inter-arrival time
        # The scale parameter for np.random.exponential is 1/lambda (mean of the distribution)
        dt_event = np.random.exponential(scale=1/rate)
        current_time += dt_event
        if current_time <= time_horizon: # Only include events within the horizon
            inter_arrival_times.append(current_time)
        else:
            break # Stop if the next event time exceeds the horizon

    event_times = np.array(inter_arrival_times)

    # Prepare data for plotting the step function of event counts
    # The number of events increases by 1 at each event_time
    time_points_plot = np.insert(event_times, 0, 0.0) # Start at time 0
    num_events_path = np.arange(len(event_times) + 1) # Cumulative count

    return event_times, num_events_path, time_points_plot

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate a single Poisson Process path
    rate_single = 2.0  # 2 events per unit of time (e.g., per hour)
    time_horizon_single = 5.0 # 5 units of time

    event_times_single, num_events_path_single, time_points_plot_single = \
        simulate_poisson_process(rate_single, time_horizon_single)

    plt.figure(figsize=(10, 6))
    plt.step(time_points_plot_single, num_events_path_single, where='post', label=f'Poisson Process (Rate={rate_single})')
    plt.title('Simulated Poisson Process Path')
    plt.xlabel('Time (t)')
    plt.ylabel('Number of Events N(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Event times for single simulation: {event_times_single}")
    print(f"Total events in {time_horizon_single} units: {len(event_times_single)}")

    # Demonstrate the Poisson distribution of event counts at a fixed time T
    num_simulations_dist = 10000
    fixed_time_T = 1.0 # At t=1, N(1) should be Poisson(rate * 1)
    rate_dist = 3.0
    event_counts_at_T = []

    for _ in range(num_simulations_dist):
        event_times_dist, _, _ = simulate_poisson_process(rate_dist, fixed_time_T)
        event_counts_at_T.append(len(event_times_dist))

    plt.figure(figsize=(10, 6))
    # Plot histogram of observed event counts
    bins = np.arange(0, max(event_counts_at_T) + 2) - 0.5 # Bins centered on integers
    plt.hist(event_counts_at_T, bins=bins, density=True, alpha=0.7, color='lightgreen', edgecolor='black',
             label=f'Observed Event Counts at t={fixed_time_T}')

    # Overlay theoretical Poisson PMF
    from scipy.stats import poisson
    k_values = np.arange(0, max(event_counts_at_T) + 1)
    pmf_values = poisson.pmf(k_values, mu=rate_dist * fixed_time_T)
    plt.plot(k_values, pmf_values, 'ro', linestyle='--', markersize=8, label=f'Theoretical Poisson({rate_dist * fixed_time_T}) PMF')

    plt.title(f'Distribution of Event Counts at Time t={fixed_time_T} (Rate={rate_dist})')
    plt.xlabel('Number of Events (k)')
    plt.ylabel('Probability')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.show()
