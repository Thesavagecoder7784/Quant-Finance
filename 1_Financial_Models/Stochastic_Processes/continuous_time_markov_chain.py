import numpy as np
import matplotlib.pyplot as plt

def simulate_ctmc(max_time, initial_state, rate_matrix, state_labels=None):
    """
    Simulates a Continuous-Time Markov Chain (CTMC) using the Gillespie algorithm (or similar).

    Args:
        max_time (float): The total simulation time.
        initial_state (int): The starting state (0-indexed).
        rate_matrix (numpy.ndarray): A square matrix where rate_matrix[i, j] is the
                                     rate of transitioning from state i to state j (for i != j).
                                     The diagonal elements are typically negative, representing
                                     the negative sum of outgoing rates from that state.
                                     Q-matrix: q_ii = -sum_{j!=i} q_ij.
        state_labels (list, optional): A list of strings for state names.
                                       Defaults to None (uses numerical indices).

    Returns:
        tuple: A tuple containing:
            - list: A list of states visited during the simulation.
            - list: A list of times at which the state changes.
    """
    num_states = rate_matrix.shape[0]

    if rate_matrix.shape[1] != num_states:
        raise ValueError("Rate matrix must be square.")
    if not (0 <= initial_state < num_states):
        raise ValueError(f"Initial state {initial_state} is out of bounds for {num_states} states.")
    if max_time < 0:
        raise ValueError("max_time must be non-negative.")

    if state_labels is None:
        state_labels = [str(i) for i in range(num_states)]
    elif len(state_labels) != num_states:
        raise ValueError("Length of state_labels must match the number of states.")

    current_state = initial_state
    current_time = 0.0

    states_history = [current_state]
    times_history = [current_time]

    while current_time < max_time:
        # Get outgoing rates from the current state (excluding self-loops)
        outgoing_rates = np.array([rate_matrix[current_state, j] for j in range(num_states) if j != current_state])
        total_outgoing_rate = np.sum(outgoing_rates)

        if total_outgoing_rate <= 1e-9: # If no outgoing transitions (absorbing state)
            current_time = max_time # Fast-forward to end
            break

        # Time until next event (exponential distribution)
        time_to_next_event = np.random.exponential(1.0 / total_outgoing_rate)
        current_time += time_to_next_event

        if current_time >= max_time:
            # If the next event occurs after max_time, we stop here
            # The last state persists until max_time
            states_history.append(current_state)
            times_history.append(max_time)
            break

        # Determine the next state based on relative probabilities
        # Normalize outgoing rates to get probabilities
        transition_probabilities = outgoing_rates / total_outgoing_rate
        possible_next_states = [j for j in range(num_states) if j != current_state]
        next_state = np.random.choice(possible_next_states, p=transition_probabilities)

        current_state = next_state
        states_history.append(current_state)
        times_history.append(current_time)

    return states_history, times_history

if __name__ == "__main__":
    # --- Example: Simple Machine Repair Model ---
    # States: 0=Working, 1=Broken
    state_labels = ["Working", "Broken"]

    # Rate matrix (Q-matrix):
    # Q[i, j] = rate from i to j (for i != j)
    # Q[i, i] = -sum_{k!=i} Q[i, k]
    # Example:
    # From Working (0): rate to Broken (1) = 0.1 (breakdown rate)
    # From Broken (1): rate to Working (0) = 0.5 (repair rate)
    rate_matrix = np.array([
        [-0.1,  0.1],  # From Working: -0.1 (sum of outgoing) to Broken 0.1
        [ 0.5, -0.5]   # From Broken: 0.5 to Working, -0.5 (sum of outgoing)
    ])

    max_simulation_time = 50.0 # units of time (e.g., hours)
    initial_state = 0 # Start with machine Working

    # --- Simulate the CTMC ---
    simulated_states, simulated_times = simulate_ctmc(
        max_simulation_time, initial_state, rate_matrix, state_labels
    )

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    # Plot the states as a step function
    plt.step(simulated_times, [state_labels[s] for s in simulated_states], where='post', marker='o', linestyle='-', markersize=5)
    plt.title('Simulation of a Continuous-Time Markov Chain (Machine State)')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yticks(range(len(state_labels)), state_labels) # Set y-ticks to state labels
    plt.xlim(0, max_simulation_time)
    plt.tight_layout()
    plt.show()

    print("Simulated States:", [state_labels[s] for s in simulated_states])
    print("Simulated Times:", [f"{t:.2f}" for t in simulated_times])
