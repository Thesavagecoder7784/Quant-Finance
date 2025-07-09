import numpy as np
import matplotlib.pyplot as plt

def simulate_dtmc(num_steps, initial_state, transition_matrix, state_labels=None):
    """
    Simulates a Discrete-Time Markov Chain (DTMC).

    Args:
        num_steps (int): The number of steps to simulate.
        initial_state (int): The starting state (0-indexed).
        transition_matrix (numpy.ndarray): A square matrix where
                                           transition_matrix[i, j] is the
                                           probability of moving from state i to state j.
                                           Rows must sum to 1.
        state_labels (list, optional): A list of strings for state names.
                                       Defaults to None (uses numerical indices).

    Returns:
        list: A list of states visited during the simulation.
    """
    num_states = transition_matrix.shape[0]

    if transition_matrix.shape[1] != num_states:
        raise ValueError("Transition matrix must be square.")
    if not np.allclose(np.sum(transition_matrix, axis=1), 1.0):
        raise ValueError("Rows of the transition matrix must sum to 1.")
    if not (0 <= initial_state < num_states):
        raise ValueError(f"Initial state {initial_state} is out of bounds for {num_states} states.")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    if state_labels is None:
        state_labels = [str(i) for i in range(num_states)]
    elif len(state_labels) != num_states:
        raise ValueError("Length of state_labels must match the number of states.")

    current_state = initial_state
    path = [current_state]

    for _ in range(num_steps):
        # Choose the next state based on the probabilities from the current state's row
        next_state = np.random.choice(num_states, p=transition_matrix[current_state, :])
        path.append(next_state)
        current_state = next_state

    return path

if __name__ == "__main__":
    # --- Example: Weather Model (Sunny, Cloudy, Rainy) ---
    # States: 0=Sunny, 1=Cloudy, 2=Rainy
    state_labels = ["Sunny", "Cloudy", "Rainy"]

    # Transition matrix P[i, j] = P(next state is j | current state is i)
    # Rows must sum to 1
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],  # From Sunny: 70% Sunny, 20% Cloudy, 10% Rainy
        [0.3, 0.4, 0.3],  # From Cloudy: 30% Sunny, 40% Cloudy, 30% Rainy
        [0.2, 0.3, 0.5]   # From Rainy: 20% Sunny, 30% Cloudy, 50% Rainy
    ])

    num_steps = 50
    initial_state = 0 # Start with Sunny

    # --- Simulate the DTMC ---
    simulated_path = simulate_dtmc(num_steps, initial_state, transition_matrix, state_labels)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    # Map numerical states to string labels for plotting
    plt.plot([state_labels[s] for s in simulated_path], marker='o', linestyle='-', markersize=5)
    plt.title('Simulation of a Discrete-Time Markov Chain (Weather)')
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yticks(range(len(state_labels)), state_labels) # Set y-ticks to state labels
    plt.tight_layout()
    plt.show()

    print(f"Simulated Path (States): {simulated_path}")
    print(f"Simulated Path (Labels): {[state_labels[s] for s in simulated_path]}")
