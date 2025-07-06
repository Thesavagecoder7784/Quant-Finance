import numpy as np
import matplotlib.pyplot as plt

def simulate_branching_process(num_generations, initial_population, offspring_dist_func, *dist_params):
    """
    Simulates a Galton-Watson Branching Process.

    In a branching process, each individual in a population reproduces independently
    according to a specified offspring distribution.

    Args:
        num_generations (int): The number of generations to simulate.
        initial_population (int): The starting number of individuals in generation 0.
        offspring_dist_func (callable): A function that generates a single random
                                        sample for the number of offspring an individual has.
                                        (e.g., np.random.poisson, np.random.binomial).
        *dist_params: Positional arguments to pass to `offspring_dist_func`.
                      For `np.random.poisson`, this would be `lam`.
                      For `np.random.binomial`, this would be `n, p`.

    Returns:
        list: A list containing the population size at each generation.
    """
    if num_generations < 0:
        raise ValueError("num_generations must be non-negative.")
    if initial_population < 0:
        raise ValueError("initial_population must be non-negative.")

    population_sizes = [initial_population]
    current_population = initial_population

    for gen in range(num_generations):
        if current_population == 0:
            # If population dies out, it stays at 0
            population_sizes.append(0)
            continue

        next_generation_population = 0
        for _ in range(current_population):
            # Each individual produces offspring independently
            try:
                offspring = offspring_dist_func(*dist_params)
            except TypeError:
                raise TypeError(f"offspring_dist_func must be a callable that accepts {len(dist_params)} arguments.")
            except Exception as e:
                raise RuntimeError(f"Error generating offspring: {e}")

            # Ensure offspring count is non-negative
            next_generation_population += max(0, int(offspring))

        current_population = next_generation_population
        population_sizes.append(current_population)

    return population_sizes

if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_generations = 20
    initial_pop = 1

    # Example 1: Poisson offspring distribution
    # Mean number of offspring per individual (lambda)
    # If lambda <= 1, population tends to die out. If lambda > 1, it tends to grow.
    poisson_lambda = 1.2 # Average 1.2 offspring per individual
    population_poisson = simulate_branching_process(
        num_generations, initial_pop, np.random.poisson, poisson_lambda
    )

    # --- Plotting Poisson Offspring ---
    plt.figure(figsize=(12, 6))
    plt.plot(population_poisson, marker='o', linestyle='-', markersize=5, label=f'Poisson Offspring (λ={poisson_lambda})')
    plt.title('Simulation of a Branching Process (Poisson Offspring)')
    plt.xlabel('Generation')
    plt.ylabel('Population Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()

    # Example 2: Binomial offspring distribution
    # n = number of trials (e.g., max possible offspring)
    # p = probability of success (e.g., probability of having an offspring)
    binomial_n = 2 # Max 2 offspring
    binomial_p = 0.6 # Probability of success for each trial
    population_binomial = simulate_branching_process(
        num_generations, initial_pop, np.random.binomial, binomial_n, binomial_p
    )

    # --- Plotting Binomial Offspring ---
    plt.figure(figsize=(12, 6))
    plt.plot(population_binomial, marker='o', linestyle='-', markersize=5, color='red', label=f'Binomial Offspring (n={binomial_n}, p={binomial_p})')
    plt.title('Simulation of a Branching Process (Binomial Offspring)')
    plt.xlabel('Generation')
    plt.ylabel('Population Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()

    # Simulate multiple paths to show variability
    num_paths = 5
    plt.figure(figsize=(12, 6))
    for _ in range(num_paths):
        path = simulate_branching_process(num_generations, initial_pop, np.random.poisson, poisson_lambda)
        plt.plot(path, alpha=0.7)
    plt.title(f'Multiple Simulations of a Branching Process ({num_paths} Paths)')
    plt.xlabel('Generation')
    plt.ylabel('Population Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    plt.show()
