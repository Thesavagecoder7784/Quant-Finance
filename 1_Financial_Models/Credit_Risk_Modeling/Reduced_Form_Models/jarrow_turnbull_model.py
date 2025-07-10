import math

def jarrow_turnbull_model(hazard_rate, time):
    """
    Calculates the survival probability and default probability using a simplified
    Jarrow-Turnbull-like reduced-form model (Poisson process).

    Args:
        hazard_rate (float): The constant hazard rate (intensity of default).
        time (float): The time horizon in years.

    Returns:
        tuple: A tuple containing:
            - survival_probability (float): Probability of survival until time t.
            - default_probability (float): Probability of default until time t.
    """
    if hazard_rate < 0 or time < 0:
        raise ValueError("Hazard rate and time must be non-negative.")

    survival_probability = math.exp(-hazard_rate * time)
    default_probability = 1 - survival_probability

    return survival_probability, default_probability

if __name__ == '__main__':
    # Example Usage:
    lambda_ = 0.02  # Constant hazard rate of 2% per year
    t = 5           # Time horizon of 5 years

    surv_prob, def_prob = jarrow_turnbull_model(lambda_, t)

    print(f"Jarrow-Turnbull (Simplified) Model Results:")
    print(f"  Hazard Rate (lambda): {lambda_}")
    print(f"  Time (t): {t}")
    print(f"  Survival Probability: {surv_prob:.4f}")
    print(f"  Default Probability: {def_prob:.4f}")

    # Example with different hazard rate
    lambda_2 = 0.05
    surv_prob_2, def_prob_2 = jarrow_turnbull_model(lambda_2, t)
    print(f"
Example with higher hazard rate (lambda={lambda_2}):")
    print(f"  Survival Probability: {surv_prob_2:.4f}")
    print(f"  Default Probability: {def_prob_2:.4f}")
