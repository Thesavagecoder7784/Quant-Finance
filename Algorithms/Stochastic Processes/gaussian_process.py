import numpy as np
import matplotlib.pyplot as plt

def squared_exponential_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Squared Exponential (RBF) kernel function.
    K(x_i, x_j) = variance * exp(-0.5 * ||x_i - x_j||^2 / length_scale^2)

    Args:
        x1 (numpy.ndarray): First set of input points (e.g., (N, D) where D is dimension).
        x2 (numpy.ndarray): Second set of input points (e.g., (M, D)).
        length_scale (float): The characteristic length scale of the kernel.
        variance (float): The amplitude of the kernel.

    Returns:
        numpy.ndarray: The kernel (covariance) matrix (N, M).
    """
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 * sq_dist / length_scale**2)

def simulate_gaussian_process(x_points, mean_function=None, kernel_function=None,
                              kernel_params=None, num_samples=1):
    """
    Simulates paths from a Gaussian Process.

    Args:
        x_points (numpy.ndarray): The input points (e.g., time points) at which to
                                  sample the GP. Shape (N, D) where N is number of points,
                                  D is dimension of input space.
        mean_function (callable, optional): A function that takes x_points and returns
                                            the mean vector. If None, assumes zero mean.
        kernel_function (callable, optional): A function that takes two sets of x_points
                                              and kernel_params and returns the covariance matrix.
                                              Defaults to squared_exponential_kernel.
        kernel_params (dict, optional): Dictionary of parameters for the kernel function.
                                        e.g., {'length_scale': 1.0, 'variance': 1.0}.
        num_samples (int): The number of sample paths to generate.

    Returns:
        numpy.ndarray: An array of shape (num_samples, len(x_points)) containing the
                       simulated GP paths.
    """
    if x_points.ndim == 1:
        x_points = x_points[:, np.newaxis] # Ensure x_points is 2D for kernel calculation

    if mean_function is None:
        mean_vector = np.zeros(x_points.shape[0])
    else:
        mean_vector = mean_function(x_points)

    if kernel_function is None:
        kernel_function = squared_exponential_kernel
    if kernel_params is None:
        kernel_params = {'length_scale': 1.0, 'variance': 1.0}

    # Compute the covariance matrix
    covariance_matrix = kernel_function(x_points, x_points, **kernel_params)

    # Add a small amount to the diagonal for numerical stability (jitter)
    covariance_matrix += 1e-6 * np.eye(covariance_matrix.shape[0])

    # Sample from the multivariate Gaussian distribution
    # np.random.multivariate_normal requires mean and covariance matrix
    # size specifies how many independent samples to draw
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=num_samples)

    return samples

if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_points = 100
    x_min, x_max = 0, 10
    x_points = np.linspace(x_min, x_max, num_points)

    # Define a custom mean function (optional, can be zero mean)
    def custom_mean_function(x):
        return np.sin(x.flatten() * 0.5) * 2 # Example: a sine wave mean

    # Kernel parameters (for Squared Exponential Kernel)
    kernel_parameters = {
        'length_scale': 1.5, # How smooth the function is
        'variance': 1.0      # Overall amplitude of the function
    }

    num_sample_paths = 5

    # --- Simulate the Gaussian Process ---
    # Using a zero mean function for simplicity in this example,
    # but you could pass `mean_function=custom_mean_function`
    gp_samples = simulate_gaussian_process(
        x_points,
        mean_function=None, # or custom_mean_function
        kernel_function=squared_exponential_kernel,
        kernel_params=kernel_parameters,
        num_samples=num_sample_paths
    )

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    for i in range(num_sample_paths):
        plt.plot(x_points, gp_samples[i, :], alpha=0.7, label=f'Sample {i+1}')

    # Plot the mean function if it was used
    if custom_mean_function is not None:
        plt.plot(x_points, custom_mean_function(x_points), 'k--', linewidth=2, label='Mean Function')

    plt.title('Simulation of a Gaussian Process (Squared Exponential Kernel)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
