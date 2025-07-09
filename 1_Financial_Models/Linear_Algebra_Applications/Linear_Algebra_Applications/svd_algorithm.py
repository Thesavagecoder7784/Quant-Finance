import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np

def run_svd(data_df: pd.DataFrame, n_components: int):
    """
    Applies Singular Value Decomposition (SVD) for dimensionality reduction
    to the input DataFrame.

    Args:
        data_df (pd.DataFrame): The input DataFrame, typically daily returns of financial assets.
                                Rows are observations (dates), columns are features (assets).
        n_components (int): The number of dimensions (components) to reduce the data to.
                            Must be less than or equal to the number of original features.

    Returns:
        pd.DataFrame: The data transformed into the reduced-dimensional space.
    """
    if data_df.empty:
        print("Input DataFrame for SVD is empty. Cannot perform SVD.")
        return pd.DataFrame()

    if n_components >= data_df.shape[1]:
        print(f"Warning: n_components ({n_components}) must be less than the number of features ({data_df.shape[1]}). "
              f"Returning original data as no reduction is possible.")
        return data_df

    # Standardize the data: SVD, like PCA, is sensitive to feature scaling.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)
    scaled_data_df = pd.DataFrame(scaled_data, index=data_df.index, columns=data_df.columns)

    # Initialize TruncatedSVD. This version is suitable for sparse data but works well
    # for dense data too, and directly provides the reduced-dimensional representation.
    svd = TruncatedSVD(n_components=n_components)

    # Fit SVD to the scaled data and transform it
    reduced_data = svd.fit_transform(scaled_data_df)

    # Create a DataFrame for the reduced data with meaningful column names
    svd_columns = [f'SVD_Comp{i+1}' for i in range(reduced_data.shape[1])]
    reduced_data_df = pd.DataFrame(data=reduced_data,
                                   columns=svd_columns,
                                   index=data_df.index)

    # The explained variance ratio attribute is also available for TruncatedSVD
    explained_variance_ratio = pd.Series(svd.explained_variance_ratio_, index=svd_columns)
    print(f"SVD Explained Variance Ratio for {n_components} components:\n{explained_variance_ratio}")
    print(f"Total SVD Explained Variance by {n_components} components: {explained_variance_ratio.sum():.4f}")


    return reduced_data_df

if __name__ == '__main__':
    # Example Usage for testing svd_algorithm.py
    print("Running example for svd_algorithm.py...")
    # Create a dummy financial dataset (e.g., daily returns of 5 stocks)
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = np.random.randn(100, 5) * 0.01 # Simulate daily returns
    dummy_df = pd.DataFrame(data, index=dates, columns=[f'Stock{i+1}' for i in range(5)])

    print("\nOriginal Data (first 5 rows):")
    print(dummy_df.head())

    # Run SVD to reduce to 2 components
    svd_reduced_data = run_svd(dummy_df, n_components=2)

    print("\nSVD Reduced Data (first 5 rows):")
    print(svd_reduced_data.head())

    # Run SVD with n_components equal to number of features (should warn and return original)
    print("\nRunning SVD with n_components = num_features (should warn):")
    svd_no_reduction = run_svd(dummy_df, n_components=5)
    print("\nSVD No Reduction Data (first 5 rows):")
    print(svd_no_reduction.head())
