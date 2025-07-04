import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def run_pca(data_df: pd.DataFrame, n_components: int = None):
    """
    Applies Principal Component Analysis (PCA) to the input DataFrame.

    Args:
        data_df (pd.DataFrame): The input DataFrame, typically daily returns of financial assets.
                                Rows are observations (dates), columns are features (assets).
        n_components (int, optional): The number of principal components to retain.
                                      If None, all components are kept. If an integer,
                                      it specifies the number of components.
                                      Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Principal components transformed data.
            - pd.Series: Explained variance ratio for each component.
            - pd.DataFrame: Principal components (loadings).
    """
    if data_df.empty:
        print("Input DataFrame for PCA is empty. Cannot perform PCA.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    # Standardize the data: PCA is sensitive to the scale of the features.
    # We standardize to ensure that features with larger values do not dominate the analysis.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)
    scaled_data_df = pd.DataFrame(scaled_data, index=data_df.index, columns=data_df.columns)

    # Initialize PCA. If n_components is None, it defaults to min(n_samples, n_features).
    # If n_components is an integer, it specifies the number of components to keep.
    pca = PCA(n_components=n_components)

    # Fit PCA to the scaled data and transform it
    principal_components = pca.fit_transform(scaled_data_df)

    # Create a DataFrame for the principal components with meaningful column names
    pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    principal_components_df = pd.DataFrame(data=principal_components,
                                           columns=pc_columns,
                                           index=data_df.index)

    # Explained variance ratio: Proportion of variance explained by each component
    explained_variance_ratio = pd.Series(pca.explained_variance_ratio_, index=pc_columns)

    # Components (loadings): These represent the weights of the original features
    # in each principal component. They indicate how much each original variable
    # contributes to each principal component.
    components_df = pd.DataFrame(pca.components_, columns=data_df.columns, index=pc_columns)

    return principal_components_df, explained_variance_ratio, components_df

if __name__ == '__main__':
    # Example Usage for testing pca_algorithm.py
    print("Running example for pca_algorithm.py...")
    # Create a dummy financial dataset (e.g., daily returns of 5 stocks)
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = np.random.randn(100, 5) * 0.01 # Simulate daily returns
    dummy_df = pd.DataFrame(data, index=dates, columns=[f'Stock{i+1}' for i in range(5)])

    print("\nOriginal Data (first 5 rows):")
    print(dummy_df.head())

    # Run PCA with 2 components
    pc_data, explained_var, loadings = run_pca(dummy_df, n_components=2)

    print("\nPrincipal Components (first 5 rows):")
    print(pc_data.head())

    print("\nExplained Variance Ratio:")
    print(explained_var)
    print(f"Total explained variance by 2 components: {explained_var.sum():.4f}")

    print("\nPrincipal Component Loadings:")
    print(loadings)

    # Run PCA with 95% explained variance
    print("\nRunning PCA to explain 95% of variance...")
    pca_95, explained_var_95, loadings_95 = run_pca(dummy_df, n_components=0.95)
    print("\nPrincipal Components (first 5 rows) for 95% variance:")
    print(pca_95.head())
    print("\nExplained Variance Ratio for 95% variance:")
    print(explained_var_95)
    print(f"Total explained variance by components (95%): {explained_var_95.sum():.4f}")
    print("\nPrincipal Component Loadings for 95% variance:")
    print(loadings_95)
