import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def estimate_pd_logistic_regression(X, y, test_size=0.3, random_state=42):
    """
    Estimates Probability of Default (PD) using Logistic Regression.

    Args:
        X (array-like): Feature matrix (e.g., financial ratios, macroeconomic indicators).
        y (array-like): Target vector (0 for no default, 1 for default).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        tuple: A tuple containing:
            - model (LogisticRegression): Trained Logistic Regression model.
            - X_test (array-like): Test feature set.
            - y_test (array-like): Test target set.
            - y_pred_proba (array-like): Predicted probabilities of default for the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LogisticRegression(solver='liblinear', random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (default)

    return model, X_test, y_test, y_pred_proba

if __name__ == "__main__":
    # Example Usage:
    # Generate some synthetic data for demonstration
    np.random.seed(42)
    num_samples = 1000
    # Features: e.g., Debt-to-Equity Ratio, Current Ratio, Profit Margin
    X = np.random.rand(num_samples, 3) * 10
    # Target: 0 for no default, 1 for default
    # Let's make default more likely for higher debt-to-equity and lower profit margin
    y = ((X[:, 0] * 0.1 + X[:, 2] * -0.05 + np.random.randn(num_samples) * 0.5) > 0.5).astype(int)

    print("Synthetic Data Generated:")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Number of defaults (y=1): {np.sum(y)}")
    print(f"Number of non-defaults (y=0): {len(y) - np.sum(y)}")

    model, X_test, y_test, y_pred_proba = estimate_pd_logistic_regression(X, y)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, (y_pred_proba > 0.5).astype(int)):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Predict PD for a new hypothetical company
    new_company_features = np.array([[7.0, 2.5, 0.1]]) # High D/E, decent current ratio, low profit margin
    predicted_pd = model.predict_proba(new_company_features)[:, 1][0]
    print(f"\nPredicted PD for new company with features {new_company_features[0]}: {predicted_pd:.4f}")
