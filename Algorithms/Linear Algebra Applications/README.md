# Linear Algebra in Finance

## Algorithms
1. Principal Component Analysis (PCA)
2. Singular Value Decomposition (SVD)

## 1. Principal Component Analysis (PCA)
### What is PCA?
Imagine you have a dataset with many features (columns) for each observation (row). For example, a spreadsheet tracking daily returns of 100 different stocks. While each stock is a separate feature, many stocks tend to move together (e.g., all tech stocks might rise or fall on the same day). This means there's a lot of redundant information or correlation between these features.

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components (PCs).

### How it works?
- Data Compression: PCA finds a lower-dimensional representation of your data that still captures most of the original information.
- Finding Main Patterns: It identifies the main "directions" or "factors" in your data along which the data varies the most.

### What PCA Does:
- Reduces Dimensionality: It transforms your original N features into K new features (principal components), where K is much smaller than N (K < N), without losing too much important information.
- Identifies Underlying Factors: The first few principal components capture the largest amount of variance in the data. In finance, these can often be interpreted as underlying market factors (e.g., a broad market factor, an industry-specific factor).
- Removes Redundancy: By creating uncorrelated components, PCA helps to eliminate multicollinearity (when features are highly correlated with each other), which can be problematic for some statistical models.
- Noise Reduction: It can help filter out random noise from your data, allowing you to focus on the more significant trends.

### Why implement PCA?
- Portfolio Management: Identify key risk factors that drive asset returns.
- Risk Management: Understand the main sources of portfolio volatility.
- Data Visualization: Project high-dimensional data into 2D or 3D for easier plotting and understanding.

Preprocessing for Machine Learning: Reduce the number of features for other algorithms, making them faster and potentially more accurate.

## 2. Singular Value Decomposition (SVD)
### What is SVD?
Singular Value Decomposition (SVD) is a powerful matrix factorization technique that decomposes any rectangular matrix into three simpler, fundamental matrices. It's a generalization of eigenvalue decomposition (which only applies to square matrices).

Mathematically, for any matrix A, SVD decomposes it into:
A=UΣV^T
 
Where:
U: An orthogonal matrix whose columns are the left singular vectors.
Σ (Sigma): A diagonal matrix containing the singular values (which are non-negative and usually ordered from largest to smallest). These singular values represent the "strength" or "importance" of each dimension.
V^T: The transpose of an orthogonal matrix whose columns are the right singular vectors.

### What SVD Does:
- Dimensionality Reduction: Similar to PCA, SVD can be used to reduce the dimensionality of data. By keeping only the largest singular values and their corresponding singular vectors, you can reconstruct an approximation of the original matrix in a lower dimension. In many practical applications with dense data, SVD on the centered data yields results equivalent to PCA.
- Noise Reduction: By discarding smaller singular values, SVD can effectively filter out noise from the data.
- Pseudoinverse Calculation: SVD is used to compute the pseudoinverse of a matrix, which is crucial for solving systems of linear equations that might not have a unique solution.
- Latent Semantic Analysis (LSA): In natural language processing, SVD is used to discover hidden (latent) semantic relationships between terms and documents.
- Recommender Systems: It's a core algorithm in many recommendation engines (e.g., Netflix's original recommendation algorithm).

### Why implement SVD?
- Robust Dimensionality Reduction: Provides a numerically stable way to reduce data dimensions, especially useful when dealing with large or noisy datasets.
- Image Compression: A classic application where SVD can significantly reduce image file size with minimal loss of visual quality.
- Solving Linear Systems: Efficiently solve overdetermined or underdetermined systems of linear equations.
- Foundation for Other Algorithms: SVD is a building block for many other advanced algorithms in machine learning and data science.