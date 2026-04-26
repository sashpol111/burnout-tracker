"""
smote.py — minimal SMOTE implementation (no imbalanced-learn dependency).

Synthetic Minority Oversampling Technique (Chawla et al., 2002).
Generates synthetic minority-class examples by interpolating between
each minority sample and one of its k nearest neighbours.
"""
import numpy as np


def smote(X, y, k=5, random_state=42):
    """
    Oversample the minority class to match the majority class size.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  — binary 0/1
    k : int — number of nearest neighbours to consider per sample
    random_state : int

    Returns
    -------
    X_res, y_res : resampled arrays with balanced classes
    """
    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(y, return_counts=True)
    minority_class  = classes[np.argmin(counts)]
    majority_class  = classes[np.argmax(counts)]
    n_majority      = counts[np.argmax(counts)]
    n_minority      = counts[np.argmin(counts)]
    n_synthetic     = n_majority - n_minority

    X_min = X[y == minority_class]

    # k-nearest neighbours for each minority sample (Euclidean)
    diffs    = X_min[:, None, :] - X_min[None, :, :]  
    sq_dists = (diffs ** 2).sum(axis=2)               
    np.fill_diagonal(sq_dists, np.inf)            
    k        = min(k, n_minority - 1)
    nn_idx   = np.argsort(sq_dists, axis=1)[:, :k]  

    # generating synthetic samples
    synthetic = np.empty((n_synthetic, X.shape[1]))
    for i in range(n_synthetic):
        base     = rng.integers(0, n_minority)
        neighbour = nn_idx[base, rng.integers(0, k)]
        lam      = rng.uniform(0, 1)
        synthetic[i] = X_min[base] + lam * (X_min[neighbour] - X_min[base])

    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.full(n_synthetic, minority_class)])

    # shuffling so minority examples aren't all at the end
    idx   = rng.permutation(len(y_res))
    return X_res[idx], y_res[idx]