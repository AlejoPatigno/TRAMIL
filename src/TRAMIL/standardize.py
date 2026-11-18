"""Leakage-free acoustic standardization: statistics from the given (training) bags only (Ec. 11, 12)."""
import numpy as np

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["5"],
    "equations": [11, 12],
    "invariants": [
        "scaler_matches_pooled_instance_statistics_and_keeps_constant_features",
        "scaler_is_fitted_on_the_bags_it_is_given_only",
    ],
}


def fit(bags_X):
    """mu_k, s_k pooled over every instance (not per subject); s_k = 1 for constant features."""
    X = np.concatenate(bags_X).astype(np.float64)
    mu, s = X.mean(0), X.std(0)
    s[s == 0] = 1.0
    return mu.astype(np.float32), s.astype(np.float32)


def apply(X, mu, s):
    return ((X - mu) / s).astype(np.float32)
