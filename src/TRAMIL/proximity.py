"""Semantic proximity R_b between responses and the eight PHQ-8 descriptors (Ec. 9, 10)."""
import numpy as np

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["4.3"],
    "equations": [9, 10],
    "invariants": ["proximity_is_cosine_mapped_to_unit_interval_without_row_normalisation"],
}


def proximity(E, Q):
    """R = (clip(E Q^T, -1, 1) + 1) / 2; no softmax over the eight columns."""
    return ((np.clip(E @ Q.T, -1.0, 1.0) + 1.0) / 2.0).astype(np.float32)
