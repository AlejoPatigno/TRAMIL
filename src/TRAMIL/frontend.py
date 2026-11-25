"""Fold-level front end: standardization (r22 Ec. 11, 12) and the PCA projection fitted on training instances only
(Ec. 94), plus the preparation of one fold's bags for the item-supervised models.

A prepared bag is a dict: Y (N, k) projected input, R (N, 8) raw prompt prior (Ec. 10), Rc its per-bag centering
(Ec. 90, training target), Rc_perm the centered permuted target (Ec. 37, seed 2026 and the subject id), G the
reference graph (Ec. 70), s (8,) item scores, y = 1[sum s >= 10], subject, logN = log N_b."""
import numpy as np
import torch

from . import standardize
from .comparators import permute_prior
from .poc_graph_penalty import prior_graph

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["4.3", "5", "6"],
    "equations": [10, 11, 12, 37, 70, 90, 94],
    "experiments": ["E1"],
    "invariants": [
        "pca_is_fitted_on_training_instances_only_with_descending_uncorrelated_scores",
        "full_dimension_is_an_orthogonal_rotation_that_preserves_standardized_norms",
        "fold_bags_never_read_held_out_instances",
        "prepared_bags_carry_targets_permuted_target_and_reference_graph",
        "prompt_source_builds_the_input_from_the_uncentered_prompt_rows_and_never_the_audio",
        "rc_scale_makes_the_zero_surrogate_penalty_one",
    ],
}

EPS = 1e-8  # v07: epsilon of Ec. 70, 71, 116


class FrontEnd:
    """Scaler then V_k: the top-k eigenvectors of the pooled covariance of the standardized training instances.
    Unsupervised, frozen after fit, refitted per training run; k = d is an orthogonal rotation."""

    def __init__(self, k):
        self.k = k

    def fit(self, bags_X):
        self.mu, self.sd = standardize.fit(bags_X)
        Xs = np.concatenate([standardize.apply(X, self.mu, self.sd) for X in bags_X]).astype(np.float64)
        w, V = np.linalg.eigh(Xs.T @ Xs / len(Xs))
        V = V[:, np.argsort(w)[::-1][: self.k]]
        V *= np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])  # deterministic sign per component
        self.V = V.astype(np.float32)
        return self

    def transform(self, X):
        return standardize.apply(X, self.mu, self.sd) @ self.V


def prepare(it, fe, source="X"):
    R = np.asarray(it["R"], np.float32)
    P = permute_prior(R, it["subject"])
    return {
        "Y": torch.from_numpy(np.ascontiguousarray(fe.transform(it[source]), np.float32)),
        "R": torch.from_numpy(R),
        "Rc": torch.from_numpy(R - R.mean(0)),
        "Rc_perm": torch.from_numpy(P - P.mean(0)),
        "G": prior_graph(torch.from_numpy(R), EPS),
        "s": torch.as_tensor(np.asarray(it["s"], np.float32)),
        "y": int(it["y"]),
        "subject": int(it["subject"]),
        "logN": float(np.log(len(R))),
    }


def rc_scale(bags):
    """vbar = mean of (R^c)^2 over every segment and item of the given (training) bags; mu_R = 1 / vbar."""
    return float(torch.cat([b["Rc"] for b in bags]).double().pow(2).mean())


def fold_bags(items, train_idx, eval_idx, k, source="X"):
    """Fit the front end on the training items only and prepare both sets. source = "X" (audio) or "R" (the
    uncentered prompt rows, prompt-only model). Returns (train bags, eval bags, {"vbar", "k", "frontend"})."""
    train_idx, eval_idx = list(train_idx), list(eval_idx)
    fe = FrontEnd(k).fit([items[i][source] for i in train_idx])
    tr = [prepare(items[i], fe, source) for i in train_idx]
    ev = [prepare(items[i], fe, source) for i in eval_idx]
    return tr, ev, {"vbar": rc_scale(tr), "k": k, "frontend": fe}
