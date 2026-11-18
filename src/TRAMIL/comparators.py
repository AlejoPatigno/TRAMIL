"""MIL comparators and controls: ABMIL, Gated ABMIL (Ec. 33, 34), permuted prior (Ec. 37), SVM on bag statistics
(Ec. 66). The learned-only bipartite (Ec. 36) is ResidualBipartite(learned_only=True) in model.py."""
import numpy as np
import torch
from torch import nn

from . import aggregation, encoder

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["11", "18.2"],
    "equations": [33, 34, 37, 66],
    "invariants": [
        "abmil_and_gated_abmil_parameter_counts_match_the_proposal_table",
        "abmil_weights_are_a_distribution_and_the_model_is_order_invariant",
        "gated_scoring_differs_from_plain_scoring_with_shared_weights",
        "permuted_prior_keeps_each_column_distribution_but_breaks_row_correspondence",
        "bag_statistics_are_mean_and_sample_std_of_dimension_176",
        "svm_baseline_separates_bag_statistics_and_standardises_inside_the_training_fold",
    ],
}

PRIOR_SEED = 2026


class ABMIL(nn.Module):
    """s_i = w_a^T tanh(V h_i + b_v), optionally gated by sigma(U_a h_i + b_u); softmax over instances."""

    def __init__(self, gated=False, h=32):
        super().__init__()
        self.encoder = encoder.Encoder(h=h)
        self.V = nn.Linear(h, h)
        self.U = nn.Linear(h, h) if gated else None
        self.w = nn.Linear(h, 1, bias=False)
        self.head = aggregation.Head(h)

    def attention(self, X):
        H = self.encoder(X)
        t = torch.tanh(self.V(H))
        if self.U is not None:
            t = t * torch.sigmoid(self.U(H))
        return torch.softmax(self.w(t).squeeze(-1), dim=0), H

    def forward(self, X, R=None):
        a, H = self.attention(X)
        return self.head(a @ H)


def permute_prior(R, subject, seed=PRIOR_SEED):
    """R~_ij = R_{pi_j(i), j}: an independent permutation per column, from (seed, subject), never the label."""
    g = np.random.default_rng([seed, int(subject)])
    return np.stack([R[g.permutation(len(R)), j] for j in range(R.shape[1])], axis=1)


def bag_stats(X):
    """s_b = [mean ; sample std] of the eGeMAPS functionals, R^176; the std of a single instance is 0."""
    sd = X.std(0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1], X.dtype)
    return np.concatenate([X.mean(0), sd]).astype(np.float32)


def fit_svm(bags_X, y, inner_seed, grid=None):
    """Standardise inside the training fold; kernel and C chosen by inner stratified folds (AUROC)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    S = np.stack([bag_stats(X) for X in bags_X])
    grid = grid or {"svc__kernel": ["linear", "rbf"], "svc__C": [0.1, 1.0, 10.0]}
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(class_weight="balanced"))])
    cv = StratifiedKFold(3, shuffle=True, random_state=inner_seed)
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, refit=True).fit(S, y).best_estimator_
