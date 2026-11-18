"""Column normalization over instances, bipartite propagation, bag representation and linear classifier
(Ec. 18-21, 24-31; permutation invariance Ec. 27)."""
import torch
from torch import nn

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["8", "8.1", "8.2", "9"],
    "equations": [18, 19, 20, 21, 24, 27, 28, 29, 30, 31],
    "invariants": [
        "attention_columns_are_positive_and_sum_to_one_over_instances",
        "node_representations_lie_in_the_convex_hull_of_instances",
        "attention_maximises_the_kl_regularised_linear_objective",
        "flatten_is_row_concatenation_in_descriptor_order",
        "classifier_is_linear_then_sigmoid_with_threshold_half",
        "model_is_invariant_to_joint_permutation_of_instances",
    ],
}


def column_softmax(L):
    """A_ij = exp(L_ij) / sum_k exp(L_kj): across instances for each node (Ec. 20)."""
    return torch.softmax(L, dim=0)


def propagate(A, H):
    """Z = A^T H, one row per node (Ec. 21)."""
    return A.T @ H


def flatten(Z):
    """z = concat(Z_1, ..., Z_8), the row order of Ec. 28."""
    return Z.reshape(-1)


class Head(nn.Module):
    """eta = w^T z + c (Ec. 29); the sigmoid of Ec. 30 is applied by the loss or by decide()."""

    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


def decide(eta):
    """y_hat = 1[sigmoid(eta) >= 0.5] (Ec. 30)."""
    return (torch.sigmoid(eta) >= 0.5).long()
