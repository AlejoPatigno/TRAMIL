"""Residual semantic prior bipartite MIL wiring (Ec. 13-31) and its learned-only control (Ec. 36)."""
import torch
from torch import nn

from . import aggregation, corrector, encoder

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["6", "7", "8", "9", "11"],
    "equations": [13, 14, 15, 16, 17, 18, 20, 21, 28, 29, 31, 36],
    "invariants": [
        "model_is_invariant_to_joint_permutation_of_instances",
        "parameter_counts_match_the_proposal_table",
    ],
}


class ResidualBipartite(nn.Module):
    """forward(X_std (N, 88), R (N, 8)) -> bag logit eta. learned_only: A = softmax_i(D) and R unused (Ec. 36)."""

    def __init__(self, learned_only=False, per_node_lambda=False):
        super().__init__()
        self.learned_only = learned_only
        self.encoder = encoder.Encoder()
        self.corrector = corrector.Corrector()
        self.intensity = None if learned_only else corrector.PriorIntensity(8 if per_node_lambda else None)
        self.head = aggregation.Head(8 * 32)

    def attention(self, X, R):
        H = self.encoder(X)
        D = self.corrector(H)
        L = D if self.learned_only else corrector.fuse(D, R, self.intensity())
        return aggregation.column_softmax(L), H

    def forward(self, X, R):
        A, H = self.attention(X, R)
        return self.head(aggregation.flatten(aggregation.propagate(A, H)))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
