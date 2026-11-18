"""Connection corrector: D = tanh(H W_v + b_v) W_o, intensity lambda = softplus(rho), fusion L = D + lambda R / tau
(Ec. 15-18), the semantic prior A0 (Ec. 22) and the per-node intensity (Ec. 57)."""
import math

import torch
from torch import nn

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["7", "8.1", "18.4"],
    "equations": [15, 16, 17, 18, 22, 23, 57],
    "invariants": [
        "corrector_starts_at_zero_correction_and_unit_prior_intensity",
        "initial_attention_equals_the_semantic_prior",
        "attention_is_the_prior_reweighted_by_exp_correction",
        "fuse_with_per_node_intensity_broadcasts_over_columns",
    ],
}

TAU = 0.10


class Corrector(nn.Module):
    def __init__(self, h=32, nodes=8):
        super().__init__()
        self.hidden = nn.Linear(h, h)
        self.out = nn.Linear(h, nodes, bias=False)
        nn.init.zeros_(self.out.weight)  # W_o = 0

    def forward(self, H):
        return self.out(torch.tanh(self.hidden(H)))


class PriorIntensity(nn.Module):
    """lambda = softplus(rho), rho_0 = log(e - 1) so lambda_0 = 1; one scalar (Ec. 17) or one per node (Ec. 57)."""

    def __init__(self, n_nodes=None):
        super().__init__()
        shape = () if n_nodes is None else (n_nodes,)
        self.rho = nn.Parameter(torch.full(shape, math.log(math.e - 1.0)))

    def forward(self):
        return nn.functional.softplus(self.rho)


def fuse(D, R, lam, tau=TAU):
    """L = D + (lambda / tau) R (Ec. 18); a vector lambda acts column-wise (Ec. 57)."""
    return D + lam / tau * R


def prior_attention(R, tau=TAU):
    """A0 = softmax over instances of R / tau (Ec. 22)."""
    return torch.softmax(R / tau, dim=0)
