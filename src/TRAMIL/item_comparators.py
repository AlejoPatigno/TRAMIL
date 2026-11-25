"""Supervision-matched H1 comparators with the eight item heads of r22 Ec. 75, trained on the item MSE (Ec. 77 with
mu_R = 0): ABMIL-8 and Gated ABMIL-8 (Ec. 33, 34), ABMIL+R-8 (Ec. 79, reads R_b at inference), the r06 residual
bipartite with item heads (graph ablation (a) of H2, reads R_b at inference), and Symptom-Guided cross-attention
(Nerella and Yarra, arXiv:2602.15578; r22 Section 11 and 18.2). Every model shares the PCA input and the dropout-ReLU
encoder of Ec. 14 (surrogate.InputEncoder).

Symptom-Guided, minimal faithful version: trainable item queries started from the frozen item embeddings q_j of
Ec. 10, LayerNorm on queries, keys and values, softmax(Q K^T / (tau_j sqrt(d_k))) over the segments with a learnable
temperature tau_j per item, the attended vector of item j to its own head. Keys and values are the encoded segments
H_b (32-d, shared encoder), so the 768-d q_j are mapped to 32 dimensions by the fixed orthonormal projection and the
scale s_0 of Ec. 81 (v07 table); no further projection matrices, linear heads of Ec. 75 instead of the paper's
two-layer heads (supervision matching, r22 Section 11)."""
import math

import numpy as np
import torch
from torch import nn

from . import corrector
from .surrogate import H, J, InputEncoder, ItemHeads

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["11", "18.2"],
    "equations": [33, 34, 75, 79, 81],
    "experiments": ["E1"],
    "invariants": [
        "parameter_counts_match_the_r22_section_11_table",
        "single_attention_models_weight_instances_on_the_simplex_and_feed_one_vector_to_all_heads",
        "per_item_attention_models_have_one_simplex_column_per_item",
        "prompt_reading_comparators_read_r_at_inference_and_the_others_do_not",
        "symptom_guided_queries_start_from_projected_item_embeddings_with_unit_mean_norm",
        "models_are_invariant_to_joint_permutation_of_instances",
    ],
}

QUERY_SEED = 2026  # seed of the fixed orthonormal projection P of Ec. 81 (not fixed by r22 or v07)


class ABMIL8(nn.Module):
    """s_i = w_a^T tanh(V h_i + b_v) [* sigmoid(U_a h_i + b_u)] [+ u^T r_i]; one softmax over the segments; the
    pooled z_b = sum_i a_i h_i feeds all eight heads (Ec. 75 with z_bj = z_b)."""

    def __init__(self, d_in, p_drop=0.0, gated=False, with_R=False):
        super().__init__()
        self.enc = InputEncoder(d_in, p_drop)
        self.V = nn.Linear(H, H)
        self.U = nn.Linear(H, H) if gated else None
        self.w = nn.Linear(H, 1, bias=False)
        self.u = nn.Linear(J, 1, bias=False) if with_R else None
        self.heads = ItemHeads()

    def pool(self, b):
        H_ = self.enc(b["Y"])
        t = torch.tanh(self.V(H_))
        if self.U is not None:
            t = t * torch.sigmoid(self.U(H_))
        sc = self.w(t).squeeze(-1)
        if self.u is not None:
            sc = sc + self.u(b["R"]).squeeze(-1)  # Ec. 79: the raw prompt rows r_bi at inference
        a = torch.softmax(sc, dim=0)
        return a, a @ H_

    def forward(self, b):
        return self.heads(self.pool(b)[1])


class R06Items(nn.Module):
    """r06 residual bipartite (L = D + lambda R / tau, column softmax) with the item heads of Ec. 75 instead of the
    257-parameter classifier; H~ = H (no graph layer)."""

    def __init__(self, d_in, p_drop=0.0):
        super().__init__()
        self.enc = InputEncoder(d_in, p_drop)
        self.corrector = corrector.Corrector()
        self.intensity = corrector.PriorIntensity()
        self.heads = ItemHeads()

    def attention(self, b, H_=None):
        H_ = self.enc(b["Y"]) if H_ is None else H_
        return torch.softmax(corrector.fuse(self.corrector(H_), b["R"], self.intensity()), dim=0)

    def forward(self, b):
        H_ = self.enc(b["Y"])
        return self.heads(self.attention(b, H_).T @ H_)


def query_projection(seed=QUERY_SEED, h=H, d=768):
    """Fixed P in R^{h x d} with orthonormal rows (Ec. 81), from a seeded Gaussian."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, h)))
    return q.T.astype(np.float32)


def project_queries(Q):
    """xi_j(0) = s_0 P q_j with s_0 = 1 / mean_j ||P q_j|| (v07 table), so the initial query norms are about 1."""
    X = torch.from_numpy(np.asarray(Q, np.float32) @ query_projection(d=np.asarray(Q).shape[1]).T)
    return X / X.norm(dim=1).mean()


class SymptomGuided(nn.Module):
    def __init__(self, Q, d_in, p_drop=0.0):
        super().__init__()
        self.enc = InputEncoder(d_in, p_drop)
        self.queries = nn.Parameter(project_queries(Q))
        self.ln_q, self.ln_k, self.ln_v = nn.LayerNorm(H), nn.LayerNorm(H), nn.LayerNorm(H)
        self.log_tau = nn.Parameter(torch.zeros(J))  # tau_j = exp(log_tau_j) > 0, tau_j(0) = 1
        self.heads = ItemHeads()

    def temperature(self):
        return self.log_tau.exp()

    def attention(self, b, H_=None):
        H_ = self.enc(b["Y"]) if H_ is None else H_
        S = self.ln_q(self.queries) @ self.ln_k(H_).T / (self.temperature()[:, None] * math.sqrt(H))
        return torch.softmax(S, dim=1).T  # (N, 8): one distribution over the segments per item

    def forward(self, b):
        H_ = self.enc(b["Y"])
        return self.heads(self.attention(b, H_).T @ self.ln_v(H_))
