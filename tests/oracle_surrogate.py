"""Test oracle: the per-bag full model before batching (frozen copy). Full model of r22 Sections 6 to 9 on real bags: the acoustic surrogate of the prompt prior (Ec. 114-116) used twice,
in the instance graph (Ec. 104, 71, 72, 73, 74) and in the readout (Ec. 15-18, 20, 21), with eight item heads (Ec. 75)
and the normalized penalty of Ec. 77. Ported from the E0 toy (poc_e0.ModelE0, poc_e0b.npenalty) with the r22
corrector and intensity in place of the toy readout.

Flags: permuted (Ec. 37 target, E2/E5 control), mu_R = 0 (untargeted, exploratory), log_n (prompt-only model of v07
E5: the input is the prompt rows and log N_b enters every item head), prior = "text" (r12 reference: reads R^c_b and
G_b at inference, no surrogate map)."""
import math

import torch
from torch import nn

from TRAMIL.poc_graph_penalty import center, prior_graph


H, J = 32, 8
TAU_E, KAPPA, TAU_A, EPS = 1.0, 5.0, 0.2, 1e-8  # v07 validation-scheme table
TAU = 0.10  # Ec. 18


class InputEncoder(nn.Module):
    """H = ReLU(Drop_p(Y) W_x + b_x) (Ec. 14): dropout on the projected scores during training only."""

    def __init__(self, d_in, p_drop=0.0, h=H):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_in, h)

    def forward(self, Y):
        return torch.relu(self.fc(self.drop(Y)))


class ItemHeads(nn.Linear):
    """s_hat_j = w_j . z_j + c_j (Ec. 75): row j of the weight is head j; a single pooled vector z (h,) feeds every
    head. PyTorch default initialization of a Linear(h, 8)."""

    def __init__(self, h=H, j=J):
        super().__init__(h, j)

    def forward(self, Z):
        return (self.weight * Z).sum(-1) + self.bias


def offdiag(n, like):
    return 1 - torch.eye(n, dtype=like.dtype, device=like.device)


class SurrogateModel(nn.Module):
    def __init__(self, d_in, p_drop=0.0, mu_R=0.0, permuted=False, prior="surrogate", log_n=False):
        super().__init__()
        assert prior in ("surrogate", "text")
        self.prior, self.mu_R, self.permuted = prior, float(mu_R), permuted
        self.enc = InputEncoder(d_in, p_drop)
        self.W_r = nn.Linear(H, J) if prior == "surrogate" else None
        self.W_a = nn.Linear(H, H, bias=False)
        self.W_g = nn.Linear(H, H, bias=False)
        self.W_v = nn.Linear(H, H)
        self.W_o = nn.Linear(H, J, bias=False)
        nn.init.zeros_(self.W_o.weight)  # W_o = 0 (Section 7.1)
        self.rho = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))  # lambda_0 = 1 (Ec. 17)
        self.heads = ItemHeads()
        self.g = nn.Parameter(torch.zeros(J)) if log_n else None

    def surrogate_rows(self, H_):
        """R_hat = sigmoid(H W_r + b_r) (Ec. 114) and its per-bag centering (Ec. 115)."""
        return center(torch.sigmoid(self.W_r(H_)))

    def parts(self, b):
        H_ = self.enc(b["Y"])
        if self.prior == "surrogate":
            Rc_hat = self.surrogate_rows(H_)
            P, E = Rc_hat, prior_graph(Rc_hat, EPS)  # Ec. 116 (prior_graph centers, a no-op here)
        else:
            Rc_hat, P, E = None, b["Rc"], b["G"]  # r12 reference: prompt prior and Ec. 70 graph at inference
        Delta = prior_graph(self.W_a(H_), EPS) / TAU_A * offdiag(len(H_), H_)  # Ec. 104, 71
        Pi = torch.softmax(Delta + KAPPA / TAU_E * E, dim=1)  # Ec. 72, 73
        Ht = H_ + torch.tanh(self.W_g(Pi @ H_))  # Ec. 74
        D = self.W_o(torch.tanh(self.W_v(Ht)))  # Ec. 15, 16
        L = D + nn.functional.softplus(self.rho) / TAU * P  # Ec. 17, 18
        A = torch.softmax(L, dim=0)  # Ec. 20
        s_hat = self.heads(A.T @ Ht)  # Ec. 21, 75
        if self.g is not None:
            s_hat = s_hat + self.g * b["logN"]
        return {"s_hat": s_hat, "H": H_, "Rc_hat": Rc_hat, "E": E, "Delta": Delta, "Pi": Pi, "A": A}

    def forward(self, b):
        return self.parts(b)["s_hat"]

    def fit_terms(self, b):
        """(s_hat, mu_R ||R_hat^c - R^c||_F^2 / (8 N_b)) with H held constant inside the penalty (Ec. 77), so its
        gradient reaches W_r, b_r only; the target is R^c_b or its permuted version (Ec. 37)."""
        out = self.parts(b)
        if self.prior != "surrogate" or self.mu_R == 0.0:
            return out["s_hat"], 0.0
        target = b["Rc_perm"] if self.permuted else b["Rc"]
        Rc_hat = self.surrogate_rows(out["H"].detach())
        return out["s_hat"], self.mu_R * (Rc_hat - target).pow(2).sum() / (J * len(target))
