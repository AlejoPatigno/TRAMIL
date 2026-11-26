"""Test oracle: the per-bag E3/E4 variants of the full model (graph ablations, free queries, binary head), written on
the frozen per-bag full model of oracle_surrogate.py. One bag at a time, no padding, no masks."""

import torch
from torch import nn

import oracle_surrogate as OS
from TRAMIL.poc_graph_penalty import prior_graph


class PerBagVariant(OS.SurrogateModel):
    def __init__(self, d_in, p_drop=0.0, mu_R=0.0, graph="full", free=None, xi0=None, binary=False, **kw):
        super().__init__(d_in, p_drop, mu_R=mu_R, **kw)
        self.graph, self.free, self.binary = graph, free, binary
        if graph == "prior":
            self.W_a = None
        if free:
            self.rho = None
            self.register_buffer("xi0", torch.zeros(OS.J, OS.H) if xi0 is None else xi0.clone())
        if binary:
            self.heads = nn.Linear(OS.J * OS.H, 1)

    def parts(self, b):
        H_ = self.enc(b["Y"])
        Rc_hat = self.surrogate_rows(H_)
        E = prior_graph(Rc_hat, OS.EPS)
        n = len(H_)
        Delta = torch.zeros(n, n) if self.W_a is None else prior_graph(self.W_a(H_), OS.EPS) / OS.TAU_A * OS.offdiag(n, H_)
        gamma = {"full": Delta + OS.KAPPA / OS.TAU_E * E, "corr": Delta, "prior": OS.KAPPA / OS.TAU_E * E}[self.graph]
        Pi = torch.softmax(gamma, dim=1)
        Ht = H_ + torch.tanh(self.W_g(Pi @ H_))
        D = self.W_o(torch.tanh(self.W_v(Ht)))
        L = D if self.rho is None else D + nn.functional.softplus(self.rho) / OS.TAU * Rc_hat
        A = torch.softmax(L, dim=0)
        Z = A.T @ Ht
        s_hat = self.heads(Z.reshape(-1)) if self.binary else self.heads(Z)
        return {"s_hat": s_hat, "H": H_, "Rc_hat": Rc_hat, "E": E, "Delta": Delta, "Pi": Pi, "A": A}


def bce(logit, y, ow):
    """Per-bag class-weighted BCE of r06: -(ow y log p + (1 - y) log (1 - p))."""
    p = torch.sigmoid(logit)
    return -(ow * y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum()


