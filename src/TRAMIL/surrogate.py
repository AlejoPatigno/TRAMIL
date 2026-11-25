"""Full model of r22 Sections 6 to 9 on real bags: the acoustic surrogate of the prompt prior (Ec. 114-116) used twice,
in the instance graph (Ec. 104, 71, 72, 73, 74) and in the readout (Ec. 15-18, 20, 21), with eight item heads (Ec. 75)
and the normalized penalty of Ec. 77. Ported from the E0 toy (poc_e0.ModelE0, poc_e0b.npenalty) with the r22
corrector and intensity in place of the toy readout.

Flags: permuted (Ec. 37 target, E2/E5 control), mu_R = 0 (untargeted, exploratory), log_n (prompt-only model of v07
E5: the input is the prompt rows and log N_b enters every item head), prior = "text" (r12 reference: reads R^c_b and
G_b at inference, no surrogate map).

Batched (T6b): every forward takes a padded batch (item_training.collate/take) with an instance mask; centering,
graphs and softmaxes run over the valid rows of each bag only (masked_center, masked_graph, masked_softmax), so the
outputs equal the per-bag model for any padding. A lone bag is run as a batch of one."""
import math

import torch
from torch import nn

from .item_training import batch_of

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["6", "6.1", "7", "8", "9", "11", "12"],
    "equations": [14, 15, 16, 17, 18, 20, 21, 37, 70, 71, 72, 73, 74, 75, 77, 104, 105, 106, 114, 115, 116],
    "experiments": ["E1", "E2", "E5"],
    "invariants": [
        "shapes_finite_outputs_and_attention_on_the_simplex",
        "fixed_constants_of_the_v07_table",
        "zero_surrogate_penalty_is_one_under_mu_r_one_over_vbar",
        "penalty_gradient_reaches_only_the_surrogate_map",
        "item_loss_still_trains_the_surrogate_map_through_the_forward_path",
        "inference_reads_audio_only",
        "model_is_invariant_to_joint_permutation_of_instances",
        "parameter_count_is_r22_full_model_plus_the_surrogate_map",
        "permuted_target_flag_trains_against_the_permuted_rows",
        "prompt_only_model_reads_prompt_rows_and_log_bag_size_in_every_head",
        "r12_reference_reads_the_prompt_prior_and_graph_at_inference",
        "every_flag_trains_one_epoch_with_finite_predictions",
        "batched_forward_equals_the_per_bag_models",
        "batched_loss_and_gradients_equal_the_per_bag_sum",
        "extra_padding_never_changes_outputs_or_gradients",
    ],
}

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


def masked_center(X, mask):
    """X - (per-bag mean over the valid rows), padded rows set to 0; X [B, N, c], mask [B, N] (Ec. 90, 115)."""
    w = mask.unsqueeze(-1).to(X.dtype)
    return (X - (X * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)) * w


def masked_graph(X, mask, eps=EPS):
    """Cosine graph of the per-bag centered rows (Ec. 70, 71, 116); entries touching a padded row are 0."""
    Xc = masked_center(X, mask)
    n = Xc.norm(dim=-1, keepdim=True)
    return (Xc @ Xc.transpose(1, 2)) / (n * n.transpose(1, 2) + eps)


def masked_softmax(L, valid, dim):
    """Softmax along dim over the valid entries only (valid broadcasts against L); padded entries get 0."""
    return torch.softmax(L.masked_fill(~valid, float("-inf")), dim=dim)


def pool(A, V):
    """sum_i A_bij V_bi per bag: [B, N, 8] x [B, N, h] -> [B, 8, h] (Ec. 21)."""
    return A.transpose(1, 2) @ V


def unbatch(out, single):
    if not single:
        return out
    if isinstance(out, dict):
        return {k: (v[0] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in out.items()}
    return out[0]


class ItemHeads(nn.Linear):
    """s_hat_j = w_j . z_j + c_j (Ec. 75): row j of the weight is head j; Z is [B, 8, h] (one vector per item) or
    [B, 1, h] (one pooled vector for every head). PyTorch default initialization of a Linear(h, 8)."""

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

    def surrogate_rows(self, H_, mask):
        """R_hat = sigmoid(H W_r + b_r) (Ec. 114) and its per-bag centering over the valid rows (Ec. 115)."""
        return masked_center(torch.sigmoid(self.W_r(H_)), mask)

    def _parts(self, b):
        m = b["mask"]
        H_ = self.enc(b["Y"])
        if self.prior == "surrogate":
            Rc_hat = self.surrogate_rows(H_, m)
            P, E = Rc_hat, masked_graph(Rc_hat, m)  # Ec. 116 (the graph re-centers, a no-op here)
        else:
            Rc_hat, P, E = None, b["Rc"], b["G"]  # r12 reference: prompt prior and Ec. 70 graph at inference
        Delta = masked_graph(self.W_a(H_), m) / TAU_A * offdiag(H_.shape[1], H_)  # Ec. 104, 71
        Pi = masked_softmax(Delta + KAPPA / TAU_E * E, m[:, None, :], dim=2)  # Ec. 72, 73: rows over valid columns
        Ht = H_ + torch.tanh(self.W_g(Pi @ H_))  # Ec. 74
        D = self.W_o(torch.tanh(self.W_v(Ht)))  # Ec. 15, 16
        L = D + nn.functional.softplus(self.rho) / TAU * P  # Ec. 17, 18
        A = masked_softmax(L, m[:, :, None], dim=1)  # Ec. 20: per item over the valid segments
        s_hat = self.heads(pool(A, Ht))  # Ec. 21, 75
        if self.g is not None:
            s_hat = s_hat + self.g * b["logN"][:, None]
        return {"s_hat": s_hat, "H": H_, "Rc_hat": Rc_hat, "E": E, "Delta": Delta, "Pi": Pi, "A": A}

    def parts(self, b):
        bb, single = batch_of(b)
        return unbatch(self._parts(bb), single)

    def forward(self, b):
        bb, single = batch_of(b)
        return unbatch(self._parts(bb)["s_hat"], single)

    def fit_terms(self, b):
        """(s_hat, mu_R ||R_hat^c_b - R^c_b||_F^2 / (8 N_b) per bag) with H held constant inside the penalty (Ec. 77),
        so its gradient reaches W_r, b_r only; the target is R^c_b or its permuted version (Ec. 37)."""
        bb, single = batch_of(b)
        out = self._parts(bb)
        if self.prior != "surrogate" or self.mu_R == 0.0:
            return unbatch(out["s_hat"], single), 0.0
        target = bb["Rc_perm"] if self.permuted else bb["Rc"]
        m = bb["mask"]
        Rc_hat = self.surrogate_rows(out["H"].detach(), m)
        err = ((Rc_hat - target) * m.unsqueeze(-1)).pow(2).sum((1, 2))
        pen = self.mu_R * err / (J * bb["N"].to(err.dtype))
        return unbatch(out["s_hat"], single), unbatch(pen, single)
