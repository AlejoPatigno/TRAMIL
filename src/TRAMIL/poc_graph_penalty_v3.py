"""Synthetic PoC v3: favorable-state existence proof of the r12 bounded acoustic residual.

Pre-registered in odd/tasks/tramil-poc-graph-penalty.md ("v3"). Model = r12 Section 6.1 + Ec. 77, on a toy encoder:
  H_b    = relu(X_b W_x + b_x), 16 -> 32                    (Ec. 14 role: non-negative instance embedding)
  G_b    = cosine of bag-centered prompt rows, eps           (Ec. 90, 70)
  G^a_b  = cosine of bag-centered rows of H_b W_a, same eps   (Ec. 104, 71), W_a 32x32
  Delta_b = (1/tau_a) G^a_b * (11^T - I)                     (Ec. 71)
  Pi_b   = rowsoftmax(Delta_b + (kappa/tau_e) G_b)           (Ec. 72, 73)
  H~_b   = H_b + tanh(Pi_b H_b W_g)                          (Ec. 74)
  readout: A_b = colsoftmax(H~_b W_o), W_o = 0 at init; s^_bj = w_j . (A_b[:, j]^T H~_b) + c_j   (item heads)
  loss   = mean_b (1/8) sum_j (s^_bj - s_bj)^2 + mu mean_b ||(G^a_b - G_b) * (11^T - I)||_F^2 / N_b^2   (Ec. 77)
           with H_b detached inside the penalty, so its gradient reaches W_a only.

Data (favorable state). Per seed, 8 cluster means with sd s = 0.5 in dims 0..3, zero in the 12 nuisance dims;
noise sd 1 in all 16 dims. Per bag N ~ U{20..60}, acoustic item z_ac uniform on 8 items, prompt item z_text equal to
z_ac except on exactly round(0.25 N) segments, moved to a different item. Item severities s_b in {0..3}^8 uniform;
X_i = m_{z_ac} + beta (s_{b,z_ac} - 1.5) u_{z_ac} + noise, u_j = m_j / ||m_j||, beta = 0.5.
Prompt rows R as in v1 (0.6 + 0.03 onehot(z_text) + N(0, 0.005^2), clipped).

PoC settings, NOT proposal values: 1/tau_a = kappa/tau_e = 5, eps = 1e-8, Adam lr 1e-2, 40 epochs, batch 8,
256 train / 64 val / 64 test bags, 5 seeds. Evaluation runs in float64 (see READINGS).
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from TRAMIL.poc_graph_penalty import SEEDS, prior_graph

DX, DH, J, D_SIG = 16, 32, 8, 4
S_CLUSTER, BETA, DISAGREE = 0.5, 0.5, 0.25
INV_TAU_A = KAPPA_TAU = 5.0
MU_GRID_V3 = [0.0, 0.01, 0.1, 1.0, 10.0, 1000.0]
N_TRAIN, N_VAL, N_TEST = 256, 64, 64
OUT = Path(__file__).resolve().parents[2] / "TRAMIL" / "Results"

READINGS = """Readings of the pre-registered spec, fixed before any v3 run:
- Encoder: one linear 16->32 + ReLU (non-negative H as in Ec. 14), trained jointly; W_a, W_g 32x32 without bias.
- Item task: severities s_bj uniform on {0,1,2,3}; the segment signal is beta (s - 1.5) along the unit direction of
  its acoustic cluster mean, beta = 0.5 (same scale as s); the 25% disagreeing segments carry the severity of their
  acoustic item. Readout: learned per-item attention pooling A = colsoftmax(H~ W_o) (W_o = 0 at init, as in r12)
  plus one linear head per item; the prompt prior enters only through G (no prior term in the readout), so the
  graph is the only place the text can act. MSE over the 8 items, unweighted (no binary label, omega_b = 1).
- Optimizer and length reused from v1/v2: Adam lr 1e-2, 40 epochs, batch 8; no epoch selection.
- Losses compared (val and test) are the item MSE without the mu term (as Ec. 39 selects without it).
- mu* per seed = argmin val loss over the mu grid (ties -> smaller mu); F1/F2 use that seed's test value at mu*.
- Arms: full model per mu; prior-only = Delta fixed to 0 (W_a unused); acoustic-only = kappa = 0 with mu = 0
  (r12: a graph without the prior must not receive it through the penalty).
- F2: on test segments with z_text != z_ac, row mass of Pi on true acoustic neighbours = sum over k != i with
  z_ac(k) = z_ac(i) of Pi_ik; pooled over all disagreeing segments of the test bags, then mean over seeds.
- F3: ||G^a - G|| = ||(G^a - G) * (11^T - I)||_F / N per test bag, mean over bags; Spearman over the 6-point mu grid
  of the seed means. For prior-only the value uses the untrained W_a and is shown for completeness only.
- F4: every arm and seed, every test bag: max|Delta| < 1/tau_a, diag(Delta) = 0, and
  exp(-2 gamma) < N Pi_ik < exp(2 gamma) with the arm's gamma (full 1/tau_a + kappa/tau_e; prior-only kappa/tau_e;
  acoustic-only 1/tau_a). Evaluation is in float64: in float32, eps = 1e-8 is below the resolution of ||p||^2, so a
  cosine of two nearly parallel rows can round to 1 and the strict bound would fail by rounding, not by the math.
- Overall PASS iff F1, F2, F3, F4 all hold on seed means."""


def cluster_means(seed):
    means = np.zeros((J, DX))
    means[:, :D_SIG] = S_CLUSTER * np.random.default_rng(seed).standard_normal((J, D_SIG))
    return means


def make_data_v3(seed, n_bags, split, noise=1.0):
    means = cluster_means(seed)
    u = means / np.linalg.norm(means, axis=1, keepdims=True)
    rng = np.random.default_rng([seed, 3, split])  # stream 3: disjoint from the v1/v2 scenario streams 0..2
    bags = []
    for _ in range(n_bags):
        n = int(rng.integers(20, 61))
        za = rng.integers(0, J, n)
        zt = za.copy()
        idx = rng.choice(n, round(DISAGREE * n), replace=False)
        zt[idx] = (za[idx] + rng.integers(1, J, len(idx))) % J
        s = rng.integers(0, 4, J)
        X = means[za] + BETA * (s[za] - 1.5)[:, None] * u[za] + noise * rng.standard_normal((n, DX))
        R = torch.tensor(np.clip(0.6 + 0.03 * np.eye(J)[zt] + 0.005 * rng.standard_normal((n, J)), 0, 1),
                         dtype=torch.float32)
        bags.append(dict(X=torch.tensor(X, dtype=torch.float32), R=R, G=prior_graph(R), s=torch.tensor(s),
                         z_text=torch.tensor(zt), z_ac=torch.tensor(za)))
    return bags


def offdiag(n, dtype):
    return 1 - torch.eye(n, dtype=dtype)


class ModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(DX, DH)
        self.W_a, self.W_g = nn.Linear(DH, DH, bias=False), nn.Linear(DH, DH, bias=False)
        self.W_o = nn.Linear(DH, J, bias=False)
        nn.init.zeros_(self.W_o.weight)
        self.head = nn.Linear(DH, J)

    def delta(self, H):
        return INV_TAU_A * prior_graph(self.W_a(H)) * offdiag(len(H), H.dtype)

    def forward(self, X, G, kappa_tau=KAPPA_TAU, prior_only=False):
        H = torch.relu(self.enc(X))
        Delta = torch.zeros(len(H), len(H), dtype=H.dtype) if prior_only else self.delta(H)
        Pi = torch.softmax(Delta + kappa_tau * G, dim=1)
        Ht = H + torch.tanh(self.W_g(Pi @ H))
        A = torch.softmax(self.W_o(Ht), dim=0)
        return (self.head.weight * (A.T @ Ht)).sum(1) + self.head.bias, Pi, Delta, H

    def alignment_penalty(self, H, G):
        Ga = prior_graph(self.W_a(H.detach()))
        return ((Ga - G) * offdiag(len(H), H.dtype)).pow(2).sum() / len(H) ** 2


def train_v3(bags, mu, kappa_tau=KAPPA_TAU, prior_only=False, seed=0, epochs=40, lr=1e-2, batch=8):
    torch.manual_seed(seed)
    model = ModelV3()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(bags)).tolist()
        for st in range(0, len(order), batch):
            loss = 0.0
            for i in order[st:st + batch]:
                b = bags[i]
                s_hat, _, _, H = model(b["X"], b["G"], kappa_tau, prior_only)
                loss = loss + (s_hat - b["s"]).pow(2).mean()
                if mu:
                    loss = loss + mu * model.alignment_penalty(H, b["G"])
            opt.zero_grad()
            (loss / len(order[st:st + batch])).backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_v3(model, bags, kappa_tau=KAPPA_TAU, prior_only=False):
    m = copy.deepcopy(model).double()
    gamma = (0.0 if prior_only else INV_TAU_A) + kappa_tau
    loss, mass, ga, mx, dg, lo, hi = [], [], [], [], [], [], []
    for b in bags:
        G = prior_graph(b["R"].double())
        s_hat, Pi, Delta, H = m(b["X"].double(), G, kappa_tau, prior_only)
        n, za = len(H), b["z_ac"]
        loss.append((s_hat - b["s"]).pow(2).mean().item())
        same = (za[:, None] == za[None, :]).double() * offdiag(n, torch.float64)
        mass.append((Pi * same).sum(1)[b["z_text"] != za])
        ga.append(m.alignment_penalty(H, G).sqrt().item())
        mx.append(Delta.abs().max().item())
        dg.append(torch.diagonal(Delta).abs().max().item())
        lo.append((n * Pi).min().item())
        hi.append((n * Pi).max().item())
    return dict(loss=float(np.mean(loss)), nbr_mass_dis=torch.cat(mass).mean().item(), ga_dist=float(np.mean(ga)),
                max_abs_delta=max(mx), delta_diag_max=max(dg), npi_min=min(lo), npi_max=max(hi), gamma=gamma,
                band_ok=bool(min(lo) > np.exp(-2 * gamma) and max(hi) < np.exp(2 * gamma)))


def _mean(runs, k):
    return float(np.mean([r[k] for r in runs]))


def _verdicts_v3(res, mu_grid=MU_GRID_V3):
    full, po, ao = res["full"], res["prior_only"], res["acoustic_only"]
    star = [min(mu_grid, key=lambda mu: (full[str(mu)][s]["val_loss"], mu)) for s in range(len(po))]
    at_star = [full[str(mu)][s] for s, mu in enumerate(star)]
    f1 = dict(mu_star=star, loss_mu_star=_mean(at_star, "loss"), loss_prior_only=_mean(po, "loss"),
              loss_acoustic_only=_mean(ao, "loss"))
    f1["pass"] = bool(f1["loss_mu_star"] < f1["loss_prior_only"] and f1["loss_mu_star"] < f1["loss_acoustic_only"])
    f2 = dict(nbr_mass_mu_star=_mean(at_star, "nbr_mass_dis"), nbr_mass_prior_only=_mean(po, "nbr_mass_dis"))
    f2["pass"] = bool(f2["nbr_mass_mu_star"] > f2["nbr_mass_prior_only"])
    ga = [_mean(full[str(mu)], "ga_dist") for mu in mu_grid]
    f3 = dict(ga_dist_per_mu=ga, spearman=float(spearmanr(mu_grid, ga).statistic))
    f3["pass"] = bool(f3["spearman"] <= -0.9)
    runs = [r for rs in full.values() for r in rs] + po + ao
    f4 = dict(max_abs_delta=max(r["max_abs_delta"] for r in runs), bound=INV_TAU_A,
              delta_diag_max=max(r["delta_diag_max"] for r in runs),
              band_ok_runs=sum(r["band_ok"] for r in runs), n_runs=len(runs))
    f4["pass"] = bool(f4["max_abs_delta"] < INV_TAU_A and f4["delta_diag_max"] == 0 and f4["band_ok_runs"] == len(runs))
    ok = f1["pass"] and f2["pass"] and f3["pass"] and f4["pass"]
    return dict(F1=f1, F2=f2, F3=f3, F4=f4, overall="PASS" if ok else "FAIL")


def _markdown_v3(res, ver):
    def cell(runs, k):
        return f"{_mean(runs, k):.4f} +- {np.std([r[k] for r in runs], ddof=1):.4f}"

    star = [res["full"][str(mu)][s] for s, mu in enumerate(ver["F1"]["mu_star"])]
    arms = [(f"mu = {mu:g}", res["full"][str(mu)]) for mu in MU_GRID_V3] + [
        ("mu* (val-selected)", star), ("prior-only (Delta = 0)", res["prior_only"]),
        ("acoustic-only (kappa = 0)", res["acoustic_only"])]
    lines = ["# Synthetic PoC v3: favorable-state existence proof of the r12 bounded acoustic residual", "",
             "**Existence proof only.** Passing says nothing about E-DAIC performance; failing means the r12 mechanism is "
             "broken in its best case. Synthetic data; PoC settings (1/tau_a = kappa/tau_e = 5, mu grid, 5 seeds, "
             "256/64/64 bags) are not proposal values.", "", READINGS, "",
             "## Per-arm results (test bags, mean +- sd over 5 seeds)", "",
             "| arm | test loss (item MSE) | val loss | F2 nbr mass, disagreeing | ‖G^a − G‖ | max\\|Delta\\| | Pi band ok |",
             "|---|---|---|---|---|---|---|"]
    for name, runs in arms:
        lines.append(f"| {name} | {cell(runs, 'loss')} | {cell(runs, 'val_loss')} | {cell(runs, 'nbr_mass_dis')} | "
                     f"{cell(runs, 'ga_dist')} | {max(r['max_abs_delta'] for r in runs):.4f} | "
                     f"{sum(r['band_ok'] for r in runs)}/{len(runs)} |")
    f1, f2, f3, f4 = ver["F1"], ver["F2"], ver["F3"], ver["F4"]
    pf = lambda v: "PASS" if v["pass"] else "FAIL"  # noqa: E731
    lines += ["", "‖G^a − G‖ of prior-only uses the untrained W_a (not part of any criterion).", "",
              "## Verdicts (pre-registered F1-F4)", "",
              f"- F1 combination: mu* per seed {f1['mu_star']}; test loss at mu* {f1['loss_mu_star']:.4f} vs prior-only "
              f"{f1['loss_prior_only']:.4f} and acoustic-only {f1['loss_acoustic_only']:.4f} (needs below both). **{pf(f1)}**",
              f"- F2 correction: neighbour mass on disagreeing segments at mu* {f2['nbr_mass_mu_star']:.4f} vs prior-only "
              f"{f2['nbr_mass_prior_only']:.4f} (needs higher). **{pf(f2)}**",
              f"- F3 alignment: ‖G^a − G‖ per mu {[round(x, 4) for x in f3['ga_dist_per_mu']]}, Spearman "
              f"{f3['spearman']:.3f} (needs <= -0.9). **{pf(f3)}**",
              f"- F4 bounded: max|Delta| {f4['max_abs_delta']:.6f} < {f4['bound']:g}, max|diag Delta| "
              f"{f4['delta_diag_max']:g}, Pi band holds in {f4['band_ok_runs']}/{f4['n_runs']} runs. **{pf(f4)}**",
              "", f"**Overall: {ver['overall']}**", "",
              "Pre-registered reading: PASS -> experiments protocol; FAIL -> revise the mechanism before any real run.", ""]
    return "\n".join(lines)


def run_v3(out_dir=OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "poc_graph_penalty_v3.md"
    md.write_text("# Synthetic PoC v3 (running)\n\n" + READINGS + "\n", encoding="utf-8")  # readings on disk before any result
    res = dict(full={str(mu): [] for mu in MU_GRID_V3}, prior_only=[], acoustic_only=[])

    def fit(tr, va, te, seed, mu, **kw):
        model = train_v3(tr, mu, seed=seed, **kw)
        return dict(val_loss=evaluate_v3(model, va, **kw)["loss"], **evaluate_v3(model, te, **kw))

    for seed in SEEDS:
        tr, va, te = (make_data_v3(seed, n, split=k) for k, n in enumerate((N_TRAIN, N_VAL, N_TEST)))
        for mu in MU_GRID_V3:
            res["full"][str(mu)].append(fit(tr, va, te, seed, mu))
        res["prior_only"].append(fit(tr, va, te, seed, 0.0, prior_only=True))
        res["acoustic_only"].append(fit(tr, va, te, seed, 0.0, kappa_tau=0.0))
        print("seed", seed, flush=True)
    ver = _verdicts_v3(res)
    (out_dir / "poc_graph_penalty_v3.json").write_text(json.dumps(dict(
        settings=dict(inv_tau_a=INV_TAU_A, kappa_tau=KAPPA_TAU, eps=1e-8, mu_grid=MU_GRID_V3, seeds=list(SEEDS),
                      n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, s_cluster=S_CLUSTER, beta=BETA, disagree=DISAGREE,
                      noise_sd=1.0, signal_dims=D_SIG, dx=DX, dh=DH, epochs=40, lr=1e-2, batch=8,
                      note="PoC settings on synthetic data, not proposal values", readings=READINGS),
        results=res, verdicts=ver), indent=2), encoding="utf-8")
    md.write_text(_markdown_v3(res, ver), encoding="utf-8")
    return ver


if __name__ == "__main__":
    print(run_v3()["overall"])
