"""Synthetic PoC: does mu*||Delta_b||_F^2/N_b^2 pull the learned segment graph Pi_b toward the prompt prior?

Model, as in r11 Section 6.1 (scaled to a 16-dim toy encoder, no other r11 component):
  R^c_b = R_b - 1 mean(R_b)                         (Ec. 90)
  G_b   = cosine of centered rows, eps in denominator (Ec. 70)
  Delta_b = (H W_q)(H W_k)^T / sqrt(D)              (Ec. 71)
  Pi_b  = rowsoftmax(Delta_b + (kappa/tau_e) G_b)   (Ec. 72, 73)
  H~_b  = H_b + tanh(Pi_b H_b W_g)                  (Ec. 74)
  loss  = task + mu * mean_b ||Delta_b||_F^2 / N_b^2 (Ec. 77 penalty)

Synthetic data. Each bag has N ~ U{20..60} segments and 8 items. Text latent z_text(i) uniform on the items;
R rows = 0.6 + 0.03*onehot(z_text) + N(0, 0.005^2), clipped to [0, 1], so raw rows are near-collinear as in the
real prior (cosine ~0.99) and only centering exposes the item structure. Acoustic latent z_ac: "agree" z_ac = z_text;
"partial" exactly round(N/2) segments moved to a different item; "disagree" z_ac independent uniform.
H_i = mu_{z_ac(i)} + N(0, I), cluster means drawn once per seed with per-dim sd 0.5.

Task (chosen so neighbour aggregation helps): per-segment 8-way classification of z_ac with a linear head on H~.
Per-segment features are noisy, so averaging over same-acoustic-cluster neighbours (Pi close to Pi_A) improves
prediction, while averaging over prompt neighbours helps only as far as z_text agrees with z_ac. Training labels
carry 10% uniform label noise; held-out task loss is the cross-entropy against the clean z_ac.

PoC settings, NOT proposal values: kappa/tau_e = 5, eps = 1e-8, mu grid {0, 1e-3, ..., 1e3}, Adam lr 1e-2,
40 epochs of 64 training bags in batches of 8, 32 held-out bags, 5 seeds.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

D, J = 16, 8
KAPPA_TAU = 5.0
MU_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
SCENARIOS = ["agree", "partial", "disagree"]
SEEDS = range(5)


def center(R):
    return R - R.mean(0, keepdim=True)


def prior_graph(R, eps=1e-8):
    Rc = center(R)
    n = Rc.norm(dim=1, keepdim=True)
    return (Rc @ Rc.T) / (n * n.T + eps)


def acoustic_adjacency(z):
    A = (z[:, None] == z[None, :]).float()
    return A / A.sum(1, keepdim=True)


def row_js(P, Q):
    M = (P + Q) / 2
    js = 0.5 * torch.xlogy(P, P / M).sum(1) + 0.5 * torch.xlogy(Q, Q / M).sum(1)
    return js.mean().item()


class GraphLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q, self.W_k, self.W_g = (nn.Linear(D, D, bias=False) for _ in range(3))

    def forward(self, H, G, kappa_tau, prior_only=False):
        Delta = torch.zeros(len(H), len(H)) if prior_only else self.W_q(H) @ self.W_k(H).T / D ** 0.5
        Pi = torch.softmax(Delta + kappa_tau * G, dim=1)
        return H + torch.tanh(self.W_g(Pi @ H)), Pi, Delta


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph, self.head = GraphLayer(), nn.Linear(D, J)

    def forward(self, H, G, kappa_tau, prior_only=False):
        Ht, Pi, Delta = self.graph(H, G, kappa_tau, prior_only)
        return self.head(Ht), Pi, Delta


def make_data(seed, scenario, n_bags, split=0, label_noise=0.1):
    rng = np.random.default_rng(seed)
    means = 0.5 * rng.standard_normal((J, D))
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario), split])
    bags = []
    for _ in range(n_bags):
        n = int(rng.integers(20, 61))
        zt = rng.integers(0, J, n)
        if scenario == "agree":
            za = zt.copy()
        elif scenario == "partial":
            za = zt.copy()
            idx = rng.choice(n, round(n / 2), replace=False)
            za[idx] = (zt[idx] + rng.integers(1, J, len(idx))) % J
        else:
            za = rng.integers(0, J, n)
        R = np.clip(0.6 + 0.03 * np.eye(J)[zt] + 0.005 * rng.standard_normal((n, J)), 0, 1)
        H = means[za] + rng.standard_normal((n, D))
        flip = rng.random(n) < label_noise
        y = np.where(flip, (za + rng.integers(1, J, n)) % J, za)
        R = torch.tensor(R, dtype=torch.float32)
        bags.append(dict(R=R, G=prior_graph(R), H=torch.tensor(H, dtype=torch.float32),
                         z_text=torch.tensor(zt), z_ac=torch.tensor(za), y=torch.tensor(y)))
    return bags


def train(bags, mu, kappa_tau=KAPPA_TAU, seed=0, epochs=40, prior_only=False, lr=1e-2, batch=8):
    torch.manual_seed(seed)
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(bags)).tolist()
        for s in range(0, len(order), batch):
            loss = 0.0
            for i in order[s:s + batch]:
                b = bags[i]
                logits, _, Delta = model(b["H"], b["G"], kappa_tau, prior_only)
                n = len(b["y"])
                loss = loss + nn.functional.cross_entropy(logits, b["y"]) + mu * Delta.pow(2).sum() / n ** 2
            opt.zero_grad()
            (loss / len(order[s:s + batch])).backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, bags, kappa_tau=KAPPA_TAU, prior_only=False):
    out = dict(js_G=[], js_A=[], loss=[], delta_rms=[])
    for b in bags:
        logits, Pi, Delta = model(b["H"], b["G"], kappa_tau, prior_only)
        out["js_G"].append(row_js(Pi, torch.softmax(kappa_tau * b["G"], dim=1)))
        out["js_A"].append(row_js(Pi, acoustic_adjacency(b["z_ac"])))
        out["loss"].append(nn.functional.cross_entropy(logits, b["z_ac"]).item())
        out["delta_rms"].append((Delta.norm() / len(b["y"])).item())
    return {k: float(np.mean(v)) for k, v in out.items()}


def _verdicts(res):
    def m(sc, arm, k):
        return float(np.mean([r[k] for r in res[sc][arm]]))

    arms = [str(mu) for mu in MU_GRID]
    rhos = [spearmanr(MU_GRID, [res["disagree"][a][s]["js_G"] for a in arms]).statistic for s in SEEDS]
    big = arms[-1]
    p1 = dict(spearman_per_seed=[float(r) for r in rhos], spearman_mean=float(np.mean(rhos)),
              js_G_at_max_mu=m("disagree", big, "js_G"), js_A_at_max_mu=m("disagree", big, "js_A"))
    p1["pass"] = bool(p1["spearman_mean"] <= -0.9 and p1["js_G_at_max_mu"] < p1["js_A_at_max_mu"])
    p2 = dict(js_A_mu0=m("disagree", "0.0", "js_A"), js_G_mu0=m("disagree", "0.0", "js_G"),
              loss_mu0=m("disagree", "0.0", "loss"), loss_prior_only=m("disagree", "prior_only", "loss"))
    p2["pass"] = bool(p2["js_A_mu0"] < p2["js_G_mu0"] and p2["loss_mu0"] < p2["loss_prior_only"])
    base = m("agree", "prior_only", "loss")
    p3 = dict(loss_prior_only=base, bound=1.05 * base, loss_per_mu={a: m("agree", a, "loss") for a in arms})
    p3["pass"] = bool(all(v <= p3["bound"] for v in p3["loss_per_mu"].values()))
    return dict(P1=p1, P2=p2, P3=p3, overall="PASS" if p1["pass"] and p2["pass"] and p3["pass"] else "FAIL")


def _markdown(res, ver):
    lines = ["# Synthetic PoC: Delta penalty pulling the segment graph toward the prompt prior", "",
             f"PoC settings (not proposal values): kappa/tau_e = {KAPPA_TAU}, seeds = {len(SEEDS)}, "
             "64 train / 32 held-out bags, N in 20..60, D = 16, 8 items. Metrics on held-out bags, mean +- sd over seeds.",
             "JS = mean row Jensen-Shannon (nats). delta_rms = ||Delta||_F / N. Task loss = CE vs clean z_ac.", ""]
    for sc in SCENARIOS:
        lines += [f"## {sc}", "", "| mu | JS(Pi, Pi_G) | JS(Pi, Pi_A) | task loss | delta_rms |", "|---|---|---|---|---|"]
        for arm, runs in res[sc].items():
            cells = [f"{np.mean([r[k] for r in runs]):.4f} +- {np.std([r[k] for r in runs], ddof=1):.4f}"
                     for k in ("js_G", "js_A", "loss", "delta_rms")]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    p1, p2, p3 = ver["P1"], ver["P2"], ver["P3"]
    lines += ["## Verdicts (pre-registered criteria)", "",
              f"- P1 pull: Spearman(mu, JS(Pi,Pi_G)) per seed {[round(r, 3) for r in p1['spearman_per_seed']]}, "
              f"mean {p1['spearman_mean']:.3f} (needs <= -0.9); at mu = {MU_GRID[-1]:g} JS to G {p1['js_G_at_max_mu']:.4f} "
              f"vs JS to A {p1['js_A_at_max_mu']:.4f} (needs G < A). **{'PASS' if p1['pass'] else 'FAIL'}**",
              f"- P2 acoustic-centered: at mu = 0 JS to A {p2['js_A_mu0']:.4f} vs JS to G {p2['js_G_mu0']:.4f} (needs A < G); "
              f"loss {p2['loss_mu0']:.4f} vs prior-only {p2['loss_prior_only']:.4f} (needs lower). **{'PASS' if p2['pass'] else 'FAIL'}**",
              f"- P3 no harm (agree): max loss over mu {max(p3['loss_per_mu'].values()):.4f} vs bound "
              f"1.05 x prior-only = {p3['bound']:.4f}. **{'PASS' if p3['pass'] else 'FAIL'}**",
              "", f"**Overall: {ver['overall']}**", ""]
    return "\n".join(lines)


def run(out_dir=Path(__file__).resolve().parents[2] / "TRAMIL" / "Results"):
    res = {}
    for sc in SCENARIOS:
        res[sc] = {str(mu): [] for mu in MU_GRID} | {"prior_only": []}
        for seed in SEEDS:
            tr, te = make_data(seed, sc, 64), make_data(seed, sc, 32, split=1)
            for mu in MU_GRID:
                res[sc][str(mu)].append(evaluate(train(tr, mu, seed=seed), te))
            res[sc]["prior_only"].append(evaluate(train(tr, 0.0, seed=seed, prior_only=True), te, prior_only=True))
            print(sc, seed, flush=True)
    ver = _verdicts(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "poc_graph_penalty.json").write_text(json.dumps(dict(
        settings=dict(kappa_tau=KAPPA_TAU, mu_grid=MU_GRID, seeds=list(SEEDS), n_train=64, n_test=32, epochs=40, lr=1e-2,
                      note="PoC settings on synthetic data, not proposal values"),
        results=res, verdicts=ver), indent=2))
    (out_dir / "poc_graph_penalty.md").write_text(_markdown(res, ver))
    return ver


if __name__ == "__main__":
    print(run()["overall"])
