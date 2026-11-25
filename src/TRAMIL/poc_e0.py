"""PoC E0: text-distilled acoustic prior on synthetic bags (experiments v05, exp:E0; proposal r22, H8).

Pre-registered in odd/tasks/tramil-poc-e0.md. Replaces the r12 proof of concept (v3), whose generator leaks item
identity into audio. Model = acoustic surrogate Ec. 114-116, graph Ec. 72 with G^, readout Ec. 18, loss Ec. 77, on a
toy encoder. Settings below are PoC settings on synthetic data, not proposal values.

Generator (E0 variant). Per seed: p (8x16) orthonormal rows supported on the 12 nuisance dims 4..15, w a unit vector
on the 4 signal dims. Per bag (rng depends on seed and split only, never on rho): N ~ U{20..60}, z_i uniform on 8
items, severities s_b in {0..3}^8 uniform, noise N(0, I_16);  X_i = rho p[z_i] + A_SEV (s_b[z_i] - 1.5) w + noise.
R = clip(0.6 + 0.03 onehot(z) + N(0, 0.005^2), 0, 1). Item identity reaches the audio only through rho p[z]; there is no
prompt/audio disagreement. Label y_b = 1[sum s_b >= 10].
"""
import copy
import json
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from torch import nn

from TRAMIL.poc_graph_penalty import center, prior_graph
from TRAMIL.poc_graph_penalty_v3 import _init_worker, constant_mse, offdiag

DX, DH, J, D_SIG = 16, 32, 8, 4
A_SEV = 0.5
INV_TAU_A = KAPPA_TAU = 5.0
TAU = 0.10
LAM0 = 0.1
EPS = 1e-8
N_TRAIN, N_VAL, N_TEST = 256, 64, 64
RHOS = (0.5, 0.0)
SEEDS_E0 = range(10)
GATE_SEEDS = range(5)
MU_R_GRID = (0.1, 1.0, 10.0)
LR_GRID = (1e-3, 3e-3, 1e-2)
WO_GRID = ("zero", "normal")
MAX_EPOCHS, BATCH = 400, 8
ALPHA, BOOT_B = 0.05, 10000
DEADLINE_MARGIN_S = 600
OUT = Path(__file__).resolve().parents[2] / "TRAMIL" / "Results"

FULL_CFG = dict(smoke=False, seeds=list(SEEDS_E0), gate_seeds=list(GATE_SEEDS),
                gate_grid=[(lr, wo) for lr in LR_GRID for wo in WO_GRID], n=(N_TRAIN, N_VAL, N_TEST), epochs=MAX_EPOCHS)
SMOKE_CFG = dict(smoke=True, seeds=[0, 1], gate_seeds=[0], gate_grid=[(1e-2, "zero")], n=(32, 16, 16), epochs=3)

READINGS = """Readings of the pre-registered E0 spec, fixed before any E0 run:
- Generator: see the module docstring. Paired across strengths: bags at rho = 0.5 and rho = 0 share every draw.
- Surrogate (Ec. 114-115): Rhat = sigmoid(W_r H + b_r), Rhat_c = Rhat - column mean over the bag; Ghat = cosine of the
  centered rows of Rhat (prior_graph, eps 1e-8). Graph (Ec. 72): Delta = (1/tau_a) prior_graph(W_a H) * offdiag,
  Pi = rowsoftmax(Delta + (kappa/tau_e) E), E = Ghat (surrogate) or G (text arm), H~ = H + tanh(W_g Pi H); 1/tau_a =
  kappa/tau_e = 5. Prior "none" is the graph-free Gate-0 model (H~ = H, no prior anywhere).
- Readout (Ec. 18): L = W_o H~ + (lam/tau) P with P = Rhat_c (surrogate) or Rc (text), tau = 0.10, lam learnable scalar
  initialised at 0.1; A = colsoftmax(L); s_j = head_j . (A[:, j]^T H~) + c_j. W_o starts at zero or N(0, sd 0.1).
  The surrogate forward never reads R, Rc or G.
- Loss (Ec. 77): omega_b mean_j (s^ - s)^2 + mu_R ||Rhat_c - Rc||_F^2 / (8 N_b), batch mean, omega_plus = M0/M1 from the
  TRAIN bags, omega_b = omega_plus if y_b else 1. mu_R only for the surrogate arms.
- Model selection: best epoch on val unweighted item MSE without the penalty (ties -> earliest), weights restored.
  Up to 400 epochs, Adam, batch 8, with the Gate-0 chosen lr and W_o init for every arm.
- Gate 0: prior none, rho = 0.5, lr x W_o-init grid, 5 seeds, val only (test bags never generated). Chosen = lowest
  seed-mean best val MSE; PASS iff below the seed-mean constant val MSE (unweighted train item mean). FAIL -> INVALID, no
  arm trained.
- Arms per (rho, seed): full = surrogate with mu_R* chosen on val over {0.1, 1, 10} (ties -> smaller); untargeted =
  surrogate mu_R = 0; r12 = text arm; constant = omega-weighted train item mean. Test metrics of a run are computed
  when it ends but no choice ever reads them; each run is cached and resumable.
- SS^w (Ec. 93) on test with omega and sbar_w from train. Surrogate error = mean over test bags of
  ||Rhat_c - Rc||_F^2 / (8 N). AUROC of sum_j s^_j against y. Per-item Spearman of Rhat_c vs Rc pooled over the test
  segments of one seed.
- Tests (seed = unit, n = 10, alpha 0.05): P1 one-sample Wilcoxon (greater) of full SS^w at rho 0.5 against 0; P2a paired
  Wilcoxon full > untargeted SS^w; P2b paired Wilcoxon full < untargeted surrogate error; Holm over the three; effect
  size = rank-biserial (positive = hypothesis direction); CI = percentile bootstrap (B = 10000, seeded) of the seed-mean
  difference (P1: of the mean). P4: bootstrap CI of full - untargeted SS^w at rho 0, supported iff it contains zero. P3:
  estimate and CI of full - r12 SS^w at rho 0.5, no threshold and no verdict.
- Verdicts: P1 supported iff adjusted p < alpha; P2 supported iff both P2a and P2b adjusted p < alpha, partial iff one,
  else not supported; inconclusive if gate 0 fails or a unit is missing. H8 carried forward iff P1, P2, P4 supported.
- smoke mode (rehearsal only): 2 seeds, 32/16/16 bags, 3 epochs, one gate seed and config; it runs the arms even when
  the gate verdict is FAIL so the whole pipeline is exercised, and the report says so."""


# ---------------- data ----------------
def e0_params(seed):
    rng = np.random.default_rng([seed, 5, 1000])  # seed-only stream, disjoint from the split streams
    Q, _ = np.linalg.qr(rng.standard_normal((DX - D_SIG, J)))
    p = np.zeros((J, DX))
    p[:, D_SIG:] = Q.T
    w = np.zeros(DX)
    w[:D_SIG] = rng.standard_normal(D_SIG)
    return p, w / np.linalg.norm(w)


def make_data_e0(seed, n_bags, split, rho):
    p, w = e0_params(seed)
    rng = np.random.default_rng([seed, 5, split])  # no rho in the stream: paired across strengths
    bags = []
    for _ in range(n_bags):
        n = int(rng.integers(20, 61))
        z = rng.integers(0, J, n)
        s = rng.integers(0, 4, J)
        noise = rng.standard_normal((n, DX))
        rnoise = rng.standard_normal((n, J))
        X = rho * p[z] + A_SEV * (s[z] - 1.5)[:, None] * w + noise
        R = torch.tensor(np.clip(0.6 + 0.03 * np.eye(J)[z] + 0.005 * rnoise, 0, 1), dtype=torch.float32)
        bags.append(dict(X=torch.tensor(X, dtype=torch.float32), R=R, Rc=center(R), G=prior_graph(R),
                         s=torch.tensor(s), y=torch.tensor(int(s.sum() >= 10)), z=torch.tensor(z)))
    return bags


def omega_plus(train_bags):
    m1 = sum(int(b["y"]) for b in train_bags)
    return (len(train_bags) - m1) / m1


def bag_weights(bags, ow):
    return [ow if int(b["y"]) else 1.0 for b in bags]


def weighted_train_mean(train_bags, ow):
    w = torch.tensor(bag_weights(train_bags, ow), dtype=torch.float64)
    S = torch.stack([b["s"] for b in train_bags]).double()
    return (w[:, None] * S).sum(0) / w.sum()


# ---------------- model ----------------
class ModelE0(nn.Module):
    def __init__(self, prior="surrogate", wo_init="zero"):
        super().__init__()
        assert prior in ("surrogate", "text", "none")
        self.prior = prior
        self.enc = nn.Linear(DX, DH)
        self.W_r = nn.Linear(DH, J)
        self.W_a, self.W_g = nn.Linear(DH, DH, bias=False), nn.Linear(DH, DH, bias=False)
        self.W_o = nn.Linear(DH, J, bias=False)
        nn.init.zeros_(self.W_o.weight)
        if wo_init == "normal":
            nn.init.normal_(self.W_o.weight, 0.0, 0.1)
        self.head = nn.Linear(DH, J)
        self.lam = nn.Parameter(torch.tensor(LAM0))

    def _surr(self, H):
        R = torch.sigmoid(self.W_r(H))
        return R, center(R), prior_graph(R, EPS)

    def surrogate(self, X):
        return self._surr(torch.relu(self.enc(X)))

    def forward(self, b):
        H = torch.relu(self.enc(b["X"]))
        Ht, P, Rc_hat = H, None, None
        if self.prior != "none":
            if self.prior == "surrogate":
                _, Rc_hat, E = self._surr(H)
                P = Rc_hat
            else:
                E, P = b["G"], b["Rc"]
            Delta = INV_TAU_A * prior_graph(self.W_a(H), EPS) * offdiag(len(H), H.dtype)
            Pi = torch.softmax(Delta + KAPPA_TAU * E, dim=1)
            Ht = H + torch.tanh(self.W_g(Pi @ H))
        L = self.W_o(Ht)
        if P is not None:
            L = L + (self.lam / TAU) * P
        A = torch.softmax(L, dim=0)
        return (self.head.weight * (A.T @ Ht)).sum(1) + self.head.bias, Rc_hat


def penalty(Rc_hat, Rc):
    return (Rc_hat - Rc).pow(2).sum() / (J * len(Rc))


@torch.no_grad()
def predict(model, bags):
    out = [model(b) for b in bags]
    return [o[0].double() for o in out], [o[1] for o in out]


def val_mse(model, bags):
    preds, _ = predict(model, bags)
    return float(np.mean([(p - b["s"]).pow(2).mean().item() for p, b in zip(preds, bags)]))


def train_e0(bags, prior, mu_R=0.0, seed=0, epochs=MAX_EPOCHS, lr=1e-2, batch=BATCH, wo_init="zero", val=None,
             omega_plus=1.0):
    """Adam on Ec. 77; with `val`, restores the best epoch (ties -> earliest) and sets model.history."""
    torch.manual_seed(seed)
    model = ModelE0(prior, wo_init)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_pen = prior == "surrogate" and mu_R > 0
    hist, best, best_v = [], None, np.inf
    for _ in range(epochs):
        order = torch.randperm(len(bags)).tolist()
        for st in range(0, len(order), batch):
            loss = 0.0
            for i in order[st:st + batch]:
                b = bags[i]
                s_hat, rc = model(b)
                loss = loss + (omega_plus if int(b["y"]) else 1.0) * (s_hat - b["s"]).pow(2).mean()
                if use_pen:
                    loss = loss + mu_R * penalty(rc, b["Rc"])
            opt.zero_grad()
            (loss / len(order[st:st + batch])).backward()
            opt.step()
        if val is not None:
            hist.append(val_mse(model, val))
            if hist[-1] < best_v:
                best_v, best = hist[-1], copy.deepcopy(model.state_dict())
    if val is not None:
        model.load_state_dict(best)
        model.history = dict(val=hist, best_epoch=int(np.argmin(hist)) + 1)
    return model


# ---------------- metrics ----------------
def ss_w(preds, bags, sbar, weights):
    num = sum(w * (p - b["s"]).pow(2).sum().item() for p, b, w in zip(preds, bags, weights))
    den = sum(w * (sbar - b["s"].double()).pow(2).sum().item() for b, w in zip(bags, weights))
    return 1.0 - num / den


def test_metrics(model, train_bags, test_bags):
    ow = omega_plus(train_bags)
    sbar = weighted_train_mean(train_bags, ow)
    preds, rcs = predict(model, test_bags)
    y = [int(b["y"]) for b in test_bags]
    out = dict(ss_w=ss_w(preds, test_bags, sbar, bag_weights(test_bags, ow)),
               auroc=float(roc_auc_score(y, [p.sum().item() for p in preds])), surr_err=float("nan"),
               spearman=[float("nan")] * J)
    if rcs[0] is not None:
        out["surr_err"] = float(np.mean([penalty(r, b["Rc"]).item() for r, b in zip(rcs, test_bags)]))
        P, T = torch.cat(rcs).numpy(), torch.cat([b["Rc"] for b in test_bags]).numpy()
        out["spearman"] = [float(spearmanr(P[:, j], T[:, j]).statistic) for j in range(J)]
    return out


# ---------------- statistics ----------------
def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def bootstrap_ci(d, B=BOOT_B, seed=0, level=0.95):
    d = np.asarray(d, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def wilcoxon_greater(d):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 0.0, 1.0
    r = wilcoxon(d, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def _row(test, d, ci_d, n):
    st, p = wilcoxon_greater(d)
    row = dict(test=test, statistic=st, p_raw=p, correction="Holm", effect_size=rank_biserial(d),
               ci=list(bootstrap_ci(ci_d)), mean_diff=float(np.mean(ci_d)), n=n, alpha=ALPHA)
    return row


def compare(arms):
    """arms: {"full|rho=0.5": {"ss_w": [...], "surr_err": [...]}, ...}. Returns the comparison rows and verdicts."""
    f5, u5, r5 = arms["full|rho=0.5"], arms["untargeted|rho=0.5"], arms["r12|rho=0.5"]
    f0, u0 = arms["full|rho=0"], arms["untargeted|rho=0"]
    a = lambda k, v: np.asarray(v[k], dtype=float)  # noqa: E731
    n = len(a("ss_w", f5))
    rows = dict(
        P1=_row("one-sample Wilcoxon, full SS^w > 0", a("ss_w", f5), a("ss_w", f5), n),
        P2a=_row("paired Wilcoxon, full > untargeted SS^w", a("ss_w", f5) - a("ss_w", u5),
                 a("ss_w", f5) - a("ss_w", u5), n),
        # direction of the hypothesis: untargeted error - full error > 0; the CI is of the plain difference full - untargeted
        P2b=_row("paired Wilcoxon, full < untargeted surrogate error", a("surr_err", u5) - a("surr_err", f5),
                 a("surr_err", f5) - a("surr_err", u5), n))
    adj = holm([rows[k]["p_raw"] for k in ("P1", "P2a", "P2b")])
    for k, v in zip(("P1", "P2a", "P2b"), adj):
        rows[k]["p_adj"] = float(v)
        rows[k]["pass"] = bool(v < ALPHA)
    d4 = a("ss_w", f0) - a("ss_w", u0)
    ci4 = bootstrap_ci(d4)
    rows["P4"] = dict(test="bootstrap CI of paired full - untargeted SS^w at rho 0", statistic=None, p_raw=None, p_adj=None,
                      correction="none", effect_size=None, ci=list(ci4), mean_diff=float(d4.mean()), n=n, alpha=ALPHA,
                      pass_=bool(ci4[0] <= 0 <= ci4[1]))
    d3 = a("ss_w", f5) - a("ss_w", r5)
    rows["P3"] = dict(test="estimate, full - r12 SS^w at rho 0.5 (no threshold)", statistic=None, p_raw=None, p_adj=None,
                      correction="none", effect_size=None, ci=list(bootstrap_ci(d3)), mean_diff=float(d3.mean()), n=n,
                      alpha=ALPHA)
    p2 = rows["P2a"]["pass"] + rows["P2b"]["pass"]
    ver = dict(P1="supported" if rows["P1"]["pass"] else "not supported",
               P2="supported" if p2 == 2 else "partially supported" if p2 == 1 else "not supported",
               P4="supported" if rows["P4"]["pass_"] else "not supported", P3="estimate only")
    ver["H8_carried_forward"] = bool(ver["P1"] == ver["P2"] == ver["P4"] == "supported")
    return rows, ver


# ---------------- units, cache, parallel ----------------
def _uid(u):
    if u["kind"] == "gate":
        return f"gate__lr={u['lr']:g}__wo={u['wo']}__s={u['seed']}"
    mu = f"__mu={u['mu']:g}" if u["arm"] in ("full",) else ""
    return f"arm__{u['arm']}__rho={u['rho']:g}__s={u['seed']}{mu}"


def _run_unit(u, runs_dir):
    path = Path(runs_dir) / f"{u['id']}.json"
    if path.exists():
        return
    n_tr, n_va, n_te = u["n"]
    seed, ep = u["seed"], u["epochs"]
    if u["kind"] == "gate":
        tr, va = make_data_e0(seed, n_tr, 0, 0.5), make_data_e0(seed, n_va, 1, 0.5)
        m = train_e0(tr, "none", seed=seed, epochs=ep, lr=u["lr"], wo_init=u["wo"], val=va, omega_plus=omega_plus(tr))
        res = dict(val=min(m.history["val"]), best_epoch=m.history["best_epoch"], curve=m.history["val"],
                   const_val=constant_mse(tr, va))
    else:
        tr, va, te = (make_data_e0(seed, n, k, u["rho"]) for k, n in enumerate(u["n"]))
        if u["arm"] == "constant":
            ow = omega_plus(tr)
            sbar = weighted_train_mean(tr, ow)
            preds = [sbar] * len(te)
            res = dict(ss_w=ss_w(preds, te, sbar, bag_weights(te, ow)),
                       auroc=float(roc_auc_score([int(b["y"]) for b in te], [0.0] * len(te))),
                       surr_err=float("nan"), spearman=[float("nan")] * J, val=constant_mse(tr, va), curve=[])
        else:
            prior = "text" if u["arm"] == "r12" else "surrogate"
            m = train_e0(tr, prior, mu_R=u["mu"], seed=seed, epochs=ep, lr=u["lr"], wo_init=u["wo"], val=va,
                         omega_plus=omega_plus(tr))
            res = dict(val=min(m.history["val"]), best_epoch=m.history["best_epoch"], curve=m.history["val"],
                       **test_metrics(m, tr, te))
    path.write_text(json.dumps(_clean(dict(res, id=u["id"])), indent=1), encoding="utf-8")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o.item() if isinstance(o, np.generic) else o


def _load(runs_dir, u):
    r = json.loads((Path(runs_dir) / f"{u['id']}.json").read_text(encoding="utf-8"))
    for k in ("surr_err",):
        if r.get(k, 0) is None:
            r[k] = float("nan")
    if "spearman" in r:
        r["spearman"] = [float("nan") if v is None else v for v in r["spearman"]]
    return r


def _low_time():
    d = os.environ.get("FORGE_SESSION_DEADLINE")
    return d is not None and float(d) - time.time() < DEADLINE_MARGIN_S


def _run_units(units, runs_dir, workers, fn=None):
    """Runs the uncached units; returns the ids still owed when the session deadline is near."""
    fn = fn or _run_unit
    runs_dir.mkdir(parents=True, exist_ok=True)
    pending = deque(u for u in units if not (runs_dir / f"{u['id']}.json").exists())
    if not pending:
        return []
    workers = workers or min(14, os.cpu_count() or 1, len(pending))
    if workers <= 1:
        while pending and not _low_time():
            fn(pending[0], runs_dir)
            pending.popleft()
        return [u["id"] for u in pending]
    inflight = set()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        while True:
            while pending and len(inflight) < workers and not _low_time():
                inflight.add(ex.submit(fn, pending.popleft(), runs_dir))
            if not inflight:
                break
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                f.result()
    return [u["id"] for u in pending]


def _partial(remaining):
    Path("partial.json").write_text(json.dumps(dict(remaining=remaining)), encoding="utf-8")
    return dict(partial=True, remaining=remaining)


def _gate_units(cfg):
    return [dict(kind="gate", lr=lr, wo=wo, seed=s, n=cfg["n"], epochs=cfg["epochs"], id=None)
            for lr, wo in cfg["gate_grid"] for s in cfg["gate_seeds"]]


def _arm_units(cfg, lr, wo):
    base = dict(kind="arm", n=cfg["n"], epochs=cfg["epochs"], lr=lr, wo=wo)
    units = []
    for rho in RHOS:
        for s in cfg["seeds"]:
            units += [dict(base, arm="full", mu=mu, rho=rho, seed=s) for mu in MU_R_GRID]
            units += [dict(base, arm=a, mu=0.0, rho=rho, seed=s) for a in ("untargeted", "r12", "constant")]
    return units


def _with_ids(units):
    for u in units:
        u["id"] = _uid(u)
    return units


def _cfg_key(lr, wo):
    return f"lr={lr:g}, W_o={wo}"


def _gate_summary(cfg, runs_dir):
    units = _with_ids(_gate_units(cfg))
    configs = {}
    for lr, wo in cfg["gate_grid"]:
        rs = [_load(runs_dir, u) for u in units if u["lr"] == lr and u["wo"] == wo]
        configs[_cfg_key(lr, wo)] = dict(lr=lr, wo_init=wo, val=float(np.mean([r["val"] for r in rs])),
                                         val_sd=float(np.std([r["val"] for r in rs], ddof=1)) if len(rs) > 1 else None,
                                         best_epoch=[r["best_epoch"] for r in rs],
                                         const_val=float(np.mean([r["const_val"] for r in rs])))
    chosen = min(configs, key=lambda k: configs[k]["val"])
    c = configs[chosen]
    return dict(configs=configs, chosen=chosen, lr=c["lr"], wo_init=c["wo_init"], val=c["val"], const_val=c["const_val"],
                verdict="PASS" if c["val"] < c["const_val"] else "FAIL")


def _aggregate(cfg, runs_dir, lr, wo):
    arms, curves = {}, {}
    for rho in RHOS:
        per = {a: [] for a in ("full", "untargeted", "r12", "constant")}
        for s in cfg["seeds"]:
            def get(arm, mu=0.0):
                return _load(runs_dir, _with_ids([dict(kind="arm", arm=arm, mu=mu, rho=rho, seed=s)])[0])
            cands = [(mu, get("full", mu)) for mu in MU_R_GRID]
            mu_star, best = min(cands, key=lambda c: (c[1]["val"], c[0]))
            per["full"].append(dict(best, mu_star=mu_star))
            for a in ("untargeted", "r12", "constant"):
                per[a].append(get(a))
        for a, rs in per.items():
            key = f"{a}|rho={rho:g}"
            arms[key] = {k: [r[k] for r in rs] for k in ("ss_w", "surr_err", "auroc", "spearman", "val")}
            if a == "full":
                arms[key]["mu_star"] = [r["mu_star"] for r in rs]
            if a != "constant":
                arms[key]["best_epoch"] = [r["best_epoch"] for r in rs]
                C = np.array([r["curve"] for r in rs])
                curves[key] = dict(mean=C.mean(0).tolist(), sd=(C.std(0, ddof=1) if len(C) > 1 else 0 * C[0]).tolist())
    return arms, curves


def _fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _markdown(js):
    g = js["gate0"]
    lines = ["# PoC E0: text-distilled acoustic prior on synthetic bags", "",
             "**Existence proof only; synthetic data; PoC settings, not proposal values.**"
             + (" **SMOKE rehearsal, not a result.**" if js["smoke"] else ""), "", READINGS, "",
             "## Gate 0 (graph-free model, rho 0.5, val only)", "",
             "| Gate 0 configuration | Seed-mean validation MSE | Constant validation MSE | Kept |", "|---|---|---|---|"]
    for k, c in g["configs"].items():
        lines.append(f"| {k} | {c['val']:.4f} | {c['const_val']:.4f} | {'yes' if k == g['chosen'] else ''} |")
    lines += ["", f"**Gate 0: {g['verdict']}** (chosen {g['chosen']}: {g['val']:.4f} vs constant {g['const_val']:.4f})", ""]
    if "arms" not in js:
        return "\n".join(lines + ["**Overall: INVALID** (gate 0 failed; no arm trained, no criterion read)", ""])
    lines += ["## Arms (test bags, seed mean +- sd)", "",
              "| Arm at strength | SS omega mean | SS omega sd | Surrogate error mean | AUROC mean |", "|---|---|---|---|---|"]
    names = dict(full="Full", untargeted="Untargeted", r12="r12 reference", constant="Constant")
    for rho in RHOS:
        for a in names:
            r = js["arms"][f"{a}|rho={rho:g}"]
            ss = np.array(r["ss_w"], dtype=float)
            se = np.array(r["surr_err"], dtype=float)
            lines.append(f"| {names[a]}, rho {rho:g} | {ss.mean():.4f} | {ss.std(ddof=1) if len(ss) > 1 else float('nan'):.4f} | "
                         f"{_fmt(None if np.all(np.isnan(se)) else float(se.mean()))} | {np.mean(r['auroc']):.4f} |")
    lines += ["", "## Comparisons (seed is the unit; CI = percentile bootstrap of the seed-mean difference)", "",
              "| Comparison | Test | Statistic | Raw p | Adjusted p | Correction | Effect size | CI | Mean diff | n | alpha |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in js["comparisons"].items():
        lines.append(f"| {k} | {c['test']} | {_fmt(c['statistic'], 2)} | {_fmt(c['p_raw'])} | {_fmt(c['p_adj'])} | "
                     f"{c['correction']} | {_fmt(c['effect_size'], 3)} | [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] | "
                     f"{c['mean_diff']:.4f} | {c['n']} | {c['alpha']} |")
    v = js["verdicts"]
    lines += ["", "## Verdicts", ""] + [f"- {k}: {x}" for k, x in v.items()] + [""]
    return "\n".join(lines)


def run_gate0(out_dir=OUT, cfg=None, workers=None):
    cfg = cfg or FULL_CFG
    out_dir = Path(out_dir)
    rem = _run_units(_with_ids(_gate_units(cfg)), out_dir / "e0_runs", workers)
    if rem:
        return _partial(rem)
    g = _gate_summary(cfg, out_dir / "e0_runs")
    (out_dir / "poc_e0_gate.json").write_text(json.dumps(_clean(g), indent=2), encoding="utf-8")
    return g


def run_e0(out_dir=OUT, smoke=False, workers=None):
    cfg = SMOKE_CFG if smoke else FULL_CFG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = out_dir / "e0_runs"
    gate = run_gate0(out_dir, cfg, workers)
    if gate.get("partial"):
        return gate
    js = dict(smoke=smoke, readings=READINGS, settings=dict(
        rhos=list(RHOS), seeds=cfg["seeds"], n=list(cfg["n"]), epochs=cfg["epochs"], mu_grid=list(MU_R_GRID),
        inv_tau_a=INV_TAU_A, kappa_tau=KAPPA_TAU, tau=TAU, alpha=ALPHA, bootstrap_B=BOOT_B), gate0=gate)
    if gate["verdict"] == "FAIL" and not smoke:
        js["verdicts"] = dict(overall="INVALID")
    else:
        rem = _run_units(_with_ids(_arm_units(cfg, gate["lr"], gate["wo_init"])), runs, workers)
        if rem:
            return _partial(rem)
        js["arms"], js["curves"] = _aggregate(cfg, runs, gate["lr"], gate["wo_init"])
        js["comparisons"], js["verdicts"] = compare(js["arms"])
        js["verdicts"]["gate0"] = gate["verdict"]
    (out_dir / "poc_e0.json").write_text(json.dumps(_clean(js), indent=2), encoding="utf-8")
    (out_dir / "poc_e0.md").write_text(_markdown(js), encoding="utf-8")
    from TRAMIL.e1_run import _stamp  # shard.json for the remote smoke verdict; no-op outside the runner
    _stamp(dict(smoke=smoke, verdicts=_clean(js["verdicts"])))
    return js


# ---------------- timing ----------------
def timing_probe(out_dir=OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    tr, va = make_data_e0(0, N_TRAIN, 0, 0.5), make_data_e0(0, N_VAL, 1, 0.5)
    train_e0(tr[:8], "none", epochs=1)  # warm-up: first-call overhead must not land on the first arm
    sec = {}
    for prior in ("none", "surrogate", "text"):
        t0 = time.perf_counter()
        train_e0(tr, prior, mu_R=1.0, seed=0, epochs=1, lr=1e-2, val=va, omega_plus=omega_plus(tr))
        sec[prior] = time.perf_counter() - t0
    runs = dict(none=30, surrogate=2 * len(SEEDS_E0) * (len(MU_R_GRID) + 1), text=2 * len(SEEDS_E0))
    workers = min(14, os.cpu_count() or 1)
    serial = lambda ep: float(sum(runs[k] * sec[k] * ep for k in runs))  # noqa: E731
    out = dict(seconds_per_epoch=sec, device="cpu (1 torch thread per worker)", torch=torch.__version__, runs=runs,
               workers_assumed=workers, epochs_upper=MAX_EPOCHS, epochs_lower=100,
               note="lower bound assumes 100 epochs per run (stand-in for best-epoch-style early stopping; the runner "
                    "always trains the full budget); parallel figures assume linear speedup over workers",
               total_serial_s=dict(upper=serial(MAX_EPOCHS), lower=serial(100)),
               total_parallel_s=dict(upper=serial(MAX_EPOCHS) / workers, lower=serial(100) / workers),
               gate_parallel_s=runs["none"] * sec["none"] * MAX_EPOCHS / min(workers, runs["none"]))
    (out_dir / "timing_e0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    if "--timing" in sys.argv:
        print(json.dumps(timing_probe(), indent=2))
    elif "--gate" in sys.argv:
        print(json.dumps(run_gate0(), indent=2))
    else:
        r = run_e0(smoke="--smoke" in sys.argv)
        print(r.get("verdicts", r))
