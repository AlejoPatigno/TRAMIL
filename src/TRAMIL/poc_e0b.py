"""PoC E0b: redesigned E0 (experiments v06, exp:E0; proposal r22, H8). Pre-registered in odd/tasks/tramil-poc-e0b.md.

Why a redesign: in E0 the surrogate arms were restored at epoch 1-2 (lr 0.01 chosen on the graph-free model), and the
penalty was negligible because Rc has variance ~1.2e-4, so E0 never tested prior recovery (per-item Spearman ~0).
Generator and model are E0's unchanged (poc_e0.make_data_e0, poc_e0.ModelE0). Changes are listed in READINGS.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from TRAMIL import poc_e0 as e0
from TRAMIL.poc_e0 import (ALPHA, J, ModelE0, _clean, _fmt, _low_time, _partial, bag_weights, bootstrap_ci, holm,  # noqa: F401
                           make_data_e0, omega_plus, rank_biserial, ss_w, val_mse, weighted_train_mean, wilcoxon_greater)
from TRAMIL.poc_graph_penalty_v3 import constant_mse

RHOS = (0.0, 0.5, 1.0, 2.0)
PRIMARY = 1.0
LR_GRID = (1e-3, 3e-3, 1e-2)
WO_GRID = ("zero", "normal")
MU_R_GRID = (0.1, 1.0, 10.0)
MAX_EPOCHS, PATIENCE, BATCH = 400, 40, 8
OUT = e0.OUT

FULL_CFG = dict(smoke=False, seeds=list(range(10)), gate_seeds=list(range(5)), lr_grid=list(LR_GRID),
                gate_grid=[(lr, wo) for lr in LR_GRID for wo in WO_GRID], mu_grid=list(MU_R_GRID),
                n=(256, 64, 64), epochs=MAX_EPOCHS, patience=PATIENCE)
SMOKE_CFG = dict(smoke=True, seeds=[0, 1], gate_seeds=[0], lr_grid=[1e-2], gate_grid=[(1e-2, "zero")],
                 mu_grid=[1.0, 10.0], n=(32, 16, 16), epochs=3, patience=2)

READINGS = """Readings of the pre-registered E0b spec, fixed before any E0b run (changes from E0 marked CHANGED):
- Generator and model: E0's, unchanged (poc_e0 docstring and READINGS). Bags are paired across strengths.
- CHANGED strengths: rho in {0, 0.5, 1, 2}; primary rho = 1 (P1, P2, P3); P4 at rho = 0; all strengths reported.
- CHANGED penalty (Ec. 77): mu_R ||Rhat_c - Rc||_F^2 / (8 N_b vbar), vbar = mean of Rc^2 over every TRAIN segment and
  item (does not depend on rho), so predicting Rhat_c = 0 scores about 1. Surrogate error on test is normalized the same
  way (1 = the zero predictor).
- CHANGED training: up to 400 epochs, early stop after 40 epochs without a val improvement; best val epoch restored
  (val unweighted item MSE without the penalty, ties -> earliest).
- CHANGED selection: lr is chosen on val per (arm, rho, seed) from {1e-3, 3e-3, 1e-2}, jointly with mu_R in
  {0.1, 1, 10} for the full arm (ties -> smaller lr, then smaller mu_R). W_o init comes from gate 0.
- Gate 0: graph-free model at rho = 1, lr x W_o grid, 5 seeds, val only; keeps W_o init; PASS iff the best seed-mean
  val MSE is below the constant's. FAIL -> INVALID.
- Arms: full (surrogate, mu_R > 0), untargeted (surrogate, mu_R = 0), r12 (text at inference), constant.
- Metrics: SS^w (Ec. 93); normalized surrogate error; AUROC; per-item Spearman of Rhat_c vs Rc pooled over the test
  segments of a seed, and its mean over the 8 items (spearman_mean).
- CHANGED tests (seed = unit, n = 10, alpha 0.05, Holm over P1, P2a, P2b, P2c): P1 one-sample Wilcoxon full SS^w > 0;
  P2a paired Wilcoxon full > untargeted SS^w; P2b full < untargeted surrogate error; P2c full > untargeted
  spearman_mean (prior recovery); rank-biserial effect size; percentile bootstrap CI (B = 10000). P4: bootstrap CI of
  full - untargeted SS^w at rho = 0 contains zero. P3: estimate of full - r12 SS^w at rho = 1. Dose response: full
  spearman_mean by rho, descriptive.
- Verdicts: P1 supported iff adjusted p < alpha; P2 supported iff P2a, P2b and P2c pass, partially supported iff one or
  two pass; H8 carried forward iff P1, P2, P4 supported.
- smoke mode (rehearsal only): 2 seeds, 32/16/16 bags, 3 epochs, reduced grids; runs the arms even if gate 0 fails."""


def rc_scale(bags):
    return float(torch.cat([b["Rc"] for b in bags]).pow(2).mean())


def npenalty(Rc_hat, Rc, vbar):
    return (Rc_hat - Rc).pow(2).sum() / (J * len(Rc) * vbar)


def train_e0b(bags, prior, mu_R=0.0, seed=0, epochs=MAX_EPOCHS, lr=1e-2, wo_init="zero", val=None, omega_plus=1.0,
              patience=PATIENCE, vbar=1.0):
    torch.manual_seed(seed)
    model = ModelE0(prior, wo_init)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_pen = prior == "surrogate" and mu_R > 0
    hist, best, best_v = [], None, np.inf
    for _ in range(epochs):
        order = torch.randperm(len(bags)).tolist()
        for st in range(0, len(order), BATCH):
            loss = 0.0
            for i in order[st:st + BATCH]:
                b = bags[i]
                s_hat, rc = model(b)
                loss = loss + (omega_plus if int(b["y"]) else 1.0) * (s_hat - b["s"]).pow(2).mean()
                if use_pen:
                    loss = loss + mu_R * npenalty(rc, b["Rc"], vbar)
            opt.zero_grad()
            (loss / len(order[st:st + BATCH])).backward()
            opt.step()
        if val is not None:
            hist.append(val_mse(model, val))
            if hist[-1] < best_v:
                best_v, best = hist[-1], copy.deepcopy(model.state_dict())
            elif len(hist) - 1 - int(np.argmin(hist)) >= patience:
                break
    if val is not None:
        model.load_state_dict(best)
        model.history = dict(val=hist, best_epoch=int(np.argmin(hist)) + 1)
    return model


def select(cands):
    return min(cands, key=lambda c: (c["val"], c["lr"], c.get("mu", 0.0)))


def test_metrics(model, tr, te):
    ow = omega_plus(tr)
    sbar = weighted_train_mean(tr, ow)
    preds, rcs = e0.predict(model, te)
    out = dict(ss_w=ss_w(preds, te, sbar, bag_weights(te, ow)),
               auroc=float(roc_auc_score([int(b["y"]) for b in te], [p.sum().item() for p in preds])),
               surr_err=float("nan"), spearman=[float("nan")] * J, spearman_mean=float("nan"))
    if rcs[0] is not None:
        vbar = rc_scale(tr)
        out["surr_err"] = float(np.mean([npenalty(r, b["Rc"], vbar).item() for r, b in zip(rcs, te)]))
        P, T = torch.cat(rcs).numpy(), torch.cat([b["Rc"] for b in te]).numpy()
        out["spearman"] = [float(spearmanr(P[:, j], T[:, j]).statistic) for j in range(J)]
        out["spearman_mean"] = float(np.nanmean(out["spearman"])) if np.any(np.isfinite(out["spearman"])) else 0.0
    return out


# ---------------- units ----------------
def _uid(u):
    if u["kind"] == "gate":
        return f"b_gate__lr={u['lr']:g}__wo={u['wo']}__s={u['seed']}"
    lr = "" if u["arm"] == "constant" else f"__lr={u['lr']:g}"
    mu = f"__mu={u['mu']:g}" if u["arm"] == "full" else ""
    return f"b_arm__{u['arm']}__rho={u['rho']:g}__s={u['seed']}{lr}{mu}"


def _ids(units):
    for u in units:
        u["id"] = _uid(u)
    return units


def _run_unit(u, runs_dir):
    path = Path(runs_dir) / f"{u['id']}.json"
    if path.exists():
        return
    seed, ep, pat = u["seed"], u["epochs"], u["patience"]
    if u["kind"] == "gate":
        tr, va = make_data_e0(seed, u["n"][0], 0, PRIMARY), make_data_e0(seed, u["n"][1], 1, PRIMARY)
        m = train_e0b(tr, "none", seed=seed, epochs=ep, lr=u["lr"], wo_init=u["wo"], val=va, omega_plus=omega_plus(tr),
                      patience=pat)
        res = dict(val=min(m.history["val"]), best_epoch=m.history["best_epoch"], curve=m.history["val"],
                   const_val=constant_mse(tr, va))
    else:
        tr, va, te = (make_data_e0(seed, n, k, u["rho"]) for k, n in enumerate(u["n"]))
        if u["arm"] == "constant":
            ow = omega_plus(tr)
            sbar = weighted_train_mean(tr, ow)
            res = dict(ss_w=ss_w([sbar] * len(te), te, sbar, bag_weights(te, ow)), auroc=0.5, surr_err=float("nan"),
                       spearman=[float("nan")] * J, spearman_mean=float("nan"), val=constant_mse(tr, va), curve=[])
        else:
            prior = "text" if u["arm"] == "r12" else "surrogate"
            m = train_e0b(tr, prior, mu_R=u["mu"], seed=seed, epochs=ep, lr=u["lr"], wo_init=u["wo"], val=va,
                          omega_plus=omega_plus(tr), patience=pat, vbar=rc_scale(tr))
            res = dict(val=min(m.history["val"]), best_epoch=m.history["best_epoch"], curve=m.history["val"],
                       **test_metrics(m, tr, te))
    path.write_text(json.dumps(_clean(dict(res, id=u["id"])), indent=1), encoding="utf-8")


def _load(runs_dir, u):
    r = json.loads((Path(runs_dir) / f"{u['id']}.json").read_text(encoding="utf-8"))
    nan = lambda v: float("nan") if v is None else v  # noqa: E731
    for k in ("surr_err", "spearman_mean"):
        r[k] = nan(r.get(k))
    r["spearman"] = [nan(v) for v in r.get("spearman", [])]
    return r


def _gate_units(cfg):
    return _ids([dict(kind="gate", lr=lr, wo=wo, seed=s, n=cfg["n"], epochs=cfg["epochs"], patience=cfg["patience"])
                 for lr, wo in cfg["gate_grid"] for s in cfg["gate_seeds"]])


def _arm_units(cfg, wo):
    base = dict(kind="arm", n=cfg["n"], epochs=cfg["epochs"], patience=cfg["patience"], wo=wo)
    units = []
    for rho in RHOS:
        for s in cfg["seeds"]:
            for lr in cfg["lr_grid"]:
                units += [dict(base, arm="full", lr=lr, mu=mu, rho=rho, seed=s) for mu in cfg["mu_grid"]]
                units += [dict(base, arm=a, lr=lr, mu=0.0, rho=rho, seed=s) for a in ("untargeted", "r12")]
            units.append(dict(base, arm="constant", lr=0.0, mu=0.0, rho=rho, seed=s))
    return _ids(units)


def _gate_summary(cfg, runs):
    units, configs = _gate_units(cfg), {}
    for lr, wo in cfg["gate_grid"]:
        rs = [_load(runs, u) for u in units if u["lr"] == lr and u["wo"] == wo]
        configs[f"lr={lr:g}, W_o={wo}"] = dict(lr=lr, wo_init=wo, val=float(np.mean([r["val"] for r in rs])),
                                               best_epoch=[r["best_epoch"] for r in rs],
                                               const_val=float(np.mean([r["const_val"] for r in rs])))
    chosen = min(configs, key=lambda k: configs[k]["val"])
    c = configs[chosen]
    return dict(configs=configs, chosen=chosen, wo_init=c["wo_init"], val=c["val"], const_val=c["const_val"],
                verdict="PASS" if c["val"] < c["const_val"] else "FAIL")


def _aggregate(cfg, runs, units):
    by_id = {u["id"]: u for u in units}
    arms = {}
    for rho in RHOS:
        per = {a: [] for a in ("full", "untargeted", "r12", "constant")}
        for s in cfg["seeds"]:
            for a in per:
                cands = [dict(_load(runs, u), lr=u["lr"], mu=u["mu"]) for u in by_id.values()
                         if u["arm"] == a and u["rho"] == rho and u["seed"] == s]
                per[a].append(select(cands))
        for a, rs in per.items():
            key = f"{a}|rho={rho:g}"
            arms[key] = {k: [r[k] for r in rs] for k in ("ss_w", "surr_err", "auroc", "spearman", "spearman_mean", "val")}
            if a != "constant":
                arms[key].update(best_epoch=[r["best_epoch"] for r in rs], lr=[r["lr"] for r in rs])
            if a == "full":
                arms[key]["mu_star"] = [r["mu"] for r in rs]
    return arms


def _row(test, d, ci_d, n):
    st, p = wilcoxon_greater(d)
    return dict(test=test, statistic=st, p_raw=p, correction="Holm", effect_size=rank_biserial(d),
                ci=list(bootstrap_ci(ci_d)), mean_diff=float(np.mean(ci_d)), n=n, alpha=ALPHA)


def compare(arms):
    k = f"rho={PRIMARY:g}"
    f, u, r = arms[f"full|{k}"], arms[f"untargeted|{k}"], arms[f"r12|{k}"]
    f0, u0 = arms["full|rho=0"], arms["untargeted|rho=0"]
    a = lambda key, v: np.asarray(v[key], dtype=float)  # noqa: E731
    n = len(a("ss_w", f))
    dss, dse, dsp = a("ss_w", f) - a("ss_w", u), a("surr_err", f) - a("surr_err", u), a("spearman_mean", f) - a("spearman_mean", u)
    rows = dict(P1=_row("one-sample Wilcoxon, full SS^w > 0", a("ss_w", f), a("ss_w", f), n),
                P2a=_row("paired Wilcoxon, full > untargeted SS^w", dss, dss, n),
                P2b=_row("paired Wilcoxon, full < untargeted normalized surrogate error", -dse, dse, n),
                P2c=_row("paired Wilcoxon, full > untargeted mean per-item Spearman", dsp, dsp, n))
    fam = ("P1", "P2a", "P2b", "P2c")
    for key, v in zip(fam, holm([rows[x]["p_raw"] for x in fam])):
        rows[key].update(p_adj=float(v), **{"pass": bool(v < ALPHA)})
    d4 = a("ss_w", f0) - a("ss_w", u0)
    ci4 = bootstrap_ci(d4)
    none = dict(statistic=None, p_raw=None, p_adj=None, correction="none", effect_size=None, n=n, alpha=ALPHA)
    rows["P4"] = dict(none, test="bootstrap CI of paired full - untargeted SS^w at rho 0", ci=list(ci4),
                      mean_diff=float(d4.mean()), pass_=bool(ci4[0] <= 0 <= ci4[1]))
    d3 = a("ss_w", f) - a("ss_w", r)
    rows["P3"] = dict(none, test=f"estimate, full - r12 SS^w at rho {PRIMARY:g} (no threshold)",
                      ci=list(bootstrap_ci(d3)), mean_diff=float(d3.mean()))
    p2 = sum(rows[x]["pass"] for x in ("P2a", "P2b", "P2c"))
    ver = dict(P1="supported" if rows["P1"]["pass"] else "not supported",
               P2="supported" if p2 == 3 else "partially supported" if p2 else "not supported",
               P4="supported" if rows["P4"]["pass_"] else "not supported", P3="estimate only")
    ver["H8_carried_forward"] = bool(ver["P1"] == ver["P2"] == ver["P4"] == "supported")
    return rows, ver


def _markdown(js):
    g = js["gate0"]
    L = ["# PoC E0b: redesigned text-distilled acoustic prior on synthetic bags", "",
         "**Existence proof only; synthetic data.**" + (" **SMOKE rehearsal, not a result.**" if js["smoke"] else ""),
         "", READINGS, "", f"## Gate 0 (graph-free, rho {PRIMARY:g}, val only)", "",
         "| Configuration | Seed-mean val MSE | Constant val MSE | Best epochs | Kept |", "|---|---|---|---|---|"]
    for k, c in g["configs"].items():
        L.append(f"| {k} | {c['val']:.4f} | {c['const_val']:.4f} | {c['best_epoch']} | {'yes' if k == g['chosen'] else ''} |")
    L += ["", f"**Gate 0: {g['verdict']}**", ""]
    if "arms" not in js:
        return "\n".join(L + ["**Overall: INVALID**", ""])
    L += ["## Arms (test bags, seed mean)", "",
          "| Arm, rho | SS^w mean | SS^w sd | Norm. surrogate error | Mean per-item Spearman | AUROC | Median best epoch |",
          "|---|---|---|---|---|---|---|"]
    for rho in RHOS:
        for a in ("full", "untargeted", "r12", "constant"):
            r = js["arms"][f"{a}|rho={rho:g}"]
            ss = np.array(r["ss_w"], dtype=float)
            se, sp = np.array(r["surr_err"], dtype=float), np.array(r["spearman_mean"], dtype=float)
            be = r.get("best_epoch")
            L.append(f"| {a}, {rho:g} | {ss.mean():.4f} | {ss.std(ddof=1) if len(ss) > 1 else float('nan'):.4f} | "
                     f"{_fmt(None if np.all(np.isnan(se)) else float(np.nanmean(se)))} | "
                     f"{_fmt(None if np.all(np.isnan(sp)) else float(np.nanmean(sp)), 3)} | {np.mean(r['auroc']):.4f} | "
                     f"{'-' if not be else int(np.median(be))} |")
    L += ["", "## Comparisons (seed = unit)", "",
          "| Comparison | Test | Statistic | Raw p | Adjusted p | Correction | Effect size | CI | Mean diff | n | alpha |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in js["comparisons"].items():
        L.append(f"| {k} | {c['test']} | {_fmt(c['statistic'], 2)} | {_fmt(c['p_raw'])} | {_fmt(c['p_adj'])} | "
                 f"{c['correction']} | {_fmt(c['effect_size'], 3)} | [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] | "
                 f"{c['mean_diff']:.4f} | {c['n']} | {c['alpha']} |")
    L += ["", "## Verdicts", ""] + [f"- {k}: {x}" for k, x in js["verdicts"].items()] + [""]
    return "\n".join(L)


def run_e0b(out_dir=OUT, smoke=False, workers=None):
    cfg = SMOKE_CFG if smoke else FULL_CFG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = out_dir / "e0b_runs"
    rem = e0._run_units(_gate_units(cfg), runs, workers, fn=_run_unit)
    if rem:
        return _partial(rem)
    gate = _gate_summary(cfg, runs)
    js = dict(smoke=smoke, readings=READINGS, gate0=gate, settings=dict(
        rhos=list(RHOS), primary=PRIMARY, seeds=cfg["seeds"], n=list(cfg["n"]), epochs=cfg["epochs"],
        patience=cfg["patience"], lr_grid=cfg["lr_grid"], mu_grid=cfg["mu_grid"]))
    if gate["verdict"] == "FAIL" and not smoke:
        js["verdicts"] = dict(overall="INVALID")
    else:
        units = _arm_units(cfg, gate["wo_init"])
        rem = e0._run_units(units, runs, workers, fn=_run_unit)
        if rem:
            return _partial(rem)
        js["arms"] = _aggregate(cfg, runs, units)
        js["comparisons"], js["verdicts"] = compare(js["arms"])
        js["verdicts"]["gate0"] = gate["verdict"]
    (out_dir / "poc_e0b.json").write_text(json.dumps(_clean(js), indent=2), encoding="utf-8")
    (out_dir / "poc_e0b.md").write_text(_markdown(js), encoding="utf-8")
    from TRAMIL.e1_run import _stamp  # shard.json for the remote smoke verdict; no-op outside the runner
    _stamp(dict(smoke=smoke, verdicts=_clean(js["verdicts"])))
    return js


if __name__ == "__main__":
    import sys
    r = run_e0b(smoke="--smoke" in sys.argv)
    print(r.get("verdicts", r))
