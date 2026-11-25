"""E1 of experiments v07 (with the E2/E5 arms that share its protocol): search -> real -> perm -> report.

Every phase enumerates independent unit specs (units), runs each through the pure run_unit(spec, data) and writes one
result file per unit under <out>/<phase>/<uid>.json; an existing file is skipped (resume). select() restricts the
units to an explicit id list or to a shard "i/n" (1-based, units[i-1::n]). dispatch.run maps units over N workers.

- search: Ec. 102 units of hpo (rep, fold, arm, k, p, inner fold), seed 11; each completed outer training set is
  reduced and frozen to <out>/hpo/hpo_r{rep}_f{fold}.json.
- real: (rep, fold, arm, seed in {11, 23, 37}) with the frozen k*, p*_m (prompt_only at k = 8, full_k768 = full model
  at k = 768 with p768): inner epoch selection (Ec. 39) on the three inner folds, refit on the outer training set for
  e* epochs, out-of-fold item predictions by subject. Bag SVM (mean + sd, binary class from the item sum): one unit
  per (rep, fold), kernel and C chosen on inner AUROC and saved.
- perm (Ec. 91 reduced protocol): permutation r in 0..R_pi-1, labels permuted across subjects with the eight item
  scores as a unit (donor permutation seed PERM_SEED + r over the subject order), repetition-1 outer partition of the
  true labels, seed 11, k*, p* frozen from hpo_r0, inner epoch selection kept; the SVM reuses the real run's kernel
  and C of the same outer fold. Unit = (perm, fold, arm).
- report: pooled statistics (pooled_stats) and the four tables of v07 'Expected report E1' -> E1_v07.json and .md.

CLI: python -m TRAMIL.e1v07_run {search,real,perm,report} [--out DIR] [--input ROOT] [--shard i/n]
     [--units id,id] [--workers N] [--limit-bags N] [--max-epochs N] [--r-pi N]."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score

from . import comparators, dispatch, e1_arms, frontend, hpo, item_training, pooled_stats, statistics, validation

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v07.md",
    "sections": ["E1", "E2", "E5"],
    "experiments": ["E1", "E2", "E5"],
    "invariants": [
        "phases_run_end_to_end_and_the_report_fills_the_expected_e1_tables",
        "real_units_cover_every_arm_seed_and_fold_with_frozen_hyperparameters",
        "phases_are_resumable_and_skip_existing_result_files",
        "shards_partition_the_units_and_an_explicit_list_restricts_them",
        "permutation_units_use_repetition_one_seed_11_and_move_item_scores_as_a_unit",
        "a_real_unit_never_reads_its_outer_test_subjects_for_fitting",
        "kernel_output_wavlm_cache_is_discovered_below_the_input_root",
        "sharded_search_kernels_feed_a_real_kernel_through_mounted_outputs",
    ],
}

PHASES = ("search", "real", "perm", "report")
SEEDS = validation.INIT_SEEDS  # 11, 23, 37
REPS, FOLDS = 3, 5
R_PI = pooled_stats.R_PI
PERM_SEED = 2026  # donor permutation of replicate r uses seed PERM_SEED + r
E1_NEURAL = hpo.KSTAR_ARMS  # full + five neural comparators
COMPARATORS = ("abmil8", "gated_abmil8", "symptom_guided", "r06_8", "abmil_r8", "svm")
PERM_MODELS = (*E1_NEURAL, "prompt_only", "svm")
LABELS = {"full": "Full model", "abmil8": "ABMIL-8", "gated_abmil8": "Gated ABMIL-8", "symptom_guided": "Symptom-Guided",
          "r06_8": "r06 residual bipartite with item heads", "abmil_r8": "ABMIL+R-8",
          "svm": "Bag-level SVM on WavLM features", "full_k768": "Full model at k = 768 (Ec. 103)",
          "prompt_only": "Prompt-only reference (E5)", "permuted": "Permuted target (E2)",
          "untargeted": "Untargeted (E2, exploratory)", "r12": "r12 reference (E2, exploratory)"}
DEFAULTS = dict(reps=REPS, seeds=SEEDS, K=hpo.K_GRID, P=hpo.P_GRID, max_epochs=item_training.MAX_EPOCHS, r_pi=R_PI,
                arms=tuple(e1_arms.ARMS), kstar_arms=hpo.KSTAR_ARMS, B=pooled_stats.B)


# ---------------- units ----------------
def uid(s):
    if s["phase"] == "search":
        return f"r{s['rep']}_f{s['fold']}_{s['arm']}_k{s['k']}_p{s['p']:g}_i{s['inner']}"
    if s["phase"] == "real":
        return f"r{s['rep']}_f{s['fold']}_{s['arm']}_s{s['seed']}"
    return f"p{s['perm']}_f{s['fold']}_{s['arm']}"


def _cfg(cfg):
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}


def units(phase, out, **cfg):
    c, out = _cfg(cfg), Path(out)
    neural = [a for a in c["arms"] if a in e1_arms.ARMS]
    specs = []
    if phase == "search":
        for rep in range(c["reps"]):
            for fold in range(FOLDS):
                specs += [{**s, "phase": "search", "model": s["arm"]}
                          for s in hpo.units(rep, fold, neural, c["K"], c["P"], c["max_epochs"])]
    elif phase == "real":
        for rep in range(c["reps"]):
            for fold in range(FOLDS):
                fr = hpo.load_frozen(out / "hpo", rep, fold)
                base = {"phase": "real", "rep": rep, "fold": fold, "max_epochs": c["max_epochs"]}
                for arm in neural:
                    k = hpo.K_PROMPT if e1_arms.SOURCE.get(arm) == "R" else fr["k_star"]
                    specs += [{**base, "arm": arm, "model": arm, "k": k, "p": fr["p_star"][arm], "seed": s}
                              for s in c["seeds"]]
                if "full" in neural and fr["p768"] is not None:
                    specs += [{**base, "arm": "full_k768", "model": "full", "k": hpo.K_REF, "p": fr["p768"], "seed": s}
                              for s in c["seeds"]]
                specs.append({**base, "arm": "svm", "model": "svm", "seed": 0})
    elif phase == "perm":
        for fold in range(FOLDS):
            fr = hpo.load_frozen(out / "hpo", 0, fold)
            svm = json.loads((out / "real" / f"r0_f{fold}_svm_s0.json").read_text())
            for r in range(c["r_pi"]):
                base = {"phase": "perm", "perm": r, "rep": 0, "fold": fold, "max_epochs": c["max_epochs"]}
                for arm in [a for a in neural if a in PERM_MODELS]:
                    k = hpo.K_PROMPT if e1_arms.SOURCE.get(arm) == "R" else fr["k_star"]
                    specs.append({**base, "arm": arm, "model": arm, "k": k, "p": fr["p_star"][arm], "seed": hpo.SEED})
                specs.append({**base, "arm": "svm", "model": "svm", "seed": 0, "grid": svm["grid"]})
    return specs


def select(specs, shard=None, only=None):
    if only is not None:
        keep = set(only)
        specs = [s for s in specs if uid(s) in keep]
    if shard:
        i, n = map(int, str(shard).split("/"))
        specs = specs[i - 1 :: n]
    return specs


# ---------------- unit execution ----------------
def permuted_items(items, r):
    """Labels permuted across subjects as a unit: subject b receives the item scores (hence class) of donor perm[b]."""
    S = item_training.permute_items(np.stack([it["s"] for it in items]), PERM_SEED + r)
    return [dict(it, s=np.asarray(s, np.float32), y=int(s.sum() >= item_training.THRESHOLD)) for it, s in zip(items, S)]


def _move(bags, dev):
    return [{k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()} for b in bags]


def _bags(data, key, items, tr, ev, k, source):
    """Prepared fold bags on the training device, cached per worker."""
    cache = data.setdefault("cache", {})
    if key not in cache:
        tb, eb, info = frontend.fold_bags(items, tr, ev, k=k, source=source)
        dev = item_training.device()
        cache[key] = (_move(tb, dev), _move(eb, dev), info["vbar"])
    return cache[key]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _items_for(spec, data):
    if spec["phase"] != "perm":
        return data["items"]
    perms = data.setdefault("perm_items", {})
    if spec["perm"] not in perms:
        perms[spec["perm"]] = permuted_items(data["items"], spec["perm"])
    return perms[spec["perm"]]


def _split(spec, data):
    """Outer test indices from the true labels (the partition of repetition rep), training indices the rest."""
    y_true = [it["y"] for it in data["items"]]
    test = validation.outer_splits(y_true, spec["rep"])[spec["fold"]]
    return np.setdiff1d(np.arange(len(y_true)), test), test


def _train_inner(spec, data, items, train, k, source, pkey):
    y = [it["y"] for it in items]
    tr, va = validation.inner_splits(y, train, spec["fold"])[spec["inner"]]
    tb, vb, vbar = _bags(data, (pkey, spec["rep"], spec["fold"], spec["inner"], k, source), items, tr, va, k, source)
    make = e1_arms.build(spec["model"], k=k, p_drop=spec["p"], vbar=vbar, Q=data.get("Q"))
    return item_training.train(make, tb, spec["max_epochs"], spec["seed"], eval_bags=vb)[1]


def run_unit(spec, data):
    """One unit; data = {"items", "Q", "cache"}. Returns the spec plus its result and compute seconds."""
    _sync()
    t0 = time.perf_counter()
    items = _items_for(spec, data)
    train, test = _split(spec, data)
    pkey = spec.get("perm", -1)
    out = dict(spec)
    if spec["model"] == "svm":
        X = [it["X"] for it in items]
        y = np.array([it["y"] for it in items])
        pipe = comparators.fit_svm([X[i] for i in train], y[train], validation.INNER_SEED + spec["fold"], spec.get("grid"))
        sc = pipe.decision_function(np.stack([comparators.bag_stats(X[i]) for i in test]))
        svc = pipe.named_steps["svc"]
        out.update(oof={str(items[i]["subject"]): float(v) for i, v in zip(test, sc)},
                   grid={"svc__kernel": [svc.kernel], "svc__C": [float(svc.C)]})
    else:
        source = e1_arms.SOURCE.get(spec["model"], "X")
        k = min(spec["k"], items[0][source].shape[1])
        if spec["phase"] == "search":
            out["curve"] = [float(v) for v in _train_inner(spec, data, items, train, k, source, pkey)]
        else:
            curves = [_train_inner({**spec, "inner": r}, data, items, train, k, source, pkey)
                      for r in range(hpo.INNER_FOLDS)]
            e = item_training.select_epoch(curves)
            tb, eb, vbar = _bags(data, (pkey, spec["rep"], spec["fold"], "outer", k, source), items, train, test, k,
                                 source)
            make = e1_arms.build(spec["model"], k=k, p_drop=spec["p"], vbar=vbar, Q=data.get("Q"))
            m, _ = item_training.train(make, tb, e, spec["seed"])
            P = item_training.predict(m, eb)
            out.update(epoch=e, curves=[[float(v) for v in c] for c in curves],
                       oof={str(items[i]["subject"]): [float(v) for v in row] for i, row in zip(test, P)})
    _sync()
    out["seconds"] = time.perf_counter() - t0
    return out


# ---------------- phases ----------------
def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _load_dir(d):
    return [json.loads(f.read_text()) for f in sorted(Path(d).glob("*.json"))] if Path(d).exists() else []


def _freeze_complete(out, c):
    have = {}
    for r in _load_dir(out / "search"):
        have.setdefault((r["rep"], r["fold"]), []).append(r)
    neural = [a for a in c["arms"] if a in e1_arms.ARMS]
    for (rep, fold), res in sorted(have.items()):
        need = len(hpo.units(rep, fold, neural, c["K"], c["P"], c["max_epochs"]))
        target = out / "hpo" / f"hpo_r{rep}_f{fold}.json"
        if len(res) == need and not target.exists():
            grid = {"K": list(c["K"]), "P": list(c["P"]), "k_ref": hpo.K_REF, "k_prompt": hpo.K_PROMPT,
                    "arms": neural, "max_epochs": c["max_epochs"], "inner_folds": hpo.INNER_FOLDS}
            hpo.freeze(res, rep, fold, out / "hpo", grid, [a for a in c["kstar_arms"] if a in neural])


def run_phase(phase, data, out, workers=1, shard=None, only=None, **cfg):
    c, out = _cfg(cfg), Path(out)
    if phase == "report":
        rep = report(data, out, **c)
        return {"ran": ["report"], "report": rep}
    _freeze_complete(out, c)  # search units imported from other shards' kernels
    specs = select(units(phase, out, **c), shard, only)
    pending = [s for s in specs if not (out / phase / f"{uid(s)}.json").exists()]
    dispatch.run(run_unit, pending, data, workers, on_result=lambda s, r: _write(out / phase / f"{uid(s)}.json", r))
    _freeze_complete(out, c)
    return {"ran": [uid(s) for s in pending], "skipped": len(specs) - len(pending)}


# ---------------- report ----------------
def _score_matrix(results, subjects, reps):
    """{arm: (Q, M) seed-ensemble score} and {arm: (Q, M, 8) item predictions} (items arms only)."""
    acc = {}
    for r in results:
        acc.setdefault(r["arm"], {}).setdefault(r["rep"], {}).setdefault(r["seed"], {}).update(r["oof"])
    scores, profiles = {}, {}
    for arm, byrep in acc.items():
        if sorted(byrep) != list(range(reps)):
            continue
        rows = []
        for q in range(reps):
            seeds = [byrep[q][s] for s in sorted(byrep[q])]
            if any(len(s) != len(subjects) for s in seeds):
                break
            rows.append(np.mean([[s[str(b)] for b in subjects] for s in seeds], axis=0))
        if len(rows) != reps:
            continue
        A = np.asarray(rows, float)
        if A.ndim == 3:
            profiles[arm], scores[arm] = A, A.sum(-1)
        else:
            scores[arm] = A
    return scores, profiles


def _nulls(perm_results, data, subjects):
    acc = {}
    for r in perm_results:
        acc.setdefault((r["arm"], r["perm"]), {}).update(r["oof"])
    out = {}
    for (arm, perm), oof in sorted(acc.items()):
        if len(oof) != len(subjects):
            continue
        yp = [it["y"] for it in permuted_items(data["items"], perm)]
        v = np.array([oof[str(b)] for b in subjects], float)
        s = v.sum(-1) if v.ndim == 2 else v
        out.setdefault(arm, []).append(pooled_stats.auroc(s, yp))
    return out


def _threshold(arm, s, y):
    pred = (s >= (0.0 if arm == "svm" else item_training.THRESHOLD)).astype(int)
    return {"ap": float(average_precision_score(y, s)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0))}


def report(data, out, **cfg):
    c, out = _cfg(cfg), Path(out)
    items = data["items"]
    subjects = [it["subject"] for it in items]
    y = np.array([it["y"] for it in items])
    S = np.stack([it["s"] for it in items]).astype(float)
    scores, profiles = _score_matrix(_load_dir(out / "real"), subjects, c["reps"])
    I = pooled_stats.bootstrap_indices(y, c["B"])
    pooled = pooled_stats.pooled(scores, y, I, ref="full")
    nulls = _nulls(_load_dir(out / "perm"), data, subjects)
    folds_q = [validation.outer_splits(y, q) for q in range(c["reps"])]

    assay = []
    for m in [m for m in (*E1_NEURAL, "svm", "prompt_only") if m in pooled and m in nulls]:
        a = pooled_stats.assay(pooled[m]["lower"], pooled[m]["A_bar"], nulls[m])
        assay.append({"model": m, "A_bar": pooled[m]["A_bar"], "lower": pooled[m]["lower"],
                      "b": int((np.asarray(nulls[m]) >= pooled[m]["A_bar"]).sum()), "nulls": nulls[m], **a})
    passes = {r["model"]: r["passes"] for r in assay}
    comps = [m for m in COMPARATORS if m in pooled]
    testable, why, m_ok = pooled_stats.testable_h1(passes.get("full", False), {m: passes.get(m, False) for m in comps})

    ss = {}
    for m, P in profiles.items():
        Sh, w, ref = pooled_stats.ss_inputs(P, S, y, folds_q)
        boot = pooled_stats.boot_ss_omega(Sh, S, w, ref, I)
        ss[m] = {"point": pooled_stats.ss_omega(Sh, S, w, ref), "lower": pooled_stats.lower(boot), "boot": boot}
    h5 = None
    if "full" in ss and "permuted" in ss:
        ell_ss = pooled_stats.lower(ss["full"]["boot"] - ss["permuted"]["boot"])
        h5 = {"lower_ss_full": ss["full"]["lower"], "ell_ss_full_minus_permuted": ell_ss,
              "supported": bool(ss["full"]["lower"] > 0 and ell_ss > 0)}
    h1 = pooled_stats.h1_verdict({m: pooled[m]["ell"] for m in comps}, m_ok, testable,
                                 h5_supported=bool(h5 and h5["supported"]))
    h1.update(testable=testable, reason=why, m_ok=m_ok, h5=h5)

    ni, praw = [], []
    for m in comps:
        o = pooled[m]
        dl = [pooled_stats.delong(scores["full"][q], scores[m][q], y) for q in range(c["reps"])]
        d_mean, v_mean = float(np.mean([d["delta"] for d in dl])), float(np.mean([d["var"] for d in dl]))
        praw.append(float((1 + (o["delta_boot"] <= -pooled_stats.DELTA_NI).sum()) / (len(o["delta_boot"]) + 1)))
        ni.append({"comparator": m, "in_M_ok": m in m_ok, "delta_bar": o["delta_bar"], "ell": o["ell"],
                   "delong_delta": d_mean, "delong_z": d_mean / np.sqrt(v_mean) if v_mean > 0 else float("nan"),
                   "rho": o["rho"], "se": o["se"], "attainable_80": pooled_stats.attainability(o["se"])["attainable_80"],
                   "clears_margin": bool(o["ell"] > -pooled_stats.DELTA_NI)})
    padj = statistics.holm(praw) if praw else []
    for row, p, pa in zip(ni, praw, padj):
        row.update(p_raw=p, p_holm=float(pa))

    desc = []
    for m in scores:
        fold_auc = [pooled_stats.auroc(scores[m][q][te], y[te]) for q in range(c["reps"]) for te in folds_q[q]]
        row = {"model": m, "fold_auroc_mean": float(np.mean(fold_auc)), "fold_auroc_sd": float(np.std(fold_auc, ddof=1)),
               **_threshold(m, scores[m].mean(0), y), "development": "not read (deferred to the end of the study)"}
        if m != "full" and "full" in scores:
            fa = [pooled_stats.auroc(scores["full"][q][te], y[te]) for q in range(c["reps"]) for te in folds_q[q]]
            d = np.array(fa) - np.array(fold_auc)
            row["fisher_ratio"] = statistics.fisher_ratio(fa, fold_auc) if d.var() > 0 else float("nan")
            row["corrected_t"] = statistics.corrected_t(d)[0] if d.var() > 0 else float("nan")
        desc.append(row)

    js = {"stage": "E1 v07", "n_subjects": len(y), "n_positive": int(y.sum()), "B": c["B"], "R_pi": c["r_pi"],
          "pooled": {m: {k: v for k, v in o.items() if not k.endswith("boot") and k != "boot"} for m, o in pooled.items()},
          "assay": assay, "h1": h1, "noninferiority": ni, "descriptive": desc,
          "ss_omega": {m: {"point": v["point"], "lower": v["lower"]} for m, v in ss.items()},
          "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
          "boot_seed": pooled_stats.BOOT_SEED, "perm_seed": PERM_SEED}
    _write(out / "E1_v07.json", js)
    (out / "E1_v07.md").write_text(_markdown(js))
    return js


def _f(x, nd=3):
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def _markdown(js):
    L = lambda m: LABELS.get(m, m)  # noqa: E731
    md = ["# E1 (experiments v07) report", "", f"Subjects {js['n_subjects']} ({js['n_positive']} positive); "
          f"B = {js['B']}; R_pi = {js['R_pi']}.", "",
          "| Model | Pooled AUROC, mean of repetitions | Bootstrap lower bound (Ec. 91) | Largest of the null AUROCs | "
          "Lower bound above the largest null | Exceedances b, point estimate | Descriptive permutation p | Passes Ec. 91 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in js["assay"]:
        md.append(f"| {L(r['model'])} | {_f(r['A_bar'])} | {_f(r['lower'])} | {_f(r['null_max'])} | "
                  f"{_f(r['lower'] > r['null_max'])} | {r['b']} | {_f(r['p'])} | {_f(r['passes'])} |")
    md += ["", "| Comparator | In M_ok | Mean paired difference (Ec. 83) | Lower bound (Ec. 85) | DeLong cross-check "
           "(Ec. 84) | Paired correlation | SE of the difference | Margin attainable (Ec. 87) | Bound above minus the "
           "margin |", "|---|---|---|---|---|---|---|---|---|"]
    for r in js["noninferiority"]:
        md.append(f"| {L(r['comparator'])} | {_f(r['in_M_ok'])} | {_f(r['delta_bar'])} | {_f(r['ell'])} | "
                  f"{_f(r['delong_delta'])} (z {_f(r['delong_z'], 2)}) | {_f(r['rho'])} | {_f(r['se'])} | "
                  f"{_f(r['attainable_80'])} | {_f(r['clears_margin'])} |")
    md += ["", "| Model, descriptive per-fold and threshold readings | Mean fold AUROC | SD fold AUROC | Fisher ratio "
           "(Ec. 62) | Corrected t (Ec. 63) | Average precision | Macro F1 | Balanced accuracy | Sensitivity | "
           "Specificity | Development AUROC | Development DeLong difference to full model |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in js["descriptive"]:
        md.append(f"| {L(r['model'])} | {_f(r['fold_auroc_mean'])} | {_f(r['fold_auroc_sd'])} | "
                  f"{_f(r.get('fisher_ratio'))} | {_f(r.get('corrected_t'), 2)} | {_f(r['ap'])} | {_f(r['macro_f1'])} | "
                  f"{_f(r['balanced_accuracy'])} | {_f(r['sensitivity'])} | {_f(r['specificity'])} | deferred | deferred |")
    h1 = js["h1"]
    md += ["", "| Comparison | Test | Statistic | Raw p | Adjusted p | Correction | Effect size | Effect size interval "
           "| n | alpha |", "|---|---|---|---|---|---|---|---|---|---|",
           f"| H1 testability (Ec. 100) | Ec. 91 on every model | testable = {_f(h1['testable'])}"
           f"{'' if h1['testable'] else ' (' + str(h1['reason']) + ')'} | | | | M_ok = {', '.join(h1['m_ok']) or '-'} "
           f"| | {js['n_subjects']} | 0.05 |",
           f"| H1 non-inferiority, conjunctive over M_ok (Ec. 86) | intersection-union, bootstrap bound | verdict = "
           f"{h1['verdict']} | | | none (IUT) | | | {js['n_subjects']} | 0.05 |"]
    for r in js["noninferiority"]:
        md.append(f"| Full against {L(r['comparator'])} | one-sided subject bootstrap (Ec. 85), H0: Delta <= -0.10 | "
                  f"ell = {_f(r['ell'])} | {_f(r['p_raw'], 4)} | {_f(r['p_holm'], 4)} | Holm | {_f(r['delta_bar'])} | "
                  f"[{_f(r['ell'])}, inf) | {js['n_subjects']} | 0.05 |")
    if js["ss_omega"]:
        md += ["", "| Model | SS^omega (Ec. 93) | Bootstrap lower bound (Ec. 99) |", "|---|---|---|"]
        md += [f"| {L(m)} | {_f(v['point'])} | {_f(v['lower'])} |" for m, v in js["ss_omega"].items()]
    md += ["", "Figures (convergence curves, ROC, null distributions, forest plot) and the development reading are "
           "deferred."]
    return "\n".join(md) + "\n"


# ---------------- remote entry ----------------
KAGGLE_INPUT = "/kaggle/input"
RESULT_DIRS = ("search", "hpo", "real", "perm")


def find_inputs(roots):
    """Labels, instances CSV, the WavLM _W.npy folder and the text _E.npy folder below the roots (os.walk). The WavLM
    cache is the output of the extract-wavlm kernel, mounted as a kernel source at
    <root>/notebooks/<owner>/<slug>/clone/TRAMIL/Data/features, deeper than a dataset."""
    from . import e1_run

    return e1_run._find(roots, depth=10, acoustic="wavlm")


def _input_roots(input_root=None):
    if input_root:
        return [input_root]
    mounted = json.loads(os.environ.get("FORGE_INPUT_DATASETS") or "{}")
    roots = [v for v in mounted.values() if v]
    if Path(KAGGLE_INPUT).is_dir():
        roots.append(KAGGLE_INPUT)
    return roots or ["TRAMIL/Data"]


def import_upstream(roots, out):
    """Copy earlier kernels' unit results (every E1_v07/{search,hpo,real,perm}/*.json found below the roots) into out;
    an existing file is never overwritten. Returns the number of files copied."""
    out, n = Path(out), 0
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        for here, dirs, _ in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(("instances_audio", "src", "."))]
            if Path(here).name != "E1_v07" or Path(here).resolve() == out.resolve():
                continue
            for sub in RESULT_DIRS:
                for f in sorted((Path(here) / sub).glob("*.json")):
                    dst = out / sub / f.name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(f, dst)
                        n += 1
            dirs[:] = []
    return n


def load_data(input_root=None, limit_bags=None):
    """WavLM bags with item scores and prompt priors, plus the descriptor embeddings Q (Symptom-Guided)."""
    from . import exp_e1, features

    env = os.environ
    if all(env.get(k) for k in ("TRAMIL_LABELS", "TRAMIL_EDAIC", "TRAMIL_FEATURES")):
        labels, edaic, feats = env["TRAMIL_LABELS"], env["TRAMIL_EDAIC"], env["TRAMIL_FEATURES"]
        text = env.get("TRAMIL_TEXT") or feats
    else:
        found = find_inputs(_input_roots(input_root))
        labels, edaic, feats, text = found["labels"], found["instances"].parent, found["features"], found["text"]
    items, info = exp_e1.load_items(labels, edaic, Path(feats), limit_bags, acoustic="wavlm", text_dir=text)
    return {"items": items, "Q": features.embed_texts(features.DESCRIPTORS)}, info


def _default_out():
    return Path(os.environ.get("FORGE_CLONE_ROOT", ".")) / "TRAMIL" / "Results" / "E1_v07"


def run(phases, out_dir=None, input_root=None, shard=None, units=None, workers=None, limit_bags=None,
        import_from_inputs=True, **cfg):
    """Remote callable (run-config.json): phases is one phase or a list run in order (a smoke chains them at a tiny
    scale). Earlier kernels' results mounted under the input roots are imported first; shard.json under the runner."""
    from . import e1_run

    phases = [phases] if isinstance(phases, str) else list(phases)
    out = Path(out_dir) if out_dir else _default_out()
    out.mkdir(parents=True, exist_ok=True)
    imported = import_upstream(_input_roots(input_root), out) if import_from_inputs else 0
    data, info = load_data(input_root, limit_bags)
    workers = workers or dispatch.default_workers(out.parent / "timing.json", item_training.device().type)
    summary = {"imported": imported, "workers": workers, "shard": shard, **info}
    for ph in phases:
        res = run_phase(ph, data, out, workers=workers, shard=shard, only=units, **cfg)
        summary[ph] = {"ran": len(res["ran"]), "skipped": res.get("skipped", 0)}
    e1_run._stamp(summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m TRAMIL.e1v07_run")
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--out", default=str(_default_out()))
    ap.add_argument("--input")
    ap.add_argument("--shard")
    ap.add_argument("--units")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit-bags", type=int)
    ap.add_argument("--max-epochs", type=int, default=item_training.MAX_EPOCHS)
    ap.add_argument("--r-pi", type=int, default=R_PI)
    a = ap.parse_args(argv)
    res = run(a.phase, a.out, a.input, a.shard, a.units.split(",") if a.units else None, a.workers, a.limit_bags,
              max_epochs=a.max_epochs, r_pi=a.r_pi)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
