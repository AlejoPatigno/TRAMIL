"""Shared hyperparameter search of r22 Ec. 102 (v07 E1 'Hyperparameter search'), run inside one outer training set.

Unit of work = (rep, fold, arm, k, p, inner fold): run_unit(spec, items) trains one model with seed 11 on the inner
training subjects and returns its per-epoch Ec. 39 curve on the inner validation subjects. Units are independent
(a parallel dispatcher can map them in any order); reduce(results) is a deterministic function of the result set:

- obj_m(k, p) = - min_e mean over the three inner folds of the class-weighted item MSE (Ec. 39, earliest epoch);
- k* = argmax_k mean over the k*-voting arms of max_p obj_m(k, p); p*_m = argmax_p obj_m(k*, p);
  ties to the smaller k, then the smaller p (Ec. 102);
- arms outside the vote (E2/E5 arms) get p* at k*; prompt_only is searched at k = 8 only (v07 E5); the k = 768
  reference of Ec. 103 is the full model searched over the same p values at k = 768 (p768).

run_fold maps the units sequentially, reduces and freezes the result to hpo_r{rep}_f{fold}.json; load_frozen reads
it back for the real run and the permutation runner."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import e1_arms, frontend, item_training, validation

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["6", "13", "18.1"],
    "equations": [39, 94, 102, 103],
    "experiments": ["E1", "E2", "E5"],
    "invariants": [
        "trial_objective_is_minus_the_ec39_inner_mse_at_the_earliest_best_epoch",
        "shared_k_is_the_argmax_over_k_of_the_arm_mean_of_the_best_p_objective",
        "ties_go_to_the_smaller_k_then_the_smaller_p",
        "arms_outside_the_shared_k_get_their_own_p_and_the_768_reference_its_own",
        "reducer_is_deterministic_under_any_result_order",
        "every_arm_gets_the_same_trial_values_and_unit_specs_are_independent",
        "a_unit_never_touches_an_outer_test_subject",
        "fold_search_writes_frozen_json_with_provenance_and_the_loader_reads_it_back",
    ],
}

K_GRID = (16, 32, 64, 128)  # operator setting 2026-11-25
P_GRID = (0.0, 0.15, 0.3, 0.45)  # T = 4, the same trial values for every arm
K_REF, K_PROMPT = 768, 8
SEED = 11  # one initialization seed for the search
INNER_FOLDS = 3
KSTAR_ARMS = ("full", "abmil8", "gated_abmil8", "symptom_guided", "r06_8", "abmil_r8")  # E1 neural arms


def units(rep, fold, arms=tuple(e1_arms.ARMS), K=K_GRID, P=P_GRID, max_epochs=item_training.MAX_EPOCHS):
    """Every independent unit of one outer training set, in a fixed order."""
    out = []
    for arm in arms:
        ks = (K_PROMPT,) if e1_arms.SOURCE.get(arm) == "R" else tuple(K) + ((K_REF,) if arm == "full" else ())
        for k in ks:
            for p in P:
                for r in range(INNER_FOLDS):
                    out.append({"rep": rep, "fold": fold, "arm": arm, "k": int(k), "p": float(p), "inner": r,
                                "seed": SEED, "max_epochs": int(max_epochs)})
    return out


def run_unit(spec, items, Q=None):
    """Train one (arm, k, p) model on inner training fold `inner` of outer fold (rep, fold); the front end is fitted on
    the inner training subjects only. k is capped at the input dimension (k = 768 is the full rotation)."""
    y = [it["y"] for it in items]
    test = validation.outer_splits(y, spec["rep"])[spec["fold"]]
    train = np.setdiff1d(np.arange(len(items)), test)
    tr, va = validation.inner_splits(y, train, spec["fold"])[spec["inner"]]
    source = e1_arms.SOURCE.get(spec["arm"], "X")
    k = min(spec["k"], items[0][source].shape[1])
    tb, vb, info = frontend.fold_bags(items, tr, va, k=k, source=source)
    make = e1_arms.build(spec["arm"], k=k, p_drop=spec["p"], vbar=info["vbar"], Q=Q)
    _, curve = item_training.train(make, tb, spec["max_epochs"], spec["seed"], eval_bags=vb)
    return {**spec, "curve": [float(c) for c in curve], "train_subjects": [items[i]["subject"] for i in tr],
            "val_subjects": [items[i]["subject"] for i in va]}


def _best(cands):
    """max objective, ties to the smaller key (k or p): cands = [(key, objective)]."""
    return min(cands, key=lambda c: (-c[1], c[0]))[0]


def reduce(results, kstar_arms=KSTAR_ARMS):
    curves = {}
    for r in results:
        curves.setdefault((r["arm"], r["k"], r["p"]), {})[r["inner"]] = r["curve"]
    table = []
    for (arm, k, p), byfold in sorted(curves.items()):
        mean = np.mean([byfold[i] for i in sorted(byfold)], axis=0)
        e = item_training.select_epoch(mean[None, :])
        table.append({"arm": arm, "k": k, "p": p, "objective": -float(mean[e - 1]), "epoch": e,
                      "inner_folds": len(byfold)})
    obj = {(t["arm"], t["k"], t["p"]): t["objective"] for t in table}
    arms = sorted({a for a, _, _ in obj})
    grid_k = sorted({k for a, k, _ in obj if a in kstar_arms and k != K_REF})
    voters = [a for a in arms if a in kstar_arms]

    def best_p(arm, k):
        return _best([(p, o) for (a, kk, p), o in obj.items() if a == arm and kk == k])

    k_star = _best([(k, float(np.mean([obj[(a, k, best_p(a, k))] for a in voters]))) for k in grid_k])
    p_star = {}
    for a in arms:
        ks = {k for aa, k, _ in obj if aa == a}
        k_a = K_PROMPT if ks == {K_PROMPT} else k_star
        if k_a in ks:
            p_star[a] = best_p(a, k_a)
    p768 = best_p("full", K_REF) if ("full", K_REF) in {(a, k) for a, k, _ in obj} else None
    return {"k_star": int(k_star), "p_star": p_star, "p768": p768, "table": table}


def _commit():
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except OSError:
        return None


def freeze(results, rep, fold, out_dir, grid, kstar_arms=KSTAR_ARMS):
    """Reduce one outer training set's unit results and write hpo_r{rep}_f{fold}.json (atomic)."""
    out = reduce(results, kstar_arms)
    js = {"rep": rep, "fold": fold, **out, "grid": {**grid, "kstar_arms": list(kstar_arms)},
          "provenance": {"seed": SEED, "commit": _commit() or os.environ.get("FORGE_CLONE_COMMIT"),
                         "outer_seed": validation.SEEDS_OUTER[rep], "inner_seed": validation.INNER_SEED + fold}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"hpo_r{rep}_f{fold}.json.tmp"
    tmp.write_text(json.dumps(js, indent=2))
    os.replace(tmp, out_dir / f"hpo_r{rep}_f{fold}.json")
    return out


def run_fold(items, rep, fold, out_dir, Q=None, arms=tuple(e1_arms.ARMS), K=K_GRID, P=P_GRID,
             max_epochs=item_training.MAX_EPOCHS, kstar_arms=KSTAR_ARMS):
    """Sequential search of one outer training set; writes and returns the frozen values."""
    res = [run_unit(s, items, Q) for s in units(rep, fold, arms, K, P, max_epochs)]
    grid = {"K": list(K), "P": list(P), "k_ref": K_REF, "k_prompt": K_PROMPT, "arms": list(arms),
            "max_epochs": max_epochs, "inner_folds": INNER_FOLDS}
    return freeze(res, rep, fold, out_dir, grid, kstar_arms)


def load_frozen(out_dir, rep, fold):
    js = json.loads((Path(out_dir) / f"hpo_r{rep}_f{fold}.json").read_text())
    return {"k_star": js["k_star"], "p_star": js["p_star"], "p768": js["p768"]}
