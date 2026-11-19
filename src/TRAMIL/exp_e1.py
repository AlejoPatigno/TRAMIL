"""E1 rehearsal: main comparison with equal text access, run at plumbing scale (stamped PILOT)."""
import functools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import bags as bagmod
from . import comparators, features, model, proximity, standardize, statistics, training, validation

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v04.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "invariants": [
        "e1_rehearsal_runs_every_arm_on_every_fold_without_leakage",
        "results_table_is_stamped_pilot_in_json_and_markdown",
    ],
}

ARMS = {
    "residual": model.ResidualBipartite,
    "learned_only": functools.partial(model.ResidualBipartite, learned_only=True),
    "abmil": comparators.ABMIL,
    "gated_abmil": functools.partial(comparators.ABMIL, gated=True),
}
NOTE = (
    "PILOT: numbers are plumbing checks of the pipeline at rehearsal scale (few epochs, few replicates, a subset of "
    "the training bags). They are not results and support no claim about H1."
)


def load_items(labels_path, edaic_dir, cache_dir, n_bags=None, seed=0, acoustic="egemaps", text_dir=None):
    """Training-split bags with their features: (X raw, R proximity, y) per subject, plus the exclusions.
    X is the 88-d eGeMAPS block (default) or the 768-d WavLM layer-6 vector (acoustic="wavlm")."""
    import pandas as pd

    edaic = Path(edaic_dir)
    lab = bagmod.load_labels(labels_path)
    allb = bagmod.build_bags(pd.read_csv(edaic / "instances_clean.csv"), edaic / "instances_audio", lab, "train")
    chosen = bagmod.stratified_subset(allb, n_bags, seed) if n_bags else allb
    Q = features.embed_texts(features.DESCRIPTORS)
    items, dropped = [], 0
    for b in chosen:
        X, E = features.bag_features(b, cache_dir, acoustic=acoustic, text_dir=text_dir)
        dropped += len(b) - len(X)
        if len(X):
            items.append({"subject": b.subject, "X": X, "R": proximity.proximity(E, Q), "y": b.y})
    return items, {"bags_requested": len(chosen), "bags_used": len(items), "instances_dropped_nonfinite": dropped}


def _neural(make, items, train_idx, test_idx, k, seeds, max_epochs):
    make = functools.partial(make, d_in=items[0]["X"].shape[1])  # input dim comes from the data
    y = [it["y"] for it in items]
    pick = lambda idx: [(items[i]["X"], items[i]["R"], items[i]["y"]) for i in idx]

    def prep(fit_idx, idx):
        mu, s = standardize.fit([items[i]["X"] for i in fit_idx])  # refitted for every training run
        return training.prepare(pick(idx), mu, s)

    probs, epochs = [], []
    for seed in seeds:
        curves = []
        for tr, va in validation.inner_splits(y, train_idx, k):
            curves.append(training.train(make, prep(tr, tr), max_epochs, seed, prep(tr, va))[1])
        e_star = validation.select_epoch(curves)
        m, _ = training.train(make, prep(train_idx, train_idx), e_star, seed)
        probs.append(training.predict(m, prep(train_idx, test_idx)))
        epochs.append(e_star)
    return validation.seed_ensemble(probs), epochs


def run_e1(items, n_folds=2, seeds=(11,), max_epochs=5, repetition=0):
    y = np.array([it["y"] for it in items])
    subj = [it["subject"] for it in items]
    folds = validation.outer_splits(y, repetition, n_folds)
    arms = {n: {"fold_auroc": [], "oof": {}, "epochs": [], "seconds": 0.0} for n in [*ARMS, "svm"]}
    out_folds = []
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(items)), test_idx)
        out_folds.append({"train": [subj[i] for i in train_idx], "test": [subj[i] for i in test_idx]})
        for name in arms:
            t0 = time.time()
            if name == "svm":
                pipe = comparators.fit_svm([items[i]["X"] for i in train_idx], y[train_idx], validation.INNER_SEED + k)
                scores = pipe.decision_function(np.stack([comparators.bag_stats(items[i]["X"]) for i in test_idx]))
            else:
                scores, e = _neural(ARMS[name], items, train_idx, test_idx, k, seeds, max_epochs)
                arms[name]["epochs"] += e
            arms[name]["seconds"] += time.time() - t0
            arms[name]["fold_auroc"].append(float(roc_auc_score(y[test_idx], scores)))
            arms[name]["oof"].update({subj[i]: float(s) for i, s in zip(test_idx, scores)})
    res = {"stage": "PILOT", "folds": out_folds, "arms": arms, "n_bags": len(items), "n_positive": int(y.sum()),
           "config": {"folds": n_folds, "repetition": repetition, "seeds": list(seeds), "max_epochs": max_epochs}}
    res["comparisons"] = _comparisons(arms)
    return res


def _comparisons(arms):
    base = np.array(arms["residual"]["fold_auroc"])
    out = {}
    for name, a in arms.items():
        if name == "residual" or len(base) < 2:
            continue
        d = base - np.array(a["fold_auroc"])
        t, p, df = statistics.corrected_t(d) if d.var(ddof=1) > 0 else (float("nan"), float("nan"), len(d) - 1)
        lo, hi, eq, below = statistics.tost(d) if d.var(ddof=1) > 0 else (float("nan"),) * 2 + (False, False)
        out[name] = {"mean_diff": float(d.mean()), "fisher_ratio": statistics.fisher_ratio(base, a["fold_auroc"])
                     if d.var(ddof=1) > 0 else float("nan"), "corrected_t": t, "p": p, "df": df,
                     "tost_90": [lo, hi], "equivalent_within_0.05": eq, "R": len(d)}
    return out


def write_results(res, out_dir, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = {**res, "note": NOTE, **(extra or {})}
    (out_dir / "E1_pilot.json").write_text(json.dumps(full, indent=2, default=float))
    rows = ["| Arm | Fold AUROC (plumbing) | Mean | Selected epochs | Seconds |", "| --- | --- | --- | --- | --- |"]
    for name, a in res["arms"].items():
        rows.append("| {} | {} | {:.3f} | {} | {:.0f} |".format(
            name, ", ".join(f"{v:.3f}" for v in a["fold_auroc"]), np.mean(a["fold_auroc"]), a["epochs"] or "-", a["seconds"]))
    md = ["# E1 rehearsal -- PILOT (plumbing check, not results)", "", NOTE, "",
          f"Bags: {res['n_bags']} ({res['n_positive']} positive); config: {res['config']}", ""]
    md += rows
    md += ["", "Paired comparisons against the residual model (R = number of fold replicates; plumbing only):", ""]
    for n, c in res["comparisons"].items():
        md.append(f"- {n}: mean diff {c['mean_diff']:.3f}, TOST 90% {c['tost_90']}, p {c['p']:.3f} (R={c['R']})")
    if extra:
        md += ["", "Run details:", ""] + [f"- {k}: {v}" for k, v in extra.items()]
    (out_dir / "E1_pilot.md").write_text("\n".join(md) + "\n")
