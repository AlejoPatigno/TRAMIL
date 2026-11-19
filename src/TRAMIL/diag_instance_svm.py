"""Diagnostic (not part of E1): instance-level linear SVM on eGeMAPS (X is the 88-d acoustic block only; text
embeddings live in the separate E array), subject labels inherited by instances, scores aggregated per subject.
Run: python -m TRAMIL.diag_instance_svm"""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import comparators, e1_run, exp_e1, validation

GRID = [0.001, 0.01, 0.1, 1.0]
AGGS = ("mean", "max", "top3")


def _agg(scores, counts):
    """Per-subject mean / max / mean of the 3 highest instance scores; instances are grouped in bag order."""
    out = {a: [] for a in AGGS}
    for s in np.split(scores, np.cumsum(counts)[:-1]):
        out["mean"].append(s.mean())
        out["max"].append(s.max())
        out["top3"].append(np.sort(s)[-3:].mean())
    return {a: np.array(v) for a, v in out.items()}


def _stack(items, idx):
    X = np.concatenate([items[i]["X"] for i in idx])
    counts = [len(items[i]["X"]) for i in idx]
    y = np.repeat([items[i]["y"] for i in idx], counts)
    return X, y, counts


def _fit(X, y, C):
    return make_pipeline(StandardScaler(), LinearSVC(C=C, class_weight="balanced", max_iter=5000)).fit(X, y)


def fit_unit(items, train_idx, test_idx, k, grid=GRID):
    """Scaler and SVM see training-subject instances only; C by subject-grouped inner folds (subject AUROC, mean agg)."""
    X, y, counts = _stack(items, train_idx)
    groups = np.repeat(np.arange(len(train_idx)), counts)
    ysub = np.array([items[i]["y"] for i in train_idx])
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=validation.INNER_SEED + k)
    folds = list(cv.split(X, y, groups))
    cv_auc = []
    for C in grid:
        aucs = []
        for a, b in folds:
            m = _fit(X[a], y[a], C)
            subs = np.unique(groups[b])
            s = _agg(m.decision_function(X[b]), np.bincount(groups[b])[subs])["mean"]
            aucs.append(roc_auc_score(ysub[subs], s))
        cv_auc.append(float(np.mean(aucs)))
    best = grid[int(np.argmax(cv_auc))]
    model = _fit(X, y, best)
    Xt, _, ct = _stack(items, test_idx)
    return {"model": model, "C": best, "cv_auc": cv_auc, "n_fit_instances": len(X),
            "train": _agg(model.decision_function(X), counts), "test": _agg(model.decision_function(Xt), ct)}


def run(items, repetitions=3, n_folds=5, grid=GRID):
    y = np.array([it["y"] for it in items])
    arms = {n: {"train_auroc": [], "val_auroc": []} for n in ("bag_svm", *AGGS_NAMES)}
    chosen_C, t0 = [], time.time()
    for r in range(repetitions):
        for k, te in enumerate(validation.outer_splits(y, r, n_folds)):
            tr = np.setdiff1d(np.arange(len(items)), te)
            res = fit_unit(items, tr, te, k, grid)
            chosen_C.append(res["C"])
            for a in AGGS:
                arms[f"instance_{a}"]["train_auroc"].append(float(roc_auc_score(y[tr], res["train"][a])))
                arms[f"instance_{a}"]["val_auroc"].append(float(roc_auc_score(y[te], res["test"][a])))
            pipe = comparators.fit_svm([items[i]["X"] for i in tr], y[tr], validation.INNER_SEED + k)
            S = lambda idx: np.stack([comparators.bag_stats(items[i]["X"]) for i in idx])
            arms["bag_svm"]["train_auroc"].append(float(roc_auc_score(y[tr], pipe.decision_function(S(tr)))))
            arms["bag_svm"]["val_auroc"].append(float(roc_auc_score(y[te], pipe.decision_function(S(te)))))
    n_inst = [len(it["X"]) for it in items]
    return {"arms": arms, "grid": list(grid), "chosen_C": chosen_C, "seconds": time.time() - t0,
            "instances_per_subject": {"min": int(min(n_inst)), "median": float(np.median(n_inst)), "max": int(max(n_inst))},
            "n_subjects": len(items), "units": repetitions * n_folds}


AGGS_NAMES = tuple(f"instance_{a}" for a in AGGS)


def report(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    e1_run._write_json(out / "diag_instance_svm.json", res)
    ips = res["instances_per_subject"]
    md = ["# Diagnostic: instance-level SVM (eGeMAPS only)", "",
          f"{res['n_subjects']} subjects, {res['units']} outer units, instances/subject min/median/max "
          f"{ips['min']}/{ips['median']:.0f}/{ips['max']}, C grid {res['grid']}, {res['seconds']:.0f} s", "",
          "| arm | mean train AUROC | mean val AUROC | per-fold val AUROC |", "| --- | --- | --- | --- |"]
    for n, a in res["arms"].items():
        md.append(f"| {n} | {np.mean(a['train_auroc']):.3f} | {np.mean(a['val_auroc']):.3f} | "
                  + ", ".join(f"{v:.3f}" for v in a["val_auroc"]) + " |")
    (out / "diag_instance_svm.md").write_text("\n".join(md) + "\n")


def main(out_dir="TRAMIL/Results"):
    found = e1_run._find(e1_run._default_roots())
    items, _ = exp_e1.load_items(found["labels"], found["instances"].parent, found["features"])
    res = run(items)
    report(res, out_dir)
    return res


if __name__ == "__main__":
    main()
