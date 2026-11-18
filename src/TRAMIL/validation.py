"""Nested subject-level validation: epoch selection (Ec. 39), seed ensemble (Ec. 40), AUROC difference (Ec. 41)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r06.md",
    "sections": ["13", "14", "18.1"],
    "equations": [39, 40, 41],
    "invariants": [
        "epoch_selection_takes_the_best_mean_inner_auroc_earliest_on_ties",
        "seed_ensemble_is_the_mean_probability",
        "outer_and_inner_splits_are_stratified_subject_partitions_with_the_declared_seeds",
        "bootstrap_difference_uses_the_same_indices_for_both_methods",
    ],
}

SEEDS_OUTER = (31415, 31416, 31417)  # experiments v04: one outer partition seed per repetition
INNER_SEED = 7301  # inner folds of outer fold k use 7301 + k
INIT_SEEDS = (11, 23, 37)


def select_epoch(curves):
    """e* = argmax_e mean over inner folds of AUROC(e), earliest on ties; epochs are 1-based."""
    return int(np.argmax(np.asarray(curves).mean(0))) + 1  # argmax returns the first maximum


def seed_ensemble(ps):
    """p_bar_b = mean over initialisation seeds (rows) of the out-of-fold probability."""
    return np.asarray(ps).mean(0)


def outer_splits(y, repetition, n_folds=5):
    """Test-index arrays of the stratified outer folds of repetition q (seed 31415 + q)."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEEDS_OUTER[repetition])
    return [te for _, te in skf.split(np.zeros(len(y)), y)]


def inner_splits(y, train_idx, k, n_folds=3):
    """(train, validation) index pairs inside one outer training set, seed 7301 + k."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=INNER_SEED + k)
    tr = np.asarray(train_idx)
    return [(tr[a], tr[b]) for a, b in skf.split(np.zeros(len(tr)), np.asarray(y)[tr])]


def bootstrap_delta(y, p_a, p_b, replicates=4999, seed=0):
    """Delta = AUROC(a) - AUROC(b) over label-stratified subject bootstrap draws shared by both methods."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    g = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.empty(replicates)
    for r in range(replicates):
        idx = np.concatenate([g.choice(pos, len(pos)), g.choice(neg, len(neg))])
        out[r] = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
    return out
