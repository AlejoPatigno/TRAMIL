import math

import numpy as np
import pytest
import torch

from TRAMIL import model, training, validation


def _items(n, seed=0, shift=2.0):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = i % 2
        X = g.standard_normal((6, 88)).astype(np.float32)
        X[:, :10] += shift * y
        out.append((X, g.random((6, 8)).astype(np.float32), y))
    return out


def test_class_weight_is_negatives_over_positives():
    assert training.class_weight(np.array([0] * 9 + [1] * 3)) == 3.0


def test_weighted_bce_matches_the_formula_of_ec_38():
    eta = torch.tensor([0.3, -1.2, 2.0, 0.1])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    w = 2.5
    p = torch.sigmoid(eta)
    manual = -(w * y * p.log() + (1 - y) * (1 - p).log()).mean()
    assert torch.isclose(training.weighted_bce(eta, y, w), manual, atol=1e-6)


def test_training_uses_adamw_lr_1e3_wd_1e4_batch_8():
    assert (training.LR, training.WEIGHT_DECAY, training.BATCH) == (1e-3, 1e-4, 8)
    opt = training.make_optimizer(model.ResidualBipartite())
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[0]["weight_decay"] == 1e-4


def test_training_learns_a_separable_toy_problem_and_is_deterministic_per_seed():
    tr, va = _items(24), _items(12, seed=1)
    mu, s = np.zeros(88, np.float32), np.ones(88, np.float32)
    ptr, pva = training.prepare(tr, mu, s), training.prepare(va, mu, s)
    m1, curve = training.train(model.ResidualBipartite, ptr, epochs=60, seed=11, eval_items=pva)
    assert len(curve) == 60 and curve[-1] > 0.9 and curve[-1] > curve[0] - 1e-9
    m2, _ = training.train(model.ResidualBipartite, ptr, epochs=60, seed=11)
    m3, _ = training.train(model.ResidualBipartite, ptr, epochs=60, seed=23)
    p1, p2, p3 = (training.predict(m, pva) for m in (m1, m2, m3))
    assert np.allclose(p1, p2, atol=1e-6) and not np.allclose(p1, p3, atol=1e-6)


def test_epoch_selection_takes_the_best_mean_inner_auroc_earliest_on_ties():
    curves = np.array([[0.5, 0.7, 0.7, 0.6], [0.5, 0.9, 0.9, 0.2], [0.5, 0.8, 0.8, 0.4]])
    assert validation.select_epoch(curves) == 2  # 1-based; epochs 2 and 3 tie
    assert validation.select_epoch(np.array([[0.1, 0.2, 0.9]])) == 3


def test_seed_ensemble_is_the_mean_probability():
    ps = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.1]])
    assert np.allclose(validation.seed_ensemble(ps), [0.4, 0.5])


def test_outer_and_inner_splits_are_stratified_subject_partitions_with_the_declared_seeds():
    y = np.array([0] * 30 + [1] * 15)
    folds = validation.outer_splits(y, repetition=0)
    assert [f.tolist() for f in folds] != [] and len(folds) == 5
    test_idx = np.concatenate(folds)
    assert sorted(test_idx.tolist()) == list(range(45))
    assert all(abs(y[f].mean() - 1 / 3) < 0.12 for f in folds)
    assert not np.array_equal(folds[0], validation.outer_splits(y, repetition=1)[0])
    tr = np.setdiff1d(np.arange(45), folds[0])
    inner = validation.inner_splits(y, tr, k=0)
    assert len(inner) == 3 and all(set(v).isdisjoint(t) for t, v in inner)
    assert all(set(t) | set(v) == set(tr.tolist()) for t, v in inner)
    assert validation.SEEDS_OUTER == (31415, 31416, 31417) and validation.INNER_SEED == 7301


def test_bootstrap_difference_uses_the_same_indices_for_both_methods():
    y = np.array([0] * 20 + [1] * 10)
    g = np.random.default_rng(3)
    a = np.where(y == 1, g.random(30) + 0.5, g.random(30))
    d = validation.bootstrap_delta(y, a, a, replicates=50, seed=0)
    assert d.shape == (50,) and np.all(d == 0.0)
    rev = validation.bootstrap_delta(y, a, 1 - a, replicates=200, seed=0)
    from sklearn.metrics import roc_auc_score

    base = roc_auc_score(y, a) - roc_auc_score(y, 1 - a)
    assert rev.mean() > 0.3 and abs(rev.mean() - base) < 0.15  # AUROC(a) - AUROC(1 - a) = 2 AUROC(a) - 1 per draw
