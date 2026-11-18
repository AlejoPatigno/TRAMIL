import numpy as np
import pytest
import torch

from TRAMIL import comparators, model


def test_abmil_and_gated_abmil_parameter_counts_match_the_proposal_table():
    assert model.n_params(comparators.ABMIL()) == 3969
    assert model.n_params(comparators.ABMIL(gated=True)) == 5025


def test_abmil_weights_are_a_distribution_and_the_model_is_order_invariant():
    for gated in (False, True):
        m = comparators.ABMIL(gated=gated)
        X = torch.randn(9, 88)
        a, H = m.attention(X)
        assert a.shape == (9,) and torch.isclose(a.sum(), torch.tensor(1.0), atol=1e-6) and (a > 0).all()
        perm = torch.randperm(9)
        assert torch.allclose(m(X), m(X[perm]), atol=1e-5)


def test_gated_scoring_differs_from_plain_scoring_with_shared_weights():
    plain, gated = comparators.ABMIL(), comparators.ABMIL(gated=True)
    gated.load_state_dict({k: v for k, v in plain.state_dict().items()}, strict=False)
    X = torch.randn(6, 88)
    assert not torch.allclose(plain.attention(X)[0], gated.attention(X)[0], atol=1e-6)


def test_permuted_prior_keeps_each_column_distribution_but_breaks_row_correspondence():
    R = np.random.default_rng(0).random((12, 8)).astype(np.float32)
    P = comparators.permute_prior(R, subject=301)
    assert np.allclose(np.sort(P, axis=0), np.sort(R, axis=0))
    assert not np.allclose(P, R)
    assert np.array_equal(P, comparators.permute_prior(R, subject=301))  # reproducible
    assert not np.array_equal(P, comparators.permute_prior(R, subject=302))  # subject-specific
    cols = [np.flatnonzero(np.all(R[:, [0]] == R[:, [j]], axis=1)) for j in range(1, 8)]
    # each column gets its own permutation: not all columns are permuted by the same index map
    maps = {tuple(np.argsort(np.argsort(P[:, j]))) for j in range(8)}
    assert len(maps) > 1


def test_bag_statistics_are_mean_and_sample_std_of_dimension_176():
    X = np.random.default_rng(1).standard_normal((7, 88)).astype(np.float32)
    s = comparators.bag_stats(X)
    assert s.shape == (176,)
    assert np.allclose(s[:88], X.mean(0), atol=1e-5) and np.allclose(s[88:], X.std(0, ddof=1), atol=1e-5)
    assert np.all(comparators.bag_stats(X[:1])[88:] == 0.0)  # one instance: std defined as 0


def test_svm_baseline_separates_bag_statistics_and_standardises_inside_the_training_fold():
    g = np.random.default_rng(2)
    y = np.array([0, 1] * 20)
    bags = [(g.standard_normal((6, 88)) + 1.5 * yy * (np.arange(88) < 10)).astype(np.float32) for yy in y]
    tr, te = np.arange(0, 28), np.arange(28, 40)
    pipe = comparators.fit_svm([bags[i] for i in tr], y[tr], inner_seed=7301)
    sc = pipe.decision_function(np.stack([comparators.bag_stats(bags[i]) for i in te]))
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y[te], sc) > 0.9
    train_stats = np.stack([comparators.bag_stats(bags[i]) for i in tr])
    assert np.allclose(pipe[0].mean_, train_stats.mean(0), atol=1e-5)  # scaler saw training bags only
    assert pipe[-1].kernel in ("linear", "rbf")
