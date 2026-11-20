import numpy as np
from sklearn.metrics import roc_auc_score

from TRAMIL import diag_instance_svm as d


def _items(n=40, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for s in range(n):
        y = s % 2
        out.append({"subject": 100 + s, "y": y, "X": g.normal(size=(g.integers(5, 12), 6)) + 3.0 + 2.0 * y * np.eye(6)[0]})
    return out


def test_fit_unit_never_touches_test_instances_and_aggregates_per_subject():
    items = _items()
    test_idx = np.arange(0, 40, 3)
    train_idx = np.setdiff1d(np.arange(40), test_idx)
    r = d.fit_unit(items, train_idx, test_idx, k=0, grid=[0.1, 1.0])
    train_X = np.concatenate([items[i]["X"] for i in train_idx])
    assert np.allclose(r["model"].named_steps["standardscaler"].mean_, train_X.mean(0))
    assert r["n_fit_instances"] == len(train_X)
    y = np.array([it["y"] for it in items])
    for agg in ("mean", "max", "top3"):
        assert r["test"][agg].shape == (len(test_idx),) and r["train"][agg].shape == (len(train_idx),)
        assert roc_auc_score(y[test_idx], r["test"][agg]) > 0.8


def test_bag_svm_permuted_run_is_deterministic_and_differs_from_unpermuted():
    items = _items(30)
    kw = dict(repetitions=1, n_folds=3)
    obs = d.bag_svm_val_auroc(items, seed=None, **kw)
    p1, p2 = d.bag_svm_val_auroc(items, seed=5, **kw), d.bag_svm_val_auroc(items, seed=5, **kw)
    assert p1 == p2 and p1 != obs and obs > 0.8
