"""PCA front end (r22 Ec. 94) and fold-level bag preparation."""
import numpy as np
import pytest
import torch

from TRAMIL import comparators, frontend


def _items(n=10, d=12, seed=0, n_inst=None):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        N = n_inst or int(g.integers(3, 9))
        s = np.full(8, 2.0 if i % 2 else 0.0, np.float32)
        out.append({"subject": 300 + i, "X": (g.standard_normal((N, d)) * np.linspace(3, 0.5, d)).astype(np.float32),
                    "R": g.random((N, 8)).astype(np.float32), "s": s, "y": int(s.sum() >= 10)})
    return out


def test_pca_is_fitted_on_training_instances_only_with_descending_uncorrelated_scores():
    items = _items()
    fe = frontend.FrontEnd(k=5).fit([it["X"] for it in items[:7]])
    Y = np.concatenate([fe.transform(it["X"]) for it in items[:7]])
    assert Y.shape[1] == 5 and np.allclose(Y.mean(0), 0, atol=1e-5)
    C = np.cov(Y.T, bias=True)
    assert np.allclose(C - np.diag(np.diag(C)), 0, atol=1e-4) and np.all(np.diff(np.diag(C)) <= 1e-6)
    other = frontend.FrontEnd(k=5).fit([it["X"] for it in items])  # held-out bags change the fit when included
    assert not np.allclose(other.transform(items[0]["X"]), fe.transform(items[0]["X"]))


def test_full_dimension_is_an_orthogonal_rotation_that_preserves_standardized_norms():
    g = np.random.default_rng(1)
    bags = [g.standard_normal((50, 768)).astype(np.float32) * 2 + 1 for _ in range(4)]
    fe = frontend.FrontEnd(k=768).fit(bags)
    X = bags[0]
    Xs = (X - fe.mu) / fe.sd
    assert np.allclose(np.linalg.norm(fe.transform(X), axis=1), np.linalg.norm(Xs, axis=1), rtol=1e-4)
    assert np.allclose(fe.V.T @ fe.V, np.eye(768), atol=1e-4)


def test_fold_bags_never_read_held_out_instances():
    items = _items()
    tr, ev = list(range(7)), [7, 8, 9]
    a_tr, a_ev, info = frontend.fold_bags(items, tr, ev, k=4)
    changed = [dict(it) for it in items]
    for i in ev:
        changed[i]["X"] = changed[i]["X"] * 100 + 7
        changed[i]["R"] = changed[i]["R"] * 0.5
    b_tr, _, info2 = frontend.fold_bags(changed, tr, ev, k=4)
    assert all(torch.equal(x["Y"], y["Y"]) for x, y in zip(a_tr, b_tr))
    assert info["vbar"] == info2["vbar"] == pytest.approx(frontend.rc_scale(a_tr))
    assert a_ev[0]["Y"].shape == (len(items[7]["X"]), 4)


def test_prepared_bags_carry_targets_permuted_target_and_reference_graph():
    items = _items()
    (b,), _, _ = frontend.fold_bags(items, [0], [], k=3)
    R = items[0]["R"]
    assert torch.allclose(b["Rc"], torch.from_numpy(R - R.mean(0)))
    P = comparators.permute_prior(R, items[0]["subject"])
    assert torch.allclose(b["Rc_perm"], torch.from_numpy(P - P.mean(0)))
    assert b["G"].shape == (len(R), len(R)) and torch.all(b["G"].abs() <= 1 + 1e-6)  # < 1 exactly; float32 rounding
    assert b["logN"] == pytest.approx(np.log(len(R))) and b["s"].shape == (8,) and b["y"] == items[0]["y"]


def test_prompt_source_builds_the_input_from_the_uncentered_prompt_rows_and_never_the_audio():
    items = _items()
    a, _, _ = frontend.fold_bags(items, range(10), [], k=8, source="R")
    changed = [dict(it, X=it["X"] + 5) for it in items]
    b, _, _ = frontend.fold_bags(changed, range(10), [], k=8, source="R")
    assert all(torch.equal(x["Y"], y["Y"]) for x, y in zip(a, b)) and a[0]["Y"].shape[1] == 8


def test_rc_scale_makes_the_zero_surrogate_penalty_one():
    items = _items(n_inst=6)
    bags, _, info = frontend.fold_bags(items, range(10), [], k=3)
    v = info["vbar"]
    err = np.mean([(b["Rc"] ** 2).sum().item() / (8 * len(b["Rc"]) * v) for b in bags])
    assert err == pytest.approx(1.0, rel=1e-5)
