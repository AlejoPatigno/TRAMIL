"""Pooled out-of-fold statistics of r22 Section 18.1 / 18.7 and the v07 validation scheme."""
import time

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from TRAMIL import poc_e0, pooled_stats as ps


def _data(M=141, n1=44, seed=0, signal=1.0):
    g = np.random.default_rng(seed)
    y = np.array([1] * n1 + [0] * (M - n1))
    return y, g


def test_rank_auroc_equals_the_pairwise_half_credit_definition_with_ties():
    y = np.array([1, 1, 0, 0, 1, 0, 0])
    s = np.array([3.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0])
    assert ps.auroc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_mean_pooled_auroc_averages_the_repetitions():
    y, g = _data()
    S = g.standard_normal((3, len(y))) + y
    assert ps.mean_auroc(S, y) == pytest.approx(np.mean([roc_auc_score(y, s) for s in S]))


def test_bootstrap_indices_are_label_stratified_reproducible_and_shared():
    y, _ = _data()
    I = ps.bootstrap_indices(y, B=50, seed=3)
    assert I.shape == (50, len(y))
    assert np.all(y[I].sum(1) == 44) and np.array_equal(I, ps.bootstrap_indices(y, B=50, seed=3))
    assert not np.array_equal(I, ps.bootstrap_indices(y, B=50, seed=4))
    assert ps.B == 9999 and ps.ALPHA == 0.05 and ps.DELTA_NI == 0.10 and ps.R_PI == 23


def test_vectorized_bootstrap_auroc_matches_the_reference_per_resample():
    y, g = _data()
    s = np.round(g.standard_normal(len(y)) + y, 1)  # ties included
    I = ps.bootstrap_indices(y, B=20, seed=0)
    want = [roc_auc_score(y[i], s[i]) for i in I]
    assert np.allclose(ps.boot_auroc(s, y, I), want)


def test_pooled_reading_gives_bounds_from_the_shared_resamples():
    y, g = _data()
    base = g.standard_normal((3, len(y)))
    scores = {"res": base + 1.2 * y, "same": base + 1.2 * y, "weak": g.standard_normal((3, len(y))) + 0.2 * y}
    I = ps.bootstrap_indices(y, B=500, seed=0)
    out = ps.pooled(scores, y, I, ref="res")
    assert out["res"]["A_bar"] == pytest.approx(ps.mean_auroc(scores["res"], y))
    assert out["res"]["lower"] == pytest.approx(np.quantile(out["res"]["boot"], 0.05))
    assert out["res"]["lower"] < out["res"]["A_bar"]
    assert out["same"]["delta_bar"] == 0.0 and np.all(out["same"]["delta_boot"] == 0)  # identical scores, shared I
    assert out["weak"]["ell"] > 0 and out["weak"]["delta_bar"] == pytest.approx(
        out["res"]["A_bar"] - out["weak"]["A_bar"])
    assert -1 <= out["weak"]["rho"] <= 1 and out["weak"]["se"] == pytest.approx(out["weak"]["delta_boot"].std(ddof=1))


def test_delong_variance_matches_a_hand_computed_example():
    # positives [3, 1] / [2, 2], negatives [2, 0] / [1, 3]: AUROC 0.75 and 0.5; by hand
    # S10 = [[0.125, 0], [0, 0]], S01 = [[0.125, -0.25], [-0.25, 0.5]] -> Var = 0.125/2 + 1.125/2 = 0.625
    y = np.array([1, 1, 0, 0])
    out = ps.delong(np.array([3.0, 1, 2, 0]), np.array([2.0, 2, 1, 3]), y)
    assert out["auc_a"] == 0.75 and out["auc_b"] == 0.5
    assert out["var"] == pytest.approx(0.625) and out["delta"] == pytest.approx(0.25)
    assert out["z"] == pytest.approx(0.25 / np.sqrt(0.625))
    lo, hi = out["ci95"]
    assert lo == pytest.approx(0.25 - 1.959964 * np.sqrt(0.625), abs=1e-5) and hi > 0.25


def test_assay_sensitivity_rule_and_descriptive_permutation_p():
    nulls = np.linspace(0.40, 0.62, 23)
    a = ps.assay(lower=0.63, point=0.70, nulls=nulls)
    assert a["passes"] and a["null_max"] == pytest.approx(0.62) and a["p"] == pytest.approx(1 / 24)
    b = ps.assay(lower=0.61, point=0.60, nulls=nulls)
    assert not b["passes"] and b["p"] == pytest.approx((1 + int((nulls >= 0.60).sum())) / 24)


def test_h1_is_testable_only_if_the_full_model_and_two_comparators_pass():
    assert ps.testable_h1(True, {"a": True, "b": True, "c": False}) == (True, None, ["a", "b"])
    t, why, ok = ps.testable_h1(False, {"a": True, "b": True})
    assert not t and "full model" in why
    t, why, ok = ps.testable_h1(True, {"a": True, "b": False})
    assert not t and "at most one" in why and ok == ["a"]


def test_non_inferiority_rule_over_the_passing_comparators():
    ell = {"a": -0.05, "b": -0.02, "c": -0.30}
    assert ps.h1_verdict(ell, ["a", "b"], testable=True)["verdict"] == "supported"
    assert ps.h1_verdict(ell, ["a", "b", "c"], testable=True)["verdict"] == "partially supported"
    assert ps.h1_verdict({"a": -0.2, "b": -0.3}, ["a", "b"], testable=True)["verdict"] == "not supported"
    assert ps.h1_verdict(ell, ["a"], testable=False)["verdict"] == "inconclusive"
    v = ps.h1_verdict(ell, ["a", "b"], testable=True, h5_supported=False)
    assert v["verdict"] == "estimate" and v["clears"] == {"a": True, "b": True}


def test_margin_attainability_of_ec_87():
    a = ps.attainability(se=0.040)
    assert a["multiplier_80"] == pytest.approx(2.4865, abs=1e-3) and a["attainable_80"] and a["attainable_50"]
    b = ps.attainability(se=0.045)
    assert not b["attainable_80"] and b["attainable_50"]
    assert ps.attainability(se=0.07)["attainable_50"] is False


def test_class_weighted_skill_score_matches_the_e0_reference_and_bootstraps():
    y, g = _data(M=40, n1=12)
    S = g.integers(0, 4, (40, 8)).astype(float)
    S_hat = S + 0.5 * g.standard_normal((40, 8))
    folds = np.array_split(g.permutation(40), 5)
    w, ref = ps.ss_reference(S, y, folds)
    point = ps.ss_omega(S_hat, S, w, ref)
    # E0 helper on one fold: same formula with one training set
    te = folds[0]
    tr = np.setdiff1d(np.arange(40), te)
    bags = [{"s": torch.tensor(S[i]), "y": int(y[i])} for i in range(40)]
    ow = poc_e0.omega_plus([bags[i] for i in tr])
    sbar = poc_e0.weighted_train_mean([bags[i] for i in tr], ow)
    want = poc_e0.ss_w([torch.tensor(S_hat[i]) for i in te], [bags[i] for i in te], sbar,
                       poc_e0.bag_weights([bags[i] for i in te], ow))
    assert ps.ss_omega(S_hat[te], S[te], w[te], ref[te]) == pytest.approx(want)
    I = ps.bootstrap_indices(y, B=300, seed=0)
    boot = ps.boot_ss_omega(S_hat, S, w, ref, I)
    assert boot.shape == (300,) and abs(np.median(boot) - point) < 0.1
    assert ps.ss_omega(ref, S, w, ref) == 0.0


def test_skill_inputs_average_predictions_weights_and_references_over_repetitions():
    y, g = _data(M=30, n1=10)
    S = g.integers(0, 4, (30, 8)).astype(float)
    S_hat_q = np.stack([S + g.standard_normal((30, 8)) for _ in range(3)])
    folds_q = [np.array_split(np.random.default_rng(q).permutation(30), 5) for q in range(3)]
    S_hat, w, ref = ps.ss_inputs(S_hat_q, S, y, folds_q)
    per = [ps.ss_reference(S, y, f) for f in folds_q]
    assert np.allclose(S_hat, S_hat_q.mean(0)) and np.allclose(w, np.mean([p[0] for p in per], 0))
    assert np.allclose(ref, np.mean([p[1] for p in per], 0))


def test_shared_bootstrap_of_eight_models_and_three_repetitions_runs_in_well_under_a_minute():
    y, g = _data()
    scores = {f"m{i}": g.standard_normal((3, len(y))) + 0.5 * y for i in range(8)}
    t0 = time.perf_counter()
    I = ps.bootstrap_indices(y)
    out = ps.pooled(scores, y, I, ref="m0")
    assert time.perf_counter() - t0 < 60 and out["m3"]["boot"].shape == (9999,)
