"""E2 to E5 analyses on synthetic data."""
import numpy as np
import pytest

from TRAMIL import e2e5_stats as st, pooled_stats

RNG = np.random.default_rng(0)


def test_bh_matches_the_step_up_definition_and_is_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(st.bh(p), [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)
    q = RNG.random(30)
    a = st.bh(q)
    assert (a >= q - 1e-12).all() and (a <= 1).all() and (np.diff(a[np.argsort(q)]) >= -1e-12).all()
    assert st.bh([]).size == 0


def _setup(M=20):
    S = RNG.integers(0, 4, (M, 8)).astype(float)
    folds = [np.arange(i, M, 4) for i in range(4)]
    return S, [folds]


def test_profile_metrics_match_hand_computed_skill_scores():
    S, fq = _setup()
    Sh = S + RNG.normal(0, 0.5, S.shape)
    P, ref = st.plain_inputs(Sh[None], S, fq)
    for te in fq[0]:
        assert np.allclose(ref[te], S[np.setdiff1d(np.arange(20), te)].mean(0))
    num, den = ((Sh - S) ** 2).sum(), ((ref - S) ** 2).sum()
    assert np.isclose(st.ss_plain(P, S, ref), 1 - num / den)
    j = 3
    assert np.isclose(st.ss_items(P, S, ref)[j], 1 - ((Sh - S)[:, j] ** 2).sum() / ((ref - S)[:, j] ** 2).sum())
    assert np.isclose(st.ss_plain(ref, S, ref), 0.0) and np.isclose(st.mae(Sh, S), np.abs(Sh - S).mean())
    assert np.isclose(st.weighted_kappa(S, S), 1.0)
    rho, p = st.item_spearman(Sh, S)
    assert (rho > 0.5).all() and (p < 0.5).all()
    flat = Sh.copy()
    flat[:, 0] = 1.0
    assert np.isnan(st.item_spearman(flat, S)[0][0])


def test_shape_skill_compares_with_the_models_own_total_spread_over_the_item_means():
    S, fq = _setup()
    _, ref = st.plain_inputs(S[None], S, fq)
    Sh = ref * (S.sum(1, keepdims=True) / ref.sum(1, keepdims=True))  # exactly the allocation baseline
    assert np.isclose(st.ss_shape(Sh, S, ref), 0.0)
    assert st.ss_shape(S, S, ref) == 1.0  # a perfect profile beats the allocation of its own total


def test_declared_profile_subjects_depend_on_the_total_only():
    S = np.zeros((50, 8))
    S[:, 0] = np.arange(50)[::-1] % 25
    subjects = list(range(1000, 1050))
    a = st.declared_profile_subjects(S, subjects)
    assert len(a) == 8 and len(set(a)) == 8
    S2 = S.copy()
    S2[:, 1] = 0  # untouched total: same panel whatever the item split
    assert a == st.declared_profile_subjects(S2, subjects)
    T = S.sum(1)
    assert np.all(np.diff([T[subjects.index(s)] for s in a]) >= 0)


def _runs(sd, M=40, R=9, seed=1):
    g = np.random.default_rng(seed)
    return g.normal(0, 1, M) + g.normal(0, sd, (R, M))


def test_log_variance_ratio_is_positive_for_the_noisier_model_and_bootstrapped_on_shared_resamples():
    y = np.array([0, 1] * 20)
    I = pooled_stats.bootstrap_indices(y, B=400)
    pt, boot = st.log_variance_ratio(_runs(1.0), _runs(0.2, seed=2), I)
    assert pt > 0 and st.lower(boot) > 0 and boot.shape == (400,)
    pt2, boot2 = st.log_variance_ratio(_runs(0.2), _runs(1.0, seed=2), I)
    assert pt2 < 0 and st.lower(boot2) < 0
    assert np.isclose(pt, np.log(st.subject_variance(_runs(1.0)).mean() / st.subject_variance(_runs(0.2, seed=2)).mean()))


def test_percentile_ranks_are_average_ranks_over_the_held_out_subjects_of_a_run():
    r = st.percentile_ranks(np.array([[3.0, 1.0, 2.0, 2.0]]))
    assert np.allclose(r, [[1.0, 0.25, 0.625, 0.625]])
    assert np.allclose(st.percentile_ranks(np.array([[10, 20, 30.0]]) * 7), [[1 / 3, 2 / 3, 1.0]])  # scale-free


def test_fisher_and_brown_forsythe_flag_the_noisier_group():
    a, b = np.random.default_rng(3).normal(0.7, 0.01, 45), np.random.default_rng(4).normal(0.7, 0.05, 45)
    r = st.fisher_brown_forsythe(a, b)
    assert r["F"] > 5 and r["p_F"] < 0.01 and r["p_W"] < 0.01 and 0 <= r["p_W"] <= 1


def _att(seed, N=12, V=9, noise=0.0, base=None):
    g = np.random.default_rng(seed)
    base = g.random((N, 8)) if base is None else base
    return np.stack([base + noise * g.random((N, 8)) for _ in range(V)])


def _ranks(cols):
    rc = [st.column_ranks(c) for c in cols]
    return np.stack([r for r, _ in rc]), np.stack([c for _, c in rc])


def test_stability_is_the_mean_pairwise_rank_correlation_over_defined_pairs():
    r, c = _ranks(_att(0))  # identical columns in every run
    assert np.allclose(st.stab_subject(r, c), 1.0)
    cols = _att(1, noise=0.0)
    cols[1::2] = -cols[1::2]  # odd runs invert the ranking: pairs of unlike parity anti-correlate
    r, c = _ranks(cols)
    s = st.stab_subject(r, c)
    assert np.allclose(s, (16 - 20) / 36, atol=1e-9)  # 16 same-parity pairs at +1, 20 mixed pairs at -1


def test_constant_columns_are_undefined_and_complete_case_and_zero_filled_readings_differ():
    cols = _att(2, noise=0.05)
    cols[:, :, 3] = 1.0 / cols.shape[1]  # item 3 uniform in every run
    r, c = _ranks(cols)
    assert c[:, 3].all()
    s = st.stab_subject(r, c)
    assert np.isnan(s[3]) and np.isfinite(np.delete(s, 3)).all()
    M = 6
    Sa = np.vstack([np.full(8, 0.8)] * M)
    Sb = np.vstack([np.full(8, 0.4)] * M)
    Sb[:, 3] = np.nan
    I = pooled_stats.bootstrap_indices(np.array([0, 1] * 3), B=200)
    out = st.stab_contrast(Sa, Sb, I)
    assert np.isclose(out["complete_case"]["diff"], 0.4) and np.isclose(out["zero_filled"]["diff"], (7 * 0.4 + 0.8) / 8)
    assert out["dropped_b"] == 1 / 8 and out["dropped_a"] == 0 and out["dropped_b_item"][3] == 1.0
    assert out["zero_filled"]["diff"] > out["complete_case"]["diff"] and out["complete_case"]["lower"] > 0
    Sb[:, :] = np.nan  # nothing defined in the other arm: complete case drops every subject
    out = st.stab_contrast(Sa, Sb, I)
    assert out["complete_case"]["n"] == 0


def test_pairwise_tv_is_zero_for_identical_runs_and_one_for_disjoint_columns():
    A = np.zeros((4, 3, 8))
    A[:, 0, :] = 1.0
    assert st.mean_pairwise_tv(A) == 0.0
    B = A.copy()
    B[1:2] = 0
    B[1:2, 2, :] = 1.0
    assert np.isclose(st.mean_pairwise_tv(B), 3 / 6)  # three of the six run pairs differ, TV = 1 each


def test_effective_attention_is_one_for_uniform_columns_and_small_for_a_spike():
    N = 10
    assert np.isclose(st.feff(np.full((N, 8), 1 / N)), 1.0)
    spike = np.zeros((N, 8))
    spike[0] = 1.0
    assert np.isclose(st.feff(spike), 1 / N)


def test_edge_diagnostics_and_surrogate_fidelity():
    N = 5
    G = np.eye(N) + 0.1
    al, rms = st.edge_diagnostics(G, G, np.zeros((N, N)))
    assert al == 0 and rms == 0
    al, rms = st.edge_diagnostics(G + 1.0, G, np.ones((N, N)) * 2)
    assert np.isclose(al, (N * N - N) / N**2) and np.isclose(rms, 2 * N / N)
    Rc = np.random.default_rng(6).random((9, 8))
    assert np.allclose(st.spearman_fidelity(Rc, Rc), 1.0) and np.allclose(st.spearman_fidelity(-Rc, Rc), -1.0)
    flat = Rc.copy()
    flat[:, 2] = 0.3
    assert np.isnan(st.spearman_fidelity(flat, Rc)[2])
