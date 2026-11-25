"""Pooled out-of-fold statistics of r22 Section 18.1 and 18.7 under the v07 validation scheme: pooled AUROC of the seed
ensemble per repetition and its mean (Ec. 82, 83), one label-stratified subject bootstrap shared by every model and
repetition (Ec. 85, B = 9999), one-sided lower bounds at alpha = 0.05, non-inferiority (Ec. 86, delta_NI = 0.10),
paired DeLong (Ec. 84, cross-check), the assay-sensitivity rule (Ec. 91 as operationalized by v07: lower bound above
the largest of the R_pi = 23 nulls, descriptive p = (b + 1) / (R_pi + 1)), testable_H1 (Ec. 100), margin attainability
(Ec. 87) and the class-weighted skill score SS^omega with its bootstrap (Ec. 93, 99). Holm adjustment for
comparator-wise statements is statistics.holm. Scores are arrays (Q repetitions, M subjects) in a fixed subject order."""
import numpy as np
from scipy import stats as sps

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["18.1", "18.7"],
    "equations": [82, 83, 84, 85, 86, 87, 91, 93, 99, 100],
    "experiments": ["E1", "E2", "E5"],
    "invariants": [
        "rank_auroc_equals_the_pairwise_half_credit_definition_with_ties",
        "mean_pooled_auroc_averages_the_repetitions",
        "bootstrap_indices_are_label_stratified_reproducible_and_shared",
        "vectorized_bootstrap_auroc_matches_the_reference_per_resample",
        "pooled_reading_gives_bounds_from_the_shared_resamples",
        "delong_variance_matches_a_hand_computed_example",
        "assay_sensitivity_rule_and_descriptive_permutation_p",
        "h1_is_testable_only_if_the_full_model_and_two_comparators_pass",
        "non_inferiority_rule_over_the_passing_comparators",
        "margin_attainability_of_ec_87",
        "class_weighted_skill_score_matches_the_e0_reference_and_bootstraps",
        "skill_inputs_average_predictions_weights_and_references_over_repetitions",
        "shared_bootstrap_of_eight_models_and_three_repetitions_runs_in_well_under_a_minute",
    ],
}

B, ALPHA, DELTA_NI, R_PI = 9999, 0.05, 0.10, 23
BOOT_SEED = 2026  # not fixed by r22 or v07; recorded with the results


def auroc(s, y):
    """Ec. 82: (1 / n1 n0) sum over positive-negative pairs of 1{s_i > s_k} + 1/2 1{s_i = s_k}, by average ranks."""
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((sps.rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mean_auroc(S, y):
    """A_bar = mean over repetitions of the pooled AUROC (Ec. 83 terms)."""
    return float(np.mean([auroc(s, y) for s in np.atleast_2d(S)]))


def bootstrap_indices(y, B=B, seed=BOOT_SEED):
    """(B, M) subject indices, label-stratified: each row resamples the positives and the negatives with replacement."""
    y = np.asarray(y)
    g = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([g.choice(pos, (B, len(pos))), g.choice(neg, (B, len(neg)))], axis=1)


def boot_auroc(s, y, I):
    """(B,) pooled AUROC on every resample, vectorized over rows by average ranks."""
    S, Y = np.asarray(s, float)[I], np.asarray(y).astype(bool)[I]
    n1 = Y.sum(1)
    n0 = Y.shape[1] - n1
    return (np.where(Y, sps.rankdata(S, axis=1), 0).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def lower(boot, alpha=ALPHA):
    """One-sided lower bound: the alpha-quantile of the bootstrap distribution (linear interpolation)."""
    return float(np.quantile(boot, alpha))


def pooled(scores, y, I, ref="res", alpha=ALPHA):
    """Per model: A_q, A_bar, its bootstrap and lower bound; against `ref`: Delta_bar = A_bar(ref) - A_bar(m), its
    bootstrap on the same resamples (Ec. 85), ell, the bootstrap SE and the correlation rho of the two A_bar*."""
    out = {}
    for m, S in scores.items():
        S = np.atleast_2d(S)
        boot = np.mean([boot_auroc(s, y, I) for s in S], axis=0)
        out[m] = {"A_q": [auroc(s, y) for s in S], "A_bar": mean_auroc(S, y), "boot": boot, "lower": lower(boot, alpha)}
    r = out[ref]
    for m, o in out.items():
        if m == ref:
            continue
        d = r["boot"] - o["boot"]
        o.update(delta_bar=r["A_bar"] - o["A_bar"], delta_boot=d, ell=lower(d, alpha), se=float(d.std(ddof=1)),
                 rho=float(np.corrcoef(r["boot"], o["boot"])[0, 1]) if d.std() > 0 else 1.0)
    return out


def delong(s_a, s_b, y):
    """Paired DeLong (Ec. 84): structural components V10 over positives and V01 over negatives with the half-credit
    kernel, sample covariances S10, S01, Var(Delta) = c^T (S10 / n1 + S01 / n0) c with c = (1, -1)."""
    y = np.asarray(y).astype(bool)
    V10, V01 = [], []
    for s in (np.asarray(s_a, float), np.asarray(s_b, float)):
        P, N = s[y][:, None], s[~y][None, :]
        psi = (P > N) + 0.5 * (P == N)
        V10.append(psi.mean(1))
        V01.append(psi.mean(0))
    S10, S01 = np.cov(np.stack(V10)), np.cov(np.stack(V01))
    c = np.array([1.0, -1.0])
    var = float(c @ (S10 / y.sum() + S01 / (~y).sum()) @ c)
    auc = [float(v.mean()) for v in V10]
    delta = auc[0] - auc[1]
    z = delta / np.sqrt(var) if var > 0 else float("nan")
    h = sps.norm.ppf(0.975) * np.sqrt(var)
    return {"auc_a": auc[0], "auc_b": auc[1], "delta": delta, "var": var, "z": z,
            "p_two_sided": float(2 * sps.norm.sf(abs(z))) if var > 0 else float("nan"), "ci95": (delta - h, delta + h)}


def assay(lower, point, nulls):
    """Ec. 91 (v07): passes iff the lower bound exceeds every null pooled AUROC; p = (b + 1) / (R_pi + 1) with b the
    number of nulls at or above the point estimate (descriptive)."""
    nulls = np.asarray(nulls, float)
    return {"passes": bool(lower > nulls.max()), "null_max": float(nulls.max()),
            "p": float(((nulls >= point).sum() + 1) / (len(nulls) + 1)), "R_pi": len(nulls)}


def testable_h1(res_passes, comparator_passes):
    """Ec. 100: (testable, reason or None, M_ok)."""
    ok = [m for m, v in comparator_passes.items() if v]
    if not res_passes:
        return False, "the full model failed Ec. 91", ok
    if len(ok) < 2:
        return False, "at most one comparator passed Ec. 91", ok
    return True, None, ok


def h1_verdict(ell, m_ok, testable, delta=DELTA_NI, h5_supported=True):
    """Ec. 86 and the v07 E1 hypothesis matrix; ell for every comparator is reported, the decision reads M_ok."""
    clears = {m: bool(ell[m] > -delta) for m in m_ok}
    if not testable:
        verdict = "inconclusive"
    elif not h5_supported:
        verdict = "estimate"  # fixed sequence of Ec. 113: H1 read as an estimate when H5 is not supported
    elif all(clears.values()):
        verdict = "supported"
    elif any(clears.values()):
        verdict = "partially supported"
    else:
        verdict = "not supported"
    return {"verdict": verdict, "clears": clears, "ell": dict(ell), "delta_NI": delta}


def attainability(se, delta=DELTA_NI, alpha=ALPHA):
    """Ec. 87: attainable when delta >= (z_{1-alpha} + z_{1-beta}) SE, at 80% and 50% power."""
    za = sps.norm.ppf(1 - alpha)
    m80, m50 = za + sps.norm.ppf(0.80), za + sps.norm.ppf(0.50)
    return {"se": float(se), "multiplier_80": float(m80), "multiplier_50": float(m50),
            "attainable_80": bool(delta >= m80 * se), "attainable_50": bool(delta >= m50 * se),
            "se_max_80": float(delta / m80), "se_max_50": float(delta / m50)}


def ss_reference(S, y, test_folds):
    """For one repetition: omega_b (omega_+ = M0/M1 of the training set of b's fold) and the class-weighted training
    item mean s_bar^{omega, tr(b)} (Ec. 93), per held-out subject."""
    S, y = np.asarray(S, float), np.asarray(y)
    w, ref = np.empty(len(y)), np.empty_like(S)
    for te in test_folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        ow = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        wt = np.where(y[tr] == 1, ow, 1.0)
        w[te] = np.where(y[te] == 1, ow, 1.0)
        ref[te] = (wt[:, None] * S[tr]).sum(0) / wt.sum()
    return w, ref


def ss_inputs(S_hat_q, S, y, folds_q):
    """Repetition-averaged inputs of Ec. 93: predictions (seed-ensembled per repetition), weights and references are
    each averaged over the Q repetitions (each repetition has its own folds)."""
    per = [ss_reference(S, y, f) for f in folds_q]
    return np.mean(S_hat_q, axis=0), np.mean([p[0] for p in per], axis=0), np.mean([p[1] for p in per], axis=0)


def _ss_terms(S_hat, S, w, ref):
    return w * ((S_hat - S) ** 2).sum(1), w * ((ref - S) ** 2).sum(1)


def ss_omega(S_hat, S, w, ref):
    num, den = _ss_terms(np.asarray(S_hat, float), np.asarray(S, float), np.asarray(w, float), np.asarray(ref, float))
    return float(1 - num.sum() / den.sum())


def boot_ss_omega(S_hat, S, w, ref, I):
    """(B,) SS^omega on the shared resamples (Ec. 99); differences of two models use the same I."""
    num, den = _ss_terms(np.asarray(S_hat, float), np.asarray(S, float), np.asarray(w, float), np.asarray(ref, float))
    return 1 - num[I].sum(1) / den[I].sum(1)
