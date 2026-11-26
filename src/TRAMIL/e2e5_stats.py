"""Analyses of E2 to E5 of experiments v07 on out-of-fold results (r22 Section 18.1 and 18.7): item-profile metrics
(Ec. 61, 88, 92, 98, 78), Benjamini-Hochberg adjustment, one-sided bootstrap p, score variance of the nine runs (Ec. 89)
with the Fisher and Brown-Forsythe descriptive readings (Ec. 64, 65), attention stability with the complete-case and
zero-filled readings (Ec. 95, 101 and the undefined-correlation rule of Section 18.1), rank-free pairwise total
variation, relative effective attention (Ec. 52) and the edge diagnostics (Ec. 107) from model parts. Pure numpy/scipy
functions of arrays; the subject bootstrap indices are those of pooled_stats (shared by every arm)."""
import warnings

import numpy as np
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score

__provenance__ = {
    "revision": "research-concept-residual-semantic-prior-bipartite-mil-r22.md",
    "sections": ["16.2", "18.1", "18.7"],
    "equations": [51, 52, 61, 64, 65, 78, 85, 88, 89, 92, 95, 98, 101, 107],
    "experiments": ["E2", "E3", "E4", "E5"],
    "invariants": [
        "bh_adjustment_matches_the_step_up_definition_and_is_monotone",
        "profile_metrics_match_hand_computed_skill_scores",
        "shape_skill_compares_with_the_models_own_total_spread_over_the_item_means",
        "stability_is_the_mean_pairwise_rank_correlation_over_defined_pairs",
        "constant_columns_are_undefined_and_complete_case_and_zero_filled_readings_differ",
        "log_variance_ratio_is_positive_for_the_noisier_model_and_bootstrapped_on_shared_resamples",
        "percentile_ranks_are_average_ranks_over_the_held_out_subjects_of_a_run",
        "pairwise_tv_is_zero_for_identical_runs",
        "effective_attention_is_one_for_uniform_columns",
        "declared_profile_subjects_are_chosen_by_total_only",
    ],
}

RTOL, ATOL = 1e-7, 1e-12  # tolerances of Ec. 44 for a constant attention column
ALPHA = 0.05


def bh(p):
    """Benjamini-Hochberg adjusted p values (step up), in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def p_one_sided(boot_diff):
    """Bootstrap p of H0: difference <= 0, (1 + #{b*: d* <= 0}) / (B + 1)."""
    d = np.asarray(boot_diff)
    return float((1 + (d <= 0).sum()) / (len(d) + 1))


def lower(boot, alpha=ALPHA):
    return float(np.quantile(boot, alpha))


# ---------------- item profile (Ec. 88, 92, 98, 61, 78) ----------------
def plain_inputs(S_hat_q, S, folds_q):
    """Repetition-averaged predictions and training item means per held-out subject (the reference of Ec. 88)."""
    S = np.asarray(S, float)
    refs = []
    for folds in folds_q:
        ref = np.empty_like(S)
        for te in folds:
            ref[te] = S[np.setdiff1d(np.arange(len(S)), te)].mean(0)
        refs.append(ref)
    return np.mean(S_hat_q, axis=0), np.mean(refs, axis=0)


def ss_plain(S_hat, S, ref):
    """SS (Ec. 88): 1 - sum (s_hat - s)^2 / sum (ref - s)^2 over subjects and items."""
    return float(1 - ((S_hat - S) ** 2).sum() / ((ref - S) ** 2).sum())


def ss_items(S_hat, S, ref):
    """Per-item skill (Ec. 92), (8,)."""
    return 1 - ((S_hat - S) ** 2).sum(0) / ((ref - S) ** 2).sum(0)


def ss_shape(S_hat, S, ref):
    """Within-model shape skill (Ec. 98): the reference spreads the model's own predicted total over the item means."""
    T = S_hat.sum(1, keepdims=True)
    tilde = ref * T / ref.sum(1, keepdims=True)
    return float(1 - ((S_hat - S) ** 2).sum() / ((tilde - S) ** 2).sum())


def mae(S_hat, S):
    """Per-subject mean absolute item error in PHQ-8 points, averaged over subjects."""
    return float(np.abs(S_hat - S).mean())


def weighted_kappa(S_hat, S):
    """Quadratic weighted kappa between the classes of Ec. 78 and the true item scores, mean over the items where it
    is defined."""
    c = np.clip(np.rint(S_hat), 0, 3).astype(int)
    k = [cohen_kappa_score(S[:, j].astype(int), c[:, j], weights="quadratic", labels=[0, 1, 2, 3]) for j in range(8)]
    k = [v for v in k if np.isfinite(v)]
    return float(np.mean(k)) if k else float("nan")


def item_spearman(S_hat, S):
    """Per-item Spearman between predicted and true item scores across subjects (Ec. 61): (rho (8,), p (8,)); a constant
    prediction gives nan."""
    rho, p = np.full(8, np.nan), np.full(8, np.nan)
    for j in range(8):
        if np.ptp(S_hat[:, j]) > 0 and np.ptp(S[:, j]) > 0:
            rho[j], p[j] = sps.spearmanr(S_hat[:, j], S[:, j])
    return rho, p


def declared_profile_subjects(S, subjects):
    """Qualitative panel of E2: one subject per decile of the true PHQ-8 total from the second to the ninth, chosen by
    total only (the subject at the middle of each decile of the totals sorted by total then identifier), before any
    prediction is read."""
    T = np.asarray(S).sum(1)
    order = sorted(range(len(T)), key=lambda i: (T[i], subjects[i]))
    n = len(order)
    return [subjects[order[int((d + 0.5) * n / 10)]] for d in range(1, 9)]


# ---------------- scores of the nine runs (Ec. 89, 64, 65) ----------------
def percentile_ranks(runs):
    """(R, M) scores -> average ranks / M within each run's held-out subjects (scale-free; ties averaged)."""
    runs = np.asarray(runs, float)
    return np.stack([sps.rankdata(r) / len(r) for r in runs])


def subject_variance(runs):
    """Per-subject variance of the score over the nine runs, (M,) (Ec. 89, sample variance)."""
    return np.var(np.asarray(runs, float), axis=0, ddof=1)


def log_variance_ratio(runs_m, runs_ref, I):
    """(point, bootstrap (B,)) of log(V_m / V_res), V = mean over subjects of the per-subject run variance (Ec. 89)."""
    vm, vr = subject_variance(runs_m), subject_variance(runs_ref)
    return float(np.log(vm.mean() / vr.mean())), np.log(vm[I].mean(1) / vr[I].mean(1))


def fisher_brown_forsythe(a_res, b_m):
    """Ec. 64, 65 on the seed-level run AUROCs (descriptive): F = s2_m / s2_res with its one-sided p, and W with its p."""
    a, b = np.asarray(a_res, float), np.asarray(b_m, float)
    F = float(b.var(ddof=1) / a.var(ddof=1)) if a.var(ddof=1) > 0 else float("nan")
    W, pw = sps.levene(a, b, center="median")
    return {"F": F, "p_F": float(sps.f.sf(F, len(b) - 1, len(a) - 1)) if np.isfinite(F) else float("nan"),
            "W": float(W), "p_W": float(pw)}


# ---------------- attention (Ec. 51, 52, 95, 101, 107) ----------------
def column_ranks(A):
    """(N, 8) attention columns -> (average ranks, constant flags) with the constant rule of Ec. 44."""
    A = np.asarray(A, float)
    const = (A.max(0) - A.min(0)) <= ATOL + RTOL * A.max(0)
    return np.stack([sps.rankdata(A[:, j]) for j in range(A.shape[1])], 1), const


def stab_subject(ranks, const):
    """Stab_bj (Ec. 95) of one subject: ranks (V, N, 8), const (V, 8) over the V runs -> (8,) mean Spearman over the
    pairs of runs where both columns are defined; nan if no pair is defined (Section 18.1)."""
    ranks = np.asarray(ranks, float)
    V, N, J = ranks.shape
    iu = np.triu_indices(V, 1)
    out = np.full(J, np.nan)
    for j in range(J):
        R = ranks[:, :, j]
        Z = R - R.mean(1, keepdims=True)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        ok = ~np.asarray(const)[:, j] & (nrm[:, 0] > 0)
        C = (Z / np.where(nrm > 0, nrm, 1)) @ (Z / np.where(nrm > 0, nrm, 1)).T
        good = ok[iu[0]] & ok[iu[1]]
        if good.any():
            out[j] = C[iu][good].mean()
    return out


def stab_contrast(Sa, Sb, I):
    """Ec. 101 for arm `a` (the full model) minus arm `b`: Sa, Sb (M, 8) Stab_bj with nan for undefined pairs.
    Complete case: items defined in both arms, subjects with none dropped. Zero-filled: undefined set to 0. Returns the
    two readings (difference, lower bound, bootstrap) and the dropped fraction per arm and per item."""
    Sa, Sb = np.asarray(Sa, float), np.asarray(Sb, float)
    both = ~np.isnan(Sa) & ~np.isnan(Sb)
    d = np.where(both, Sa - Sb, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = np.nanmean(d, axis=1)
    ok = ~np.isnan(cc)
    c0 = np.nan_to_num(cc)
    boot_cc = (c0[I] * ok[I]).sum(1) / np.maximum(ok[I].sum(1), 1)
    z = (np.nan_to_num(Sa) - np.nan_to_num(Sb)).mean(1)
    boot_z = z[I].mean(1)
    return {"complete_case": {"diff": float(c0[ok].mean()) if ok.any() else float("nan"), "lower": lower(boot_cc), "boot": boot_cc, "n": int(ok.sum())},
            "zero_filled": {"diff": float(z.mean()), "lower": lower(boot_z), "boot": boot_z, "n": len(z)},
            "dropped_a": float(np.isnan(Sa).mean()), "dropped_b": float(np.isnan(Sb).mean()),
            "dropped_a_item": np.isnan(Sa).mean(0).tolist(), "dropped_b_item": np.isnan(Sb).mean(0).tolist()}


def mean_pairwise_tv(cols):
    """Rank-free companion of Ec. 95: mean over the pairs of runs of the total variation distance between the attention
    columns, averaged over the eight items. cols: (V, N, 8)."""
    cols = np.asarray(cols, float)
    iu = np.triu_indices(len(cols), 1)
    tv = 0.5 * np.abs(cols[iu[0]] - cols[iu[1]]).sum(1)  # (pairs, 8)
    return float(tv.mean())


def feff(A):
    """Ec. 52: F_eff = (1 / 8N) sum_j exp(entropy of column j); A is (N, 8) on the simplex per column."""
    A = np.asarray(A, float)
    H = -(A * np.log(np.where(A > 0, A, 1))).sum(0)
    return float(np.exp(H).sum() / (A.shape[1] * len(A)))


def edge_diagnostics(G_hat, G, Delta):
    """Ec. 107 for one bag: Al_b = ||(G_hat - G) * (1 1^T - I)||_F^2 / N^2 and RMS^Delta_b = ||Delta||_F / N."""
    N = len(G)
    off = 1 - np.eye(N)
    return (float((((G_hat - G) * off) ** 2).sum() / N**2), float(np.linalg.norm(Delta) / N))


def spearman_fidelity(Rc_hat, Rc):
    """Surrogate fidelity of one subject: per-item Spearman between R_hat^c and R^c across the segments, (8,) with nan
    for a constant column."""
    out = np.full(8, np.nan)
    for j in range(8):
        if len(Rc) > 2 and np.ptp(Rc_hat[:, j]) > 0 and np.ptp(Rc[:, j]) > 0:
            out[j] = sps.spearmanr(Rc_hat[:, j], Rc[:, j])[0]
    return out
