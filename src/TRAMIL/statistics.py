"""Paired-replicate statistics of E1: Fisher ratio (Ec. 62), corrected resampled t (Ec. 63), two one-sided tests
against the equivalence margin (Ec. 69) and Holm correction."""
import numpy as np
from scipy import stats as sps

__provenance__ = {
    "revision": "experiments-experimental-plan-for-the-residual-semantic-prior-bipartite-v04.md",
    "sections": ["E1"],
    "experiments": ["E1"],
    "equations": [62, 63, 69],
    "invariants": [
        "fisher_ratio_is_squared_mean_gap_over_summed_variances",
        "corrected_t_deflates_the_ordinary_paired_t_by_the_overlap_factor",
        "tost_interval_is_the_90_percent_corrected_interval_inside_the_margin",
        "tost_decision_agrees_with_the_two_one_sided_p_values",
        "holm_adjustment_is_step_down_and_monotone",
    ],
}

TEST_TO_TRAIN = 0.25  # n_te / n_tr of the 5-fold outer partition
DELTA = 0.05  # equivalence margin in AUROC, fixed before any run


def fisher_ratio(a, b):
    """J_F = (mu_a - mu_b)^2 / (var_a + var_b) with sample variances (descriptive)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) ** 2 / (a.var(ddof=1) + b.var(ddof=1)))


def _se(d, ratio):
    d = np.asarray(d, float)
    return np.sqrt((1.0 / len(d) + ratio) * d.var(ddof=1))


def corrected_t(d, ratio=TEST_TO_TRAIN):
    """Corrected resampled t over the paired differences d_r (Ec. 63): (t, two-sided p, degrees of freedom)."""
    d = np.asarray(d, float)
    t = d.mean() / _se(d, ratio)
    df = len(d) - 1
    return float(t), float(2 * sps.t.sf(abs(t), df)), df


def tost(d, delta=DELTA, alpha=0.05, ratio=TEST_TO_TRAIN):
    """(lower, upper, equivalent, refuted_below): the 1 - 2 alpha interval of Ec. 69 against (-delta, delta)."""
    d = np.asarray(d, float)
    half = sps.t.ppf(1 - alpha, len(d) - 1) * _se(d, ratio)
    lo, hi = d.mean() - half, d.mean() + half
    return float(lo), float(hi), bool(lo > -delta and hi < delta), bool(hi < -delta)


def tost_p(d, delta=DELTA, ratio=TEST_TO_TRAIN):
    """p-value of the two one-sided tests: the larger of the two one-sided corrected-t p-values."""
    d = np.asarray(d, float)
    se, df = _se(d, ratio), len(d) - 1
    p_lower = sps.t.sf((d.mean() + delta) / se, df)  # H0: mean <= -delta
    p_upper = sps.t.sf((delta - d.mean()) / se, df)  # H0: mean >= delta
    return float(max(p_lower, p_upper))


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj
