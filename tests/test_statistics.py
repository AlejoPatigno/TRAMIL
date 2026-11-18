import numpy as np
from scipy import stats as sps

from TRAMIL import statistics as st


def test_fisher_ratio_is_squared_mean_gap_over_summed_variances():
    assert st.fisher_ratio([1, 2, 3], [3, 4, 5]) == 2.0  # (2)^2 / (1 + 1)
    assert st.fisher_ratio([1, 2, 3], [1, 2, 3]) == 0.0


def test_corrected_t_deflates_the_ordinary_paired_t_by_the_overlap_factor():
    d = np.array([0.03, 0.01, 0.05, -0.02, 0.04, 0.02, 0.0, 0.03, 0.01, 0.06, -0.01, 0.02, 0.03, 0.04, 0.01])
    R = len(d)
    t, p, df = st.corrected_t(d)
    ordinary = sps.ttest_1samp(d, 0.0)
    assert df == R - 1
    assert np.isclose(t, ordinary.statistic * np.sqrt((1 / R) / (1 / R + 0.25)), rtol=1e-9)
    assert abs(t) < abs(ordinary.statistic) and p > ordinary.pvalue
    manual = d.mean() / np.sqrt((1 / R + 0.25) * d.var(ddof=1))
    assert np.isclose(t, manual)


def test_tost_interval_is_the_90_percent_corrected_interval_inside_the_margin():
    d = np.array([0.004, -0.006, 0.01, 0.0, -0.003, 0.007, 0.002, -0.001, 0.005, 0.0, -0.004, 0.003, 0.006, -0.002, 0.001])
    lo, hi, eq, below = st.tost(d, delta=0.05)
    R = len(d)
    half = sps.t.ppf(0.95, R - 1) * np.sqrt((1 / R + 0.25) * d.var(ddof=1))
    assert np.isclose(lo, d.mean() - half) and np.isclose(hi, d.mean() + half)
    assert eq and not below
    # an interval crossing the margin is inconclusive, not equivalent
    wide = d * 12
    assert not st.tost(wide, delta=0.05)[2]
    # entirely below -delta: refuted
    assert st.tost(np.full(15, -0.2) + 0.001 * np.arange(15), delta=0.05)[3]


def test_tost_decision_agrees_with_the_two_one_sided_p_values():
    g = np.random.default_rng(0)
    for scale in (0.005, 0.02, 0.05):
        d = g.normal(0.0, scale, 15)
        _, _, eq, _ = st.tost(d, delta=0.05, alpha=0.05)
        assert eq == (st.tost_p(d, delta=0.05) < 0.05)


def test_holm_adjustment_is_step_down_and_monotone():
    adj = st.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert np.allclose(st.holm([0.5]), [0.5])
    assert max(st.holm([0.4, 0.9, 0.8])) <= 1.0
