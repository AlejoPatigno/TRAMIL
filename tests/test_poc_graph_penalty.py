import numpy as np
import torch

from TRAMIL import poc_graph_penalty as pg


def _offdiag(M):
    M = np.asarray(M)
    return M[~np.eye(len(M), dtype=bool)]


def test_centering_spreads_the_prior_graph_off_diagonals():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 8, 40)
    R = torch.tensor(0.6 + 0.03 * np.eye(8)[z] + 0.005 * rng.standard_normal((40, 8)), dtype=torch.float32)
    raw = torch.nn.functional.normalize(R, dim=1)
    assert np.median(_offdiag(raw @ raw.T)) > 0.99  # raw rows near-identical
    G = pg.prior_graph(R)
    assert torch.allclose(pg.center(R).mean(0), torch.zeros(8), atol=1e-6)
    off = _offdiag(G)
    assert np.percentile(off, 5) < -0.05 and np.percentile(off, 95) > 0.5


def test_pi_rows_sum_to_one_and_zero_delta_gives_the_prior_softmax():
    torch.manual_seed(0)
    H, R = torch.randn(25, pg.D), torch.rand(25, 8)
    G = pg.prior_graph(R)
    layer = pg.GraphLayer()
    _, Pi, _ = layer(H, G, kappa_tau=3.0)
    assert torch.allclose(Pi.sum(1), torch.ones(25), atol=1e-5)
    _, Pi0, Delta0 = layer(H, G, kappa_tau=3.0, prior_only=True)
    assert torch.count_nonzero(Delta0) == 0
    assert torch.allclose(Pi0, torch.softmax(3.0 * G, dim=1))


def test_row_js_is_zero_for_identical_rows_and_positive_otherwise():
    P = torch.softmax(torch.randn(10, 10), dim=1)
    assert pg.row_js(P, P) < 1e-7
    A = pg.acoustic_adjacency(torch.tensor([0, 0, 1, 1, 2]))
    assert torch.allclose(A.sum(1), torch.ones(5))
    assert 0 < pg.row_js(P[:5, :5], A) <= np.log(2) + 1e-6


def test_generator_scenarios_have_the_stated_agreement():
    for scenario, lo, hi in [("agree", 1.0, 1.0), ("partial", 0.45, 0.55), ("disagree", 0.08, 0.17)]:
        bags = pg.make_data(seed=1, scenario=scenario, n_bags=40)
        zt = torch.cat([b["z_text"] for b in bags])
        za = torch.cat([b["z_ac"] for b in bags])
        frac = (zt == za).float().mean().item()
        assert lo <= frac <= hi, (scenario, frac)
        assert all(20 <= len(b["z_text"]) <= 60 for b in bags)


def test_very_large_mu_drives_delta_to_zero():
    bags = pg.make_data(seed=0, scenario="disagree", n_bags=8)
    torch.manual_seed(0)
    model0 = pg.Model()
    before = pg.evaluate(model0, bags, 5.0)["delta_rms"]
    model = pg.train(bags, mu=1e4, kappa_tau=5.0, seed=0, epochs=60)
    after = pg.evaluate(model, bags, 5.0)["delta_rms"]
    assert before > 0.05 and after < 0.05 * before  # Adam step floor keeps a small residual


def test_default_cluster_sd_reproduces_v1_generator_output():
    # fingerprint recorded from the v1 generator (commit 82db92f) before s was added
    for bags in (pg.make_data(3, "partial", 2), pg.make_data(3, "partial", 2, s=0.5)):
        assert np.allclose([float(b["H"].sum()) for b in bags], [-34.3952, -15.6295], atol=2e-4)
        assert np.allclose(bags[0]["H"][0, :3].tolist(), [-0.458343, -4.319282, 0.695930], atol=1e-5)
        assert bags[0]["z_ac"][:5].tolist() == [0, 6, 3, 2, 2]


def test_larger_cluster_sd_separates_acoustic_clusters_relative_to_noise():
    def sep(s):
        bags = pg.make_data(0, "disagree", 20, s=s)
        H, z = torch.cat([b["H"] for b in bags]), torch.cat([b["z_ac"] for b in bags])
        means = torch.stack([H[z == j].mean(0) for j in range(pg.J)])
        within = torch.cat([H[z == j] - means[j] for j in range(pg.J)]).var().item()
        return means.var(0).mean().item() / within

    ratios = [sep(s) for s in (0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] > 10 * ratios[0]
    a, b = pg.make_data(0, "disagree", 3, s=0.5), pg.make_data(0, "disagree", 3, s=4.0)
    assert all(torch.equal(x["z_ac"], y["z_ac"]) and torch.equal(x["R"], y["R"]) for x, y in zip(a, b))


def test_row_js_is_finite_when_both_rows_have_exact_zeros():
    P = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    A = pg.acoustic_adjacency(torch.tensor([0, 0, 1]))[:2]  # rows [0.5, 0.5, 0]
    js = pg.row_js(P, A)
    assert np.isfinite(js)
    expected = 0.5 * (0.5 * (np.log(1 / 0.75) + 0.5 * np.log(0.5 / 0.75) + 0.5 * np.log(2)))  # row 2 is 0
    assert abs(js - expected) < 1e-6
    assert pg.row_js(A, A) == 0.0
