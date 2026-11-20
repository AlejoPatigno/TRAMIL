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
