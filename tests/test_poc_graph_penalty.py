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


# ---- v3: r12 bounded acoustic residual (Ec. 71, 77, 104-106) ----
from TRAMIL import poc_graph_penalty_v3 as v3  # noqa: E402


def test_v3_delta_is_bounded_with_masked_diagonal():
    torch.manual_seed(0)
    model = v3.ModelV3().double()
    torch.nn.init.eye_(model.W_a.weight)
    H = torch.rand(30, v3.DH, dtype=torch.float64)
    Delta = model.delta(H)
    assert torch.count_nonzero(torch.diagonal(Delta)) == 0
    assert Delta.abs().max().item() < v3.INV_TAU_A
    assert torch.allclose(Delta, Delta.T)
    # exactly parallel centered rows: cosine strictly below 1 thanks to eps, so |Delta| < 1/tau_a still holds
    Hc = H - H.mean(0)
    H2 = torch.stack([Hc[0], 2 * Hc[0], -Hc[0]] + list(Hc[3:])) + 5.0
    D2 = model.delta(H2)
    assert 0.999 * v3.INV_TAU_A < D2[0, 1].item() < v3.INV_TAU_A
    assert D2.abs().max().item() < v3.INV_TAU_A


def test_v3_pi_stays_within_the_ec105_band():
    torch.manual_seed(1)
    model = v3.ModelV3().double()
    X = torch.randn(40, v3.DX, dtype=torch.float64)
    G = pg.prior_graph(torch.rand(40, 8, dtype=torch.float64))
    for kappa_tau in (v3.KAPPA_TAU, 0.0):
        _, Pi, _, _ = model(X, G, kappa_tau)
        gamma = v3.INV_TAU_A + kappa_tau
        nPi = 40 * Pi
        assert nPi.min().item() > np.exp(-2 * gamma) and nPi.max().item() < np.exp(2 * gamma)
        assert torch.allclose(Pi.sum(1), torch.ones(40, dtype=torch.float64))
    _, Pi0, D0, _ = model(X, G, v3.KAPPA_TAU, prior_only=True)
    assert torch.count_nonzero(D0) == 0 and torch.allclose(Pi0, torch.softmax(v3.KAPPA_TAU * G, 1))


def test_v3_alignment_penalty_is_zero_when_acoustic_graph_equals_prior_off_diagonal():
    torch.manual_seed(2)
    R = torch.rand(25, 8, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(v3.DH, 8, dtype=torch.float64))  # orthonormal columns
    H = R @ Q.T  # centered H W_a = R^c Q^T with W_a = I, so G^a = G exactly (Ec. 106 attainability)
    model = v3.ModelV3().double()
    torch.nn.init.eye_(model.W_a.weight)
    G = pg.prior_graph(R)
    assert model.alignment_penalty(H, G).item() < 1e-20
    G_diag = G + torch.diag(torch.rand(25, dtype=torch.float64))  # diagonal is masked out
    assert model.alignment_penalty(H, G_diag).item() < 1e-20
    assert model.alignment_penalty(H, -G).item() > 0.1


def test_v3_alignment_penalty_gradient_reaches_W_a_only():
    torch.manual_seed(3)
    model = v3.ModelV3()
    b = v3.make_data_v3(0, 1, split=0)[0]
    _, _, _, H = model(b["X"], b["G"])
    model.alignment_penalty(H, b["G"]).backward()
    assert model.W_a.weight.grad is not None and model.W_a.weight.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        if name != "W_a.weight":
            assert p.grad is None or p.grad.abs().sum() == 0, name


def test_v3_generator_agreement_and_signal_dimensions():
    bags = v3.make_data_v3(seed=1, n_bags=60, split=0)
    zt, za = torch.cat([b["z_text"] for b in bags]), torch.cat([b["z_ac"] for b in bags])
    assert 0.73 <= (zt == za).float().mean().item() <= 0.77
    assert all(20 <= len(b["z_ac"]) <= 60 for b in bags)
    means = v3.cluster_means(1)
    assert means.shape == (v3.J, v3.DX)
    assert np.all(means[:, 4:] == 0) and np.all(np.abs(means[:, :4]) > 0)
    quiet = v3.make_data_v3(seed=1, n_bags=5, split=0, noise=0.0)
    assert all(torch.count_nonzero(b["X"][:, 4:]) == 0 for b in quiet)
    assert [len(v3.make_data_v3(0, n, split=k)) for k, n in enumerate((v3.N_TRAIN, v3.N_VAL, v3.N_TEST))] == [256, 64, 64]


def test_v3_item_severity_targets_drive_the_segment_signal():
    means = v3.cluster_means(4)
    u = means / np.linalg.norm(means, axis=1, keepdims=True)
    for b in v3.make_data_v3(seed=4, n_bags=5, split=2, noise=0.0):
        s = b["s"].numpy()
        assert s.shape == (v3.J,) and set(s.tolist()) <= {0, 1, 2, 3}
        za = b["z_ac"].numpy()
        expected = means[za] + v3.BETA * (s[za] - 1.5)[:, None] * u[za]
        assert np.allclose(b["X"].numpy(), expected, atol=1e-6)


def test_v3_mu_star_is_chosen_on_val_and_criteria_use_test():
    def run(val, loss, mass=0.5, ga=0.1):
        return dict(val_loss=val, loss=loss, nbr_mass_dis=mass, ga_dist=ga, max_abs_delta=1.0,
                    delta_diag_max=0.0, band_ok=True)

    grid = [0.0, 1.0]
    res = {"full": {"0.0": [run(1.0, 9.0, ga=0.3), run(2.0, 0.1, ga=0.3)],
                    "1.0": [run(2.0, 0.1, mass=0.9), run(1.0, 9.0, mass=0.9)]},
           "prior_only": [run(0, 5.0, mass=0.2)] * 2, "acoustic_only": [run(0, 6.0)] * 2}
    ver = v3._verdicts_v3(res, grid)
    assert ver["F1"]["mu_star"] == [0.0, 1.0]
    assert ver["F1"]["loss_mu_star"] == 9.0 and not ver["F1"]["pass"]  # test loss at val-chosen mu, not test-chosen
    assert abs(ver["F2"]["nbr_mass_mu_star"] - 0.7) < 1e-12
    assert abs(ver["F3"]["spearman"] + 1) < 1e-9 and ver["F3"]["pass"]
    assert ver["F4"]["pass"] and ver["overall"] == "FAIL"


# ---- v3b: validity gate (oracles calibrate the signal amplitude a) ----
def test_v3b_default_amplitude_reproduces_v3_and_a_scales_the_severity_signal():
    for x, y in zip(v3.make_data_v3(2, 3, split=1), v3.make_data_v3(2, 3, split=1, a=v3.BETA)):
        assert torch.equal(x["X"], y["X"])
    means = v3.cluster_means(5)
    u = means / np.linalg.norm(means, axis=1, keepdims=True)
    for b in v3.make_data_v3(5, 3, split=0, noise=0.0, a=4.0):
        za, s = b["z_ac"].numpy(), b["s"].numpy()
        assert np.allclose(b["X"].numpy(), means[za] + 4.0 * (s[za] - 1.5)[:, None] * u[za], atol=1e-5)


def test_v3b_oracle_nb_averages_true_acoustic_neighbours_per_item():
    b = v3.make_data_v3(6, 1, split=0)[0]
    F = v3.nb_features(b)
    assert F.shape == (v3.J, v3.DX)
    for j in range(v3.J):
        members = b["z_ac"] == j
        expected = b["X"][members].mean(0) if members.any() else torch.zeros(v3.DX)
        assert torch.allclose(F[j], expected)
    shuffled = dict(b, z_text=b["z_text"].flip(0), R=b["R"].flip(0))  # text side is ignored
    assert torch.equal(v3.nb_features(shuffled), F)
    full = lambda bags: [b for b in bags if len(set(b["z_ac"].tolist())) == v3.J]  # noqa: E731  absent item: unobservable
    W = v3.fit_oracle_nb(full(v3.make_data_v3(6, 64, split=0, noise=0.0)))
    assert v3.oracle_nb_mse(W, full(v3.make_data_v3(6, 16, split=1, noise=0.0))) < 1e-6  # noiseless: exact


def test_v3b_oracle_seg_uses_no_graph_aggregation():
    torch.manual_seed(7)
    model = v3.ModelV3()
    b = v3.make_data_v3(7, 1, split=0)[0]
    out1 = model(b["X"], b["G"], graph=False)[0]
    out2 = model(b["X"], -b["G"], graph=False)[0]
    assert torch.equal(out1, out2)
    out1.sum().backward()
    assert model.W_a.weight.grad is None and model.W_g.weight.grad is None


def test_v3b_gate_takes_the_smallest_passing_a_or_none():
    g = lambda nb, seg, const: dict(nb=nb, seg=seg, const=const)  # noqa: E731
    gates = {0.5: g(1.2, 1.25, 1.25), 1.0: g(0.9, 0.95, 1.25), 2.0: g(0.5, 0.9, 1.25), 4.0: g(0.1, 0.9, 1.25)}
    assert v3.choose_a(gates) == 2.0  # 1.0 passes V1 (0.9 <= 1.0) but fails V2 (0.9 > 0.855)
    assert v3.choose_a({1.0: g(1.0, 2.0, 1.25)}) == 1.0  # V1 boundary is inclusive
    assert v3.choose_a({0.5: g(1.2, 1.25, 1.25), 4.0: g(0.9, 0.95, 1.0)}) is None


def test_v3b_uniform_row_expectation():
    b = dict(z_ac=torch.tensor([0, 0, 0, 1]), z_text=torch.tensor([0, 1, 0, 0]))
    # disagreeing segments: 1 (2 neighbours of 3) and 3 (0 of 3)
    assert abs(v3.uniform_mass_dis([b]) - (2 / 3 + 0) / 2) < 1e-12


def test_v3b_verdicts_add_constant_and_uniform_comparisons():
    def run(val, loss, mass, ga=0.1):
        return dict(val_loss=val, loss=loss, nbr_mass_dis=mass, ga_dist=ga, max_abs_delta=1.0,
                    delta_diag_max=0.0, band_ok=True)

    res = {"full": {"0.0": [run(1.0, 1.0, 0.30, ga=0.3)], "1.0": [run(2.0, 0.5, 0.9)]},
           "prior_only": [run(0, 1.2, 0.2)], "acoustic_only": [run(0, 1.1, 0.5)],
           "constant": [dict(loss=0.9, val_loss=0.9)], "uniform_mass_dis": [0.25]}
    ver = v3._verdicts_v3b(res, [0.0, 1.0], a=2.0)
    assert ver["F1"]["pass"] and not ver["F1p"]["pass"] and ver["F1p"]["loss_constant"] == 0.9
    assert ver["F2"]["pass"] and ver["F2b"]["pass"] and ver["F2b"]["uniform_mass_dis"] == 0.25
    assert ver["overall"] == "FAIL" and ver["a"] == 2.0
    res["constant"] = [dict(loss=1.05, val_loss=1.05)]
    res["uniform_mass_dis"] = [0.35]
    ver = v3._verdicts_v3b(res, [0.0, 1.0], a=2.0)
    assert ver["F1p"]["pass"] and not ver["F2b"]["pass"] and ver["overall"] == "FAIL"
    res["uniform_mass_dis"] = [0.2]
    assert v3._verdicts_v3b(res, [0.0, 1.0], a=2.0)["overall"] == "PASS"
