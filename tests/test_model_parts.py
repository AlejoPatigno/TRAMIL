import math

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from TRAMIL import aggregation, corrector, encoder, model, proximity, standardize

torch.manual_seed(0)


def _unit(n, d=768, seed=0):
    g = np.random.default_rng(seed)
    x = g.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- proximity (Ec. 9, 10)
def test_proximity_is_cosine_mapped_to_unit_interval_without_row_normalisation():
    E = _unit(5)
    Q = np.concatenate([E[:1], -E[:1], _unit(6, seed=3)])
    R = proximity.proximity(E, Q)
    assert R.shape == (5, 8) and R.min() >= 0.0 and R.max() <= 1.0
    assert math.isclose(R[0, 0], 1.0, abs_tol=1e-5) and math.isclose(R[0, 1], 0.0, abs_tol=1e-5)
    assert not np.allclose(R.sum(1), 1.0)  # no softmax over the eight columns
    assert np.allclose(R, (np.clip(E @ Q.T, -1, 1) + 1) / 2, atol=1e-6)


# --- standardisation (Ec. 11, 12)
def test_scaler_matches_pooled_instance_statistics_and_keeps_constant_features():
    g = np.random.default_rng(1)
    bags = [g.standard_normal((n, 88)).astype(np.float32) * 3 + 2 for n in (4, 9, 2)]
    bags[0][:, 5] = 7.0
    for b in bags[1:]:
        b[:, 5] = 7.0
    mu, s = standardize.fit(bags)
    ref = StandardScaler().fit(np.concatenate(bags))
    assert np.allclose(mu, ref.mean_, atol=1e-5) and np.allclose(s, ref.scale_, atol=1e-5)
    assert s[5] == 1.0
    out = standardize.apply(bags[1], mu, s)
    assert np.allclose(out[:, 5], 0.0)


def test_scaler_is_fitted_on_the_bags_it_is_given_only():
    a = [np.zeros((3, 88), np.float32) + 1.0, np.zeros((3, 88), np.float32) + 3.0]
    b = [np.zeros((3, 88), np.float32) + 100.0]
    mu_a, _ = standardize.fit(a)
    mu_ab, _ = standardize.fit(a + b)
    assert np.allclose(mu_a, 2.0) and not np.allclose(mu_ab, mu_a)


# --- encoder (Ec. 13, 14)
def test_encoder_is_a_nonnegative_32d_relu_projection():
    enc = encoder.Encoder()
    H = enc(torch.randn(11, 88))
    assert H.shape == (11, 32) and (H >= 0).all() and (H == 0).any()
    X = torch.randn(3, 88)
    assert torch.allclose(enc(X), torch.relu(X @ enc.fc.weight.T + enc.fc.bias), atol=1e-6)


# --- corrector (Ec. 15-17, 22, 23, 57)
def test_corrector_starts_at_zero_correction_and_unit_prior_intensity():
    c = corrector.Corrector()
    H = torch.relu(torch.randn(6, 32))
    D = c(H)
    assert D.shape == (6, 8) and torch.all(D == 0)
    assert c.out.bias is None
    lam = corrector.PriorIntensity()
    assert math.isclose(float(lam()), 1.0, abs_tol=1e-6) and float(lam()) > 0
    assert corrector.PriorIntensity(n_nodes=8)().shape == (8,)


def test_initial_attention_equals_the_semantic_prior():
    N = 9
    R = torch.rand(N, 8)
    c, lam = corrector.Corrector(), corrector.PriorIntensity()
    A = aggregation.column_softmax(corrector.fuse(c(torch.relu(torch.randn(N, 32))), R, lam()))
    assert torch.allclose(A, corrector.prior_attention(R), atol=1e-6)


def test_attention_is_the_prior_reweighted_by_exp_correction():
    N = 7
    R, D = torch.rand(N, 8), torch.randn(N, 8)
    lam = torch.tensor(1.7)
    A = aggregation.column_softmax(corrector.fuse(D, R, lam))
    A0 = corrector.prior_attention(R)
    num = A0**lam * torch.exp(D)
    assert torch.allclose(A, num / num.sum(0, keepdim=True), atol=1e-5)  # Ec. 23
    assert not torch.allclose(A, A0, atol=1e-3)  # the reweighting is not the identity here


def test_fuse_with_per_node_intensity_broadcasts_over_columns():
    R, D = torch.rand(5, 8), torch.randn(5, 8)
    lam = torch.linspace(0.5, 2.0, 8)
    L = corrector.fuse(D, R, lam)
    assert torch.allclose(L[:, 3], D[:, 3] + lam[3] / corrector.TAU * R[:, 3])


# --- aggregation (Ec. 18-21, 24-31)
def test_attention_columns_are_positive_and_sum_to_one_over_instances():
    A = aggregation.column_softmax(torch.randn(12, 8) * 3)
    assert (A > 0).all() and torch.allclose(A.sum(0), torch.ones(8), atol=1e-6)


def test_node_representations_lie_in_the_convex_hull_of_instances():
    H = torch.relu(torch.randn(10, 32))
    A = aggregation.column_softmax(torch.randn(10, 8))
    Z = aggregation.propagate(A, H)
    assert Z.shape == (8, 32)
    assert (Z <= H.max(0).values + 1e-6).all() and (Z >= H.min(0).values - 1e-6).all()


def test_attention_maximises_the_kl_regularised_linear_objective():
    N = 6
    R, D = torch.rand(N, 8), torch.randn(N, 8)
    lam = 1.3
    P = torch.softmax(lam * R / corrector.TAU, dim=0)
    A = aggregation.column_softmax(corrector.fuse(D, R, torch.tensor(lam)))

    def obj(a):
        return (a * D).sum(0) - (a * (a / P).log()).sum(0)

    base = obj(A)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        a = torch.softmax(A.log() + 0.3 * torch.randn(N, 8, generator=g), dim=0)
        assert (obj(a) <= base + 1e-5).all()


def test_flatten_is_row_concatenation_in_descriptor_order():
    Z = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32)
    z = aggregation.flatten(Z)
    assert z.shape == (256,) and z[32] == Z[1, 0] and z[255] == Z[7, 31]


def test_classifier_is_linear_then_sigmoid_with_threshold_half():
    head = aggregation.Head(256)
    z = torch.rand(256)
    eta = head(z)
    assert torch.allclose(eta, z @ head.fc.weight[0] + head.fc.bias[0], atol=1e-6)
    assert aggregation.decide(torch.tensor([-0.1, 0.0, 0.2])).tolist() == [0, 1, 1]


# --- wiring
def test_model_is_invariant_to_joint_permutation_of_instances():
    m = model.ResidualBipartite()
    # make the corrector non-trivial so the invariance is not an artefact of D = 0
    torch.nn.init.normal_(m.corrector.out.weight, std=0.5)
    X, R = torch.randn(14, 88), torch.rand(14, 8)
    perm = torch.randperm(14)
    assert not torch.allclose(m.corrector(m.encoder(X)), torch.zeros(14, 8))
    a, b = m(X, R), m(X[perm], R[perm])
    assert torch.allclose(a, b, atol=1e-5)
    # control: permuting only X (not R) breaks the pairing and changes the output
    assert not torch.allclose(a, m(X[perm], R), atol=1e-5)


def test_parameter_counts_match_the_proposal_table():
    assert model.n_params(model.ResidualBipartite()) == 4418
    assert model.n_params(model.ResidualBipartite(learned_only=True)) == 4417
